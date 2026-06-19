#!/usr/bin/env python3
"""
GIFT-Eval official evaluation script for contrastive forecasting model.

Uses the gift-eval library's Dataset class and gluonts evaluate_model()
to produce results directly comparable to the GIFT-Eval leaderboard.

Wraps our backbone + forecasting head as a gluonts-compatible predictor
that produces QuantileForecast objects (point forecast used for all quantiles).

Usage:
    GIFT_EVAL=~/workspaces/gift-eval-data \
    CUDA_VISIBLE_DEVICES=0 python eval_gift_eval_official.py \
        --backbone-path ../../checkpoints/tiny_fresh_best_gap.pth \
        --head-path ../../checkpoints/forecasting_head_50k_best_ema0.130.pth \
        --output-dir ../../results/contrastive_tiny

Output:
    - all_results.csv matching the official GIFT-Eval submission format
    - summary.txt with per-config MASE and aggregate scores
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import torch

# Suppress pandas FutureWarnings about deprecated freq aliases
warnings.filterwarnings("ignore", category=FutureWarning)

# --- gluonts imports ---
from gluonts.dataset import Dataset as GluonTSDataset
from gluonts.dataset.util import forecast_start
from gluonts.ev.metrics import (
    MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model, Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import get_seasonality

# --- gift_eval import ---
from gift_eval.data import Dataset as GiftDataset

# --- our model imports ---
# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import ConfigurableModel
from src.forecasting_head import (
    ForecastingHead,
    FORECAST_LEN,
    forecast_autoregressive,
    forecast_with_strategy,
)


# ============================================================================
# Configuration
# ============================================================================

BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
)

HEAD_CONFIG = dict(
    H=512, hidden_dim=128, num_gru_layers=2,
    forecast_len=FORECAST_LEN, dropout=0.1,
)

T_RAW = 1024       # Backbone context window
BACKBONE_C = 4     # Backbone expects 4 channels
MODEL_NAME = "contrastive_tiny"

# All 97 dataset configs from the GIFT-Eval benchmark
SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W "
    "hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W "
    "ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

MED_LONG_DATASETS = (
    "electricity/15T electricity/H "
    "solar/10T solar/H "
    "kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T "
    "M_DENSE/H "
    "ett1/15T ett1/H "
    "ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T "
    "bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

DATASET_PROPERTIES = {
    "m4_yearly": {"domain": "Econ/Fin", "frequency": "A", "num_variates": 1},
    "m4_quarterly": {"domain": "Econ/Fin", "frequency": "Q", "num_variates": 1},
    "m4_monthly": {"domain": "Econ/Fin", "frequency": "M", "num_variates": 1},
    "m4_weekly": {"domain": "Econ/Fin", "frequency": "W", "num_variates": 1},
    "m4_daily": {"domain": "Econ/Fin", "frequency": "D", "num_variates": 1},
    "m4_hourly": {"domain": "Econ/Fin", "frequency": "H", "num_variates": 1},
    "electricity": {"domain": "Energy", "frequency": "W", "num_variates": 1},
    "ett1": {"domain": "Energy", "frequency": "W", "num_variates": 7},
    "ett2": {"domain": "Energy", "frequency": "W", "num_variates": 7},
    "solar": {"domain": "Energy", "frequency": "W", "num_variates": 1},
    "hospital": {"domain": "Healthcare", "frequency": "M", "num_variates": 1},
    "covid_deaths": {"domain": "Healthcare", "frequency": "D", "num_variates": 1},
    "us_births": {"domain": "Healthcare", "frequency": "M", "num_variates": 1},
    "saugeen": {"domain": "Nature", "frequency": "M", "num_variates": 1},
    "temperature_rain": {"domain": "Nature", "frequency": "D", "num_variates": 1},
    "kdd_cup_2018": {"domain": "Nature", "frequency": "D", "num_variates": 1},
    "jena_weather": {"domain": "Nature", "frequency": "D", "num_variates": 21},
    "car_parts": {"domain": "Sales", "frequency": "M", "num_variates": 1},
    "restaurant": {"domain": "Sales", "frequency": "D", "num_variates": 1},
    "hierarchical_sales": {"domain": "Sales", "frequency": "W-WED", "num_variates": 1},
    "loop_seattle": {"domain": "Transport", "frequency": "D", "num_variates": 1},
    "sz_taxi": {"domain": "Transport", "frequency": "H", "num_variates": 1},
    "m_dense": {"domain": "Transport", "frequency": "D", "num_variates": 1},
    "bitbrains_fast_storage": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 2},
    "bitbrains_rnd": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 2},
    "bizitobs_application": {"domain": "Web/CloudOps", "frequency": "10S", "num_variates": 2},
    "bizitobs_service": {"domain": "Web/CloudOps", "frequency": "10S", "num_variates": 2},
    "bizitobs_l2c": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 7},
}


# Standard quantile levels used by GIFT-Eval
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# All metric objects used in official evaluation
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=QUANTILE_LEVELS),
]

CSV_HEADER = [
    "dataset", "model",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS", "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain", "num_variates",
]


# ============================================================================
# Predictor wrapper
# ============================================================================

class ContrastiveForecasterPredictor(RepresentablePredictor):
    """Wraps our backbone + forecasting head as a gluonts predictor.

    For each input series, runs autoregressive rollout and wraps
    the point forecast as a QuantileForecast (all quantiles = point forecast,
    since our model is deterministic).
    """

    def __init__(
        self,
        backbone: ConfigurableModel,
        head: ForecastingHead,
        prediction_length: int,
        device: torch.device,
        t_raw: int = T_RAW,
        backbone_c: int = BACKBONE_C,
        quantile_levels: Optional[List[float]] = None,
        strategy: str = 'A1',
    ):
        super().__init__(prediction_length=prediction_length)
        self.backbone = backbone
        self.head = head
        self.device = device
        self.t_raw = t_raw
        self.backbone_c = backbone_c
        self.quantile_levels = quantile_levels or QUANTILE_LEVELS
        self.strategy = strategy

        # Build forecast_keys: "mean" + quantile strings
        self.forecast_keys = ["mean"] + [str(q) for q in self.quantile_levels]

    def predict(self, dataset: GluonTSDataset, **kwargs) -> Iterator[Forecast]:
        for item in dataset:
            yield self.predict_item(item)

    def predict_item(self, item) -> QuantileForecast:
        target = np.asarray(item["target"], dtype=np.float32)

        # Handle NaN in context: forward-fill then back-fill
        if np.isnan(target).any():
            target = target.copy()
            mask = np.isnan(target)
            if mask.all():
                target[:] = 0.0
            else:
                first_valid = np.where(~mask)[0][0]
                target[:first_valid] = target[first_valid]
                for i in range(1, len(target)):
                    if np.isnan(target[i]):
                        target[i] = target[i - 1]

        # Prepare context: truncate/pad to t_raw, expand to backbone_c channels
        context = self._prepare_context(target)  # (1, t_raw, backbone_c)
        context = context.to(self.device)

        # Forecast using selected strategy
        with torch.no_grad():
            forecast_raw = forecast_with_strategy(
                self.strategy, self.backbone, self.head, context,
                horizon=self.prediction_length, device=self.device,
            )

        # forecast_raw is either:
        #   (prediction_length, C)              — MSE head (point forecast)
        #   (num_quantiles, prediction_length, C) — quantile head
        if forecast_raw.ndim == 3:
            # Quantile output. Channel 0, transpose to (Q, prediction_length).
            quantile_arrays = forecast_raw[:, :, 0].astype(np.float64)
            # Replace any NaN/inf with 0
            quantile_arrays = np.nan_to_num(
                quantile_arrays, nan=0.0, posinf=0.0, neginf=0.0)
            # Median (q=0.5 = level 5 of 9 → index 4) is the point/mean.
            median_idx = self.quantile_levels.index(0.5)
            point_forecast = quantile_arrays[median_idx]
            # forecast_keys are ["mean"] + str(quantile_levels). Build aligned arrays.
            forecast_arrays = np.stack(
                [point_forecast] +
                [quantile_arrays[self.quantile_levels.index(q)]
                 for q in self.quantile_levels],
                axis=0,
            )
        else:
            # MSE point forecast — broadcast across all quantile keys.
            point_forecast = forecast_raw[:, 0].astype(np.float64)
            point_forecast = np.nan_to_num(
                point_forecast, nan=0.0, posinf=0.0, neginf=0.0)
            n_keys = len(self.forecast_keys)
            forecast_arrays = np.stack([point_forecast] * n_keys, axis=0)

        # Compute start_date for the forecast (= end of context + 1)
        fstart = forecast_start(item)

        return QuantileForecast(
            forecast_arrays=forecast_arrays,
            forecast_keys=self.forecast_keys,
            start_date=fstart,
            item_id=item.get("item_id", None),
        )

    def _prepare_context(self, target: np.ndarray) -> torch.Tensor:
        """Prepare univariate series for backbone.

        Truncates/pads to t_raw and replicates to backbone_c channels.
        """
        n = len(target)
        if n >= self.t_raw:
            context = target[-self.t_raw:]
        else:
            pad_len = self.t_raw - n
            context = np.concatenate([
                np.full(pad_len, target[0], dtype=np.float32),
                target,
            ])

        # (t_raw,) -> (1, t_raw, 1) — single channel, no replication needed
        # encoder/transformer/rollout all operate per-channel independently
        context_t = torch.from_numpy(context).float()
        context_t = context_t.unsqueeze(0).unsqueeze(-1)  # (1, t_raw, 1)
        return context_t


# ============================================================================
# Dataset iteration helpers
# ============================================================================

def get_all_dataset_configs():
    """Build the list of (ds_name, term) pairs for all 97 configs."""
    short_list = SHORT_DATASETS.split()
    med_long_list = MED_LONG_DATASETS.split()
    all_ds = list(set(short_list + med_long_list))

    configs = []
    for ds_name in sorted(all_ds):
        configs.append((ds_name, "short"))
        if ds_name in med_long_list:
            configs.append((ds_name, "medium"))
            configs.append((ds_name, "long"))

    return configs


def get_ds_config_name(ds_name, term):
    """Get the official config name: 'pretty_name/freq/term'."""
    if "/" in ds_name:
        ds_key = ds_name.split("/")[0].lower()
        ds_freq = ds_name.split("/")[1]
    else:
        ds_key = ds_name.lower()
        ds_freq = DATASET_PROPERTIES[
            PRETTY_NAMES.get(ds_key, ds_key)]["frequency"]

    ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    return f"{ds_key}/{ds_freq}/{term}", ds_key


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Official GIFT-Eval evaluation")
    p.add_argument("--backbone-path", required=True)
    p.add_argument("--head-path", required=True)
    p.add_argument("--output-dir", default="results/contrastive_tiny")
    p.add_argument("--device", default="cuda")
    p.add_argument("--test-only", type=int, default=0,
                   help="If >0, evaluate only this many configs (for testing)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing partial all_results.csv")
    p.add_argument("--strategy", default="A1",
                   choices=["A1", "A2", "B1", "B2", "B3", "B3R", "B4"],
                   help="Forecast rollout strategy (default: A1)")
    p.add_argument("--forecast-len", type=int, default=128,
                   help="Head forecast length: 128 (default) or 16 for W-heads")
    p.add_argument("--encoder-type", default=None,
                   choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv"],
                   help="Override backbone encoder type (must match checkpoint)")
    p.add_argument("--rev-norm-kind", default="ewma",
                   choices=["ewma", "revin", "none"],
                   help="Reversible norm variant — MUST match the backbone's "
                        "training-time choice. Both have 0 params so state_dict "
                        "doesn't disambiguate. Default 'ewma'.")
    p.add_argument("--rev-norm-span", type=int, default=32,
                   help="Span for RevEWMNorm (only used when "
                        "--rev-norm-kind=ewma). Must match training-time value.")
    p.add_argument("--patch-stats", default="auto",
                   choices=["auto", "none", "diff", "raw"],
                   help="Backbone patch-stats setting; 'auto' (default) "
                        "detects from the encoder's input width.")
    p.add_argument("--config-filter", default=None,
                   help="If set, only evaluate configs whose '<dataset>/<term>' "
                        "name matches this regex. Useful for cheap screens — "
                        "e.g. --config-filter 'ett[12]/(15T|W)|solar/10T|m4_hourly/H' "
                        "for the 6-config periodic focus set.")
    # Backbone architecture overrides — match the values used when training
    # the backbone. If omitted, the historical defaults (T_RAW=1024,
    # BACKBONE_C=4, H=512, nhead=8, num_layers=6) are used and the
    # state_dict load will fail with a size mismatch on a non-default backbone.
    p.add_argument("--t-raw", type=int, default=None,
                   help="Backbone context window. Overrides T_RAW (default 1024).")
    p.add_argument("--n-channels", "--backbone-c", dest="n_channels", type=int,
                   default=None,
                   help="Backbone input channel count C. Overrides "
                        "BACKBONE_C / BACKBONE_CONFIG['C'] (default 4).")
    p.add_argument("--d-model", type=int, default=None,
                   help="Backbone hidden width H. Overrides "
                        "BACKBONE_CONFIG['H'] (default 512).")
    p.add_argument("--n-heads", type=int, default=None,
                   help="Backbone transformer heads. Overrides "
                        "BACKBONE_CONFIG['nhead'] (default 8).")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Backbone transformer depth. Overrides "
                        "BACKBONE_CONFIG['num_layers'] (default 6).")
    p.add_argument("--forecaster-d-model", type=int, default=None,
                   help="Backbone forecaster bottleneck dim. None=same as d-model.")
    p.add_argument("--forecaster-n-heads", type=int, default=None,
                   help="Backbone forecaster heads. None=same as n-heads.")
    p.add_argument("--head-nhead", type=int, default=6,
                   help="Transformer head: number of attention heads "
                        "(default 6, mirrors backbone). Used only when the "
                        "head state dict is auto-detected as a transformer.")
    p.add_argument("--head-causal", default="true",
                   choices=["true", "false"],
                   help="Transformer head: causal mask (default true, "
                        "matches training-time semantics). State dict "
                        "is identical for causal vs bidir — pass the same "
                        "value used at training time.")
    return p.parse_args()


def load_models(args, device):
    """Load backbone and forecasting head."""
    # Backbone architecture overrides (CLI > defaults). HEAD_CONFIG['H']
    # follows the backbone's H so the head input width matches.
    global T_RAW, BACKBONE_C
    if args.t_raw is not None:
        T_RAW = args.t_raw
    if args.n_channels is not None:
        BACKBONE_CONFIG["C"] = args.n_channels
        BACKBONE_C = args.n_channels
    if args.d_model is not None:
        BACKBONE_CONFIG["H"] = args.d_model
        HEAD_CONFIG["H"] = args.d_model
    if args.n_heads is not None:
        BACKBONE_CONFIG["nhead"] = args.n_heads
    if args.num_layers is not None:
        BACKBONE_CONFIG["num_layers"] = args.num_layers
    if args.forecaster_d_model is not None:
        BACKBONE_CONFIG["forecaster_d_model"] = args.forecaster_d_model
    if args.forecaster_n_heads is not None:
        BACKBONE_CONFIG["forecaster_n_heads"] = args.forecaster_n_heads
    if args.encoder_type is not None:
        BACKBONE_CONFIG["encoder_type"] = args.encoder_type
    # Auto-detect freq_emb_dim and seasonality_emb_dim so backbones
    # trained with either / both axes load cleanly without CLI flags.
    sd = torch.load(args.backbone_path, map_location=device, weights_only=True)
    w = sd.get("freq_embedding.embedding.weight")
    BACKBONE_CONFIG["freq_emb_dim"] = (w.shape[1] if w is not None else 0)
    sw = sd.get("seasonality_embedding.embedding.weight")
    BACKBONE_CONFIG["seasonality_emb_dim"] = (sw.shape[1] if sw is not None else 0)
    if "log_inv_tau" in sd:
        BACKBONE_CONFIG["learnable_tau"] = True
        print(f"  [eval] auto-detected learnable τ "
              f"(log_inv_tau={sd['log_inv_tau'].item():.4f}, "
              f"τ={float((-sd['log_inv_tau']).exp()):.4f}) from backbone checkpoint")
    # Auto-detect num_encoder_layers from transformer.encoder_layers.<N>.* keys
    # (encoder-forecaster backbones from 2026-05-10). When absent → 0, which
    # matches the pre-encoder-stage default ConfigurableModel build.
    enc_layer_idxs = set()
    for k in sd:
        if k.startswith("transformer.encoder_layers."):
            try:
                enc_layer_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if enc_layer_idxs:
        BACKBONE_CONFIG["num_encoder_layers"] = max(enc_layer_idxs) + 1
        print(f"  [eval] auto-detected num_encoder_layers="
              f"{BACKBONE_CONFIG['num_encoder_layers']} from backbone checkpoint")
    # Auto-detect the b1024 collapse-fix norms (#322): QK-norm (q_norm/k_norm) and
    # attention-output RMSNorm (attn_out_rms) add per-layer params to encoder +
    # forecaster layers; build with the matching flags so _qk_aon backbones load
    # cleanly. Absent keys -> flags stay False (older backbones unaffected).
    if any(k.endswith(".q_norm.weight") for k in sd):
        BACKBONE_CONFIG["qk_norm"] = True
        print("  [eval] auto-detected qk_norm=True from backbone checkpoint")
    if any(k.endswith(".attn_out_rms.weight") for k in sd):
        BACKBONE_CONFIG["attn_out_norm"] = True
        print("  [eval] auto-detected attn_out_norm=True from backbone checkpoint")
    # Auto-detect #350 bilinear-main-loss W. main_w.weight in the state_dict ⇒
    # build the backbone with main_loss_bilinear=True so the matrix is
    # registered and Wᵀ is applied to the forecaster output in
    # extract_forecaster_latents / rollout_latent. Init value is irrelevant —
    # load_state_dict overwrites.
    if "main_w.weight" in sd:
        BACKBONE_CONFIG["main_loss_bilinear"] = True
        print("  [eval] auto-detected main_loss_bilinear=True (#350) from backbone checkpoint")
    # Auto-detect a CPC multi-step forecaster (#316). Two families:
    #   transformer.cpc_layers.<N>.*  → 'cpc'        (K transformer-1L heads, #1)
    #   transformer.cpc_heads.<N>.*   → 'linear_cpc' (K linear heads, #2/#3)
    lin_idxs = set()
    for k in sd:
        if k.startswith("transformer.cpc_heads."):
            try:
                lin_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if lin_idxs:
        BACKBONE_CONFIG["forecaster_kind"] = "linear_cpc"
        BACKBONE_CONFIG["cpc_k_steps"] = max(lin_idxs) + 1
        print(f"  [eval] auto-detected linear_cpc forecaster "
              f"(K={BACKBONE_CONFIG['cpc_k_steps']}) from checkpoint")
    cpc_head_idxs = set()
    for k in sd:
        if k.startswith("transformer.cpc_layers."):
            try:
                cpc_head_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if cpc_head_idxs:
        BACKBONE_CONFIG["forecaster_kind"] = "cpc"
        BACKBONE_CONFIG["cpc_k_steps"] = max(cpc_head_idxs) + 1
        w = sd.get("transformer.cpc_down.0.weight")
        if w is not None:
            BACKBONE_CONFIG["forecaster_d_model"] = w.shape[0]
        print(f"  [eval] auto-detected cpc forecaster "
              f"(K={BACKBONE_CONFIG['cpc_k_steps']}, "
              f"d={BACKBONE_CONFIG.get('forecaster_d_model')}) from checkpoint")
    BACKBONE_CONFIG["rev_norm_kind"] = args.rev_norm_kind
    if args.rev_norm_kind == "ewma":
        BACKBONE_CONFIG["rev_norm_span"] = args.rev_norm_span
    if BACKBONE_CONFIG["freq_emb_dim"] > 0:
        print(f"  [eval] auto-detected freq_emb_dim="
              f"{BACKBONE_CONFIG['freq_emb_dim']} from backbone checkpoint")
    if BACKBONE_CONFIG["seasonality_emb_dim"] > 0:
        print(f"  [eval] auto-detected seasonality_emb_dim="
              f"{BACKBONE_CONFIG['seasonality_emb_dim']} from backbone checkpoint")
    if args.rev_norm_kind != "ewma":
        print(f"  [eval] using rev_norm_kind={args.rev_norm_kind}")
    # Auto-detect patch_stats from the encoder's input width.
    if args.patch_stats == "auto":
        from src.norm import PATCH_STATS_DIM
        W = BACKBONE_CONFIG["W"]
        ref = sd.get("encoder.skip.weight")
        if ref is None:
            ref = sd.get("encoder.linear1.weight")
        if ref is None:
            args.patch_stats = "none"
        else:
            extra = (ref.shape[1] - W
                     - BACKBONE_CONFIG["freq_emb_dim"]
                     - BACKBONE_CONFIG["seasonality_emb_dim"])
            if extra == 0:
                args.patch_stats = "none"
            elif extra == PATCH_STATS_DIM:
                args.patch_stats = "diff"
            else:
                raise ValueError(
                    f"Unexpected encoder in_features={ref.shape[1]}: extra "
                    f"width={extra} doesn't match W ({W}) + freq_emb_dim "
                    f"({BACKBONE_CONFIG['freq_emb_dim']}) + seasonality_emb_dim "
                    f"({BACKBONE_CONFIG['seasonality_emb_dim']}) + 0 or "
                    f"{PATCH_STATS_DIM}.")
        print(f"  [eval] auto-detected patch_stats={args.patch_stats}")
    BACKBONE_CONFIG["patch_stats_kind"] = args.patch_stats
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    # CPC InfoNCE backbones (#344) carry a learnable `cpc_w1.*` used ONLY by
    # the pretraining loss; strip it for strict load. The #350 main_w stays —
    # it IS used at eval (Wᵀ applied to f_t in extract_forecaster_latents and
    # per-step in rollout_latent), and the backbone was built above with
    # main_loss_bilinear=True to receive it.
    sd = {k: v for k, v in sd.items() if not k.startswith("cpc_w1")}
    backbone.load_state_dict(sd)
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head_config = dict(HEAD_CONFIG)
    head_config['forecast_len'] = args.forecast_len
    head_sd = torch.load(args.head_path, map_location=device, weights_only=True)
    # Auto-detect quantile vs MSE: forecast_head.weight shape distinguishes
    #   MSE     : (forecast_len,                hidden_dim or H)
    #   quantile: (num_quantiles * forecast_len, hidden_dim or H)
    # Auto-detect GRU vs linear-probe: presence of `gru.*` keys in the state
    # dict. Linear probe is a single nn.Linear (PR ?? — qhead-improvements).
    fh_w = head_sd.get("forecast_head.weight")
    fh_out = fh_w.shape[0] if fh_w is not None else 0
    is_quantile = fh_out == args.forecast_len * 9
    is_gaussian = fh_out == args.forecast_len * 2
    is_transformer = any(
        k.startswith("transformer.layers.") for k in head_sd)
    is_linear = (not is_transformer
                 and not any(k.startswith("gru.") for k in head_sd))
    H = head_config["H"]
    if is_transformer and is_gaussian:
        layer_indices = sorted({
            int(k.split(".")[2]) for k in head_sd
            if k.startswith("transformer.layers.")})
        num_layers = max(layer_indices) + 1 if layer_indices else 6
        nhead = getattr(args, "head_nhead", 6)
        causal = getattr(args, "head_causal", "true") == "true"
        from src.forecasting_head import TransformerGaussianForecastingHead
        head = TransformerGaussianForecastingHead(
            H=H, num_layers=num_layers, nhead=nhead,
            forecast_len=args.forecast_len, causal=causal)
        print(f"  [eval] auto-detected transformer-gauss head "
              f"({num_layers}L H={H} nhead={nhead} "
              f"{'causal' if causal else 'bidir'})")
    elif is_transformer:
        if not is_quantile:
            raise ValueError(
                f"transformer head with forecast_head.weight shape[0]="
                f"{fh_out} doesn't match quantile (={args.forecast_len*9}) "
                f"or gaussian (={args.forecast_len*2})")
        layer_indices = sorted({
            int(k.split(".")[2]) for k in head_sd
            if k.startswith("transformer.layers.")})
        num_layers = max(layer_indices) + 1 if layer_indices else 6
        # nhead default = 6 (matches backbone H=384/64); CLI override available.
        nhead = getattr(args, "head_nhead", 6)
        causal = getattr(args, "head_causal", "true") == "true"
        from src.forecasting_head import TransformerQuantileForecastingHead
        head = TransformerQuantileForecastingHead(
            H=H, num_layers=num_layers, nhead=nhead,
            forecast_len=args.forecast_len, causal=causal)
        print(f"  [eval] auto-detected transformer-q head "
              f"({num_layers}L H={H} nhead={nhead} "
              f"{'causal' if causal else 'bidir'})")
    elif is_linear and is_quantile:
        from src.forecasting_head import LinearQuantileForecastingHead
        head = LinearQuantileForecastingHead(
            H=H, forecast_len=args.forecast_len)
        print(f"  [eval] auto-detected linear-probe quantile head")
    elif is_linear:
        from src.forecasting_head import LinearForecastingHead
        head = LinearForecastingHead(H=H, forecast_len=args.forecast_len)
        print(f"  [eval] auto-detected linear-probe MSE head")
    elif is_quantile:
        from src.forecasting_head import QuantileForecastingHead
        head = QuantileForecastingHead(**head_config)
        print(f"  [eval] auto-detected GRU quantile head (9 levels)")
    else:
        head = ForecastingHead(**head_config)
        print(f"  [eval] auto-detected GRU MSE head")
    head.load_state_dict(head_sd)
    head = head.to(device)
    head.eval()

    return backbone, head


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load models
    print("Loading models...")
    backbone, head = load_models(args, device)
    print(f"  Backbone: {args.backbone_path}")
    print(f"  Head: {args.head_path}")
    print(f"  Strategy: {args.strategy} (forecast_len={args.forecast_len})")

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "all_results.csv")
    summary_path = os.path.join(args.output_dir, "summary.txt")

    # Get all dataset configs
    all_configs = get_all_dataset_configs()
    if args.config_filter:
        import re
        pattern = re.compile(args.config_filter)
        kept = [(d, t) for (d, t) in all_configs if pattern.search(f"{d}/{t}")]
        skipped = len(all_configs) - len(kept)
        print(f"  --config-filter {args.config_filter!r}: kept {len(kept)} of "
              f"{len(all_configs)} configs (skipped {skipped})")
        all_configs = kept
    print(f"\nTotal configs to evaluate: {len(all_configs)}")

    # Handle resume: read existing results
    done_configs = set()
    existing_rows = []
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                done_configs.add(row[0])
                existing_rows.append(row)
        print(f"  Resuming: {len(done_configs)} configs already done")

    # Open CSV for writing
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)

        # Write back existing rows if resuming
        for row in existing_rows:
            writer.writerow(row)

        # Evaluate each config
        results_for_summary = []
        count = 0

        for ds_name, term in all_configs:
            config_name, ds_key = get_ds_config_name(ds_name, term)

            if config_name in done_configs:
                for row in existing_rows:
                    if row[0] == config_name:
                        results_for_summary.append(
                            (config_name, float(row[5])))
                        break
                continue

            if args.test_only and count >= args.test_only:
                break

            t0 = time.time()
            count += 1

            try:
                # Load dataset using gift_eval library
                to_univariate_check = GiftDataset(
                    name=ds_name, term=term, to_univariate=False)
                to_univariate = to_univariate_check.target_dim > 1
                dataset = GiftDataset(
                    name=ds_name, term=term, to_univariate=to_univariate)

                season_length = get_seasonality(dataset.freq)

                # Tag the backbone with this task's freq + seasonality so
                # extract_*_latents picks them up as defaults — no need to
                # thread kwargs through every forecast strategy.
                from src.freq_embedding import (
                    gluonts_freq_to_id, seasonality_to_id,
                )
                backbone._eval_freq_id = gluonts_freq_to_id(dataset.freq)
                backbone._eval_seasonality_id = seasonality_to_id(season_length)

                # Create predictor for this dataset
                predictor = ContrastiveForecasterPredictor(
                    backbone=backbone,
                    head=head,
                    prediction_length=dataset.prediction_length,
                    device=device,
                    quantile_levels=QUANTILE_LEVELS,
                    strategy=args.strategy,
                )

                # Evaluate using gluonts official function
                res = evaluate_model(
                    predictor,
                    test_data=dataset.test_data,
                    metrics=METRICS,
                    batch_size=512,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=season_length,
                )

                mase_val = res["MASE[0.5]"][0]

                # Write to CSV
                props = DATASET_PROPERTIES[ds_key]
                writer.writerow([
                    config_name,
                    MODEL_NAME,
                    res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    res["MSIS"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    props["domain"],
                    props["num_variates"],
                ])
                csvfile.flush()

                elapsed = time.time() - t0
                results_for_summary.append((config_name, mase_val))
                print(f"  [{count:3d}/{len(all_configs)}] "
                      f"{config_name:45s} "
                      f"MASE={mase_val:8.4f}  ({elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{count:3d}/{len(all_configs)}] "
                      f"{config_name:45s} "
                      f"FAILED: {e}  ({elapsed:.1f}s)")
                import traceback
                traceback.print_exc()

    # -------------------------------------------------------------------
    # Summary: compute aggregate MASE relative to Seasonal Naive
    # -------------------------------------------------------------------
    print(f"\nResults saved to {csv_path}")

    # Load seasonal naive reference results (check multiple locations)
    sn_candidates = [
        os.path.expanduser(
            "~/workspaces/gift-eval/results/seasonal_naive/all_results.csv"),
        os.path.join(project_root, "gift-eval-ref/seasonal_naive/all_results.csv"),
    ]
    sn_mase = {}
    for sn_path in sn_candidates:
        if os.path.exists(sn_path):
            sn_df = pd.read_csv(sn_path)
            for _, row in sn_df.iterrows():
                sn_mase[row["dataset"]] = row["eval_metrics/MASE[0.5]"]
            break

    # Compute relative MASE and geometric mean
    with open(summary_path, "w") as sf:
        def tee(msg):
            print(msg)
            sf.write(msg + "\n")

        tee("=" * 90)
        tee(f"{'GIFT-Eval Official Results':^90}")
        tee("=" * 90)
        tee(f"{'Config':<45} {'MASE':>8} {'SN_MASE':>8} "
            f"{'Relative':>10}")
        tee("-" * 90)

        log_ratios = []
        for config_name, mase_val in sorted(results_for_summary):
            sn_val = sn_mase.get(config_name, None)
            if sn_val is not None and sn_val > 0:
                relative = mase_val / sn_val
                log_ratios.append(np.log(relative))
                tee(f"{config_name:<45} {mase_val:>8.4f} "
                    f"{sn_val:>8.4f} {relative:>10.4f}")
            else:
                tee(f"{config_name:<45} {mase_val:>8.4f} "
                    f"{'N/A':>8} {'N/A':>10}")

        tee("-" * 90)

        if log_ratios:
            gm_relative = np.exp(np.mean(log_ratios))
            n_configs = len(log_ratios)
            tee(f"\nAggregate GM-Relative MASE ({n_configs} configs): "
                f"{gm_relative:.4f}")
            tee("")
            tee("Leaderboard comparison:")
            tee(f"  Sundial:    0.673")
            tee(f"  TimesFM:    0.680")
            tee(f"  PatchTST:   0.762")
            tee(f"  Chronos:    0.786")
            tee(f"  Moirai:     0.809")
            tee(f"  Naive:      1.000")
            tee(f"  ** Ours:    {gm_relative:.3f} **")
        else:
            tee("\nNo configs with matching Seasonal Naive reference.")

        tee("=" * 90)

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
