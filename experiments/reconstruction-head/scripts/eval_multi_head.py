#!/usr/bin/env python3
"""
Multi-head single-pass GIFT-Eval evaluation.

Runs the backbone + latent rollout ONCE per test item, then decodes
with all heads. One pass over the dataset, not one per head.

Usage:
    GIFT_EVAL=~/gift-eval-data PYTHONPATH=. python eval_multi_head.py \
        --backbone-path checkpoints/tiny_v2_best_gap.pth \
        --heads R2:checkpoints/R2_encoder_recon_w16_best.pth:16:B4 \
                R4:checkpoints/R5_encoder_recon_w128_best.pth:128:B3R \
        --device cuda
"""

import argparse
import csv
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from gluonts.dataset.util import forecast_start
from gluonts.ev.metrics import (
    MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset as GiftDataset

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import ConfigurableModel
from src.forecasting_head import (
    ForecastingHead,
    extract_encoder_latents,
    rollout_latent,
    _get_denorm_stats,
)

# -- Constants ----------------------------------------------------------------

BACKBONE_CONFIG = dict(
    C=4, H=512, W=16, encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)
T_RAW = 1024
C = 4
W = 16
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = [
    MSE(forecast_type="mean"), MSE(forecast_type=0.5),
    MAE(), MASE(), MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(), ND(),
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

# -- Dataset config -----------------------------------------------------------

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
    "saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts",
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


# -- Head config --------------------------------------------------------------

class HeadConfig:
    def __init__(self, name, path, forecast_len, strategy):
        self.name = name
        self.path = path
        self.forecast_len = forecast_len
        self.strategy = strategy  # 'B4', 'B3R', 'B1', 'B2', 'B3'


# -- Decode (vectorized) ------------------------------------------------------

def decode_rolled(head, e_ctx, rolled_f, n_ctx, strategy,
                  mean_c, stdev_c, horizon):
    """Decode rolled latents into a forecast (vectorized, no Python loops).

    Notation
    --------
    p[i]  = patch i, W raw values starting at i*W
    e[i]  = encoder latent for p[i]
    f[i]  ≈ e[i]  (contrastive training)

    Head input (same for ALL head types):
        [ e[0], ..., e[k],  f[k+1], ..., f[k+m] ]
          context (n_ctx)    rolled (m tokens)

    The head reconstructs the patch each latent represents.
    Forecast = head output at rolled positions, assembled by strategy.

    Args:
        head: ForecastingHead (.forecast_len = output values per position)
        e_ctx: (BC, n_ctx, H)  encoder latents
        rolled_f: (BC, m, H)   rolled forecaster latents
        n_ctx: number of context patches (k+1)
        strategy: 'B4'|'B3R'|'B3'|'B1'|'B2'
        mean_c, stdev_c: denormalization stats (C, 1) or None
        horizon: raw values to forecast

    Returns:
        (horizon, num_series) denormalized forecast
    """
    output_len = head.forecast_len                                  # 16 or 128
    BC = e_ctx.size(0)

    # -- Head forward: [e_ctx, rolled_f] → output at every position --
    seq = torch.cat([e_ctx, rolled_f], dim=1)                       # (BC, n_ctx+m, H)
    rolled_out = head(seq)[:, n_ctx:, :]                            # (BC, m, output_len)

    # -- Assemble forecast (vectorized per strategy) --
    if strategy in ('B3R', 'B3'):
        # Block decode: first position in each group of stride tokens
        stride = output_len // W                                    # 8 for 128
        block_idx = torch.arange(0, rolled_out.size(1), stride)     # [0, 8, 16, ...]
        blocks = rolled_out[:, block_idx, :]                        # (BC, n_blocks, output_len)
        flat = blocks.reshape(BC, -1)                               # (BC, n_blocks * output_len)
    elif strategy == 'B1':
        # Per-token W, but last uses full output_len
        n_w_tokens = max((horizon - 1) // W, 0)                    # tokens that give W values
        last_need = horizon - n_w_tokens * W                        # remaining for last token
        parts = []
        if n_w_tokens > 0:
            parts.append(rolled_out[:, :n_w_tokens, :W].reshape(BC, -1))
        parts.append(rolled_out[:, n_w_tokens, :last_need])
        flat = torch.cat(parts, dim=1)                              # (BC, horizon)
    else:
        # B2, B4: W values per position
        flat = rolled_out[:, :, :W].reshape(BC, -1)                 # (BC, m*W)

    forecast_norm = flat[:, :horizon]                               # (BC, horizon)

    # -- Denormalize --
    if mean_c is not None:
        forecast = forecast_norm * stdev_c.clamp(min=1e-5) + mean_c # (BC, horizon)
    else:
        forecast = forecast_norm

    return forecast.cpu().T.numpy()                                 # (horizon, num_series)


# -- Item preparation ---------------------------------------------------------

def prepare_context(item, device):
    """Extract and pad context from a GluonTS test item."""
    target = np.asarray(item["target"], dtype=np.float32)

    # Forward-fill NaNs
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

    # Pad or crop to T_RAW
    n = len(target)
    if n >= T_RAW:
        context = target[-T_RAW:]
    else:
        context = np.concatenate([
            np.full(T_RAW - n, target[0], dtype=np.float32), target])

    # (1, T_RAW, C) — replicate univariate across C channels
    context_t = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(-1)
    return context_t.repeat(1, 1, C).to(device)


def make_forecast(point_forecast, item, quantile_levels):
    """Wrap a point forecast array into a QuantileForecast."""
    point = np.nan_to_num(point_forecast.astype(np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)
    keys = ["mean"] + [str(q) for q in quantile_levels]
    arrays = np.stack([point] * len(keys), axis=0)
    return QuantileForecast(
        forecast_arrays=arrays, forecast_keys=keys,
        start_date=forecast_start(item),
        item_id=item.get("item_id", None),
    )


# -- Single-pass predictor ----------------------------------------------------

class MultiHeadPredictor(RepresentablePredictor):
    """Shared-rollout predictor: caches all head outputs on first head pass.

    On the first head (idx=0), computes the rollout + all 8 decodings per
    item and caches the forecasts. Subsequent heads just return cached results.
    This avoids recomputing the expensive rollout 8 times.
    """

    def __init__(self, backbone, heads, head_cfgs, prediction_length,
                 device):
        super().__init__(prediction_length=prediction_length)
        self.backbone = backbone
        self.heads = heads
        self.head_cfgs = head_cfgs
        self.device = device
        self._current_head_idx = 0
        self._cache = {}          # {head_idx: [forecast_array, ...]}
        self._item_counter = 0    # sequential item index within current pass

        # Max rolled tokens needed across all heads
        self._max_tokens = 0
        for cfg in head_cfgs:
            if cfg.strategy in ('B3R', 'B3'):
                stride = cfg.forecast_len // W
                n_blocks = math.ceil(prediction_length / cfg.forecast_len)
                self._max_tokens = max(self._max_tokens, n_blocks * stride)
            else:
                self._max_tokens = max(self._max_tokens,
                                       math.ceil(prediction_length / W))

    def predict(self, dataset, **kwargs):
        self._item_counter = 0
        for item in dataset:
            yield self.predict_item(item)

    def predict_item(self, item):
        idx = self._current_head_idx

        if idx == 0:
            # First head: compute rollout once, decode ALL heads, cache
            context_t = prepare_context(item, self.device)

            with torch.no_grad():
                e_ctx, _ = extract_encoder_latents(self.backbone, context_t)  # (BC, n_ctx, H)
                mean_c, stdev_c = _get_denorm_stats(self.backbone, C)
                n_ctx = e_ctx.size(1)                                         # k+1
                rolled_f = rollout_latent(self.backbone, e_ctx,
                                          self._max_tokens)                   # (BC, m, H)

                for h_idx, (head, cfg) in enumerate(
                        zip(self.heads, self.head_cfgs)):
                    raw = decode_rolled(head, e_ctx, rolled_f, n_ctx,
                                        cfg.strategy, mean_c, stdev_c,
                                        self.prediction_length)              # (horizon, B)
                    self._cache.setdefault(h_idx, []).append(raw[:, 0])

        # Retrieve cached forecast for this head + item
        point = self._cache[idx][self._item_counter]
        self._item_counter += 1
        return make_forecast(point, item, QUANTILE_LEVELS)

    def clear_cache(self):
        self._cache.clear()


# -- Dataset helpers ----------------------------------------------------------

def get_all_dataset_configs():
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
    if "/" in ds_name:
        ds_key = ds_name.split("/")[0].lower()
        ds_freq = ds_name.split("/")[1]
    else:
        ds_key = ds_name.lower()
        ds_freq = DATASET_PROPERTIES[PRETTY_NAMES.get(ds_key, ds_key)]["frequency"]
    ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    return f"{ds_key}/{ds_freq}/{term}", ds_key


# -- Main ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-head single-pass GIFT-Eval")
    p.add_argument("--backbone-path", required=True)
    p.add_argument("--heads", nargs="+", required=True,
                   help="Head specs: NAME:PATH:FLEN:STRATEGY "
                        "(e.g. R2:ckpt/R2.pth:16:B4)")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--device", default="cuda")
    p.add_argument("--test-only", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load backbone
    print("Loading backbone...")
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    backbone.load_state_dict(
        torch.load(args.backbone_path, map_location=device, weights_only=True))
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Parse and load heads
    head_cfgs = []
    heads = []
    for spec in args.heads:
        parts = spec.split(":")
        name, path, flen, strategy = parts
        cfg = HeadConfig(name, path, int(flen), strategy)
        head_cfgs.append(cfg)

        head = ForecastingHead(H=512, hidden_dim=128, num_gru_layers=2,
                               forecast_len=int(flen), dropout=0.1)
        head.load_state_dict(
            torch.load(path, map_location=device, weights_only=True))
        head = head.to(device).eval()
        heads.append(head)
        print(f"  Head {name}: {path} (flen={flen}, strategy={strategy})")

    # Prepare output dirs and CSV writers
    writers = {}
    csv_files = {}
    for cfg in head_cfgs:
        out_dir = os.path.join(args.output_dir, cfg.name)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "all_results.csv")
        f = open(csv_path, "w", newline="")
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        writers[cfg.name] = w
        csv_files[cfg.name] = f

    all_configs = get_all_dataset_configs()
    print(f"\nTotal configs: {len(all_configs)}, Heads: {len(heads)}")

    results_for_summary = {cfg.name: [] for cfg in head_cfgs}
    count = 0

    for ds_name, term in all_configs:
        config_name, ds_key = get_ds_config_name(ds_name, term)
        if args.test_only and count >= args.test_only:
            break
        count += 1
        t0 = time.time()

        try:
            to_univ_check = GiftDataset(name=ds_name, term=term, to_univariate=False)
            to_univariate = to_univ_check.target_dim > 1
            dataset = GiftDataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)

            # Shared-rollout predictor: rollout once, decode all 8 heads
            predictor = MultiHeadPredictor(
                backbone=backbone, heads=heads, head_cfgs=head_cfgs,
                prediction_length=dataset.prediction_length,
                device=device)

            # Evaluate each head — head 0 computes+caches all, heads 1-7 reuse
            for idx, cfg in enumerate(head_cfgs):
                predictor._current_head_idx = idx

                res = evaluate_model(
                    predictor, test_data=dataset.test_data,
                    metrics=METRICS, batch_size=512,
                    axis=None, mask_invalid_label=True,
                    allow_nan_forecast=False, seasonality=season_length,
                )

                mase_val = res["MASE[0.5]"][0]
                props = DATASET_PROPERTIES[ds_key]
                writers[cfg.name].writerow([
                    config_name, f"contrastive_tiny_{cfg.name}",
                    res["MSE[mean]"][0], res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0], res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0], res["sMAPE[0.5]"][0],
                    res["MSIS"][0], res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0], res["ND[0.5]"][0],
                    res["mean_weighted_sum_quantile_loss"][0],
                    props["domain"], props["num_variates"],
                ])
                csv_files[cfg.name].flush()
                results_for_summary[cfg.name].append((config_name, mase_val))

            predictor.clear_cache()
            elapsed = time.time() - t0
            mase_strs = " | ".join(
                f"{cfg.name}={results_for_summary[cfg.name][-1][1]:.4f}"
                for cfg in head_cfgs)
            print(f"  [{count:3d}/{len(all_configs)}] {config_name:45s} "
                  f"{mase_strs}  ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{count:3d}/{len(all_configs)}] {config_name:45s} "
                  f"FAILED: {e}  ({elapsed:.1f}s)")
            import traceback
            traceback.print_exc()

    # Close CSV files
    for f in csv_files.values():
        f.close()

    # Summary: GM-Relative MASE
    sn_candidates = [
        os.path.expanduser("~/workspaces/gift-eval/results/seasonal_naive/all_results.csv"),
        os.path.join(project_root, "gift-eval-ref/seasonal_naive/all_results.csv"),
    ]
    sn_mase = {}
    for sn_path in sn_candidates:
        if os.path.exists(sn_path):
            import pandas as pd
            sn_df = pd.read_csv(sn_path)
            for _, row in sn_df.iterrows():
                sn_mase[row["dataset"]] = row["eval_metrics/MASE[0.5]"]
            break

    print("\n" + "=" * 60)
    for cfg in head_cfgs:
        log_ratios = []
        for config_name, mase_val in results_for_summary[cfg.name]:
            sn_val = sn_mase.get(config_name)
            if sn_val and sn_val > 0:
                log_ratios.append(np.log(mase_val / sn_val))
        if log_ratios:
            gm = np.exp(np.mean(log_ratios))
            print(f"  {cfg.name}: GM-Relative MASE = {gm:.4f} ({len(log_ratios)} configs)")

        out_dir = os.path.join(args.output_dir, cfg.name)
        with open(os.path.join(out_dir, "summary.txt"), "w") as sf:
            if log_ratios:
                sf.write(f"GM-Relative MASE ({len(log_ratios)} configs): "
                         f"{np.exp(np.mean(log_ratios)):.4f}\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
