#!/usr/bin/env python3
"""
Multi-head single-pass GIFT-Eval evaluation.

Runs the backbone + latent rollout ONCE per test item, then decodes
with multiple heads in parallel. Avoids recomputing the expensive
autoregressive rollout for each head.

Usage:
    GIFT_EVAL=~/gift-eval-data PYTHONPATH=. python eval_multi_head.py \
        --backbone-path checkpoints/tiny_v2_best_gap.pth \
        --heads R2:checkpoints/R2_encoder_recon_w16_best.pth:encoder:16:B4 \
                R4:checkpoints/R5_encoder_recon_w128_best.pth:encoder:128:B3R \
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
from typing import List, Optional

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

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

from gift_eval.data import Dataset as GiftDataset

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import ConfigurableModel
from src.forecasting_head import (
    ForecastingHead,
    W,
    extract_encoder_latents,
    extract_forecaster_latents,
    rollout_latent,
    _get_denorm_stats,
    _denormalize,
)

BACKBONE_CONFIG = dict(
    C=4, H=512, W=16, encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)
T_RAW = 1024
BACKBONE_C = 4
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

# Dataset config (same as eval_gift_eval_official.py)
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


class HeadConfig:
    """Parsed head configuration."""
    def __init__(self, name, path, recon_mode, forecast_len, strategy):
        self.name = name
        self.path = path
        self.recon_mode = recon_mode  # 'encoder' or 'forecaster'
        self.forecast_len = forecast_len
        self.strategy = strategy  # 'B4' or 'B3R'


def decode_with_head(head, head_cfg, e_bc, f_ctx, future_f, T_ctx,
                     mean_c, stdev_c, horizon, W_bb):
    """Decode rolled latents with a single head."""
    forecast_len = head.forecast_len

    if head_cfg.recon_mode == 'encoder':
        # Encoder recon: use e_bc for context (matches training)
        full_seq = torch.cat([e_bc, future_f], dim=1)
    else:
        # Forecaster recon: use f_ctx, skip duplicate first rolled token
        full_seq = torch.cat([f_ctx, future_f[:, 1:, :]], dim=1)

    all_preds_raw = head(full_seq)

    if head_cfg.recon_mode == 'encoder':
        future_preds = all_preds_raw[:, T_ctx:, :]
    else:
        future_preds = all_preds_raw[:, T_ctx:, :]  # adjusted by skip

    # Extract predictions based on strategy
    all_preds = []
    remaining = horizon

    if head_cfg.strategy == 'B3R':
        # Encoder recon block decode: take FIRST position in each group of 8
        tokens_per_chunk = forecast_len // W_bb
        n_chunks = math.ceil(horizon / forecast_len)
        for chunk_i in range(n_chunks):
            token_idx = chunk_i * tokens_per_chunk
            if token_idx >= future_preds.size(1):
                break
            pred_norm = future_preds[:, token_idx, :]
            n_take = min(forecast_len, remaining)
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break
    elif head_cfg.strategy == 'B3':
        # Prediction block decode: take LAST position in each group of 8
        tokens_per_chunk = forecast_len // W_bb
        n_chunks = math.ceil(horizon / forecast_len)
        for chunk_i in range(n_chunks):
            token_idx = (chunk_i + 1) * tokens_per_chunk - 2  # -2: adjusted for skip
            if token_idx < 0 or token_idx >= future_preds.size(1):
                break
            pred_norm = future_preds[:, token_idx, :]
            n_take = min(forecast_len, remaining)
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break
    elif head_cfg.strategy == 'B1':
        # Decode all: W per position, last gets all forecast_len
        for i in range(future_preds.size(1)):
            pred_norm = future_preds[:, i, :]
            if remaining <= forecast_len:
                n_take = remaining
            else:
                n_take = W_bb
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break
    elif head_cfg.strategy == 'B2':
        # Crop to W at each position (128 head, take only W)
        for i in range(future_preds.size(1)):
            pred_norm = future_preds[:, i, :]
            n_take = min(W_bb, remaining)
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break
    else:
        # B4: one token at a time, take W values
        for i in range(future_preds.size(1)):
            pred_norm = future_preds[:, i, :]
            n_take = min(W_bb, remaining)
            pred_actual = _denormalize(pred_norm[:, :n_take], mean_c, stdev_c)
            all_preds.append(pred_actual.cpu())
            remaining -= n_take
            if remaining <= 0:
                break

    if not all_preds:
        return np.zeros((horizon, e_bc.size(0) // BACKBONE_C))

    forecast = torch.cat(all_preds, dim=1)
    return forecast.T.numpy()


class MultiHeadPredictor(RepresentablePredictor):
    """Predictor that runs one rollout, decodes with multiple heads."""

    def __init__(self, backbone, heads, head_cfgs, prediction_length,
                 device, quantile_levels=None):
        super().__init__(prediction_length=prediction_length)
        self.backbone = backbone
        self.heads = heads
        self.head_cfgs = head_cfgs
        self.device = device
        self.quantile_levels = quantile_levels or QUANTILE_LEVELS
        self.forecast_keys = ["mean"] + [str(q) for q in self.quantile_levels]
        self._current_head_idx = 0  # set externally per evaluation

    def predict(self, dataset, **kwargs):
        for item in dataset:
            yield self.predict_item(item)

    def predict_item(self, item):
        target = np.asarray(item["target"], dtype=np.float32)
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

        n = len(target)
        if n >= T_RAW:
            context = target[-T_RAW:]
        else:
            context = np.concatenate([
                np.full(T_RAW - n, target[0], dtype=np.float32), target])

        context_t = torch.from_numpy(context).float().unsqueeze(0).unsqueeze(-1)
        context_t = context_t.repeat(1, 1, BACKBONE_C).to(self.device)

        with torch.no_grad():
            # Single rollout for all heads
            B, T_ctx_raw, C = context_t.shape
            W_bb = self.backbone.W

            # Extract encoder + forecaster latents
            e_bc, x_norm = extract_encoder_latents(self.backbone, context_t)
            f_ctx, _ = extract_forecaster_latents(self.backbone, context_t)
            mean_c, stdev_c = _get_denorm_stats(self.backbone, C)
            T_ctx = e_bc.size(1)

            # Compute max rollout tokens needed across all heads
            max_tokens = 0
            for cfg in self.head_cfgs:
                if cfg.strategy in ('B3R', 'B3'):
                    tokens_per_chunk = cfg.forecast_len // W_bb
                    n_chunks = math.ceil(self.prediction_length / cfg.forecast_len)
                    need = n_chunks * tokens_per_chunk
                    if cfg.recon_mode != 'encoder':
                        need += 1  # for skip_first_rolled
                    max_tokens = max(max_tokens, need)
                else:
                    need = math.ceil(self.prediction_length / W_bb)
                    if cfg.recon_mode != 'encoder':
                        need += 1  # for skip_first_rolled
                    max_tokens = max(max_tokens, need)

            # Single rollout
            future_f = rollout_latent(self.backbone, e_bc, max_tokens)

            # Decode with current head
            idx = self._current_head_idx
            forecast_raw = decode_with_head(
                self.heads[idx], self.head_cfgs[idx],
                e_bc, f_ctx, future_f, T_ctx,
                mean_c, stdev_c, self.prediction_length, W_bb)

        point_forecast = forecast_raw[:, 0].astype(np.float64)
        point_forecast = np.nan_to_num(point_forecast, nan=0.0, posinf=0.0, neginf=0.0)

        n_keys = len(self.forecast_keys)
        forecast_arrays = np.stack([point_forecast] * n_keys, axis=0)
        fstart = forecast_start(item)

        return QuantileForecast(
            forecast_arrays=forecast_arrays,
            forecast_keys=self.forecast_keys,
            start_date=fstart,
            item_id=item.get("item_id", None),
        )


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


def parse_args():
    p = argparse.ArgumentParser(description="Multi-head single-pass GIFT-Eval")
    p.add_argument("--backbone-path", required=True)
    p.add_argument("--heads", nargs="+", required=True,
                   help="Head specs: NAME:PATH:MODE:FLEN:STRATEGY "
                        "(e.g. R2:ckpt/R2.pth:encoder:16:B4)")
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
        name, path, mode, flen, strategy = parts
        cfg = HeadConfig(name, path, mode, int(flen), strategy)
        head_cfgs.append(cfg)

        head = ForecastingHead(H=512, hidden_dim=128, num_gru_layers=2,
                               forecast_len=int(flen), dropout=0.1)
        head.load_state_dict(
            torch.load(path, map_location=device, weights_only=True))
        head = head.to(device).eval()
        heads.append(head)
        print(f"  Head {name}: {path} (mode={mode}, flen={flen}, strategy={strategy})")

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

            predictor = MultiHeadPredictor(
                backbone=backbone, heads=heads, head_cfgs=head_cfgs,
                prediction_length=dataset.prediction_length,
                device=device, quantile_levels=QUANTILE_LEVELS,
            )

            # Evaluate each head using the shared predictor
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

    # Summary
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

        # Save summary
        out_dir = os.path.join(args.output_dir, cfg.name)
        with open(os.path.join(out_dir, "summary.txt"), "w") as sf:
            if log_ratios:
                sf.write(f"GM-Relative MASE ({len(log_ratios)} configs): "
                         f"{np.exp(np.mean(log_ratios)):.4f}\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
