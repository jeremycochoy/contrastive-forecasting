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
    rev_norm_span=32,
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
        # forecast_raw: (prediction_length, C) -- take channel 0
        point_forecast = forecast_raw[:, 0].astype(np.float64)

        # Replace any NaN/inf in forecast with 0
        point_forecast = np.nan_to_num(
            point_forecast, nan=0.0, posinf=0.0, neginf=0.0)

        # Build QuantileForecast: all quantiles = point forecast (deterministic)
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
    return p.parse_args()


def load_models(args, device):
    """Load backbone and forecasting head."""
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    backbone.load_state_dict(
        torch.load(args.backbone_path, map_location=device,
                    weights_only=True))
    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head_config = dict(HEAD_CONFIG)
    head_config['forecast_len'] = args.forecast_len
    head = ForecastingHead(**head_config)
    head.load_state_dict(
        torch.load(args.head_path, map_location=device, weights_only=True))
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
