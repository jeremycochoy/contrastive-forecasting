#!/usr/bin/env python3
"""
GIFT-Eval evaluation script for the contrastive forecasting model.

Loads a frozen backbone + trained forecasting head, then evaluates on
GIFT-Eval benchmark datasets. Computes MASE (primary metric) and
optionally Seasonal Naive MASE for reference.

Usage:
    python scripts/eval_gift_eval.py \
        --backbone-path checkpoints/tiny_best_gap.pth \
        --head-path checkpoints/forecast_head_best.pth \
        --gift-eval-dir ~/workspaces/gift-eval-data/ \
        --device cuda:1

Output:
    - CSV file with columns: dataset, frequency, horizon, MASE, SeasonalNaive_MASE
    - Summary table printed to console
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

from src.models import ConfigurableModel, count_parameters
from src.forecasting_head import (
    ForecastingHead,
    FORECAST_LEN,
    forecast_autoregressive,
)

# -- Backbone architecture (must match checkpoint) ---------------------------
BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)

HEAD_CONFIG = dict(
    H=512, hidden_dim=128, num_gru_layers=2, forecast_len=FORECAST_LEN, dropout=0.1,
)

T_RAW = 1024  # Context window for backbone
BACKBONE_C = 4  # Backbone expects 4 channels


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate contrastive forecasting on GIFT-Eval")
    p.add_argument("--backbone-path", required=True,
                   help="Path to trained backbone checkpoint")
    p.add_argument("--head-path", required=True,
                   help="Path to trained forecasting head checkpoint")
    p.add_argument("--gift-eval-dir", required=True,
                   help="Path to local GIFT-Eval data directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-csv", default="gift_eval_results.csv",
                   help="Path for output results CSV")
    p.add_argument("--max-datasets", type=int, default=None,
                   help="Limit number of datasets to evaluate (for testing)")
    p.add_argument("--max-instances", type=int, default=None,
                   help="Limit number of test instances per dataset")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mase(y_true, y_pred, y_train, seasonality=1):
    """Mean Absolute Scaled Error.

    Args:
        y_true: (horizon,) ground truth
        y_pred: (horizon,) predictions
        y_train: (n_train,) training/context portion for naive scaling
        seasonality: seasonal period for naive baseline

    Returns:
        MASE score (lower is better)
    """
    n = len(y_train)
    if n <= seasonality:
        # Fallback: use step-1 naive
        seasonality = 1

    # Naive seasonal forecast error on training set
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = naive_errors.mean()

    if scale < 1e-10:
        # Constant series -- MASE is undefined; return MAE as fallback
        return np.mean(np.abs(y_true - y_pred))

    forecast_errors = np.abs(y_true - y_pred)
    return np.mean(forecast_errors) / scale


def seasonal_naive_forecast(y_train, horizon, seasonality=1):
    """Seasonal naive baseline: repeat last season."""
    n = len(y_train)
    if n < seasonality:
        seasonality = 1
    forecast = np.empty(horizon)
    for h in range(horizon):
        idx = n - seasonality + (h % seasonality)
        forecast[h] = y_train[idx]
    return forecast


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _guess_seasonality(freq_str):
    """Guess seasonality from frequency string."""
    freq = freq_str.lower().strip()
    mapping = {
        "h": 24, "1h": 24, "hourly": 24,
        "d": 7, "1d": 7, "daily": 7,
        "w": 52, "1w": 52, "weekly": 52,
        "m": 12, "1m": 12, "monthly": 12,
        "q": 4, "1q": 4, "quarterly": 4,
        "y": 1, "1y": 1, "yearly": 1, "annual": 1,
        "t": 60, "1t": 60, "minutely": 60, "min": 60,
        "15t": 96, "15min": 96,
        "30t": 48, "30min": 48,
        "b": 5, "business": 5,
    }
    return mapping.get(freq, 1)


def load_gift_eval_datasets(gift_eval_dir):
    """Load GIFT-Eval datasets from local directory.

    Tries two approaches:
    1. gift_eval library (if installed)
    2. Direct HuggingFace datasets loading from local files

    Yields: (dataset_name, frequency, horizon, test_instances)
    where test_instances is a list of dicts with keys:
        'context': np.array, 'target': np.array, 'seasonality': int
    """
    try:
        from datasets import load_from_disk, load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed. "
              "Install with: pip install datasets")
        sys.exit(1)

    # Walk through gift-eval directory structure
    # Expected structure: gift_eval_dir/<dataset_name>/<frequency>/
    # Each contains a HF dataset with train/test splits

    # Try loading the dataset catalog
    configs = []

    # Check for Salesforce/GiftEval style: single dataset with configs
    try:
        from datasets import get_dataset_config_names
        config_names = get_dataset_config_names(
            gift_eval_dir, trust_remote_code=True)
        for config_name in config_names:
            configs.append(config_name)
    except Exception:
        pass

    if configs:
        # Load each config
        for config_name in configs:
            try:
                ds = load_dataset(
                    gift_eval_dir, config_name, trust_remote_code=True)
            except Exception as e:
                print(f"  Warning: failed to load config {config_name}: {e}")
                continue

            # Extract metadata
            parts = config_name.split("/")
            dataset_name = parts[0] if parts else config_name
            freq_str = parts[1] if len(parts) > 1 else "unknown"

            test_split = ds.get("test", ds.get("train"))
            if test_split is None:
                continue

            instances = []
            for row in test_split:
                target = np.array(row.get("target", []), dtype=np.float32)
                # Some datasets store past as "past_values" or "input"
                context = np.array(
                    row.get("past_values",
                            row.get("input",
                                    row.get("context", []))),
                    dtype=np.float32)

                horizon = len(target)
                seasonality = row.get(
                    "seasonality",
                    _guess_seasonality(freq_str))

                if len(context) == 0 or len(target) == 0:
                    continue

                instances.append({
                    "context": context,
                    "target": target,
                    "seasonality": int(seasonality) if not isinstance(
                        seasonality, int) else seasonality,
                })

            if instances:
                horizon = len(instances[0]["target"])
                yield dataset_name, freq_str, horizon, instances
    else:
        # Walk directory structure
        for entry in sorted(os.listdir(gift_eval_dir)):
            entry_path = os.path.join(gift_eval_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            # Check if this directory contains a dataset directly
            try:
                ds = load_from_disk(entry_path)
                test_split = ds.get("test", ds) if hasattr(ds, "get") else ds

                instances = []
                for row in test_split:
                    target = np.array(row.get("target", []), dtype=np.float32)
                    context = np.array(
                        row.get("past_values",
                                row.get("input",
                                        row.get("context", []))),
                        dtype=np.float32)
                    freq_str = row.get("frequency", "unknown")
                    seasonality = row.get(
                        "seasonality", _guess_seasonality(freq_str))

                    if len(context) == 0 or len(target) == 0:
                        continue

                    instances.append({
                        "context": context,
                        "target": target,
                        "seasonality": int(seasonality) if not isinstance(
                            seasonality, int) else seasonality,
                    })

                if instances:
                    horizon = len(instances[0]["target"])
                    yield entry, freq_str, horizon, instances

            except Exception:
                # Try subdirectories (frequency-level)
                for sub_entry in sorted(os.listdir(entry_path)):
                    sub_path = os.path.join(entry_path, sub_entry)
                    if not os.path.isdir(sub_path):
                        continue
                    try:
                        ds = load_from_disk(sub_path)
                        test_split = (ds.get("test", ds)
                                      if hasattr(ds, "get") else ds)

                        instances = []
                        for row in test_split:
                            target = np.array(
                                row.get("target", []), dtype=np.float32)
                            context = np.array(
                                row.get("past_values",
                                        row.get("input",
                                                row.get("context", []))),
                                dtype=np.float32)
                            seasonality = row.get(
                                "seasonality",
                                _guess_seasonality(sub_entry))

                            if len(context) == 0 or len(target) == 0:
                                continue

                            instances.append({
                                "context": context,
                                "target": target,
                                "seasonality": int(seasonality)
                                if not isinstance(seasonality, int)
                                else seasonality,
                            })

                        if instances:
                            horizon = len(instances[0]["target"])
                            yield (f"{entry}/{sub_entry}", sub_entry,
                                   horizon, instances)
                    except Exception:
                        continue


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def pad_channels(x, target_c=4):
    """Pad or chunk a series to match backbone's expected C channels.

    Args:
        x: (T,) or (T, C_orig) numpy array

    Returns:
        torch.Tensor of shape (1, T, target_c)
    """
    if x.ndim == 1:
        x = x[:, None]  # (T, 1)

    T, C_orig = x.shape
    x_t = torch.from_numpy(x).float()  # (T, C_orig)

    if C_orig == target_c:
        return x_t.unsqueeze(0)  # (1, T, C)
    elif C_orig < target_c:
        # Repeat channels to fill
        repeats = (target_c + C_orig - 1) // C_orig
        x_t = x_t.repeat(1, repeats)[:, :target_c]
        return x_t.unsqueeze(0)
    else:
        # C_orig > target_c: use first target_c channels
        # (caller should handle multivariate properly)
        return x_t[:, :target_c].unsqueeze(0)


def prepare_context(context, T_raw=1024, target_c=4):
    """Prepare context series for backbone inference.

    Pads/truncates to T_raw and adjusts channels to target_c.

    Args:
        context: (n,) numpy array (univariate)

    Returns:
        x: (1, T_raw, target_c) tensor
    """
    n = len(context)

    if n >= T_raw:
        # Use last T_raw values
        context = context[-T_raw:]
    else:
        # Pad by repeating the first value
        pad_len = T_raw - n
        context = np.concatenate([
            np.full(pad_len, context[0], dtype=context.dtype),
            context
        ])

    return pad_channels(context, target_c)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device)

    # -- Load models -----------------------------------------------------------
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    sd = torch.load(args.backbone_path, map_location=device, weights_only=True)
    # Strip pretraining-only state (#344 cpc_w1, #353 EMA teacher).
    sd = {k: v for k, v in sd.items()
          if not k.startswith("cpc_w1")
          and not k.startswith("teacher_")}
    backbone.load_state_dict(sd)
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print(f"Backbone loaded from {args.backbone_path}")

    head = ForecastingHead(**HEAD_CONFIG)
    head.load_state_dict(
        torch.load(args.head_path, map_location=device, weights_only=True))
    head = head.to(device)
    head.eval()
    print(f"Forecasting head loaded from {args.head_path}")

    # -- Prepare output --------------------------------------------------------
    results = []
    csv_file = open(args.output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "dataset", "frequency", "horizon",
        "MASE", "SeasonalNaive_MASE", "n_instances"
    ])

    # -- Evaluate datasets -----------------------------------------------------
    print(f"\nLoading GIFT-Eval datasets from {args.gift_eval_dir}")
    dataset_count = 0

    for dataset_name, freq_str, horizon, instances in \
            load_gift_eval_datasets(args.gift_eval_dir):

        if args.max_datasets and dataset_count >= args.max_datasets:
            break
        dataset_count += 1

        mase_scores = []
        naive_mase_scores = []

        n_eval = len(instances)
        if args.max_instances:
            n_eval = min(n_eval, args.max_instances)

        t0 = time.time()

        for i in range(n_eval):
            inst = instances[i]
            context = inst["context"]
            target = inst["target"]
            seasonality = inst["seasonality"]
            h = len(target)

            # Prepare context for backbone
            x_ctx = prepare_context(context, T_raw=T_RAW, target_c=BACKBONE_C)
            x_ctx = x_ctx.to(device)

            # Forecast
            with torch.no_grad():
                forecast = forecast_autoregressive(
                    backbone, head, x_ctx, horizon=h, device=device)

            # Extract channel 0 (the actual univariate forecast)
            pred = forecast[:, 0]

            # Compute MASE
            score = mase(target, pred, context, seasonality=seasonality)
            mase_scores.append(score)

            # Seasonal naive baseline
            naive_pred = seasonal_naive_forecast(context, h, seasonality)
            naive_score = mase(target, naive_pred, context,
                               seasonality=seasonality)
            naive_mase_scores.append(naive_score)

        elapsed = time.time() - t0
        avg_mase = np.mean(mase_scores) if mase_scores else float("nan")
        avg_naive = np.mean(naive_mase_scores) if naive_mase_scores else float("nan")

        results.append({
            "dataset": dataset_name,
            "frequency": freq_str,
            "horizon": horizon,
            "MASE": avg_mase,
            "SeasonalNaive_MASE": avg_naive,
            "n_instances": n_eval,
        })

        csv_writer.writerow([
            dataset_name, freq_str, horizon,
            f"{avg_mase:.4f}", f"{avg_naive:.4f}", n_eval
        ])
        csv_file.flush()

        print(f"  {dataset_name:40s} | freq={freq_str:6s} | h={horizon:4d} | "
              f"MASE={avg_mase:8.4f} | Naive={avg_naive:8.4f} | "
              f"n={n_eval:4d} | {elapsed:.1f}s")

    csv_file.close()
    print(f"\nResults saved to {args.output_csv}")

    # -- Summary table ---------------------------------------------------------
    if results:
        print(f"\n{'='*80}")
        print(f"{'GIFT-Eval Summary':^80}")
        print(f"{'='*80}")
        print(f"{'Dataset':<35} {'Freq':>6} {'H':>5} "
              f"{'MASE':>8} {'Naive':>8} {'Ratio':>8}")
        print(f"{'-'*80}")

        total_mase, total_naive, n_total = 0, 0, 0
        for r in results:
            ratio = (r["MASE"] / r["SeasonalNaive_MASE"]
                     if r["SeasonalNaive_MASE"] > 0 else float("nan"))
            print(f"{r['dataset']:<35} {r['frequency']:>6} {r['horizon']:>5} "
                  f"{r['MASE']:>8.4f} {r['SeasonalNaive_MASE']:>8.4f} "
                  f"{ratio:>8.4f}")
            if not np.isnan(r["MASE"]):
                total_mase += r["MASE"] * r["n_instances"]
                total_naive += r["SeasonalNaive_MASE"] * r["n_instances"]
                n_total += r["n_instances"]

        if n_total > 0:
            overall_mase = total_mase / n_total
            overall_naive = total_naive / n_total
            overall_ratio = overall_mase / overall_naive if overall_naive > 0 else float("nan")
            print(f"{'-'*80}")
            print(f"{'OVERALL (weighted)':35} {'':>6} {'':>5} "
                  f"{overall_mase:>8.4f} {overall_naive:>8.4f} "
                  f"{overall_ratio:>8.4f}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
