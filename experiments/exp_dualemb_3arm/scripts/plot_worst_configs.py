"""Plot the worst-MASE GIFT-Eval configs across all 3 arms.

Two outputs:

* `plots/gift_eval_worst_configs.png` — 1-2 worst configs per domain
  (13 panels), high resolution. Each shows context + truth + 3 model
  forecasts + seasonal-naive baseline.
* `plots/gift_eval_all_failures.png` — every config where all 3 arms
  have MASE > 1 (~73 panels), one big grid.

Each title carries the per-arm MASE plus the test-set SN_MASE we
recompute on the fly (the GIFT-Eval bundle's seasonal-naive sidecar
isn't shipped with the data we have).

Run from repo root with the GIFT-Eval data under $GIFT_EVAL:

    GIFT_EVAL=~/gift-eval-data PYTHONPATH=. \\
    python experiments/exp_dualemb_3arm/scripts/plot_worst_configs.py
"""
from __future__ import annotations

import csv
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from gift_eval.data import Dataset as GiftDataset
from gluonts.time_feature import get_seasonality

from src.models import ConfigurableModel
from src.forecasting_head import (
    ForecastingHead, QuantileForecastingHead, forecast_with_strategy,
)
from src.freq_embedding import gluonts_freq_to_id, seasonality_to_id


# ── GIFT-Eval name resolution (mirrors eval_gift_eval_official.py) ─────────

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
    "SZ_TAXI/15T M_DENSE/H "
    "ett1/15T ett1/H ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
DATASET_FREQ_FALLBACK = {  # for single-freq datasets
    "m4_yearly": "A", "m4_quarterly": "Q", "m4_monthly": "M",
    "m4_weekly": "W", "m4_daily": "D", "m4_hourly": "H",
    "hospital": "M", "covid_deaths": "D",
    "temperature_rain": "D", "car_parts": "M",
    "restaurant": "D",
    "bizitobs_application": "10S", "bizitobs_service": "10S",
}


def _all_configs():
    """Return list of (gift_eval_name, term) for all 97 GIFT-Eval configs."""
    short_list = SHORT_DATASETS.split()
    med_long_list = MED_LONG_DATASETS.split()
    out = []
    for ds_name in sorted(set(short_list + med_long_list)):
        out.append((ds_name, "short"))
        if ds_name in med_long_list:
            out.append((ds_name, "medium"))
            out.append((ds_name, "long"))
    return out


def _csv_name(ds_name: str, term: str) -> str:
    """Produce the dataset-key string as it appears in all_results.csv."""
    if "/" in ds_name:
        ds_key, ds_freq = ds_name.split("/", 1)
        ds_key = ds_key.lower()
    else:
        ds_key = ds_name.lower()
        ds_freq = DATASET_FREQ_FALLBACK.get(
            PRETTY_NAMES.get(ds_key, ds_key), "")
    ds_key = PRETTY_NAMES.get(ds_key, ds_key)
    return f"{ds_key}/{ds_freq}/{term}"


def csv_name_to_pair(csv_name: str):
    """Reverse-lookup: 'solar/10T/long' → ('solar/10T', 'long')."""
    for ds_name, term in _all_configs():
        if _csv_name(ds_name, term) == csv_name:
            return ds_name, term
    raise KeyError(f"Could not resolve {csv_name} to (ds_name, term)")


HERE = Path(__file__).resolve().parent.parent
SYNC_DIR = Path("sync_dualemb_3arm/checkpoints")
OUT_PATH = HERE / "plots" / "gift_eval_worst_configs.png"
OUT_PATH_ALL = HERE / "plots" / "gift_eval_all_failures.png"
RESULTS_DIR = HERE / "results"

# Backbone hyper-params shared across arms.
BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", intermediate_dim=128,
    num_layers=6, nhead=8, ffn_mult=4, dropout=0.1,
    activation="gelu", depthwise_conv=3,
    norm_type="layernorm",
)

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
HEAD_CONFIG = dict(H=512, hidden_dim=128, num_gru_layers=2, forecast_len=16)
T_RAW = 1024
BACKBONE_C = 4

ARMS = [
    ("revin",   "RevIN",          "revin", None,  "#d62728"),
    ("ewma512", "EWMA span=512",  "ewma",  512,   "#1f77b4"),
    ("ewma128", "EWMA span=128",  "ewma",  128,   "#2ca02c"),
]

# (display_name, gift_eval_name, term, mase_revin, mase_e512, mase_e128)
WORST_CONFIGS = [
    ("Econ/Fin: m4_yearly/A/short",       "m4_yearly",                          "short",  16.44, 10.69,  7.03),
    ("Econ/Fin: m4_weekly/W/short",       "m4_weekly",                          "short",  13.59, 10.67,  4.69),
    ("Energy: solar/10T/long",            "solar/10T",                          "long",    2.54,  2.98,  2.45),
    ("Energy: solar/10T/medium",          "solar/10T",                          "medium",  2.42,  2.44,  2.64),
    ("Healthcare: covid_deaths/D/short",  "covid_deaths",                       "short", 190.35, 80.25, 70.81),
    ("Healthcare: us_births/W/short",     "us_births/W",                        "short",   2.38,  2.41,  2.43),
    ("Nature: saugeen/D/short",           "saugeenday/D",                       "short",   5.16,  3.96,  4.98),
    ("Nature: temperature_rain/D/short",  "temperature_rain_with_missing",      "short",   1.57,  1.78,  1.75),
    ("Sales: car_parts/M/short",          "car_parts_with_missing",             "short",   1.26,  1.12,  1.10),
    ("Transport: loop_seattle/H/medium",  "LOOP_SEATTLE/H",                     "medium",  1.68,  1.78,  1.82),
    ("Transport: loop_seattle/H/long",    "LOOP_SEATTLE/H",                     "long",    1.68,  1.64,  1.62),
    ("Web/CloudOps: bizitobs_application/10S/long",   "bizitobs_application",   "long",   10.84, 14.95, 10.56),
    ("Web/CloudOps: bizitobs_application/10S/medium", "bizitobs_application",   "medium", 10.08, 12.74, 10.19),
]


def load_arm(checkpoint_dir: Path, arm_id: str, rev_norm_kind: str, rev_norm_span):
    """Load backbone + head for one arm. Auto-detects emb dims from state-dict."""
    bb_path = checkpoint_dir / f"tiny_dualemb_{arm_id}_FINAL.pth"
    head_path = checkpoint_dir / f"R1q_dualemb_{arm_id}_FINAL.pth"

    sd = torch.load(bb_path, map_location="cpu", weights_only=True)
    cfg = dict(BACKBONE_CONFIG)
    cfg["rev_norm_kind"] = rev_norm_kind
    if rev_norm_kind == "ewma":
        cfg["rev_norm_span"] = rev_norm_span
    fw = sd.get("freq_embedding.embedding.weight")
    cfg["freq_emb_dim"] = fw.shape[1] if fw is not None else 0
    sw = sd.get("seasonality_embedding.embedding.weight")
    cfg["seasonality_emb_dim"] = sw.shape[1] if sw is not None else 0
    cfg["patch_stats_kind"] = "none"
    backbone = ConfigurableModel(**cfg)
    backbone.load_state_dict(sd)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head_sd = torch.load(head_path, map_location="cpu", weights_only=True)
    fh_w = head_sd.get("forecast_head.weight")
    is_quantile = (
        fh_w is not None and fh_w.shape[0] == HEAD_CONFIG["forecast_len"] * 9
    )
    if is_quantile:
        head = QuantileForecastingHead(**HEAD_CONFIG)
    else:
        head = ForecastingHead(**HEAD_CONFIG)
    head.load_state_dict(head_sd)
    head.eval()
    return backbone, head


def prepare_context(target: np.ndarray) -> torch.Tensor:
    """Truncate/pad univariate target to T_RAW; replicate to backbone_c=4."""
    a = np.asarray(target, dtype=np.float32)
    # Forward-fill NaN
    if np.isnan(a).any():
        mask = np.isnan(a)
        if mask.all():
            a[:] = 0.0
        else:
            first = np.where(~mask)[0][0]
            a[:first] = a[first]
            for i in range(1, len(a)):
                if np.isnan(a[i]):
                    a[i] = a[i - 1]
    n = len(a)
    if n >= T_RAW:
        ctx = a[-T_RAW:]
    else:
        ctx = np.concatenate([np.full(T_RAW - n, a[0], dtype=np.float32), a])
    t = torch.from_numpy(ctx).unsqueeze(0).unsqueeze(-1)  # (1, T_RAW, 1)
    return t


def first_test_instance(dataset: GiftDataset):
    """Return (context, target) for the first instance of dataset.test_data."""
    # GIFT-Eval test_data is an iterable of (input_dict, label_dict) pairs;
    # input_dict["target"] is the context, label_dict["target"] is the truth.
    for it in dataset.test_data:
        # gift_eval returns a TestData object; iterate it:
        if hasattr(it, "input") and hasattr(it, "label"):
            ctx = np.asarray(it.input["target"], dtype=np.float32)
            tgt = np.asarray(it.label["target"], dtype=np.float32)
        else:
            inp, lab = it
            ctx = np.asarray(inp["target"], dtype=np.float32)
            tgt = np.asarray(lab["target"], dtype=np.float32)
        # gift_eval univariate: 1D arrays; multivariate: 2D, take channel 0.
        if ctx.ndim == 2:
            ctx = ctx[0]
            tgt = tgt[0]
        return ctx, tgt
    raise RuntimeError("dataset.test_data has no instances")


def seasonal_naive(context_1d: np.ndarray, horizon: int, period: int):
    """Standard seasonal-naive forecast: y[h] = ctx[-period + (h % period)]."""
    if period <= 0 or period >= len(context_1d):
        period = 1
    n = len(context_1d)
    out = np.empty(horizon, dtype=np.float32)
    for h in range(horizon):
        out[h] = context_1d[n - period + (h % period)]
    return out


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, period: int):
    """gluonts-style MASE: forecast error / in-sample seasonal-naive scale."""
    n = len(y_train)
    s = period if period > 0 and n > period else 1
    naive_errs = np.abs(y_train[s:] - y_train[:-s])
    scale = float(np.mean(naive_errs)) if naive_errs.size else 0.0
    if scale <= 0:
        return float("nan")
    fc_errs = np.abs(y_pred[: len(y_true)] - y_true[: len(y_pred)])
    return float(np.mean(fc_errs) / scale)


def run_arm(backbone, head, ctx_tensor, freq_id, seasonality_id, horizon):
    """Run B4 inference, return (point_forecast, lower, upper) at quantile 0.5 / 0.1 / 0.9."""
    backbone._eval_freq_id = freq_id
    backbone._eval_seasonality_id = seasonality_id
    out = forecast_with_strategy("B4", backbone, head, ctx_tensor, horizon, "cpu")
    if out.ndim == 3:
        # Quantile head: (Q, horizon, C)
        Q = out.shape[0]
        # 9 levels at 0.1..0.9
        median_idx = QUANTILE_LEVELS.index(0.5)
        lower_idx = QUANTILE_LEVELS.index(0.1)
        upper_idx = QUANTILE_LEVELS.index(0.9)
        median = out[median_idx, :, 0]
        lower = out[lower_idx, :, 0]
        upper = out[upper_idx, :, 0]
        return median, lower, upper
    # MSE head: (horizon, C)
    return out[:, 0], None, None


def _load_mase_per_config():
    """Read MASE values per arm per config from the CSVs in `results/`."""
    arms = ["revin", "ewma512", "ewma128"]
    data: dict[str, dict[str, float]] = {a: {} for a in arms}
    domains: dict[str, str] = {}
    for arm in arms:
        path = RESULTS_DIR / f"gift_eval_{arm}" / "all_results.csv"
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    v = float(row["eval_metrics/MASE[0.5]"])
                    if math.isfinite(v) and v > 0:
                        data[arm][row["dataset"]] = v
                        domains[row["dataset"]] = row.get("domain", "?")
                except (ValueError, KeyError):
                    pass
    common = sorted(set(data["revin"]) & set(data["ewma512"]) & set(data["ewma128"]))
    return data, domains, common


def _plot_one_panel(ax, display, name, term, arm_models, mase_lookup):
    """Plot one config: context + truth + 3 forecasts + seasonal-naive."""
    try:
        check = GiftDataset(name=name, term=term, to_univariate=False)
        to_univariate = check.target_dim > 1
        ds = GiftDataset(name=name, term=term, to_univariate=to_univariate)
        horizon = ds.prediction_length
        freq_str = ds.freq
        season = get_seasonality(freq_str)
        freq_id = gluonts_freq_to_id(freq_str)
        seas_id = seasonality_to_id(season)

        ctx, tgt = first_test_instance(ds)
        ctx_tensor = prepare_context(ctx)

        # Plot last min(2*horizon, 256) of context
        n_show = min(max(2 * horizon, 96), len(ctx), 512)
        ctx_show = ctx[-n_show:]
        t_ctx = np.arange(-n_show, 0)
        t_fc = np.arange(0, horizon)
        truth = np.asarray(tgt[:horizon], dtype=np.float32)

        ax.plot(t_ctx, ctx_show, color="black", linewidth=0.9, label="context")
        ax.plot(t_fc, truth, color="black", linewidth=1.6, label="truth")

        # Seasonal-naive forecast and its MASE (computed over the same target).
        sn_fc = seasonal_naive(np.asarray(ctx, dtype=np.float32), horizon, season)
        sn_mase = mase(truth, sn_fc, np.asarray(ctx, dtype=np.float32), season)
        ax.plot(t_fc, sn_fc, color="gray", linewidth=1.1, linestyle="--",
                label=f"SN (MASE={sn_mase:.2f})")

        # 3 model arms.
        for arm_id, label, kind, span, color in ARMS:
            bb, head = arm_models[arm_id]
            fc, lo, hi = run_arm(bb, head, ctx_tensor, freq_id, seas_id, horizon)
            fc = fc[:horizon]
            ax.plot(t_fc, fc, color=color, linewidth=1.1, alpha=0.85,
                    label=f"{label} ({mase_lookup[arm_id]:.2f})")
            if lo is not None and hi is not None:
                ax.fill_between(t_fc, lo[:horizon], hi[:horizon],
                                color=color, alpha=0.08)

        ax.axvline(0, color="gray", linestyle=":", linewidth=0.6)
        ax.set_title(f"{display}  (freq={freq_str}, seas={season}, h={horizon})",
                     fontsize=8)
        ax.legend(fontsize=6, loc="best", framealpha=0.85)
        ax.grid(alpha=0.25)
        print(f"  {display}: ok (h={horizon}, SN_MASE={sn_mase:.2f})")
        return True
    except Exception as e:
        ax.set_title(f"{display}  (FAILED: {e})", fontsize=8, color="red")
        print(f"  {display}: FAILED {e}")
        return False


def main():
    sync_root = Path(
        os.environ.get("SYNC_ROOT")
        or "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
    )
    checkpoint_dir = sync_root / SYNC_DIR

    print("Loading 3 arms...")
    arm_models = {}
    for arm_id, label, kind, span, color in ARMS:
        bb, head = load_arm(checkpoint_dir, arm_id, kind, span)
        arm_models[arm_id] = (bb, head)
        print(f"  {label}: ok")

    data, domains, common = _load_mase_per_config()
    arms = ["revin", "ewma512", "ewma128"]

    # ---------- Plot 1: 13 curated worst configs ---------------------------
    print("\n=== Plot 1: 13 curated worst configs ===")
    n = len(WORST_CONFIGS)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.6 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, (display, name, term, mase_r, mase_e5, mase_e1) in enumerate(WORST_CONFIGS):
        ax = axes_flat[i]
        mase_lookup = {"revin": mase_r, "ewma512": mase_e5, "ewma128": mase_e1}
        _plot_one_panel(ax, display, name, term, arm_models, mase_lookup)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(
        "Worst GIFT-Eval configs (top 1-2 per domain, all 3 arms MASE > 1): "
        "context + truth + SN + 3 model forecasts",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180)
    print(f"Saved {OUT_PATH}")

    # ---------- Plot 2: every fail-all config ------------------------------
    fail_all = [c for c in common if all(data[a][c] > 1.0 for a in arms)]
    # Sort by domain, then by GM-MASE descending for visual flow.
    def _gm(c):
        return math.exp(sum(math.log(data[a][c]) for a in arms) / 3.0)
    fail_all = sorted(fail_all, key=lambda c: (domains[c], -_gm(c)))
    print(f"\n=== Plot 2: all {len(fail_all)} fail-all configs ===")
    cols = 3
    rows = math.ceil(len(fail_all) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3.0 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, csv_name in enumerate(fail_all):
        ax = axes_flat[i]
        try:
            ds_name, term = csv_name_to_pair(csv_name)
        except KeyError as e:
            ax.set_title(f"{csv_name}  ({e})", fontsize=8, color="red")
            print(f"  {csv_name}: name resolution FAILED")
            continue
        display = f"{domains[csv_name]}: {csv_name}"
        mase_lookup = {a: data[a][csv_name] for a in arms}
        _plot_one_panel(ax, display, ds_name, term, arm_models, mase_lookup)
    for j in range(len(fail_all), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(
        f"All {len(fail_all)} GIFT-Eval configs where every arm has MASE > 1: "
        "context + truth + SN + 3 model forecasts",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT_PATH_ALL, dpi=130)
    print(f"Saved {OUT_PATH_ALL}")


if __name__ == "__main__":
    main()
