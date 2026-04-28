"""Plot the worst-MASE GIFT-Eval configs across all 3 arms.

For each of 13 hand-picked configs (top 1-2 worst per domain where every
arm has MASE > 1.0), runs B4 inference with each backbone+head, and
plots context + ground truth + 3 forecasts side by side. Helps see WHERE
the model fails (weak periodicity, explosive trend, spike-driven, etc.).

Run from repo root with the GIFT-Eval data under $GIFT_EVAL:

    GIFT_EVAL=~/gift-eval-data PYTHONPATH=. \\
    python experiments/exp_dualemb_3arm/scripts/plot_worst_configs.py
"""
from __future__ import annotations

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


HERE = Path(__file__).resolve().parent.parent
SYNC_DIR = Path("sync_dualemb_3arm/checkpoints")
OUT_PATH = HERE / "plots" / "gift_eval_worst_configs.png"

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


def main():
    # Sync dir lives in the MAIN checkout per CLAUDE.md, regardless of where
    # this script is run from (worktree or main).
    sync_root = Path(
        os.environ.get("SYNC_ROOT")
        or "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
    )
    checkpoint_dir = sync_root / SYNC_DIR

    # Load all 3 arms once
    print("Loading 3 arms...")
    arm_models = {}
    for arm_id, label, kind, span, color in ARMS:
        bb, head = load_arm(checkpoint_dir, arm_id, kind, span)
        arm_models[arm_id] = (bb, head)
        print(f"  {label}: ok")

    # Build figure: 13 configs, one row each, 3 columns (no, just one wide subplot per config)
    n = len(WORST_CONFIGS)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.0 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (display, name, term, mase_r, mase_e5, mase_e1) in enumerate(WORST_CONFIGS):
        ax = axes_flat[i]
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

            # Last 256 of context for plotting (long contexts swamp)
            n_show = min(256, len(ctx))
            ctx_show = ctx[-n_show:]

            t_ctx = np.arange(-n_show, 0)
            t_fc = np.arange(0, horizon)

            ax.plot(t_ctx, ctx_show, color="black", linewidth=1.0, label="context")
            ax.plot(t_fc, tgt[:horizon], color="black", linewidth=1.4,
                    linestyle="-", label="truth")

            mase_lookup = {"revin": mase_r, "ewma512": mase_e5, "ewma128": mase_e1}
            for arm_id, label, kind, span, color in ARMS:
                bb, head = arm_models[arm_id]
                fc, lo, hi = run_arm(bb, head, ctx_tensor, freq_id, seas_id, horizon)
                fc = fc[:horizon]
                ax.plot(t_fc, fc, color=color, linewidth=1.2, alpha=0.85,
                        label=f"{label} (MASE={mase_lookup[arm_id]:.2f})")
                if lo is not None and hi is not None:
                    ax.fill_between(t_fc, lo[:horizon], hi[:horizon],
                                    color=color, alpha=0.10)

            ax.axvline(0, color="gray", linestyle=":", linewidth=0.6)
            ax.set_title(f"{display}  (freq={freq_str}, seas={season}, h={horizon})",
                         fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.25)
            print(f"  {display}: ok (h={horizon})")
        except Exception as e:
            ax.set_title(f"{display}  (FAILED: {e})", fontsize=9, color="red")
            print(f"  {display}: FAILED {e}")

    # Hide any unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Worst GIFT-Eval configs (all 3 arms MASE > 1.0): "
                 "context + truth + 3 forecasts",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
