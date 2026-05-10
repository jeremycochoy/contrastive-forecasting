#!/usr/bin/env python3
"""
4-arm comparison: square loss vs baseline at τ=0.10 and τ=0.20.
Baselines run to 50k steps; square arms now also run to 100k steps.
Produces:
  plots/4arm_auc_top1.png
  plots/4arm_dim_usage.png
  plots/4arm_logscale.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
PLOTS_DIR = os.path.join(EXPERIMENT_DIR, "plots")
DATA_DIR  = os.path.join(EXPERIMENT_DIR, "data")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Load data ---
# Baselines: full 50k stitched CSVs (steps 1–50000).
# Squares:   full 100k stitched CSVs (steps 1–100000) produced by
#            scripts/stitch_square_100k_csvs.py.
paths = {
    "baseline_0.10": os.path.join(DATA_DIR, "tau_sweep_0_10_50k_losses.csv"),
    "baseline_0.20": os.path.join(DATA_DIR, "tau_sweep_0_20_50k_losses.csv"),
    "square_0.10":   os.path.join(DATA_DIR, "loss_ext_square_tau_0_10_100k_full_losses.csv"),
    "square_0.20":   os.path.join(DATA_DIR, "loss_ext_square_tau_0_20_100k_full_losses.csv"),
}

dfs = {}
for name, path in paths.items():
    df = pd.read_csv(path)
    dfs[name] = df
    print(f"{name}: {len(df)} rows, steps {df.step.min()}–{df.step.max()}")

# --- Style ---
# 4 distinct hues so noisy curves don't blend.
# τ=0.10: cool (blue/cyan); τ=0.20: warm (red/orange).
# Solid = baseline, dashed = square (loss family within τ).
styles = {
    "baseline_0.10": {"color": "#1f77b4", "linestyle": "-",  "label": "baseline τ=0.10"},  # blue
    "square_0.10":   {"color": "#17becf", "linestyle": "--", "label": "square τ=0.10"},    # cyan
    "baseline_0.20": {"color": "#d62728", "linestyle": "-",  "label": "baseline τ=0.20"},  # red
    "square_0.20":   {"color": "#ff7f0e", "linestyle": "--", "label": "square τ=0.20"},    # orange
}

ARM_ORDER = ["baseline_0.10", "square_0.10", "baseline_0.20", "square_0.20"]
WINDOW = 500


def rolling_mean(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ------------------------------------------------------------------ Figure 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "AUC and Top-1: square (100k) vs baseline (50k), τ=0.10 and τ=0.20",
    fontsize=13,
)

for arm in ARM_ORDER:
    df = dfs[arm]
    st = styles[arm]
    for ax, col, ylim in [
        (axes[0], "auc",  (0.86, 0.92)),
        (axes[1], "top1", (0.70, 0.79)),
    ]:
        raw = df[col]
        sm  = rolling_mean(raw, WINDOW)
        ax.plot(df["step"], raw, color=st["color"], linestyle=st["linestyle"],
                alpha=0.15, linewidth=0.8)
        ax.plot(df["step"], sm,  color=st["color"], linestyle=st["linestyle"],
                label=st["label"], linewidth=1.6)

for ax, col, ylim, title in [
    (axes[0], "auc",  (0.86, 0.92), "AUC"),
    (axes[1], "top1", (0.70, 0.79), "Top-1"),
]:
    ax.set_ylim(ylim)
    ax.set_xlabel("Step")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

fig.tight_layout()
out1 = os.path.join(PLOTS_DIR, "4arm_auc_top1.png")
fig.savefig(out1, dpi=150)
print(f"Saved: {out1}")
plt.close(fig)


# ------------------------------------------------------------------ Figure 2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Dimension usage: square (100k) vs baseline (50k), τ=0.10 and τ=0.20",
    fontsize=13,
)

for arm in ARM_ORDER:
    df = dfs[arm]
    st = styles[arm]
    for ax, col in [(axes[0], "u_batch"), (axes[1], "u_temporal")]:
        raw = df[col]
        sm  = rolling_mean(raw, WINDOW)
        ax.plot(df["step"], raw, color=st["color"], linestyle=st["linestyle"],
                alpha=0.15, linewidth=0.8)
        ax.plot(df["step"], sm,  color=st["color"], linestyle=st["linestyle"],
                label=st["label"], linewidth=1.6)

for ax, col, title in [
    (axes[0], "u_batch",    "U_batch"),
    (axes[1], "u_temporal", "U_temporal"),
]:
    ax.set_xlabel("Step")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

fig.tight_layout()
out2 = os.path.join(PLOTS_DIR, "4arm_dim_usage.png")
fig.savefig(out2, dpi=150)
print(f"Saved: {out2}")
plt.close(fig)


# ------------------------------------------------------------------ Figure 3: log-log
# Same 4 metrics in raw form, log-x (step) and log-y axes.
# AUC/Top-1 panels keep the linear-plot ylim band, displayed with log spacing.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Log-log convergence: square (100k) vs baseline (50k), τ=0.10 and τ=0.20",
    fontsize=13,
)

# (axis, column, title, ylim_or_None)
panels = [
    (axes[0, 0], "auc",        "AUC",        (0.86, 0.92)),
    (axes[0, 1], "top1",       "Top-1",      (0.70, 0.79)),
    (axes[1, 0], "u_batch",    "U_batch",    None),
    (axes[1, 1], "u_temporal", "U_temporal", None),
]

for arm in ARM_ORDER:
    df = dfs[arm]
    st = styles[arm]
    # avoid log(0) on step axis
    mask = df["step"] > 0
    step = df.loc[mask, "step"]
    for ax, col, _title, _ylim in panels:
        raw = df.loc[mask, col]
        sm  = rolling_mean(df[col], WINDOW).loc[mask]
        ax.plot(step, raw, color=st["color"], linestyle=st["linestyle"],
                alpha=0.15, linewidth=0.8)
        ax.plot(step, sm,  color=st["color"], linestyle=st["linestyle"],
                label=st["label"], linewidth=1.6)

for ax, _col, title, ylim in panels:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=500)
    ax.set_xlabel("Step (log scale)")
    ax.set_ylabel(title + " (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)

fig.tight_layout()
out3 = os.path.join(PLOTS_DIR, "4arm_logscale.png")
fig.savefig(out3, dpi=150)
print(f"Saved: {out3}")
plt.close(fig)


# ------------------------------------------------------------------ Statistics
# Per-step values are very noisy — single-row "final" numbers can be off by
# ±0.02 AUC. We report:
#   1. late-window means at the COMMON late window (40001–50000), which is
#      the last 10k of baseline coverage and where all 4 arms have data;
#   2. additional late-window means for each square arm at its OWN last 10k
#      (90001–100000), the analogous "late plateau" for the longer runs.

def window_summary(df, lo, hi):
    d = df[(df["step"] > lo) & (df["step"] <= hi)]
    return {
        "n":          len(d),
        "auc":        (d.auc.mean(),  d.auc.std()),
        "top1":       (d.top1.mean(), d.top1.std()),
        "u_batch":    d.u_batch.mean(),
        "u_temporal": d.u_temporal.mean(),
    }


print("\n=== Late-window means @ COMMON 40001–50000 (apples-to-apples) ===")
print(f"{'Arm':<20} {'n':>6} {'AUC':>15} {'Top-1':>15} {'U_batch':>9} {'U_temporal':>11}")
for arm in ARM_ORDER:
    s = window_summary(dfs[arm], 40000, 50000)
    print(f"{arm:<20} {s['n']:>6d} "
          f"{s['auc'][0]:.4f}±{s['auc'][1]:.4f}  "
          f"{s['top1'][0]:.4f}±{s['top1'][1]:.4f}  "
          f"{s['u_batch']:>9.4f} {s['u_temporal']:>11.4f}")

print("\n=== Late-window means @ OWN 90001–100000 (squares only) ===")
print(f"{'Arm':<20} {'n':>6} {'AUC':>15} {'Top-1':>15} {'U_batch':>9} {'U_temporal':>11}")
for arm in ("square_0.10", "square_0.20"):
    s = window_summary(dfs[arm], 90000, 100000)
    print(f"{arm:<20} {s['n']:>6d} "
          f"{s['auc'][0]:.4f}±{s['auc'][1]:.4f}  "
          f"{s['top1'][0]:.4f}±{s['top1'][1]:.4f}  "
          f"{s['u_batch']:>9.4f} {s['u_temporal']:>11.4f}")

# Welch tests over the FULL overlap window (5001–50000), where all 4 arms
# have data and we are well past warmup.
print("\n=== Welch t-tests on AUC and Top-1 (overlap window: steps 5001–50000) ===")
comparisons = [
    ("baseline_0.10", "square_0.10",   "baseline τ=0.10 vs square τ=0.10"),
    ("baseline_0.20", "square_0.20",   "baseline τ=0.20 vs square τ=0.20"),
    ("baseline_0.10", "baseline_0.20", "baseline τ=0.10 vs baseline τ=0.20 (sanity)"),
    ("square_0.10",   "square_0.20",   "square τ=0.10 vs square τ=0.20"),
]

for arm_a, arm_b, label in comparisons:
    print(f"\n  {label}")
    da = dfs[arm_a][(dfs[arm_a]["step"] > 5000) & (dfs[arm_a]["step"] <= 50000)]
    db = dfs[arm_b][(dfs[arm_b]["step"] > 5000) & (dfs[arm_b]["step"] <= 50000)]
    for col in ["auc", "top1"]:
        a = da[col].values
        b = db[col].values
        t, p = stats.ttest_ind(a, b, equal_var=False)
        sig = "YES" if p < 0.05 else "NO"
        print(
            f"    {col}: "
            f"A={a.mean():.4f}±{a.std():.4f} (n={len(a)})  "
            f"B={b.mean():.4f}±{b.std():.4f} (n={len(b)})  "
            f"t={t:.3f}  p={p:.2e}  sig@0.05={sig}"
        )
