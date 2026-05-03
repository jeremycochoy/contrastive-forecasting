"""Bar-chart comparison of GIFT-Eval geometric means across the τ-sweep
arms + learnable τ.

Three bar groups (one per metric: GM-MASE, GM-MAPE_SN, GM-CRPS_SN). Within
each group, one bar per arm. τ=0.07 has no eval (run halted) so its bar
appears as an explicit empty placeholder. The Aksu Moirai-Small reference
(GM-MAPE_SN=0.882, GM-CRPS_SN=0.642) is overlaid as a horizontal dashed
line per metric where applicable.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(os.path.dirname(HERE), "plots", "eval_metrics_bars.png")

ARMS = [
    ("τ=0.05",      "tab:blue",   f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau005/results/all_results.csv"),
    ("τ=0.07",      "tab:orange", f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau007/results/all_results.csv"),
    ("τ=0.20",      "tab:green",  f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau020/results/all_results.csv"),
    ("learnable τ", "tab:red",    f"{ROOT}/sync_realonly_4096_smaller_learnable_tau/learnable/results/all_results.csv"),
]

METRICS = [
    ("GM-MASE",    "eval_metrics/MASE[0.5]",    None),
    ("GM-MAPE_SN", "eval_metrics/SN_MAPE_ratio", 0.882),
    ("GM-CRPS_SN", "eval_metrics/SN_WQL_ratio",  0.642),
]


def gm(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) == 0:
        return float("nan")
    return float(np.exp(np.log(s).mean()))


def arm_metric(path, col):
    if not os.path.exists(path):
        return float("nan")
    df = pd.read_csv(path)
    if col not in df.columns or len(df) == 0:
        return float("nan")
    return gm(df[col])


fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
n_arms = len(ARMS)
labels = [a[0] for a in ARMS]
colors = [a[1] for a in ARMS]

for ax, (mname, mcol, aksu_target) in zip(axes, METRICS):
    vals = [arm_metric(p, mcol) for _, _, p in ARMS]
    xs = np.arange(n_arms)
    bars = ax.bar(xs, [0 if np.isnan(v) else v for v in vals], color=colors, alpha=0.85)
    for x, v, bar in zip(xs, vals, bars):
        if np.isnan(v):
            ax.text(x, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.05,
                    "no eval", ha="center", va="bottom", fontsize=9, color="gray", style="italic")
        else:
            ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    if aksu_target is not None:
        ax.axhline(aksu_target, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Aksu MOIRAI-S = {aksu_target}")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(mname)
    ax.set_ylabel(mname)
    ax.grid(True, axis="y", alpha=0.3)
    if all(np.isnan(v) or v > 0 for v in vals):
        # padded y-axis for label space above bars
        max_val = max((v for v in vals if not np.isnan(v)), default=1.0)
        ax.set_ylim(0, max_val * 1.18)

fig.suptitle("GIFT-Eval geometric means — small_v1 (61k rows, 47-epoch memorization regime)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Wrote {OUT_PNG}")
