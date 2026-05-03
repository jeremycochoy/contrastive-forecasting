"""Backbone + head loss curves across τ ∈ {0.05, 0.07, 0.20} + learnable τ.

Reads the local sync directories in the main checkout and produces a 2-row
log-log figure: backbone contrastive loss (top) and quantile head loss
(bottom). Each curve is a 100-step rolling mean of the raw per-step loss;
the raw loss is plotted faintly behind for context.

Saved under the same dir as this script (experiments/.../plots/).
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(os.path.dirname(HERE), "plots", "loss_curves.png")

BACKBONE = [
    ("τ=0.05",      "tab:blue",   f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau005/checkpoints/tiny_realonly_4096_smaller_tau005_losses.csv"),
    ("τ=0.07",      "tab:orange", f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau007/checkpoints/tiny_realonly_4096_smaller_tau007_losses.csv"),
    ("τ=0.20",      "tab:green",  f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau020/checkpoints/tiny_realonly_4096_smaller_tau020_losses.csv"),
    ("learnable τ", "tab:red",    f"{ROOT}/sync_realonly_4096_smaller_learnable_tau/learnable/checkpoints/tiny_realonly_4096_smaller_learnable_tau_losses.csv"),
]

HEAD = [
    ("τ=0.05",      "tab:blue",   f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau005/checkpoints/R1q_realonly_4096_smaller_tau005_losses.csv"),
    ("τ=0.07",      "tab:orange", f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau007/checkpoints/R1q_realonly_4096_smaller_tau007_losses.csv"),
    ("τ=0.20",      "tab:green",  f"{ROOT}/sync_realonly_4096_smaller_tau_sweep/tau020/checkpoints/R1q_realonly_4096_smaller_tau020_losses.csv"),
    ("learnable τ", "tab:red",    f"{ROOT}/sync_realonly_4096_smaller_learnable_tau/learnable/checkpoints/R1q_realonly_4096_smaller_learnable_tau_losses.csv"),
]

ROLL = 100


def load(path):
    """Load a losses CSV, merging the .prev rotation if present, dedupe by step."""
    parts = []
    prev = path + ".prev"
    if os.path.exists(prev):
        try:
            parts.append(pd.read_csv(prev))
        except Exception as e:
            print(f"  warn: could not read {prev}: {e}", file=sys.stderr)
    if os.path.exists(path):
        parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame(columns=["step", "loss"])
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)


def plot_panel(ax, sources, ylabel, title, ymin):
    for label, color, path in sources:
        df = load(path)
        if len(df) == 0:
            continue
        smooth = df["loss"].rolling(ROLL, min_periods=1).mean()
        ax.plot(df["step"], df["loss"], color=color, alpha=0.10, linewidth=0.5)
        gap_note = ""
        if df["step"].iloc[0] > 100:
            gap_note = f", missing 0–{int(df['step'].iloc[0])-1}"
        ax.plot(df["step"], smooth, color=color, linewidth=1.6,
                label=f"{label}  (n={len(df):,}, last={df['loss'].iloc[-1]:.4f}{gap_note})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel(f"{ylabel} (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)


fig, (ax_bb, ax_head) = plt.subplots(2, 1, figsize=(12, 9))
plot_panel(ax_bb, BACKBONE,
           ylabel="contrastive loss",
           title="Backbone — τ-sweep + learnable τ (smaller arch, EWMA-128, bs=96, T=4096, small_v1)",
           ymin=4.0)
plot_panel(ax_head, HEAD,
           ylabel="quantile head loss",
           title="Forecasting head (R1q) — τ=0.05 / 0.07 / 0.20 / learnable",
           ymin=0.05)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Wrote {OUT_PNG}")
