#!/usr/bin/env python3
"""Plot the τ=0.03 arm-1 training trajectory: loss + 6 backbone metrics.

Reads sync_tau_sweep/checkpoints/tau_sweep_0_03_losses.csv (live, in-flight
file — handles current step <50k). Smooths with a 200-step rolling mean.
Saves to experiments/2026-05-08_exp_tau_sweep/plots/arm1_curves.png.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
CSV = REPO / "sync_tau_sweep/checkpoints/tau_sweep_0_03_losses.csv"
PLOT = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/arm1_curves.png"
PLOT.parent.mkdir(parents=True, exist_ok=True)


def smooth(x, w=200):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def main():
    rows = []
    with open(CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        print("No data yet")
        return
    step = np.array([int(r["step"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    r2_r = np.array([float(r["r2_random"]) for r in rows])
    r2_n = np.array([float(r["r2_naive"]) for r in rows])
    u_t = np.array([float(r["u_temporal"]) for r in rows])
    u_b = np.array([float(r["u_batch"]) for r in rows])
    auc = np.array([float(r["auc"]) for r in rows])
    top1 = np.array([float(r["top1"]) for r in rows])

    fig, axs = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    axs = axs.ravel()

    def panel(ax, y, title, color, ylim=None, hline=None, hline_label=None):
        sm = smooth(y)
        ax.plot(step, y, color=color, alpha=0.15, linewidth=0.5)
        if len(sm) > 0:
            ax.plot(step[len(step) - len(sm):], sm, color=color, linewidth=1.5,
                    label=f"{title} (smoothed 200)")
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=0.8,
                       label=hline_label or f"{hline:.4f}")
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    panel(axs[0], loss, "loss", "blue")
    panel(axs[1], r2_r, "R²_random", "green", ylim=(0.0, 1.0),
          hline=0.6839, hline_label="backbone-beta 167k = 0.6839")
    panel(axs[2], r2_n, "R²_naive", "green", ylim=(0.0, 1.0),
          hline=0.6080, hline_label="backbone-beta 167k = 0.6080")
    panel(axs[3], u_t, "U_temporal (per-slice)", "red",
          hline=0.0375, hline_label="backbone-beta 167k = 0.0375")
    panel(axs[4], u_b, "U_batch (per-slice)", "red",
          hline=0.0762, hline_label="backbone-beta 167k = 0.0762")
    panel(axs[5], auc, "AUC", "orange", ylim=(0.75, 1.0),
          hline=0.8966, hline_label="backbone-beta 167k = 0.8966")
    panel(axs[6], top1, "Top-1", "orange", ylim=(0.6, 1.0),
          hline=0.7531, hline_label="backbone-beta 167k = 0.7531")
    # Empty 8th panel — use it for U_b log scale to see early dynamics
    axs[7].plot(step, u_b, color="red", alpha=0.2, linewidth=0.5)
    sm = smooth(u_b)
    if len(sm) > 0:
        axs[7].plot(step[len(step) - len(sm):], sm, color="red", linewidth=1.5)
    axs[7].axhline(0.0762, color="gray", linestyle="--", linewidth=0.8,
                   label="backbone-beta 167k = 0.0762")
    axs[7].axhline(0.0427, color="black", linestyle=":", linewidth=0.8,
                   label="default init U_b = 0.0427")
    axs[7].set_yscale("log")
    axs[7].set_title("U_batch (log scale)")
    axs[7].grid(alpha=0.3, which="both")
    axs[7].legend(loc="best", fontsize=8)

    # Hide the unused 9th panel (3x3 grid, 8 panels used)
    axs[8].set_visible(False)
    for ax in axs[5:8]:
        ax.set_xlabel("step")
    fig.suptitle(
        f"τ=0.03 arm — current step {step[-1]} / 50000 "
        f"({step[-1] / 50000 * 100:.0f}%) — "
        f"reference: backbone-beta_167k (learnable τ→0.072)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=110, bbox_inches="tight")
    print(f"Saved {PLOT} ({step[-1]} steps)")


if __name__ == "__main__":
    main()
