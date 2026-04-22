#!/usr/bin/env python3
"""Plot training curves for the CONTROL and MIX arms side by side.

Reads the two _losses.csv files and plots loss + gap in log-space on twin
axes. Also plots the EMA for each to make short 30k runs comparable.

Usage:
    python experiments/periodic-synth-mix/scripts/plot_training_curves.py \\
        --ctrl sync_periodic_synth/checkpoints/tiny_v3c_ctrl_losses.csv \\
        --mix  sync_periodic_synth/checkpoints/tiny_v3c_mix_losses.csv \\
        --out  experiments/periodic-synth-mix/plots/training_curves.png
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def _ema(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * x[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctrl = pd.read_csv(args.ctrl)
    mix = pd.read_csv(args.mix)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100, sharex=True)

    # Loss panel
    ax1.plot(ctrl["step"], ctrl["loss"], alpha=0.2, color="C0", linewidth=0.5)
    ax1.plot(ctrl["step"], _ema(ctrl["loss"].values), color="C0", label="CONTROL")
    ax1.plot(mix["step"], mix["loss"], alpha=0.2, color="C1", linewidth=0.5)
    ax1.plot(mix["step"], _ema(mix["loss"].values), color="C1", label="MIX")
    ax1.set_ylabel("contrastive loss")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gap panel
    ax2.plot(ctrl["step"], ctrl["gap"], alpha=0.2, color="C0", linewidth=0.5)
    ax2.plot(ctrl["step"], _ema(ctrl["gap"].values), color="C0", label="CONTROL")
    ax2.plot(mix["step"], mix["gap"], alpha=0.2, color="C1", linewidth=0.5)
    ax2.plot(mix["step"], _ema(mix["gap"].values), color="C1", label="MIX")
    ax2.set_ylabel("FF − FP gap")
    ax2.set_xlabel("step")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Training curves — CONTROL vs MIX")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.out}")

    # Short stats
    for name, df in [("CONTROL", ctrl), ("MIX", mix)]:
        last = df.tail(500)
        print(f"{name:>7s} last-500 mean loss={last['loss'].mean():.4f} "
              f"mean gap={last['gap'].mean():.4f} "
              f"final step={df['step'].max()}")


if __name__ == "__main__":
    main()
