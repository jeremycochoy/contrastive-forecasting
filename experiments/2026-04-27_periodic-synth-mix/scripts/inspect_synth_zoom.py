#!/usr/bin/env python3
"""Zoomed-in view of periodic synth samples.

Plots each sample over ~3 periods so the waveform is visible (full-T plot
is dominated by aliasing when the period is short vs 1024 samples).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.synthetic_periodic import generate_periodic_batch, primitive_name


def main():
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "plots"))
    os.makedirs(out_dir, exist_ok=True)

    N = 36
    T = 1024
    SEED = 20260422

    X, meta = generate_periodic_batch(
        batch_size=N, T_raw=T, C=1, seed=SEED, return_meta=True)
    x = X.squeeze(-1).numpy()

    # Two figures: full series + zoom on ~3 periods starting at t=0
    fig, axes = plt.subplots(6, 6, figsize=(24, 16), dpi=80)
    for i, ax in enumerate(axes.flat):
        P = float(meta['spp'][i])
        # show first ~3 periods, at least 64 points
        zoom_len = min(T, max(64, int(3 * P)))
        ax.plot(x[i, :zoom_len], linewidth=1.2)
        title = (
            f"#{i:02d} {primitive_name(meta['primitive'][i])} "
            f"P={P:.1f} phase={meta['phase'][i]:.2f} "
            f"{'env' if meta['use_env'][i] else 'no-env'} "
            f"gain={meta['env_gain'][i]:.2f} "
            f"scale={meta['scale'][i]:.2f}"
        )
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        # mark period boundaries
        phase = meta['phase'][i]
        k = 0
        while True:
            xpos = (k - phase) * P
            if xpos > zoom_len:
                break
            if xpos >= 0:
                ax.axvline(xpos, color='r', linewidth=0.5, alpha=0.4)
            k += 1
    plt.tight_layout()
    zoom_path = os.path.join(out_dir, "inspect_zoom.png")
    plt.savefig(zoom_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {zoom_path}")

    # One panel per long-period series (P > 128) — these would be aliased on
    # the 10x10 grid but are clear at full-T.
    long_idx = np.where(meta['spp'] > 128)[0]
    if len(long_idx) > 0:
        n_long = min(len(long_idx), 12)
        fig, axes = plt.subplots(4, 3, figsize=(15, 12), dpi=80)
        for ax, idx in zip(axes.flat, long_idx[:n_long]):
            ax.plot(x[idx], linewidth=1.0)
            P = float(meta['spp'][idx])
            title = (
                f"#{idx:02d} {primitive_name(meta['primitive'][idx])} "
                f"P={P:.1f} "
                f"{'env' if meta['use_env'][idx] else 'no-env'} "
                f"gain={meta['env_gain'][idx]:.2f}"
            )
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3)
        # Hide unused axes
        for ax in axes.flat[n_long:]:
            ax.axis('off')
        plt.tight_layout()
        long_path = os.path.join(out_dir, "inspect_long_period.png")
        plt.savefig(long_path, bbox_inches="tight")
        plt.close()
        print(f"wrote {long_path}")


if __name__ == "__main__":
    main()
