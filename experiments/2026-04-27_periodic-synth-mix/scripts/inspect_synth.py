#!/usr/bin/env python3
"""Draw 100 sample series from the periodic synthesizer and inspect them.

Outputs
-------
- experiments/2026-04-27_periodic-synth-mix/plots/inspect_grid.png
    10x10 grid of series with primitive name + key params in title.
- experiments/2026-04-27_periodic-synth-mix/plots/inspect_metadata.txt
    Per-series metadata + first/last/min/max/mean/std and a head/tail
    sample of values so I can read numbers (graphs are coarse).

Usage: python3.11 experiments/2026-04-27_periodic-synth-mix/scripts/inspect_synth.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from repo root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.synthetic_periodic import generate_periodic_batch, primitive_name


def main():
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "plots"))
    os.makedirs(out_dir, exist_ok=True)

    N = 100
    T = 1024
    SEED = 20260422  # today's date; reproducible

    # Draw N independent series. We ask for batch_size=N, C=1 so we get
    # exactly N distinct primitives without reshaping games.
    X, meta = generate_periodic_batch(
        batch_size=N, T_raw=T, C=1, seed=SEED, return_meta=True)
    x = X.squeeze(-1).numpy()              # [N, T]

    # -- Grid plot ------------------------------------------------------------
    fig, axes = plt.subplots(10, 10, figsize=(20, 20), dpi=80)
    for i, ax in enumerate(axes.flat):
        ax.plot(x[i], linewidth=0.7)
        title = (
            f"#{i:03d} {primitive_name(meta['primitive'][i])[:3]} "
            f"P={meta['spp'][i]:.1f} "
            f"{'E' if meta['use_env'][i] else '-'} "
            f"g={meta['env_gain'][i]:.1f} "
            f"s={meta['scale'][i]:.1f}"
        )
        ax.set_title(title, fontsize=6)
        ax.tick_params(labelsize=4)
        # Show period markers
        P = meta['spp'][i]
        if P < T / 2:
            for k in range(1, min(int(T / P), 5)):
                ax.axvline(k * P + meta['phase'][i] * P, color='r',
                           linewidth=0.3, alpha=0.3)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "inspect_grid.png")
    plt.savefig(grid_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {grid_path}")

    # -- Metadata text dump ---------------------------------------------------
    meta_path = os.path.join(out_dir, "inspect_metadata.txt")
    with open(meta_path, "w") as f:
        f.write("# 100 synthetic periodic samples\n")
        f.write(f"# seed={SEED} T={T}\n\n")

        counts = np.bincount(meta["primitive"], minlength=3)
        f.write(f"# primitive counts: sin={counts[0]} square={counts[1]} saw={counts[2]}\n")
        f.write(f"# envelope used on {int(meta['use_env'].sum())}/{N} series\n\n")

        for i in range(N):
            y = x[i]
            flip_s = "True" if meta['sign_flip'][i] else "False"
            env_s = "True" if meta['use_env'][i] else "False"
            f.write(
                f"[{i:03d}] {primitive_name(meta['primitive'][i]):>8s} "
                f"P={meta['spp'][i]:7.2f} phase={meta['phase'][i]:.3f} "
                f"flip={flip_s:>5s} "
                f"env={env_s:>5s} "
                f"gain={meta['env_gain'][i]:7.3f} "
                f"scale={meta['scale'][i]:8.3f}\n"
            )
            f.write(
                f"        min={y.min():.4g} max={y.max():.4g} "
                f"mean={y.mean():.4g} std={y.std():.4g}\n"
            )
            # Head + tail sample (12 values each)
            head = " ".join(f"{v:7.3f}" for v in y[:12])
            tail = " ".join(f"{v:7.3f}" for v in y[-12:])
            f.write(f"        head: {head}\n")
            f.write(f"        tail: {tail}\n\n")

    print(f"wrote {meta_path}")

    # -- Aggregate sanity print to stdout -------------------------------------
    print()
    print(f"{'seed':>6s} {N}  T={T}")
    print(f"primitive counts: sin={counts[0]} square={counts[1]} saw={counts[2]}")
    print(f"envelope fraction: {meta['use_env'].mean():.3f}")
    print(f"spp range: [{meta['spp'].min():.2f}, {meta['spp'].max():.2f}]")
    print(f"scale range: [{meta['scale'].min():.3f}, {meta['scale'].max():.3f}]")
    print(f"value range: [{x.min():.3g}, {x.max():.3g}]")
    print(f"finite: {bool(np.isfinite(x).all())}")


if __name__ == "__main__":
    main()
