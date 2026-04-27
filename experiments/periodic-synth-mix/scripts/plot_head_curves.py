#!/usr/bin/env python3
"""Plot head training curves (MSE reconstruction loss)."""
from __future__ import annotations
import argparse, os
import numpy as np, pandas as pd


def _ema(x, a=0.01):
    out = np.empty_like(x, dtype=float); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (1 - a) * out[i - 1] + a * x[i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctrl = pd.read_csv(args.ctrl)
    mix = pd.read_csv(args.mix)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    ax.plot(ctrl["step"], ctrl["loss"], alpha=0.15, color="C0", linewidth=0.5)
    ax.plot(ctrl["step"], _ema(ctrl["loss"].values), color="C0",
            label=f"CONTROL (final EMA={_ema(ctrl['loss'].values)[-1]:.4f})")
    ax.plot(mix["step"], mix["loss"], alpha=0.15, color="C1", linewidth=0.5)
    ax.plot(mix["step"], _ema(mix["loss"].values), color="C1",
            label=f"MIX (final EMA={_ema(mix['loss'].values)[-1]:.4f})")
    ax.set_yscale("log")
    ax.set_ylabel("head MSE loss (log)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title("R1 forecasting-head training loss — CONTROL vs MIX")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    plt.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
