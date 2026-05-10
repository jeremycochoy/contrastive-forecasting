#!/usr/bin/env python3
"""Per-sample variance plot — transformer encoder vs τ=0.10 GRU baseline.

Reads two per-sample CSVs (one per arm, written by `eval_held_out.py`)
and shows the distribution shape for AUC and Top-1: a strip-jitter of
all 50 held-out batch values per arm with mean ± stdev error bars
overlaid. Lets the eye check whether the gap is uniform across batches
or comes from a few outliers.

Output:
    experiments/2026-05-10_exp_transformer_encoder/plots/
    held_out_persample_auc_top1.png
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKTREE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MAIN_CHECKOUT = os.environ.get(
    "CFCAST_REPO_ROOT", "/home/jupyter/contrastive-forecasting")

EXP_DIR = os.path.join(WORKTREE_ROOT, "experiments",
                       "2026-05-10_exp_transformer_encoder")
RESULTS_DIR = os.path.join(MAIN_CHECKOUT, "experiments",
                           "2026-05-10_exp_transformer_encoder", "results")
OUT_PATH = os.path.join(EXP_DIR, "plots", "held_out_persample_auc_top1.png")

ARMS = [
    # (label, persample_csv_basename, color)
    ("τ=0.10 GRU baseline",
     "tau_sweep_0_10_baseline_metrics_persample_n50.csv",
     "#1f77b4"),
    ("τ=0.10 transformer encoder",
     "transformer_encoder_tau_0_10_50k_metrics_persample_n50.csv",
     "#ff7f0e"),
]


def main() -> None:
    arms = []
    for label, fname, color in ARMS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        arms.append((label, df, color))
        print(f"loaded {label}: N={len(df)}, "
              f"AUC={df.auc.mean():.4f}±{df.auc.std(ddof=0):.4f}, "
              f"Top1={df.top1.mean():.4f}±{df.top1.std(ddof=0):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax, key, title in [
            (axes[0], "auc", "AUC (held-out, N=50 batches)"),
            (axes[1], "top1", "Top-1 (held-out, N=50 batches)")]:
        for i, (label, df, color) in enumerate(arms):
            vals = df[key].values
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                       s=22, color=color, alpha=0.5, edgecolors="none",
                       label=None)
            mean = vals.mean()
            std = vals.std(ddof=0)
            sem = std / np.sqrt(len(vals))
            # Mean line + ±1 stdev whiskers + ±1 SEM as a thicker bar
            ax.errorbar(i, mean, yerr=std, fmt="_", color=color,
                        markersize=24, capsize=8, capthick=1.5,
                        elinewidth=1.0, label=f"{label}\nmean={mean:.4f}  σ={std:.4f}  SEM={sem:.4f}")
            ax.errorbar(i, mean, yerr=sem, fmt="none", color=color,
                        capsize=14, capthick=2.5, elinewidth=2.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([a[0] for a in arms], rotation=0, fontsize=9)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", alpha=0.3)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    fig.suptitle("Held-out per-sample variance · 50 disjoint B=256 batches", fontsize=11)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
