#!/usr/bin/env python3
"""τ-sweep multisample plot — 6 metric panels, mean ± stdev across batches.

Reads `results/tau_sweep_metrics_multisample.csv` (or the N=50 variant if
available; pass `--csv path` to override). Renders 6 metric panels
(R²_random, R²_naive, U_temporal, U_batch, AUC, Top-1) with a categorical
x-axis: arms left-to-right at fixed positions (no log scaling), a wide
shaded box per arm spanning [mean − stdev, mean + stdev], a thick line at
the mean, and `backbone-beta_167k` as a dashed reference.

Output: experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_eval_multisample.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO / "experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_multisample.csv"
OUT = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_eval_multisample.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Display arms in τ-numerical order, with the learnable-τ arm placed
# adjacent to its CONVERGED value (τ ≈ 0.069 ≈ τ=0.07) — its behaviour
# matches its converged τ, not its init. Single τ=0.20 entry uses the v2
# checkpoint (the run that has both FINAL.pth and full trajectory CSV).
ARMS = [
    ("τ=0.03",                     "tau_sweep_0_03",           "#1f77b4"),
    ("τ=0.05",                     "tau_sweep_0_05",           "#2ca02c"),
    ("τ=0.07",                     "tau_sweep_0_07",           "#9467bd"),
    ("τ=0.10 → 0.069 (learnable)", "tau_sweep_learnable_0_10", "#17becf"),
    ("τ=0.10",                     "tau_sweep_0_10",           "#d62728"),
    ("τ=0.20",                     "tau_sweep_0_20_v2",        "#ff7f0e"),
]

BETA = dict(r2_random=0.6839, r2_naive=0.6080, u_temporal=0.0375,
            u_batch=0.0762, auc=0.8966, top1=0.7531)

# R² panels zoomed to the data-relevant window (R²_random ∈ ~[0.6, 0.85];
# R²_naive ∈ ~[0.45, 0.8]). U / AUC / Top-1 panels keep auto-scale.
METRICS = [
    ("r2_random",   "R²_random",  (0.60, 0.85)),
    ("r2_naive",    "R²_naive",   (0.45, 0.80)),
    ("u_temporal",  "U_temporal", None),
    ("u_batch",     "U_batch",    None),
    ("auc",         "AUC",        None),
    ("top1",        "Top-1",      None),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    rows: dict[str, dict] = {}
    n_samples_seen: set[int] = set()
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            rows[r["name"]] = {k: (float(v) if v not in ("", None) else None)
                               for k, v in r.items() if k not in ("name", "encoder_type")}
            rows[r["name"]]["name"] = r["name"]
            rows[r["name"]]["encoder_type"] = r["encoder_type"]
            if "n_samples" in r and r["n_samples"]:
                n_samples_seen.add(int(float(r["n_samples"])))
    n_samples = max(n_samples_seen) if n_samples_seen else 0

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()

    arm_labels = [a[0] for a in ARMS]
    arm_names = [a[1] for a in ARMS]
    arm_colors = [a[2] for a in ARMS]
    xs = np.arange(len(arm_labels))
    box_half = 0.36

    for ax, (key, title, ylim) in zip(axs, METRICS):
        for x, name, color in zip(xs, arm_names, arm_colors):
            r = rows.get(name)
            if r is None:
                continue
            mean = r[f"{key}_mean"]
            std = r[f"{key}_std"]
            # Shaded box: [mean - std, mean + std]
            ax.bar(x, height=2 * std, bottom=mean - std,
                   width=2 * box_half, color=color, alpha=0.40,
                   edgecolor=color, linewidth=1.2, zorder=2)
            # Mean line on top
            ax.hlines(mean, x - box_half, x + box_half,
                      color=color, linewidth=2.6, zorder=3)
            # Annotate mean ± std numerically
            ax.annotate(f"{mean:.4f}\n±{std:.4f}", (x, mean + std),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7)
        # Reference line for backbone-beta_167k (single-batch on the same
        # held-out eval batch; here as a context anchor only).
        ax.axhline(BETA[key], color="gray", linestyle="--", linewidth=0.9,
                   label=f"backbone-β 167k = {BETA[key]:.4f}")
        ax.set_title(f"{title} — held-out eval (mean ± stdev)")
        ax.set_xticks(xs)
        ax.set_xticklabels(arm_labels, rotation=20, ha="right", fontsize=9)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"τ-sweep — held-out eval (N={n_samples} disjoint batches × B=256). "
        "Boxes span ±1 stdev across batches; line at the mean.",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
