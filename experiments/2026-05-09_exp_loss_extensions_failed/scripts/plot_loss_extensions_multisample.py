#!/usr/bin/env python3
"""Loss-extension multisample plot — 6 panels of mean ± stdev across N=10 batches.

Reads `results/loss_extensions_metrics_multisample.csv`. Renders 6 metric
panels as bar charts with error bars showing stdev across the N=10 held-out
batches.

Output: experiments/2026-05-09_exp_loss_extensions/plots/loss_extensions_eval_multisample.png
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_CSV = REPO / "experiments/2026-05-09_exp_loss_extensions/results/loss_extensions_metrics_multisample.csv"
OUT = REPO / "experiments/2026-05-09_exp_loss_extensions/plots/loss_extensions_eval_multisample.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (display_label, csv_name, color)
ARMS = [
    ("baseline τ=0.20",            "tau_sweep_0_20",            "#1f77b4"),
    ("Exp 3 +(h_t,f_t) pos",       "exp3_pos_htft_tau_0_20",    "#d62728"),
    ("Exp 4-only +f-cross-bc neg", "exp4_only_fnegs_tau_0_20",  "#2ca02c"),
    ("Exp 5 +skip-f neg (t↔t+2)",  "exp5_skip_fnegs_tau_0_20",  "#9467bd"),
]

METRICS = [
    ("r2_random",   "R²_random"),
    ("r2_naive",    "R²_naive"),
    ("u_temporal",  "U_temporal"),
    ("u_batch",     "U_batch"),
    ("auc",         "AUC"),
    ("top1",        "Top-1"),
]


def main() -> None:
    rows: dict[str, dict] = {}
    with open(EVAL_CSV) as f:
        for r in csv.DictReader(f):
            rows[r["name"]] = {k: (float(v) if v not in ("", None) else None)
                               for k, v in r.items() if k not in ("name", "loss_shape", "encoder_type")}

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    labels = [a[0] for a in ARMS]
    names = [a[1] for a in ARMS]
    colors = [a[2] for a in ARMS]
    xs = np.arange(len(ARMS))

    for ax, (key, title) in zip(axs, METRICS):
        means = [rows[n][f"{key}_mean"] if n in rows else None for n in names]
        stds = [rows[n][f"{key}_std"] if n in rows else None for n in names]
        keep = [(x, m, s, c) for x, m, s, c in zip(xs, means, stds, colors)
                if m is not None]
        ax.bar([k[0] for k in keep], [k[1] for k in keep],
               yerr=[k[2] for k in keep],
               color=[k[3] for k in keep], edgecolor="black", linewidth=0.5,
               capsize=4, error_kw=dict(ecolor="black", lw=1.0))
        ax.set_title(f"{title} — held-out (mean ± stdev, N=10)")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        for x, m, s, _c in keep:
            ax.annotate(f"{m:.4f}\n±{s:.4f}", (x, m),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7)

    fig.suptitle(
        "Loss extensions — held-out eval mean ± stdev "
        "(N=10 disjoint batches × B=256). Error bars are population stdev "
        "across the 10 batches.",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
