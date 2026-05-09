#!/usr/bin/env python3
"""τ-sweep multisample plot — 6 panels of mean ± stdev across N=10 batches.

Reads `results/tau_sweep_metrics_multisample.csv`. Renders 6 metric panels
(R²_random, R²_naive, U_temporal, U_batch, AUC, Top-1) on a log-x τ axis
with explicit ticks at the swept values, error bars showing stdev across
the N=10 batches, and `backbone-beta_167k` reference as a dashed line.

The 7 fixed-τ arms are plotted at their declared τ. The
`tau_sweep_learnable_0_10` arm converged from init=0.10 to τ ≈ 0.069 by
step 15k; we plot it at 0.069 with a distinct marker so it does not
collide with `tau_sweep_0_07`.

Output: experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_eval_multisample.png
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_CSV = REPO / "experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_multisample.csv"
OUT = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_eval_multisample.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (display_label, csv_name, plotted_tau, marker, color)
ARMS = [
    ("τ=0.03",                  "tau_sweep_0_03",            0.03,  "o", "#1f77b4"),
    ("τ=0.05",                  "tau_sweep_0_05",            0.05,  "o", "#2ca02c"),
    ("τ=0.07",                  "tau_sweep_0_07",            0.07,  "o", "#9467bd"),
    ("τ=0.10",                  "tau_sweep_0_10",            0.10,  "o", "#d62728"),
    ("τ=0.20",                  "tau_sweep_0_20",            0.20,  "o", "#ff7f0e"),
    ("τ=0.20 v2",               "tau_sweep_0_20_v2",         0.20,  "s", "#bcbd22"),
    ("learnable_τ → 0.069",     "tau_sweep_learnable_0_10",  0.069, "D", "#17becf"),
]

BETA = dict(r2_random=0.6839, r2_naive=0.6080, u_temporal=0.0375,
            u_batch=0.0762, auc=0.8966, top1=0.7531)

METRICS = [
    ("r2_random",   "R²_random",  None),
    ("r2_naive",    "R²_naive",   None),
    ("u_temporal",  "U_temporal", None),
    ("u_batch",     "U_batch",    None),
    ("auc",         "AUC",        None),
    ("top1",        "Top-1",      None),
]


def main() -> None:
    rows: dict[str, dict] = {}
    with open(EVAL_CSV) as f:
        for r in csv.DictReader(f):
            rows[r["name"]] = {k: (float(v) if v not in ("", None) else None)
                               for k, v in r.items() if k not in ("name", "encoder_type")}
            rows[r["name"]]["name"] = r["name"]
            rows[r["name"]]["encoder_type"] = r["encoder_type"]

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    for ax_i, (ax, (key, title, _)) in enumerate(zip(axs, METRICS)):
        for label, name, tau, marker, color in ARMS:
            r = rows.get(name)
            if r is None:
                continue
            mean = r[f"{key}_mean"]
            std = r[f"{key}_std"]
            ax.errorbar([tau], [mean], yerr=[std], fmt=marker, color=color,
                        markersize=9, capsize=5, linewidth=1.4,
                        markeredgecolor="black", markeredgewidth=0.6,
                        label=label)
        ax.axhline(BETA[key], color="gray", linestyle="--", linewidth=0.9,
                   label=f"backbone-β 167k = {BETA[key]:.4f}")
        ax.set_xscale("log")
        ax.set_title(f"{title} — held-out (mean ± stdev, N=10)")
        ax.set_xlabel("τ")
        # Explicit ticks at the swept values. 0.069 (learnable converged)
        # is too close to 0.07 to label legibly so we drop its tick.
        tick_taus = [0.03, 0.05, 0.07, 0.10, 0.20]
        ax.set_xticks(tick_taus)
        ax.set_xticklabels([f"{t:g}" for t in tick_taus], fontsize=9)
        ax.set_xlim(0.025, 0.25)
        ax.grid(alpha=0.3, which="both")
        # Only put the per-arm legend on the first panel; share the axis on
        # the rest. (β reference line is a separate label per panel.)
        if ax_i == 0:
            ax.legend(loc="lower right", fontsize=7)
        else:
            # show only the β reference line on other panels
            handles, labels_ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles[-1:], labels_[-1:], loc="best", fontsize=7)

    fig.suptitle(
        "τ-sweep — held-out eval mean ± stdev (N=10 disjoint batches × B=256). "
        "Error bars are population stdev across the 10 batches.",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
