#!/usr/bin/env python3
"""Forest plot of every paired-bootstrap Δ in the experiment: Δ = GM(left arm) −
GM(right arm) on GIFT-Eval full-97, with the 90% interval from resampling the
97-task list. One panel per head × checkpoint; one row per arm comparison.
Filled marker = interval excludes zero (reliable); open marker = interval spans
zero. Reads results/pairwise_table.csv, writes plots/deltas_forest.png.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-15_no_encoder_redo/results"
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "deltas_forest.png")

CELLS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
# (A key, B key, row label) — Δ = GM(B) − GM(A), matching pairwise_table.csv.
# Ordered top→bottom; plotted bottom-up so the first row sits at the top.
ROWS = [
    ("base_enc3", "noenc_base",    "no-enc base − enc-3 base"),
    ("base_enc6", "noenc_base",    "no-enc base − enc-6 base"),
    ("cpc_enc3",  "noenc_cpc",     "no-enc +CPC − enc-3 +CPC"),
    ("cpc_enc6",  "noenc_cpc",     "no-enc +CPC − enc-6 +CPC"),
    ("noenc_base", "noenc_cpc",    "no-enc +CPC − no-enc base"),
    ("noenc_base", "noenc_cpcall", "no-enc +CPC_All − no-enc base"),
    ("noenc_cpc",  "noenc_cpcall", "no-enc +CPC_All − no-enc +CPC"),
    ("cpc_enc3",  "noenc_cpcall",  "no-enc +CPC_All − enc-3 +CPC"),
    ("cpc_enc6",  "noenc_cpcall",  "no-enc +CPC_All − enc-6 +CPC"),
]


def main():
    d = {(r["A"], r["B"], r["head"], r["ckpt"]):
         (float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"]))
         for r in csv.DictReader(open(f"{RES}/pairwise_table.csv"))}
    ys = list(range(len(ROWS)))[::-1]  # first row at top
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.5), sharey=True)
    for ax, (head, ckpt) in zip(axes, CELLS):
        for y, (a, b, _) in zip(ys, ROWS):
            delta, lo, hi = d[(a, b, head, ckpt)]
            reliable = not (lo < 0 < hi)
            ax.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]], fmt="o",
                        color="#1f77b4" if reliable else "0.6",
                        mfc="#1f77b4" if reliable else "white",
                        ms=6, capsize=3, lw=1.4)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_title(f"{head} head · {ckpt}", fontsize=11)
        ax.set_xlabel("Δ GM-Relative MASE")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[2] for r in ROWS], fontsize=9)
    fig.suptitle("Paired-bootstrap Δ = GM(left) − GM(right), 90% interval over the 97 tasks "
                 "(filled = excludes 0)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
