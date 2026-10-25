"""The terms of the total loss as curves, one panel per run: the mini run
grn_ema099995_f6_rep01 (yellow in the score figure: 6 forecaster layers, the
L_rep weight 0.1 at each step), grn_ema099995 (magenta, 3 forecaster layers)
and grn_ema099995_f6 (grey, 6 forecaster layers). Magenta and grey have the
L_rep weight 1 to 0 over the first 10,000 steps. The log has no L_rep after
that. The other loss flags are the same (see plot_loss_composition_099995.py).

Each panel has four curves, each term as it enters the total loss:
L_align at depth 0 (column l_align), L_align at depths 1 to 3 (their sum: the
loss minus the other terms), L_rep x its weight, and SIGReg on e plus SIGReg on
h. Each value is the mean over log-spaced bins of steps (BINS bins from 100 to
665k), drawn only when the run logged all the steps of the bin. Writes
results/loss_curves_099995_binned.tsv and plots/loss_curves_099995.png.

Usage: python3 plot_loss_curves_099995.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

from plot_loss_composition_099995 import RUNS as COMPOSITION_RUNS, RUNS_DIR, logged_parts

STUDY = Path(__file__).resolve().parent.parent
TABLE = STUDY / "results" / "loss_curves_099995_binned.tsv"
OUT = STUDY / "plots" / "loss_curves_099995.png"
FIRST, TOTAL, BINS = 100, 665_000, 80
TICKS_K = [0.1, 1, 10, 40, 100, 240, 665]
Y_TOP = 1.7  # L_rep x its weight of magenta and grey goes above it before 9k
MINI_DIR = RUNS_DIR / "grn_ema099995_f6_rep01/ckpt"
# run -> ([(loss file, last step taken from it)], panel title)
RUNS = {
    "grn_ema099995_f6_rep01": ([(MINI_DIR / "grn_ema099995_f6_rep01_losses.csv", 40_000),
                                (MINI_DIR / "grn_ema099995_f6_rep01_r2_losses.csv", 180_000),
                                (MINI_DIR / "grn_ema099995_f6_rep01_r3_losses.csv", None)],
                               "Yellow: 6 forecaster layers, L_rep weight 0.1 at each step"),
    "grn_ema099995": (COMPOSITION_RUNS["grn_ema099995"][0],
                      "Magenta: 3 forecaster layers, L_rep weight 1 to 0 by 10k"),
    "grn_ema099995_f6": (COMPOSITION_RUNS["grn_ema099995_f6"][0],
                         "Grey: 6 forecaster layers, L_rep weight 1 to 0 by 10k"),
}
# term -> (columns of logged_parts summed, colour, label)
TERMS = {
    "align_rest": ([5], "#fd8d3c", "L_align, depths 1 to 3 (sum)"),
    "align_d0": ([2], "#a63603", "L_align, depth 0"),
    "rep": ([1], "#3182bd", "L_rep x its weight"),
    "sigreg": ([3, 4], "#31a354", "SIGReg on e + SIGReg on h"),
}


def binned_terms(steps, parts):
    """(bin centre, mean of each term) of each log-spaced bin logged in full."""
    terms = np.column_stack([parts[:, cols].sum(axis=1) for cols, _, _ in TERMS.values()])
    edges = np.geomspace(FIRST, TOTAL, BINS + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (steps >= low) & (steps < high)
        if steps[-1] >= high and inside.any():
            rows.append((np.sqrt(low * high), *terms[inside].mean(axis=0)))
    return rows


def write_table():
    with open(TABLE, "w") as f:
        f.write("run\tbin_centre_step\t" + "\t".join(TERMS) + "\n")
        for run, (files, _) in RUNS.items():
            for row in binned_terms(*logged_parts(files)):
                f.write(f"{run}\t{row[0]:.0f}\t" + "\t".join(f"{v:.6g}" for v in row[1:]) + "\n")


def read_table():
    """{run: {column: array}} from TABLE."""
    with open(TABLE) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {run: {k: np.array([float(r[k]) for r in rows if r["run"] == run])
                  for k in ["bin_centre_step", *TERMS]} for run in RUNS}


def draw_run(ax, cols, title):
    """The four terms of one run. A zero (L_rep after 10k) is not drawn. A
    note gives the first L_rep value when it lies above the panel."""
    for term, (_, colour_hex, label) in TERMS.items():
        values = np.where(cols[term] > 0, cols[term], np.nan)
        ax.plot(cols["bin_centre_step"], values, color=colour_hex, lw=2.2, label=label)
    if cols["rep"][0] > Y_TOP:
        ax.text(FIRST * 1.15, Y_TOP * 0.97, f"L_rep x its weight: {cols['rep'][0]:.1f} at step 100, "
                "above the panel up to 9k", fontsize=8.5, va="top", color=TERMS["rep"][1])
    ax.set_ylim(0, Y_TOP)
    ax.set_xscale("log")
    ax.set_xlim(FIRST * 0.9, TOTAL * 1.1)
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k * 1000:g}" if k < 1 else f"{k:g}k" for k in TICKS_K])
    ax.set_xlabel("Training step (batch 64, log scale)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.3)


def main():
    write_table()
    table = read_table()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for ax, (run, (_, title)) in zip(axes, RUNS.items()):
        draw_run(ax, table[run], title)
    axes[0].set_ylabel("Term of the loss, mean over log-spaced bins of steps")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle("Terms of the loss of GRN+EMA 0.99995", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
