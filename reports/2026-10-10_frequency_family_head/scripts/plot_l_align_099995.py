"""L_align through training of grn_ema099995 (magenta), grn_ema099995_f6
(grey, 6 forecaster layers) and BLK (cf419_cos200k): the column `l_align` of
their loss files, one row per step, as the mean over log-spaced bins of steps
(BINS bins from 1k to 665k). A bin is drawn only when the run logged all of
its steps. A run that went on from a checkpoint (file `_r2`) takes the steps
of that file after the checkpoint. Writes results/l_align_099995_binned.tsv
and plots/l_align_099995.png.

Usage: python3 plot_l_align_099995.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

STUDY = Path(__file__).resolve().parent.parent
RUNS_DIR = Path("/home/jupyter/cf_runs")
BLK_DIR = Path("/home/jupyter/checkpoints_backup/cf-412/vast_lr100x/cf-419c/cos200k/leg_665k")
TABLE = STUDY / "results" / "l_align_099995_binned.tsv"
OUT = STUDY / "plots" / "l_align_099995.png"
FIRST, TOTAL, BINS = 1000, 665_000, 60
TICKS_K = [1, 10, 40, 100, 200, 400, 665]
# run -> ([(loss file, last step taken from it)], colour, line width, label)
RUNS = {
    "grn_ema099995": ([(RUNS_DIR / "grn_ema099995/ckpt/grn_ema099995_losses.csv", 180_000),
                       (RUNS_DIR / "grn_ema099995/ckpt_r2/grn_ema099995_r2_losses.csv", None)],
                      "#ff00ff", 2.5, "GRN+EMA 0.99995 (stopped at 391k)"),
    "grn_ema099995_f6": ([(RUNS_DIR / "grn_ema099995_f6/ckpt/grn_ema099995_f6_losses.csv", None)],
                         "#7f7f7f", 2.5, "GRN+EMA 0.99995, 6 forecaster layers (in training)"),
    "cf419_cos200k": ([(BLK_DIR / "cf419_cos200k_losses.csv", 140_000),
                       (BLK_DIR / "cf419_cos200k_r2_losses.csv", None)],
                      "#1a1a1a", 2.5, "BLK"),
}


def logged_l_align(files):
    """(steps, l_align) of the loss ``files``; each file gives the steps after
    the last step of the file before it, up to its own limit."""
    steps, values, after = [], [], 0
    for path, last in files:
        with open(path) as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                if step > after and (last is None or step <= last):
                    steps.append(step)
                    values.append(float(row["l_align"]))
        after = last if last is not None else steps[-1]
    return np.array(steps), np.array(values)


def binned(steps, values):
    """(bin centre, mean) of each log-spaced bin that the run logged in full."""
    edges = np.geomspace(FIRST, TOTAL, BINS + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (steps >= low) & (steps < high)
        if steps[-1] >= high and inside.any():
            rows.append((np.sqrt(low * high), values[inside].mean()))
    return rows


def write_table():
    with open(TABLE, "w") as f:
        f.write("run\tbin_centre_step\tl_align_mean\n")
        for run, (files, _, _, _) in RUNS.items():
            for centre, mean in binned(*logged_l_align(files)):
                f.write(f"{run}\t{centre:.0f}\t{mean:.5f}\n")


def read_table():
    """{run: (bin centres, means)} from TABLE."""
    points = {run: ([], []) for run in RUNS}
    with open(TABLE) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            points[row["run"]][0].append(float(row["bin_centre_step"]))
            points[row["run"]][1].append(float(row["l_align_mean"]))
    return points


def style_axes(ax):
    ax.set_xscale("log")
    ax.set_xlim(FIRST * 0.9, TOTAL * 1.1)
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k}k" for k in TICKS_K])
    ax.set(xlabel="Training step (batch 64, log scale)",
           ylabel="L_align, mean over log-spaced bins of steps",
           title="L_align through training")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)


def main():
    write_table()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run, (centres, means) in read_table().items():
        _, colour_hex, width, label = RUNS[run]
        ax.plot(centres, means, color=colour_hex, lw=width, label=label)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
