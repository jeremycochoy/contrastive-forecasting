"""The EMA momentum of the teacher of grn_ema099995 (magenta) and
grn_ema099995_f6 (grey, 6 forecaster layers): the column `ema_tau` of their
loss files every 1,000 steps, and the plan of both runs (0.99995 to 1.0 in a
straight line over 665,000 steps). grn_ema099995 stopped at step 391,000 (the
owner, 10-21). Writes results/ema_tau_099995_logged.tsv and
plots/ema_schedule_099995.png.

Usage: python3 plot_ema_099995.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

STUDY = Path(__file__).resolve().parent.parent
RUNS_DIR = Path("/home/jupyter/cf_runs")
TABLE = STUDY / "results" / "ema_tau_099995_logged.tsv"
OUT = STUDY / "plots" / "ema_schedule_099995.png"
TOTAL = 665_000
START, END = 0.99995, 1.0
EVERY = 1000
# run -> (loss files in step order, colour, line width, label). grn_ema099995
# went on from its 180k checkpoint in ckpt_r2: a later file wins on a step.
RUNS = {
    "grn_ema099995": (
        [RUNS_DIR / "grn_ema099995/ckpt/grn_ema099995_losses.csv",
         RUNS_DIR / "grn_ema099995/ckpt_r2/grn_ema099995_r2_losses.csv"],
        "#ff00ff", 7, "GRN+EMA 0.99995"),
    "grn_ema099995_f6": (
        [RUNS_DIR / "grn_ema099995_f6/ckpt/grn_ema099995_f6_losses.csv"],
        "#7f7f7f", 2.5, "GRN+EMA 0.99995, 6 forecaster layers"),
}


def logged_tau(paths):
    """{step: ema_tau} at every EVERY steps of the loss files ``paths``."""
    tau = {}
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                if step % EVERY == 0:
                    tau[step] = float(row["ema_tau"])
    return tau


def write_table():
    """TABLE: run, step, ema_tau."""
    with open(TABLE, "w") as f:
        f.write("run\tstep\tema_tau\n")
        for run, (paths, _, _, _) in RUNS.items():
            for step, tau in sorted(logged_tau(paths).items()):
                f.write(f"{run}\t{step}\t{tau:.10f}\n")


def read_table():
    """{run: (steps, ema_tau values)} from TABLE."""
    points = {run: ([], []) for run in RUNS}
    with open(TABLE) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            points[row["run"]][0].append(int(row["step"]))
            points[row["run"]][1].append(float(row["ema_tau"]))
    return points


def draw_runs(ax, points):
    """The logged momentum of each run, and its value at its last step."""
    for run, (_, colour_hex, width, label) in RUNS.items():
        steps, taus = points[run]
        last = f"{steps[-1] // 1000}k"
        ax.plot(steps, taus, color=colour_hex, lw=width, solid_capstyle="butt",
                label=f"{label}: logged, 0 to {last}")
        ax.annotate(f"{taus[-1]:.6f} at {last}", (steps[-1], taus[-1]), xytext=(8, -16),
                    textcoords="offset points", fontsize=10, color=colour_hex)


def main():
    write_table()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot([0, TOTAL], [START, END], color="black", ls="--", lw=1,
            label="Plan of both runs: 0.99995 to 1.0 in a straight line over 665k")
    draw_runs(ax, read_table())
    ax.set_xlim(0, TOTAL * 1.02)
    ax.set_xticks([0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, TOTAL])
    ax.set_xticklabels(["0", "100k", "200k", "300k", "400k", "500k", "600k", "665k"])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax.set(xlabel="Training step (batch 64)", ylabel="EMA momentum of the teacher",
           title="EMA momentum of the teacher, from 0.99995")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
