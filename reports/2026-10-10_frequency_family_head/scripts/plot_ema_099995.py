"""The EMA of the teacher of grn_ema099995 (magenta) and grn_ema099995_f6
(grey, 6 forecaster layers), as a half-life: the number of steps for the
teacher to move half way to a fixed student, ln 0.5 / ln(momentum).

  our runs  the column `ema_tau` of their loss files every 1,000 steps. The
            plan of both runs: 0.99995 to 1.0 in a straight line over
            665,000 steps. grn_ema099995 stopped at step 391,000 (the owner,
            10-21). The plan of grn_ema099995_f6 after its last logged step.
  I-JEPA    0.996 to 1.0 in a straight line over its training, batch 2048
  V-JEPA    0.998 to 1.0 in a straight line over 1.25 times its training,
            batch 3072
The EMA scaling rule (Busbridge and others 2023, How to Scale Your EMA)
moves the JEPA momentum to batch 64: momentum ** (64 / batch size). The JEPA
plans are drawn against the fraction of their training. Writes
results/ema_tau_099995_logged.tsv and plots/ema_schedule_099995.png.

Usage: python3 plot_ema_099995.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullLocator

STUDY = Path(__file__).resolve().parent.parent
RUNS_DIR = Path("/home/jupyter/cf_runs")
TABLE = STUDY / "results" / "ema_tau_099995_logged.tsv"
OUT = STUDY / "plots" / "ema_schedule_099995.png"
TOTAL = 665_000
START, END = 0.99995, 1.0
BATCH = 64
EVERY = 1000
F6 = "grn_ema099995_f6"
# run -> (loss files in step order, colour, line width, label). grn_ema099995
# went on from its 180k checkpoint in ckpt_r2: a later file wins on a step.
RUNS = {
    "grn_ema099995": (
        [RUNS_DIR / "grn_ema099995/ckpt/grn_ema099995_losses.csv",
         RUNS_DIR / "grn_ema099995/ckpt_r2/grn_ema099995_r2_losses.csv"],
        "#ff00ff", 7, "GRN+EMA 0.99995 (stopped)"),
    F6: (
        [RUNS_DIR / "grn_ema099995_f6/ckpt/grn_ema099995_f6_losses.csv"],
        "#7f7f7f", 2.5, "GRN+EMA 0.99995, 6 forecaster layers"),
}
# name -> (start, end, ramp length in fractions of the training, batch size,
# line style)
JEPA = {
    "I-JEPA": (0.996, 1.0, 1.0, 2048, "--"),
    "V-JEPA": (0.998, 1.0, 1.25, 3072, "-."),
}
Y_TICKS = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]


def half_life(tau):
    """Steps for the teacher to move half way to a fixed student."""
    return np.log(0.5) / np.log(tau)


def plan_tau(steps):
    """The momentum of the plan of our runs at ``steps``."""
    return START + (END - START) * np.asarray(steps) / TOTAL


def jepa_tau(fraction, start, end, span, batch):
    """The momentum of a JEPA plan at ``fraction`` of its training, moved to
    batch 64 by the EMA scaling rule."""
    return (start + (end - start) * fraction / span) ** (BATCH / batch)


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
    """{run: (steps, ema_tau values)} from TABLE, as arrays."""
    points = {run: ([], []) for run in RUNS}
    with open(TABLE) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            points[row["run"]][0].append(int(row["step"]))
            points[row["run"]][1].append(float(row["ema_tau"]))
    return {run: (np.array(s), np.array(t)) for run, (s, t) in points.items()}


def draw_runs(ax, points):
    """The logged half-life of our runs, and the plan of grn_ema099995_f6."""
    for run, (_, colour_hex, width, label) in RUNS.items():
        steps, taus = points[run]
        last = f"{steps[-1] // 1000}k"
        ax.plot(steps, half_life(taus), color=colour_hex, lw=width, solid_capstyle="butt",
                label=f"{label}: logged, 0 to {last}")
        ax.annotate(f"{half_life(taus[-1]) / 1000:.1f}k steps at {last}",
                    (steps[-1], half_life(taus[-1])), xytext=(8, -16),
                    textcoords="offset points", fontsize=10, color=colour_hex)
    steps = np.linspace(points[F6][0][-1], TOTAL, 1000)[:-1]
    ax.plot(steps, half_life(plan_tau(steps)), color=RUNS[F6][1], ls=":", lw=2.5,
            label=f"{RUNS[F6][3]}: plan, {START} to 1 over 665k")


def draw_jepa(ax):
    """The JEPA plans at batch 64, over the fraction of their training."""
    fraction = np.linspace(0, 1, 1000)[:-1]
    for name, (start, end, span, batch, style) in JEPA.items():
        taus = jepa_tau(fraction, start, end, span, batch)
        own_end, end_64 = start + (end - start) / span, jepa_tau(1, start, end, span, batch)
        ax.plot(fraction * TOTAL, half_life(taus), color="black", ls=style, lw=1.5,
                label=f"{name}, batch {batch}: {start:g} to {own_end:g}; "
                      f"at batch 64: {taus[0]:.7g} to {end_64:.7g}")


def style_axes(ax):
    ax.set_yscale("log")
    ax.set_ylim(Y_TICKS[0] * 0.8, Y_TICKS[-1])
    ax.set_yticks(Y_TICKS)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda y, _: f"{y / 1e6:g}M" if y >= 1e6 else f"{y / 1e3:g}k"))
    ax.set_xlim(0, TOTAL * 1.02)
    ax.set_xticks([0, 100_000, 200_000, 300_000, 400_000, 500_000, 600_000, TOTAL])
    ax.set_xticklabels(["0", "100k", "200k", "300k", "400k", "500k", "600k", "665k"])
    ax.set(xlabel="Training step (batch 64); JEPA: the same fraction of its training",
           ylabel="Half-life of the teacher, in steps (log scale)",
           title="EMA of the teacher: steps to move half way to the student")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5)


def main():
    write_table()
    fig, ax = plt.subplots(figsize=(11, 6))
    draw_runs(ax, read_table())
    draw_jepa(ax)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
