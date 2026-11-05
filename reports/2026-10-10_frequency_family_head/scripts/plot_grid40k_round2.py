"""The training of the second round of the 40k grid (the user, 11-02 and 11-03),
for the view only: the decisions use the forecast GM-Relative MASE. The best
cell of the grid (lr 1e-4, EMA start 0.99995) and the changes from it: the
losses of blue (L_rep 0.05, the L_align depth weights 1/2^j), the lr schedule
(C: a cosine from 1e-4 to 1e-5 over the 40k steps; W: a warmup to 1e-3 over 4k
steps, then a cosine to 1e-4), batch 128 at the lr 1e-4, 2e-4, 4e-4 and 8e-4, A:
batch 128 with the warmup of W, then a cosine to 4e-4, and the SIGReg gate: batch
128 at lr 4e-4 with the SIGReg window fix of #438, and batch 256 at the lr 4e-4, 8e-4,
1e-3 and 1.6e-3, and at lr 1e-3 with L_rep 0.05, with the fixed rollout of #446
and #447 (the old SIGReg window), and at lr 1e-3 for 80k steps with the fixed SIGReg
window and the input init 0.01 at L_rep 0.1 and 0.05. Three panels on one step axis, means over bins
of 500 steps:

  1. The lr of each step, from the flags of the run (the formula of
     scheduled_lr in train.py; the optimizer of each 20k save has this lr).
  2. L_align at depth 0, before the depth weights (column l_align): it
     compares all runs.
  3. The total loss of the runs with the loss weights and the batch of the best
     cell (lr 1e-4 with EMA start 0.99995, C and W). The L_rep of batch 128 has more negatives.

The top of the loss panels fits the steps after the warmup (4k); a note gives
the runs whose first steps lie above it. The legend gives the forecast
GM-Relative MASE at 40k of each run. Writes plots/grid40k_round2_dynamics.png.

Usage: python3 plot_grid40k_round2.py
"""
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_grid40k import ROUND2_COLOURS, run_score

STUDY = Path(__file__).resolve().parent.parent
GRID = Path("/home/jupyter/cf_runs/grid40k")
OUT = STUDY / "plots" / "grid40k_round2_dynamics.png"
STEPS = 40000
BIN = 500
AFTER_WARMUP = 4000
CONSTANT = (1e-4, 1e-4, STEPS, 0)
# run -> (label, lr schedule (lr, final lr, end of the anneal, warmup steps),
# the loss weights and the batch of lr 1e-4, EMA start 0.99995); the colours: ROUND2_COLOURS
RUNS = {
    "g40_lr1em4_tau99995": ("lr 1e-4, EMA start 0.99995, constant", CONSTANT, True),
    "g40b_lr1em4_tau99995_rep005h": ("losses of blue: L_rep 0.05, depths 1/2^j", CONSTANT, False),
    "g40c_lr1em4_tau99995_cos": ("C: cosine from 1e-4 to 1e-5", (1e-4, 1e-5, STEPS, 0), True),
    "g40w_lr1em3_wu4k_cos": ("W: warmup to 1e-3 over 4k, cosine to 1e-4", (1e-3, 1e-4, STEPS, 4000), True),
    "g40b_lr1em4_tau99995_bs128": ("batch 128, lr 1e-4", CONSTANT, False),
    "g40b_lr2em4_tau99995_bs128": ("batch 128, lr 2e-4", (2e-4, 2e-4, STEPS, 0), False),
    "g40b_lr4em4_tau99995_bs128": ("batch 128, lr 4e-4", (4e-4, 4e-4, STEPS, 0), False),
    "g40w_lr1em3_wu4k_cos4em4_bs128": ("A: batch 128, warmup to 1e-3 over 4k, cosine to 4e-4",
                                       (1e-3, 4e-4, STEPS, 4000), False),
    "g40b_lr8em4_tau99995_bs128": ("batch 128, lr 8e-4", (8e-4, 8e-4, STEPS, 0), False),
    "g40b_lr4em4_tau99995_bs128_sigfix": ("SIGReg gate: batch 128, lr 4e-4, fixed SIGReg window",
                                          (4e-4, 4e-4, STEPS, 0), False),
    "g40b_lr4em4_tau99995_bs256_rollfix": ("fixed rollout, batch 256, lr 4e-4", (4e-4, 4e-4, STEPS, 0), False),
    "g40b_lr8em4_tau99995_bs256_rollfix": ("fixed rollout, batch 256, lr 8e-4", (8e-4, 8e-4, STEPS, 0), False),
    "g40b_lr1em3_tau99995_bs256_rollfix": ("fixed rollout, batch 256, lr 1e-3", (1e-3, 1e-3, STEPS, 0), False),
    "g40b_lr1p6em3_tau99995_bs256_rollfix": ("fixed rollout, batch 256, lr 1.6e-3",
                                             (1.6e-3, 1.6e-3, STEPS, 0), False),
    "g40b_lr1em3_tau99995_bs256_rep005_rollfix": ("fixed rollout, batch 256, lr 1e-3, L_rep 0.05",
                                                  (1e-3, 1e-3, STEPS, 0), False),
    "g80b_lr1em3_tau99995_bs256_rep01_sigfix_in0p01_rollfix": ("80k steps, fixed SIGReg window, input init 0.01, batch 256, lr 1e-3, L_rep 0.1",
                                                        (1e-3, 1e-3, STEPS, 0), False),
    "g80b_lr1em3_tau99995_bs256_rep005_sigfix_in0p01_rollfix": ("80k steps, fixed SIGReg window, input init 0.01, batch 256, lr 1e-3, L_rep 0.05",
                                                         (1e-3, 1e-3, STEPS, 0), False),
}


def scheduled_lr(step, lr_start, lr_final, total, warmup):
    """The lr of train.py: a linear warmup, then one cosine to `lr_final` at `total`."""
    if step < warmup:
        return lr_start * step / warmup
    t = min(max(step - warmup, 0), total - warmup)
    return lr_final + 0.5 * (lr_start - lr_final) * (1 + math.cos(math.pi * t / (total - warmup)))


def binned_losses(run):
    """Means of the total loss and of L_align at depth 0 over bins of BIN steps."""
    paths = [GRID / folder / run / f"{run}_losses.csv" for folder in ("ckpt", "ckpt_box")]
    frame = pd.read_csv(next(p for p in paths if p.exists()), usecols=["step", "loss", "l_align"])
    bins = frame.groupby((frame.step - 1) // BIN)[["loss", "l_align"]].mean()
    bins.index = bins.index * BIN + BIN / 2
    return bins


def legend_label(run):
    score = run_score(run)
    return f"{RUNS[run][0]}: {'no score yet' if score is None else f'{score:.4f}'}"


def draw_lr(ax):
    steps = np.arange(0, STEPS + 1, 50)
    for run, (name, schedule, _) in RUNS.items():
        if schedule != CONSTANT or run == "g40_lr1em4_tau99995":
            label = "lr 1e-4, constant (EMA start 0.99995, losses of blue, batch 128 at lr 1e-4)" \
                if schedule == CONSTANT else name
            ax.plot(steps, [scheduled_lr(s, *schedule) for s in steps], color=ROUND2_COLOURS[run], lw=2,
                    label=label)
    ax.set(yscale="log", ylabel="learning rate", title="Learning rate")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def above_note(curves, top):
    """The runs with bins above `top`, each with the end step of its last such bin."""
    parts = [f"{RUNS[run][0].split(':')[0]} to {(series[series > top].index.max() + BIN / 2) / 1000:.1f}k"
             for run, series in curves.items() if (series > top).any()]
    return textwrap.fill("above the panel: " + "; ".join(parts) + " steps", 110)


def draw_loss(ax, column, runs, title):
    curves = {run: binned_losses(run)[column] for run in runs}
    steady = pd.concat([c[c.index > AFTER_WARMUP] for c in curves.values()])
    low = min(c.min() for c in curves.values()) * 0.97
    top = steady.max() * 1.03
    for run, series in curves.items():
        ax.plot(series.index, series, color=ROUND2_COLOURS[run], lw=2, label=legend_label(run))
    ax.set_ylim(low, top)
    ax.set(ylabel="mean over 500 steps", title=title)
    ax.text(0.99, 0.03, above_note(curves, top), transform=ax.transAxes, ha="right",
            fontsize=8, color="#555555")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def main():
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    draw_lr(axes[0])
    draw_loss(axes[1], "l_align", list(RUNS),
              "L_align at depth 0, before the depth weights. Legend: forecast GM-Relative MASE at 40k")
    draw_loss(axes[2], "loss", [r for r, v in RUNS.items() if v[2]],
              "Total loss (the runs at batch 64 with the loss weights of lr 1e-4, EMA start 0.99995)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("training step (EMA ramp to 1.0 over the 40k steps)")
    fig.suptitle("40k grid, second round: the training of lr 1e-4, EMA start 0.99995 and of the changes from it\n"
                 "(view only; decisions use the forecast)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
