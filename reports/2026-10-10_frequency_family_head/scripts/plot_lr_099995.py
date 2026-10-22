"""The learning rate of grn_ema099995 (magenta) and grn_ema099995_f6 (grey),
of BLK, and the Moirai recipe of MPE with its peak at the rate of the grey run.

  magenta, grey  5.6e-5 at each step (--lr 5.6e-5, no warmup, no anneal)
  BLK            5.6e-5 to 1e-6 in one cosine over 200,000 steps, then 1e-6
  Moirai         the recipe of MPE: 0 to 1e-3 in a straight line over 10,000
                 steps at batch 256, then one cosine to 0 at step 166,000.
                 Here its peak is 5.6e-5, the rate of the grey run, and its
                 steps are batch-64 steps (4 for one step at batch 256): the
                 warmup ends at 40,000 and the cosine at 664,000.
The formulas are those of `scheduled_lr` in
experiments/2026-04-27_freq-embedding/scripts/train.py. The steps are on a
log scale from 1k, the first stop of the runs. Writes plots/lr_schedule_099995.png.

Usage: python3 plot_lr_099995.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

STUDY = Path(__file__).resolve().parent.parent
OUT = STUDY / "plots" / "lr_schedule_099995.png"
TOTAL = 665_000
LR = 5.6e-5
STEPS_PER_256 = 4  # batch-64 steps for one step at batch 256
FIRST = 1000  # the first step on the log axis: the first stop of the runs
TICKS_K = [1, 10, 40, 100, 200, 400, 665]


def cosine_lr(step, lr_start, lr_final, total):
    """`cosine_lr` of train.py: one cosine anneal, held after its end."""
    t = np.clip(step, 0, total)
    return lr_final + 0.5 * (lr_start - lr_final) * (1.0 + np.cos(np.pi * t / total))


def moirai_lr(step):
    """The Moirai recipe of MPE at batch-64 ``step``, with its peak at LR."""
    warmup, total = 10_000 * STEPS_PER_256, 166_000 * STEPS_PER_256
    anneal = cosine_lr(step - warmup, LR, 0.0, total - warmup)
    return np.where(step < warmup, LR * step / warmup, anneal)


# (label, colour, line width, line style, learning rate at each step)
LINES = [
    ("GRN+EMA 0.99995: 5.6e-5 at each step", "#ff00ff", 7, "-",
     lambda step: np.full_like(step, LR)),
    ("GRN+EMA 0.99995, 6 forecaster layers: the same", "#7f7f7f", 2.5, "-",
     lambda step: np.full_like(step, LR)),
    ("BLK: cosine from 5.6e-5 to 1e-6 by 200k, then 1e-6", "#1a1a1a", 2.5, "-",
     lambda step: cosine_lr(step, LR, 1e-6, 200_000)),
    ("Moirai recipe of MPE, peak at 5.6e-5: straight line from 0 by 40k, "
     "then cosine to 0 at 664k", "#000080", 2.5, "--", moirai_lr),
]


def style_axes(ax):
    ax.set_xscale("log")
    ax.set_xlim(FIRST * 0.9, TOTAL * 1.1)
    ax.xaxis.set_major_locator(FixedLocator([k * 1000 for k in TICKS_K]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{k}k" for k in TICKS_K])
    ax.set_ylim(0, 8.5e-5)  # room for the legend above the lines
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "0" if y == 0 else f"{y * 1e5:g}e-5"))
    ax.set(xlabel="Training step (batch 64, log scale); Moirai: 4 steps for one step at batch 256",
           ylabel="Learning rate", title="Learning rate through training")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9.5)


def main():
    steps = np.geomspace(FIRST, TOTAL, 2001)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, colour_hex, width, style, lr_of in LINES:
        ax.plot(steps, lr_of(steps), color=colour_hex, lw=width, ls=style, label=label)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
