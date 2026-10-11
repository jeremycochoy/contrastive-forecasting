"""The learning rate and the EMA momentum of the teacher of three runs: the
new run (`abc_gift`), ABC and BLK. Writes plots/lr_and_ema_schedules.png.

The learning rate comes from the flags of each run, with the formula of
`cosine_lr` in experiments/2026-04-27_freq-embedding/scripts/train.py.
ABC and `abc_gift` keep 5.6e-5 from the first step. BLK goes from 5.6e-5 to
1e-6 in one cosine over 200,000 steps, then keeps 1e-6. No run has a warmup.

The EMA momentum comes from the column `ema_tau` of the loss files of each
run. results/ema_tau_logged.tsv gives the steps that each run logged and
the values of that column. No run has `--ema-tau-end`, so the momentum stays
at `--ema-tau`.

Usage: python3 plot_schedules.py
"""
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

STUDY = Path(__file__).resolve().parent.parent
MOIRAI = STUDY.parent / "2026-09-06_moirai_small_size"
sys.path.insert(0, str(MOIRAI / "scripts"))
from run_style import P, colour, tagged  # noqa: E402

EMA_TABLE = STUDY / "results" / "ema_tau_logged.tsv"
OUT = STUDY / "plots" / "lr_and_ema_schedules.png"
NEW_RUN_COLOUR = "#2ca02c"  # the colour of the new run in plot_scores.py
ABC, BLK = P + "_lr10x", "cf419_cos200k"


def cosine_lr(step, lr_start, lr_final, total):
    """`cosine_lr` of train.py: one cosine anneal, held after its end."""
    t = np.clip(step, 0, total)
    return lr_final + 0.5 * (lr_start - lr_final) * (1.0 + np.cos(math.pi * t / total))


# run: (legend label, colour, line style, learning rate at each step)
RUNS = {
    "abc_gift": ("New run: the ABC recipe on the new data", NEW_RUN_COLOUR, "-",
                 lambda step: np.full_like(step, 5.6e-5, dtype=float)),
    ABC: (tagged(ABC, "old data"), colour(ABC), (0, (6, 4)),
          lambda step: np.full_like(step, 5.6e-5, dtype=float)),
    BLK: (tagged(BLK, "new data"), colour(BLK), "-",
          lambda step: cosine_lr(step, 5.6e-5, 1e-6, 200_000)),
}


def read_ema_table():
    """{run: (first step, last step, the logged EMA momentum)}."""
    with open(EMA_TABLE) as fh:
        rows = csv.DictReader(fh, delimiter="\t")
        return {r["run"]: (int(r["first_step"]), int(r["last_step"]),
                           float(r["ema_tau_values"])) for r in rows}


def draw_lr(ax, ema):
    for run, (label, colour_hex, style, lr_of) in RUNS.items():
        steps = np.linspace(0, ema[run][1], 2000)
        ax.plot(steps, lr_of(steps), color=colour_hex, ls=style, lw=2.6,
                label=f"Learning rate. {label}")
    ax.set_yscale("log")
    ax.set_ylim(2e-7, 2e-4)
    ax.set_ylabel("Learning rate (log scale, solid and dashed lines)")


def draw_ema(ax, ema):
    """One dotted line for each run. The three lines have the same value, so
    their dots alternate, and the legend shows a plain dotted line."""
    handles = []
    for offset, (run, (label, colour_hex, _, _)) in enumerate(RUNS.items()):
        first, last, tau = ema[run]
        ax.plot([first, last], [tau, tau], color=colour_hex, ls=(offset * 3, (2, 7)), lw=3.0)
        handles.append(Line2D([], [], color=colour_hex, ls=":", lw=3.0,
                              label=f"EMA momentum of the teacher. {label}"))
    ax.text(340_000, 0.906, "EMA momentum of the teacher: 0.9 at each logged step of the 3 runs",
            ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("EMA momentum of the teacher (dotted lines)")
    return handles


def main():
    ema = read_ema_table()
    fig, ax = plt.subplots(figsize=(12, 7))
    right = ax.twinx()
    draw_lr(ax, ema)
    ema_handles = draw_ema(right, ema)
    ax.set_xlabel("Training step (batch 64)")
    ax.set_xlim(0, 680_000)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_title("Learning rate and EMA momentum")
    ax.grid(alpha=0.3)
    handles = ax.get_legend_handles_labels()[0] + ema_handles
    ax.legend(handles, [h.get_label() for h in handles], fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
