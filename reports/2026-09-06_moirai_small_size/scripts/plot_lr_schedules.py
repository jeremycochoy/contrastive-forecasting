"""The three lr schedules of the runs in this report, against data seen.

The rates follow scheduled_lr in experiments/2026-04-27_freq-embedding/
scripts/train.py: a linear warmup, then one cosine that holds its final rate
after its last step. Writes plots/lr_schedules.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_style import P, colour

STUDY = Path(__file__).resolve().parent.parent
OUT = STUDY / "plots" / "lr_schedules.png"
FLOOR = 1e-7  # a log axis cannot show 0: the Moirai rate ends on this line


def cosine(step, start, final, total):
    """One cosine from start to final over total steps, held after its end."""
    t = np.clip(step, 0, total)
    return final + 0.5 * (start - final) * (1 + np.cos(np.pi * t / total))


def moirai_lr(step):
    """lr 1e-3: a linear warmup over 10,000 steps, then a cosine to 0 at 166,000."""
    return np.where(step < 10000, 1e-3 * step / 10000, cosine(step - 10000, 1e-3, 0.0, 156000))


def warmup_line(step, down_end):
    """Our warmup recipe: lr 0 to 1e-3 over 10,000 steps, then a straight line
    to 5.6e-5 at down_end, then 5.6e-5."""
    down = 1e-3 + (5.6e-5 - 1e-3) * np.clip((step - 10000) / (down_end - 10000), 0, 1)
    return np.where(step < 10000, 1e-3 * step / 10000, down)


def draw_schedules(ax):
    seen = np.linspace(0, 700000, 3501)
    moirai = seen <= 664000  # batch 256: one step holds 4 batch-64 steps
    ax.plot(seen[moirai], np.maximum(moirai_lr(seen[moirai] / 4), FLOOR), "--",
            color=colour("cf421f_moirai_native"), lw=2.6,
            label="Moirai recipe: lr 1e-3, warmup over 10k steps, cosine to 0 at 166k steps, "
                  "batch 256.  OMB, OMF, OBM, MPM, MPE")
    ax.plot(seen, np.maximum(warmup_line(seen, 20000), FLOOR), color=colour("cf412ow2"), lw=2.6,
            label="Our warmup recipe: lr 0 to 1e-3 over 10k steps, a straight line to 5.6e-5 "
                  "at 20k, then 5.6e-5, batch 64.  OWF, OWR")
    ax.plot(seen, np.maximum(warmup_line(seen, 40000), FLOOR), ":", color=colour("cf412ol2"), lw=2.6,
            label="The same, down to 5.6e-5 at 40k, batch 64.  OWL")
    ax.plot(seen, np.maximum(warmup_line(seen / 4, 20000), FLOOR), "-.", color=colour("cf412bw"),
            lw=2.6, label="Our warmup recipe at batch 256 (10k and 20k steps hold 40k and 80k "
                          "batch-64 steps of data).  OBW")
    ax.plot(seen, cosine(seen, 5.6e-5, 1e-6, 200000), color=colour(P + "_cos200k"), lw=2.6,
            label="Cyan recipe: lr cosine 5.6e-5→1e-6 by 200k, then 1e-6, batch 64.  "
                  "OCB, OCF, OEF, BLK, BMS (CYN starts at 5e-5)")
    ax.plot(seen, np.full_like(seen, 5.6e-5), color=colour(P + "_lr10x"), lw=2.6,
            label="lr 5.6e-5, batch 64.  ABC, OAF (OAL: batch 256)")


def mark_events(ax):
    ax.axvline(665000, color="#555555", ls=":", lw=1.2)
    ax.text(655000, 2e-3, "one pass over the data", fontsize=9, color="#555555",
            ha="right", va="top")
    ax.annotate("warmup ends", (40000, 1e-3), xytext=(60000, 1.6e-3), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("1e-6 from 200k on", (200000, 1e-6), xytext=(240000, 3e-7), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("0 at 166k steps", (664000, FLOOR), xytext=(470000, 1.6e-7), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))


def style_panel(ax, end, step, title):
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 2.5e-3)
    ax.set_xlim(0, end)
    ticks = range(0, end + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    ax.set_xlabel("Data seen, in batch-64 steps. One step at batch 256 counts 4.")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")


def main():
    fig, (near, full) = plt.subplots(1, 2, figsize=(15.5, 8.2), sharey=True,
                                     gridspec_kw=dict(width_ratios=[1, 2]))
    for ax in (near, full):
        draw_schedules(ax)
    mark_events(full)
    style_panel(near, 100000, 20000, "The first 100k steps")
    style_panel(full, 700000, 100000, "To 700k steps")
    near.set_ylabel("Learning rate (log scale)")
    fig.suptitle("The lr schedules of the runs, against data seen", fontsize=13)
    handles, labels = full.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9.5, loc="lower center", framealpha=0.94)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")


main()
