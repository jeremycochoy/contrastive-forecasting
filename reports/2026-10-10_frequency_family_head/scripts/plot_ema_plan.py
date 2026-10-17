"""The EMA momentum of the teacher through training: our runs, and the
schedules of I-JEPA, V-JEPA and TS-JEPA. Writes plots/ema_schedule_plan.png.

  GRN       0.9 at each step (--ema-tau 0.9, no schedule)
  BLK+EMA   0.9 to 1.0 in a straight line over 665,000 steps
  GRN+EMA   0.98 to 1.0 in a straight line over the whole run: 1.0 only at
            step 665,000 (the owner, 10-12; run grn_ema098); the same from
            0.99 (grn_ema099), from 0.95 (grn_ema095), from 0.999
            (grn_ema0999, the owner, 10-15), from 0.9998 (grn_ema09998,
            the owner's grant, 10-16) and from 0.99995 (grn_ema099995, the
            owner, 10-17)
  BLK+EMA cosine  0.9 to 1.0 on a cosine over the first 200,000 steps, the
            span of the learning-rate cosine of BLK, then 1.0: a frozen
            teacher (the owner, 10-13; run blk_emacos)
  I-JEPA    0.996 to 1.0 in a straight line over its training
  V-JEPA    0.998 to 1.0 in a straight line over 1.25 times its training, so
            0.9996 at its last step
  TS-JEPA   0.998 at each step
The JEPA schedules are drawn against the fraction of their training.

Usage: python3 plot_ema_plan.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parent.parent
OUT = STUDY / "plots" / "ema_schedule_plan.png"
TOTAL = 665_000
MIDDLE = 332_500


COSINE_SPAN = 200_000  # the span of the learning-rate cosine of BLK


def cosine_ramp(step, start, end, span):
    """One cosine from ``start`` at step 0 to ``end`` at ``span``, then ``end``."""
    t = np.clip(step, 0, span) / span
    return end - (end - start) * 0.5 * (1.0 + np.cos(np.pi * t))


def draw_ours(ax, steps):
    ax.plot(steps, np.full_like(steps, 0.9), color="#2ca02c", lw=2.2, label="GRN: 0.9 at each step")
    ax.plot(steps, 0.9 + 0.1 * steps / TOTAL, color="#ff7f0e", lw=2.2,
            label="BLK+EMA: 0.9 to 1.0 in a straight line")
    for start, colour in ((0.95, "#8c564b"), (0.98, "#6a3d9a"), (0.99, "#e377c2"),
                          (0.999, "#bcbd22"), (0.9998, "#008080"), (0.99995, "#ff00ff")):
        ax.plot(steps, start + (1.0 - start) * steps / TOTAL, color=colour, lw=2.6,
                label=f"GRN+EMA {start}: {start} to 1.0 in a straight line")
    ax.plot(steps, cosine_ramp(steps, 0.9, 1.0, COSINE_SPAN), color="#d62728", lw=3.2,
            label="BLK+EMA cosine: 0.9 to 1.0 on a cosine by 200k, then 1.0")
    ax.axvline(COSINE_SPAN, color="0.5", lw=1.0, ls=":")


def draw_jepa(ax, steps):
    frac = steps / TOTAL
    ax.plot(steps, 0.996 + 0.004 * frac, color="0.35", lw=1.6, ls="--", label="I-JEPA: 0.996 to 1.0")
    ax.plot(steps, 0.998 + 0.002 * frac / 1.25, color="0.35", lw=1.6, ls="-.",
            label="V-JEPA: 0.998 to 0.9996")
    ax.plot(steps, np.full_like(steps, 0.998), color="0.35", lw=1.6, ls=":", label="TS-JEPA: 0.998")


def style(ax, low):
    ax.set_xlim(0, 680_000)
    ax.set_ylim(low, 1.0015)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_xlabel("Training step (batch 64); JEPA: the same fraction of its training")
    ax.grid(alpha=0.3)


def main():
    steps = np.linspace(0, TOTAL, 2001)
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax in (full, zoom):
        draw_ours(ax, steps)
        draw_jepa(ax, steps)
    style(full, 0.88)
    style(zoom, 0.975)
    full.set_ylabel("EMA momentum of the teacher")
    full.set_title("All the schedules (dotted line: end of the learning-rate cosine of BLK)")
    zoom.set_title("The same, from 0.975")
    for step in (100_000, 140_000, 180_000):
        value = float(cosine_ramp(np.array(step), 0.9, 1.0, COSINE_SPAN))
        full.annotate(f"{value:.3f} at {step / 1000:.0f}k", (step, value), xytext=(step + 25_000, value - 0.012),
                      fontsize=9, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))
    full.legend(loc="lower right", fontsize=9)
    fig.suptitle("EMA momentum of the teacher: our runs and JEPA")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)


if __name__ == "__main__":
    main()
