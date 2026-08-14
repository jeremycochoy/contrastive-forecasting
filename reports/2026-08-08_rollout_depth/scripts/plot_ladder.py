#!/usr/bin/env python3
"""#373 figure 2 — the card's `ladder.png`: GM-Relative MASE against backbone
train step, all 14 cells at k = 3, both heads.

Horizontal axis: the backbone train step, 40k / 100k / 200k. Vertical axis:
GM-Relative MASE over the 97 GIFT-Eval configs. One line per cell per head,
labelled at its end, so no line is identified by colour alone. The key on the
right prints every cell's recipe, so the figure needs nothing outside itself.

The grey rule is the frontier before this study: the lowest GM-Relative MASE
any of the three parent reports printed, from `published.best_published()`.
The band is the head-seed band of `ema_sched_ladder.md`. `plot_frontier.py`
draws the same rule from the same call.

Usage: plot_ladder.py [--results DIR] --out plots/ladder.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.lines import Line2D                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_colours as C                                  # noqa: E402
import published                                          # noqa: E402
import r2_ladder as L                                     # noqa: E402

GREY, PARITY, INK, SOFT, GRID = "#8f8e8a", "#6f6e6a", "#0b0b0b", "#52514e", "#e6e5e1"


def recipe(cell):
    arm, align, ema = L.CELL_ARM[cell]
    tgt = "no L_align" if align == "none" else f"L_align→{align}"
    ema = "EMA 0.9→1.0" if ema == "scheduled" else "EMA 0.9"
    return f"{arm} · {tgt} · {ema}"


def spread(ys, gap):
    """Push overlapping end labels apart, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for n, i in enumerate(order[1:], start=1):
        prev = out[order[n - 1]]
        if out[i] - prev < gap:
            out[i] = prev + gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base, bcell, bhead, bstop = published.best_published()
    band = published.NOISE_BAND
    xs = {40: 0, 100: 1, 200: 2}

    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=0.30)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    key_ax = fig.add_subplot(gs[0, 2])
    key_ax.axis("off")

    lo, hi = 9.9, 0.0
    drawn = 0
    for ax, head in zip(axes, L.HEADS):
        ends = []
        for cell in L.CELLS:
            pts = {s: L.score(cell, s, head, a.results) for s in L.STOPS}
            pts = {s: v for s, v in pts.items() if v is not None}
            if not pts:
                continue
            drawn += 1
            ss = sorted(pts)
            col = C.ladder_colour(cell)
            ax.plot([xs[s] for s in ss], [pts[s] for s in ss], color=col,
                    lw=1.9, marker="o", ms=4.5, mec="white", mew=0.8, zorder=3)
            lo, hi = min(lo, *pts.values()), max(hi, *pts.values())
            ends.append((xs[ss[-1]], pts[ss[-1]], cell, col))

        ax.axhspan(base - band, base + band, color=GREY, alpha=0.20, zorder=0)
        ax.axhline(base, color=GREY, lw=2.0, zorder=1)
        ax.set_title(f"{head}-encoder head", loc="left", fontsize=11, color=INK)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["40k", "100k", "200k"])
        ax.set_xlim(-0.14, 2.62)
        ax.set_xlabel("backbone train step")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax._ends = ends

    top = hi + 0.035
    bot = min(lo, base - band) - 0.02
    for ax in axes:
        ax.set_ylim(bot, top)
        gap = (top - bot) * 0.031
        ends = ax._ends
        ys = spread([e[1] for e in ends], gap)
        for (x, y, cell, col), yy in zip(ends, ys):
            ax.annotate(cell, (x, y), xytext=(x + 0.12, yy),
                        textcoords="data", fontsize=8.5, color=col,
                        va="center", ha="left", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                        shrinkA=1, shrinkB=1))
    axes[0].set_ylabel("GM-Relative MASE, 97 GIFT-Eval configs (lower is better)")

    axes[0].annotate(f"frontier before this study {base:.4f}", (0.0, base),
                     xytext=(2, -11), textcoords="offset points", fontsize=8,
                     color=SOFT)

    key_ax.text(0, 1.0, "the 14 cells", fontsize=9.5, color=INK, va="top",
                fontweight="bold", transform=key_ax.transAxes)
    for i, cell in enumerate(L.CELLS):
        y = 0.945 - i * 0.058
        key_ax.plot([0.0, 0.055], [y, y], color=C.ladder_colour(cell), lw=2.6,
                    transform=key_ax.transAxes, clip_on=False)
        key_ax.text(0.075, y, f"{cell}", fontsize=8.5, color=INK, va="center",
                    fontweight="bold", transform=key_ax.transAxes)
        key_ax.text(0.205, y, recipe(cell), fontsize=8, color=SOFT,
                    va="center", transform=key_ax.transAxes)
    key_ax.text(0, 0.10,
                f"grey rule: the best GM-Relative MASE published before this\n"
                f"study: {base:.4f}, cell {bcell} {L.CELL_ARM[bcell][0]}, "
                f"{bhead}-encoder head,\nbb{bstop}k, from the EMA-schedule "
                f"ladder report.\nband: the ±{band:.4f} head-seed band of that "
                f"same report.",
                fontsize=7.6, color=SOFT, va="top",
                transform=key_ax.transAxes)

    fig.suptitle("GM-Relative MASE against backbone train step, 14 cells at "
                 "rollout depth k = 3", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, dpi=140)
    print(f"wrote {a.out}  ({drawn} lines, baseline {base:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
