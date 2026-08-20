#!/usr/bin/env python3
"""#407 — the card's deliverable: A4's score against backbone train step.

One axes, because the card asks one question: does the score keep falling
when A4 sees the rest of the data? A second panel would ask the reader to
hold two questions.

Four channels, four facts, none of them carrying two:

  colour       the HEAD. Blue is the student encoder, orange the teacher.
               Both hues clear the lightness band, the chroma floor, the
               all-pairs CVD separation and the 3:1 contrast floor against
               white, and they are the same two #373 used for the same two
               heads.
  fill         WHO MEASURED IT. A hollow marker is #373's published point
               at 40k, 100k or 200k. A filled marker is this card's, at
               300k, 450k or 665k. One backbone trajectory, so one line
               joins them.
  grey rule    1.0660, the project's best before this card — A4's student
               head at 200,000 steps. Read off #373's own score file, not
               typed here.

Direct labels on every point, not a value axis the reader has to trace:
the whole figure is six points per head, and the differences it has to
carry are in the third decimal. For the same reason the axis does not
reach seasonal-naive parity at 1.0. A rule 0.07 below every point would
take four fifths of the axis and leave the card's question in a band too
thin to read.

Usage:
  plot_full_pass.py [--results DIR] [--parent DIR] --out plots/full_pass.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                       # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.lines import Line2D                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_pass as FP                                    # noqa: E402

# #373's head palette, unchanged, so a reader who learned it there keeps it.
HEAD_COLOUR = {"student": "#2a78d6", "teacher": "#eb6834"}
INK, INK_SOFT, GRID, RULE = "#0b0b0b", "#52514e", "#e6e5e1", "#8f8e8a"


def label_side(curves):
    """`{(head, step): +1 above / -1 below}`, so two labels never collide.

    At a step where both heads scored, the higher point takes the space
    above and the lower one the space below. The two heads land within
    0.001 of each other at 40k, which is closer than one label is tall.
    """
    sides = {}
    for head, points in curves.items():
        for step, value in points.items():
            others = [c[step] for h, c in curves.items()
                      if h != head and step in c]
            sides[(head, step)] = -1 if any(o > value for o in others) else 1
    return sides


def draw(ax, head, points, parent_stops, sides):
    """One head's curve, with #373's points hollow and this card's filled."""
    colour = HEAD_COLOUR[head]
    xs = sorted(points)
    ax.plot([x / 1000 for x in xs], [points[x] for x in xs],
            lw=2.0, color=colour, zorder=3, solid_capstyle="round")
    for x in xs:
        published = x in parent_stops
        ax.plot(x / 1000, points[x], marker="o", ms=8.5, color=colour,
                mfc="white" if published else colour, mec=colour, mew=2.0,
                zorder=4, clip_on=False)
        side = sides[(head, x)]
        # A white stroke behind the digits: a steep segment passes through
        # the space a label needs, and the label has to win.
        ax.annotate(f"{points[x]:.4f}", (x / 1000, points[x]),
                    textcoords="offset points", xytext=(0, 11 * side - 4),
                    ha="center", va="bottom" if side > 0 else "top",
                    fontsize=8, color=INK_SOFT, zorder=5,
                    path_effects=[pe.withStroke(linewidth=3.0,
                                                foreground="white")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=FP.RESULTS,
                    help="this study's results directory")
    ap.add_argument("--parent", default=FP.PARENT_RESULTS,
                    help="#373's results directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    curves = {h: FP.curve(h, a.results, a.parent) for h in FP.HEADS}
    best = FP.best_before(a.parent)
    values = [v for c in curves.values() for v in c.values()] + [best]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.axhline(best, color=RULE, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax.annotate(f"best before #407: {best:.4f}", (0.995, best),
                xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 6),
                ha="right", fontsize=8.5, color=RULE)

    sides = label_side(curves)
    for head in FP.HEADS:
        draw(ax, head, curves[head], set(FP.PARENT_STOPS), sides)

    ax.set_xlabel("backbone train step (thousands)", fontsize=10, color=INK)
    ax.set_ylabel("GM-Relative MASE, 97 GIFT-Eval configs (lower is better)",
                  fontsize=10, color=INK)
    ax.set_title("A4 to one full pass over small_v1",
                 fontsize=12.5, color=INK, loc="left", pad=12)
    ticks = [s // 1000 for s in FP.PARENT_STOPS + FP.STOPS]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}k" for t in ticks])
    ax.set_xlim(0, FP.STOPS[-1] / 1000 * 1.06)
    # `or 0.01` covers the first figure of the study, where the only value
    # is the rule itself and the range is zero.
    span = (max(values) - min(values)) or 0.01
    ax.set_ylim(min(values) - 0.22 * span, max(values) + 0.22 * span)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)

    handles = [Line2D([], [], color=HEAD_COLOUR[h], lw=2.0, marker="o",
                      ms=7, mec=HEAD_COLOUR[h], label=f"{h} head")
               for h in FP.HEADS]
    handles.append(Line2D([], [], color=INK_SOFT, lw=0, marker="o", ms=7,
                          mfc="white", mec=INK_SOFT, mew=1.8,
                          label="published by #373"))
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_SOFT,
              loc="upper right", handletextpad=0.6, borderaxespad=1.2)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    for head in FP.HEADS:
        row = "  ".join(f"bb{s // 1000}k={v:.4f}"
                        for s, v in sorted(curves[head].items()))
        print(f"  {head:<8} {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
