#!/usr/bin/env python3
"""#373 round 2 — the headline figure: `k = 3` minus `k = 0`, per cell.

One panel per stop. Inside a panel, one row per cell and one bar per head.
A bar left of the rule is a cell the depth improved.

Encoding, one channel per question:

  bar direction   the sign. Lower GM-Relative MASE is better, so a bar that
                  points left is an improvement. Polarity is already in the
                  geometry, so colour is not spent on it.
  colour          the HEAD. Two hues, slots 1 and 2 of the validated
                  categorical theme in `cell_colours.py`.
  grey band       the parents' pooled head-seed band, +/-0.0384. It bounds
                  the head seed alone. Each bar here is the difference of
                  two INDEPENDENT backbone trainings, and that spread is not
                  measured, so the band is a floor on the noise and not a
                  significance test.
  open bar        no published `k = 0` for that head, so no delta exists.
                  Group B's two parents publish the student head only.

Usage:
  r2_plot_delta.py [--results DIR] --out plots/k3_vs_k0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.patches import Patch                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import published                                          # noqa: E402
import r2_ladder as L                                     # noqa: E402

HEAD_COLOUR = {"student": "#2a78d6", "teacher": "#eb6834"}
INK, INK_SOFT, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#d8d7d2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    stops = [s for s in L.STOPS
             if any(L.score(c, s, h, a.results) is not None
                    for c in L.CELLS for h in L.HEADS)]
    if not stops:
        print("  no scored stop yet — no delta figure")
        return 0

    cells = list(reversed(L.CELLS))          # A1 at the top
    width = max(8.0, 4.6 * len(stops))
    fig, axes = plt.subplots(1, len(stops), figsize=(width, 6.4),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes[0]

    for ax, stop in zip(axes, stops):
        ax.axvspan(-published.NOISE_BAND, published.NOISE_BAND,
                   color=BAND, zorder=0, lw=0)
        ax.axvline(0, color=INK_SOFT, lw=1.0, zorder=2)
        for i, cell in enumerate(cells):
            for j, head in enumerate(L.HEADS):
                y = i + (0.19 if head == "student" else -0.19)
                s3 = L.score(cell, stop, head, a.results)
                if s3 is None:
                    continue
                s0 = L.baseline(cell, stop, head)
                if s0 is None:
                    # No baseline: mark that the cell was measured, and that
                    # no delta exists, rather than leaving the row blank.
                    ax.plot([0], [y], marker="o", ms=5, mfc="white",
                            mec=HEAD_COLOUR[head], mew=1.4, zorder=4)
                    continue
                d = s3 - s0
                ax.barh(y, d, height=0.34, color=HEAD_COLOUR[head],
                        zorder=3, lw=0)
                ax.text(d + (0.012 if d >= 0 else -0.012), y, f"{d:+.3f}",
                        va="center", ha="left" if d >= 0 else "right",
                        fontsize=7, color=INK_SOFT, zorder=5)
        ax.set_title(f"bb{stop}k", fontsize=11, color=INK)
        ax.set_xlabel("GM-Relative MASE, k = 3 minus k = 0", fontsize=9,
                      color=INK_SOFT)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=8)

    axes[0].set_yticks(range(len(cells)))
    axes[0].set_yticklabels(cells, fontsize=9, color=INK)
    axes[0].set_ylim(-0.7, len(cells) - 0.3)
    # Room for the value labels, which sit outside the bar end.
    lo, hi = axes[0].get_xlim()
    pad = 0.16 * (hi - lo)
    axes[0].set_xlim(lo - pad, hi + pad)

    handles = [Patch(facecolor=HEAD_COLOUR[h], label=f"{h} encoder")
               for h in L.HEADS]
    handles.append(Patch(facecolor=BAND, label="head-seed band ±0.0384"))
    handles.append(plt.Line2D([], [], ls="", marker="o", ms=5, mfc="white",
                              mec=INK_SOFT, mew=1.4,
                              label="measured, no published k = 0"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("left of the rule, the depth helped", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"  {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
