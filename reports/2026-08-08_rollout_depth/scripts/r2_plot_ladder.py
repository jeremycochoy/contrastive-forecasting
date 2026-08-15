#!/usr/bin/env python3
"""#373 round 2 — every cell's ladder, `k = 3` on top of the published `k = 0`.

Fourteen small multiples, one per cell, because fourteen curves in one axes
would need fourteen hues and a categorical palette is validated on four.
Inside a panel colour carries the HEAD and line style carries the DEPTH, so
a reader learns two channels once and reads all fourteen panels with them.

  solid, coloured   this study, k = 3
  dashed, grey      the card's published k = 0
  horizontal rule   seasonal-naive parity at 1.0. Below it the model beats
                    the naive forecaster on the 97-config geometric mean.

Group B's two parents publish the student head only, so a group-B panel
carries one grey curve and two coloured ones.

Usage:
  r2_plot_ladder.py [--results DIR] --out plots/ladder.png
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
import published                                          # noqa: E402
import cell_config as CC                                   # noqa: E402
import r2_ladder as L                                     # noqa: E402

HEAD_COLOUR = {"student": "#2a78d6", "teacher": "#eb6834"}
INK, INK_SOFT, GRID, PARITY = "#0b0b0b", "#52514e", "#e6e5e1", "#8f8e8a"
K0_GREY = "#9a9995"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fig, axes = plt.subplots(2, 7, figsize=(16.5, 6.9), sharex=True, sharey=True)
    flat = axes.flatten()

    lo, hi = 10.0, 0.0
    for ax, cell in zip(flat, L.CELLS):
        for head in L.HEADS:
            k0 = published.PUBLISHED.get(cell, {}).get(head, {})
            if k0:
                xs = sorted(k0)
                ax.plot(xs, [k0[x] for x in xs], ls=(0, (5, 2)), lw=1.6,
                        color=K0_GREY, marker="o", ms=4, mfc="white",
                        mec=K0_GREY, zorder=2)
                lo, hi = min(lo, *k0.values()), max(hi, *k0.values())
            pts = {s: L.score(cell, s, head, a.results) for s in L.STOPS}
            pts = {s: v for s, v in pts.items() if v is not None}
            if not pts:
                continue
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], ls="solid", lw=2.0,
                    color=HEAD_COLOUR[head], marker="o", ms=5, zorder=3)
            lo, hi = min(lo, *pts.values()), max(hi, *pts.values())

        ax.axhline(1.0, color=PARITY, lw=0.9, zorder=1)
        ax.set_title(f"{cell}  {CC.base_arm(cell)}", fontsize=8.5, color=INK)
        ax.text(0.5, 0.02,
                f"L_align {CC.align_target(cell)} · {CC.ema_words(cell)}",
                transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.5, color=INK_SOFT)
        ax.grid(color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=7)
        ax.set_xticks(L.STOPS)
        ax.set_xticklabels([f"{s}k" for s in L.STOPS])

    if hi > lo:
        pad = 0.08 * (hi - lo)
        flat[0].set_ylim(min(lo - pad, 0.98), hi + pad)
    for ax in axes[:, 0]:
        ax.set_ylabel("GM-Relative MASE", fontsize=8, color=INK_SOFT)
    for ax in axes[1, :]:
        ax.set_xlabel("backbone step", fontsize=8, color=INK_SOFT)

    handles = [
        Line2D([], [], color=HEAD_COLOUR["student"], lw=2, label="k = 3, student"),
        Line2D([], [], color=HEAD_COLOUR["teacher"], lw=2, label="k = 3, teacher"),
        Line2D([], [], color=K0_GREY, lw=1.6, ls=(0, (5, 2)), marker="o", ms=4,
               mfc="white", label="published k = 0"),
        Line2D([], [], color=PARITY, lw=0.9, label="seasonal-naive parity"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.135))
    # Panel width has room for the arm and no more, so the loss terms each
    # arm trains on go once, under the legend.
    seen = {}
    for cell in L.CELLS:
        seen.setdefault(CC.arm(cell), []).append(cell)
    fig.text(0.012, 0.004, "\n".join(
        f"{a}  ({', '.join(cs)}):  "
        f"{CC.terms(cs[0], target=False, short=True)}"
        for a, cs in sorted(seen.items())),
        ha="left", va="bottom", fontsize=6.6, color=INK_SOFT)
    fig.suptitle("each cell's ladder: k = 3 against its own published k = 0",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.165, 1, 0.955))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=165)
    print(f"  {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
