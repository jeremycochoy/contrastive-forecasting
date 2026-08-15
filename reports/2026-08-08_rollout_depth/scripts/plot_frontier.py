#!/usr/bin/env python3
"""#373 figure 1 — the best GM-Relative MASE each cell reached, against the
frontier the project held before this study.

One row per cell, two marks per row: the student-encoder head and the
teacher-encoder head. The value is that cell's LOWEST GM-Relative MASE over
the stops it was scored at, and the stop that gave it is printed beside the
mark. Rows are sorted by the better of the two heads, best at the top.

The grey rule is the frontier before this study: the lowest GM-Relative MASE
any of the three parent reports printed, from `published.best_published()`.
The band around it is the head-seed band of `ema_sched_ladder.md`, which
bounds the head seed alone.

Usage: plot_frontier.py [--results DIR] --out plots/frontier.png
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

STUDENT, TEACHER = "#2a78d6", "#eb6834"
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GREY, PARITY = "#8f8e8a", "#6f6e6a"


def recipe(cell):
    """The cell's configuration in words, so no row is a bare code."""
    return CC.recipe(cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base, bcell, bhead, bstop = published.best_published()
    band = published.NOISE_BAND

    rows = []
    for cell in L.CELLS:
        best = {}
        for head in L.HEADS:
            pts = {s: L.score(cell, s, head, a.results) for s in L.STOPS}
            pts = {s: v for s, v in pts.items() if v is not None}
            if pts:
                s = min(pts, key=lambda s: pts[s])
                best[head] = (pts[s], s)
        if best:
            rows.append((cell, best))
    rows.sort(key=lambda r: min(v for v, _ in r[1].values()), reverse=True)

    fig, ax = plt.subplots(figsize=(15.0, 7.0))
    ax.axvspan(base - band, base + band, color=GREY, alpha=0.20, zorder=0)
    ax.axvline(base, color=GREY, lw=2.0, zorder=1)
    ax.axvline(1.0, color=PARITY, lw=1.0, ls=(0, (4, 3)), zorder=1)

    for i, (cell, best) in enumerate(rows):
        vals = [v for v, _ in best.values()]
        ax.plot([min(vals), max(vals)], [i, i], color=SOFT, lw=0.9, zorder=2)
        for head, (v, stop) in best.items():
            c = STUDENT if head == "student" else TEACHER
            m = "o" if head == "student" else "^"
            ax.plot([v], [i], marker=m, ms=9, color=c, mec="white", mew=0.9,
                    zorder=4, clip_on=False)
        lo = min(best.items(), key=lambda kv: kv[1][0])
        ax.annotate(f"{lo[1][0]:.4f} at bb{lo[1][1]}k",
                    (lo[1][0], i), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([recipe(c) for c, _ in rows], fontsize=7.6)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel("best GM-Relative MASE over the cell's stops, 97 GIFT-Eval "
                  "configs (lower is better)")
    ax.set_title("Every cell's best score, against the frontier before this "
                 "study", fontsize=12, color=INK, pad=26)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    top = ax.get_xlim()[1]
    ax.annotate(f"frontier before this study {base:.4f}\n"
                f"{bcell}, {bhead} head, bb{bstop}k — its row above",
                (base, len(rows) - 0.35), xytext=(-4, 2),
                textcoords="offset points", fontsize=7.6, color=SOFT,
                ha="right", va="bottom")
    ax.annotate("seasonal-naive parity 1.0", (1.0, len(rows) - 0.35),
                xytext=(4, 2), textcoords="offset points", fontsize=8,
                color=PARITY, va="bottom")

    key = [Line2D([], [], marker="o", ls="none", ms=9, color=STUDENT,
                  label="student-encoder head"),
           Line2D([], [], marker="^", ls="none", ms=9, color=TEACHER,
                  label="teacher-encoder head"),
           Line2D([], [], color=GREY, lw=8, alpha=0.35,
                  label=f"head-seed band ±{band:.4f} around the frontier")]
    ax.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(a.out, dpi=140)
    print(f"wrote {a.out}  ({len(rows)} cells, baseline {base:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
