#!/usr/bin/env python3
"""#373 — teacher encoder minus student encoder.

The parent's `encoder_delta.png`, unchanged in form: the teacher head's
GM-Relative MASE minus the student head's. Below zero means the teacher
encoder gives the better head.

Two panels. The left one is the whole grid, every cell at k = 3 at every
stop it reached. The right one is the five arms that also trained a k = 0,
so it is the only place the depth itself can move the gap.

The card asks for this figure because the depth changes how much the
teacher enters the loss: every h index of a duplicated term shifts with the
depth, so MoCo keys and teacher targets are re-read k+1 times per step.

Reads results/splits.csv (`all` rows), written by split_scores.py.

Usage: plot_encoder_delta.py --splits results/splits.csv \\
           --out plots/encoder_delta.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
import r2_ladder as L2                                  # noqa: E402
import runs as R                                       # noqa: E402
from published import NOISE_BAND                       # noqa: E402

plt.rcParams.update(cc.rc())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--results", default=L2.RESULTS)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data = {}
    with open(args.splits) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "all":
                continue
            run = R.resolve(r["stop"])
            if run is None or run.role != "depth":
                continue
            data[(run.arm, run.k, run.head)] = float(r["gm_rel_mase"])

    pts = []
    for arm in R.ARM_ORDER:
        for k in sorted({kk for a, kk, _h in data if a == arm}):
            s, t = data.get((arm, k, "student")), data.get((arm, k, "teacher"))
            if s is not None and t is not None:
                pts.append((arm, k, t - s))
    if not pts:
        raise SystemExit("ABORT: no (arm, depth) has both heads in "
                         f"{args.splits}")

    # The whole grid: every cell, every stop, at k = 3. The panel above it
    # covers the arms that also trained a k = 0, which is five of them; this
    # one covers all 14 cells, which is what the card's figure asks for.
    grid = []
    for cell in L2.CELLS:
        for stop in L2.STOPS:
            s = L2.score(cell, stop, "student", args.results)
            t = L2.score(cell, stop, "teacher", args.results)
            if s is not None and t is not None:
                grid.append((cell, stop, t - s))

    lim = max(max(abs(d) for *_x, d in pts + grid) * 1.25, NOISE_BAND * 1.28)
    fig, (axg, ax) = plt.subplots(
        1, 2, figsize=(12.4, 0.34 * max(len(grid), len(pts)) + 2.4),
        gridspec_kw={"width_ratios": [1.0, 1.0]})

    axg.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.BAND, zorder=0)
    axg.axvline(0.0, color=cc.INK_SOFT, linewidth=1.0)
    for y, (cell, stop, d) in enumerate(grid):
        axg.barh(y, d, height=0.62, color=cc.ladder_colour(cell),
                 edgecolor="white", linewidth=0.6)
        axg.text(d + (0.0012 if d >= 0 else -0.0012), y, f"{d:+.4f}",
                 va="center", ha="left" if d >= 0 else "right", fontsize=7.4)
    axg.set_yticks(range(len(grid)))
    axg.set_yticklabels([f"{c}  bb{s}k" for c, s, _d in grid], fontsize=7.6)
    axg.invert_yaxis()
    axg.set_xlim(-lim, lim)
    axg.set_xlabel("teacher head minus student head, GM-Relative MASE\n"
                   "(negative = the teacher encoder wins)", fontsize=9)
    axg.set_title("Every cell at k = 3, every stop", loc="left", fontsize=10)

    # The band the report reads every bar against, shaded here with the same
    # colour screen_bb100k.png and stop_delta.png give it.
    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.BAND, zorder=0)
    ax.axvline(0.0, color=cc.INK_SOFT, linewidth=1.0)
    for y, (arm, k, d) in enumerate(pts):
        ax.barh(y, d, height=0.55, color=cc.face(arm),
                alpha=1.0 if k else 0.45, edgecolor=cc.colour(arm),
                linewidth=1.4, hatch="///" if cc.hollow(arm) else None)
        ax.text(d + (0.0012 if d >= 0 else -0.0012), y, f"{d:+.4f}",
                va="center", ha="left" if d >= 0 else "right", fontsize=7.4)
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([f"{arm}  k = {k}" for arm, k, _d in pts], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("teacher head minus student head, GM-Relative MASE\n"
                  "(negative = the teacher encoder wins)", fontsize=9)
    ax.legend(handles=[Line2D([], [], color=cc.INK_SOFT, linewidth=8,
                              alpha=0.45, label="k = 0"),
                       Line2D([], [], color=cc.INK_SOFT, linewidth=8,
                              label="k > 0"),
                       Patch(facecolor=cc.BAND,
                             label=f"head-seed band ±{NOISE_BAND}")],
              loc="lower right", fontsize=8, facecolor="#ffffff",
              edgecolor="none", framealpha=0.95)
    ax.set_title("The five arms that trained both depths, bb40k", loc="left",
                 fontsize=10)
    fig.suptitle("Which encoder the head is trained on", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(grid)} cell-stop(s), {len(pts)} depth pair(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
