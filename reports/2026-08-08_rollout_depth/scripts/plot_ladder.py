#!/usr/bin/env python3
"""#373 — where k = 3 at bb40k sits on the k = 0 trajectory.

The parent's `ladder.png` is GM-Relative MASE against backbone step. This
study has one stop, so a ladder of its own would be a scatter. What it can
draw instead is the question that one stop answers: the published k = 0
trajectory of each cell, over every stop its parent report reached, with
this study's two bb40k points marked on it.

Read it as: does three depths at 40k steps buy what the k = 0 cell needed
100k or 200k steps to reach?

The k = 0 marker at bb40k is this study's own retrained gate where one
exists, and the published number otherwise; the two are distinguished in
the legend.

Usage: plot_ladder.py --results <results dir> --out plots/ladder.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                     # noqa: E402
import runs as R                              # noqa: E402                              # noqa: E402
from published import PUBLISHED                        # noqa: E402

SCORE_RE = re.compile(r"^score_([AB]\d+)_k(\d+)_bb(\d+)k_(student|teacher)\.txt$")
plt.rcParams.update(cc.rc())


def read_scores(results):
    """`{(arm, k, stop, head): value}` over the depth runs."""
    out = {}
    for p in Path(results).glob("score_*.txt"):
        run = R.resolve(p.name[len("score_"):-len(".txt")])
        if run is None or run.role != "depth":
            continue
        try:
            out[(run.arm, run.k, run.stop, run.head)] = \
                float(p.read_text().strip())
        except ValueError:
            continue
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    sc = read_scores(args.results)
    arms = [a for a in R.ARM_ORDER if any(x == a for x, _k, _s, _h in sc)]
    if not arms:
        raise SystemExit("no score file yet — no ladder")

    heads = ["student", "teacher"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)

    for ax, head in zip(axes, heads):
        for arm in arms:
            col = cc.colour(arm)
            # The published trajectory belongs to the CELL. Both of B5's
            # backbone seeds are the same cell, so it is drawn once.
            pub = ({} if cc.hollow(arm)
                   else PUBLISHED.get(cc.cell_of(arm), {}).get(head, {}))
            if pub:
                xs = sorted(pub)
                ax.plot([x * 1000 for x in xs], [pub[x] for x in xs],
                        color=col, linestyle=cc.style(0), linewidth=1.6,
                        marker="o", markersize=4, markerfacecolor="white")
            # Two seeds of one cell would land on top of each other at
            # x = 40000, so nudge the second one along the step axis. It is
            # not a different budget; the offset is legibility only.
            x = 40000 + (1600 if cc.hollow(arm) else 0)
            k0 = sc.get((arm, 0, 40, head))
            if k0 is not None:
                ax.plot(x, k0, color=col, marker="D", markersize=7,
                        markerfacecolor="white", linestyle="none")
            for k in sorted({kk for a, kk, _s, h in sc
                             if a == arm and h == head and kk}):
                kk = sc[(arm, k, 40, head)]
                ax.plot(x, kk, color=col, marker="*", markersize=14,
                        linestyle="none",
                        markerfacecolor="white" if k == 1 else col)
                if k0 is not None:
                    ax.annotate("", xy=(x, kk), xytext=(x, k0),
                                arrowprops=dict(arrowstyle="->", color=col,
                                                linewidth=1.2))
        # The lowest number the three parent reports publish for any of the
        # 14 cells at any stop. A point below it is a new best for this
        # protocol.
        best = min(v for c in PUBLISHED.values() for h in c.values()
                   for v in h.values())
        ax.axhline(best, color=cc.PARITY, linewidth=1.1,
                   linestyle=(0, (4, 3)), zorder=0)
        ax.annotate(
            f"best published number of the 14 cells, any stop ({best:.4f})",
            (0.99, best), xycoords=("axes fraction", "data"), fontsize=7.5,
            color=cc.INK_SOFT, ha="right", va="bottom",
            bbox=dict(fc="#ffffff", ec="none", pad=0.8))
        ax.set_xlabel("backbone step")
        ax.set_title(f"{head} head", loc="left")
    axes[0].set_ylabel("GM-Relative MASE (97 configs)")

    axes[1].legend(handles=[
        Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(0), marker="o",
               markerfacecolor="white", label="published k = 0 trajectory"),
        Line2D([], [], color=cc.INK_SOFT, marker="D", markerfacecolor="white",
               linestyle="none", label="this study, retrained k = 0"),
        Line2D([], [], color=cc.INK_SOFT, marker="*", markersize=12,
               linestyle="none", label="this study, k > 0")]
        + [Line2D([], [], color=cc.colour(a),
                   label=cc.label(a) + ("  ✗ retracted" if a in R.RETRACTED
                                        else ""))
           for a in arms],
        loc="best", frameon=False, fontsize=8)

    fig.suptitle("This study's bb40k runs against the published k = 0 "
                 "trajectories", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(arms)} arm(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
