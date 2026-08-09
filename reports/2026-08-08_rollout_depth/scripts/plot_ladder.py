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
import cell_colours as cc                              # noqa: E402
from published import PUBLISHED                        # noqa: E402

SCORE_RE = re.compile(r"^score_([AB]\d+)_k(\d+)_bb(\d+)k_(student|teacher)\.txt$")
plt.rcParams.update(cc.rc())


def read_scores(results):
    out = {}
    for p in Path(results).glob("score_*.txt"):
        m = SCORE_RE.match(p.name)
        if not m:
            continue
        try:
            out[(m.group(1), int(m.group(2)), int(m.group(3)), m.group(4))] = \
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
    cells = sorted({c for c, _k, _s, _h in sc},
                   key=lambda c: cc.ORDER.index(c) if c in cc.ORDER else 99)
    if not cells:
        raise SystemExit("no score file yet — no ladder")

    heads = ["student", "teacher"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)

    for ax, head in zip(axes, heads):
        for cell in cells:
            col = cc.colour(cell)
            pub = PUBLISHED.get(cell, {}).get(head, {})
            if pub:
                xs = sorted(pub)
                ax.plot([x * 1000 for x in xs], [pub[x] for x in xs],
                        color=col, linestyle=cc.style(0), linewidth=1.6,
                        marker="o", markersize=4, markerfacecolor="white")
            k0 = sc.get((cell, 0, 40, head))
            if k0 is not None:
                ax.plot(40000, k0, color=col, marker="D", markersize=7,
                        markerfacecolor="white", linestyle="none")
            k3 = sc.get((cell, 3, 40, head))
            if k3 is not None:
                ax.plot(40000, k3, color=col, marker="*", markersize=14,
                        linestyle="none")
                if k0 is not None:
                    ax.annotate("", xy=(40000, k3), xytext=(40000, k0),
                                arrowprops=dict(arrowstyle="->", color=col,
                                                linewidth=1.2))
        ax.set_xlabel("backbone step")
        ax.set_title(f"{head} head")
    axes[0].set_ylabel("GM-Relative MASE (97 configs)")

    axes[1].legend(handles=[
        Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(0), marker="o",
               markerfacecolor="white", label="published k = 0 trajectory"),
        Line2D([], [], color=cc.INK_SOFT, marker="D", markerfacecolor="white",
               linestyle="none", label="this study, retrained k = 0"),
        Line2D([], [], color=cc.INK_SOFT, marker="*", markersize=12,
               linestyle="none", label="this study, k = 3")]
        + [Line2D([], [], color=cc.colour(c), label=cc.label(c))
           for c in cells],
        loc="best", frameon=False, fontsize=8)

    fig.suptitle("k = 3 at bb40k against the published k = 0 trajectory",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(cells)} cell(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
