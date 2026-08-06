#!/usr/bin/env python3
"""#393 — the head-seed spread as error bars, and which moves survive it.

Reads `results/seed_spread.csv` (written by scripts/seed_spread.py) and draws
the one thing the extend rule could not see: how far GM-Relative MASE moves
when only the head seed changes.

One panel: where the three seeds land, per cell per head. The bar spans min
to max across seeds 20260722 / 20260723 / 20260724, the dot is the mean, and
the open diamond is the bb40k value the rule compared against. The distance
between diamond and dot is the change the rule read; the bar is how much of
it is seed.

Colour is the run, from `scripts/run_colours.py`, the same colour the run
carries in every other figure of the report. Head is the marker shape.

Usage:  python3 scripts/plot_seed_spread.py [--spread FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import run_colours  # noqa: E402
from cell_label import label as cell_label  # noqa: E402

INK = run_colours.INK
INK_SOFT = run_colours.INK_SOFT
HEAD_MARKER = run_colours.HEAD_MARKER
SEEDS = ["seed_20260722", "seed_20260723", "seed_20260724"]


def read_spread(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)]
    return [r for r in rows if r.get("bb40k") and r.get("mean")]


def f(row: dict, key: str):
    v = (row.get(key) or "").strip()
    return float(v) if v else None


def draw(rows: list[dict], path: str) -> None:
    # The report's fixed run order; within a run, student above teacher.
    rank = {c: i for i, c in enumerate(run_colours.ORDER)}
    rows.sort(key=lambda r: (rank.get(r["cell"], len(rank)), r["head"]))
    labels, ys = [], []
    for i, r in enumerate(rows):
        labels.append(f"{cell_label(r['cell'], short=True)}  {r['head']}")
        ys.append(i)

    fig, ax = plt.subplots(figsize=(8.6, 6.6))

    complete = all(int(r["n_seeds"]) == len(SEEDS) for r in rows)

    # --- left: where the three seeds land ----------------------------------
    for y, r in zip(ys, rows):
        vals = [f(r, s) for s in SEEDS]
        have = [v for v in vals if v is not None]
        colour = run_colours.colour(r["cell"])
        lo, hi = min(have), max(have)
        ax.plot([lo, hi], [y, y], "-", color=colour, lw=2, alpha=0.85,
                solid_capstyle="round", zorder=2)
        ax.plot(have, [y] * len(have), "|", color=colour, ms=9, mew=1.4,
                alpha=0.9, zorder=3)
        ax.plot([f(r, "mean")], [y], HEAD_MARKER[r["head"]], color=colour,
                ms=8, mec="white", mew=1.2, zorder=4)
        bb = f(r, "bb40k")
        ax.plot([bb, f(r, "mean")], [y, y], ":", color=INK_SOFT, lw=1.1,
                zorder=1)
        ax.plot([bb], [y], "D", mfc="white", mec=INK, mew=1.3, ms=6, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(run_colours.colour(r["cell"]))
    ax.invert_yaxis()
    ax.set_xlabel("GM-Relative MASE  (lower is better)")
    ax.set_title("Head-seed spread at bb100k, three seeds per run per "
                 "head\n"
                 "bar = min to max across seeds, hollow diamond = bb40k",
                 fontsize=10)
    ax.grid(alpha=0.22, axis="x")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        # These keys carry the marker SHAPE, so they are drawn the way the
        # mean marks are drawn: filled, white edge. A hollow key here reads
        # as the hollow diamond, which is the bb40k mark and not a head.
        Line2D([], [], color=INK, marker="o", ms=7, lw=0, mec="white",
               mew=1.2, label="student encoder (filled circle)"),
        Line2D([], [], color=INK, marker="s", ms=7, lw=0, mec="white",
               mew=1.2, label="teacher encoder (filled square)"),
        Line2D([], [], marker="D", ms=6, lw=0, mfc="white", mec=INK, mew=1.3,
               label="bb40k value (hollow diamond)"),
    ]
    # Below the panels: with twelve rows there is no corner of either axes
    # that is reliably empty, and a legend over a data row hides the answer.
    leg = fig.legend(handles=handles, fontsize=8, ncol=3, loc="lower center",
                     frameon=False, bbox_to_anchor=(0.5, 0.0),
                     title="one colour per run, the report's colour code")
    leg.get_title().set_fontsize(8)

    if not complete:
        fig.suptitle("PARTIAL — not every seed has finished", fontsize=11,
                     y=0.985)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    res = os.path.join(EXP_DIR, "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--spread", default=os.path.join(res, "seed_spread.csv"))
    p.add_argument("--out", default=os.path.join(EXP_DIR, "plots",
                                                 "seed_spread.png"))
    a = p.parse_args()

    if not os.path.exists(a.spread):
        print(f"plot_seed_spread: no {a.spread}", file=sys.stderr)
        return 1
    rows = read_spread(a.spread)
    if not rows:
        print(f"plot_seed_spread: no usable row in {a.spread}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    draw(rows, a.out)
    print(f"[out] {a.out}  ({len(rows)} cell-head rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
