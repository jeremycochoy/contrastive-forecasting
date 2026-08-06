#!/usr/bin/env python3
"""#393 — the head-seed spread as error bars, and which moves survive it.

Reads `results/seed_spread.csv` (written by scripts/seed_spread.py) and draws
the one thing the extend rule could not see: how far GM-Relative MASE moves
when only the head seed changes.

  left   where the three seeds land, per cell per head. The bar spans
         min to max across seeds 20260722 / 20260723 / 20260724, the dot is
         the mean, and the open diamond is the bb40k value the rule compared
         against. The distance between diamond and dot is the change the
         rule read; the bar is how much of it is seed.
  right  that change with the same spread as its error bar. The rule is a
         strict `<`, so it is exactly the SIGN of this quantity. The grey
         band is this study's own largest head-seed range, pooled over both
         head budgets (scripts/noise_band.py), because these changes have a
         15,000-step head at one end and a 30,000-step head at the other.

Colour is the answer, not the identity: a move clear of its own spread is
blue, a move the spread covers is orange. Head is carried by marker shape as
well, so neither panel depends on colour alone.

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

import noise_band  # noqa: E402
from cell_label import label as cell_label  # noqa: E402

# Two categorical hues, validated together: ΔE 21.9 under protanopia and 27.9
# under normal vision against the light surface, both inside the lightness
# band and above the chroma floor. Ink is furniture, not a third series.
RESOLVED = "#2c6fb5"
UNRESOLVED = "#c2571b"
INK = "#3f3f3f"
INK_SOFT = "#9a9a9a"
HEAD_MARKER = {"student": "o", "teacher": "s"}
SEEDS = ["seed_20260722", "seed_20260723", "seed_20260724"]


def read_spread(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)]
    return [r for r in rows if r.get("bb40k") and r.get("mean")]


def f(row: dict, key: str):
    v = (row.get(key) or "").strip()
    return float(v) if v else None


def draw(rows: list[dict], path: str) -> None:
    # Cells keep the order seed_spread.py emits, which is the order the card
    # names them; within a cell, student above teacher.
    labels, ys = [], []
    for i, r in enumerate(rows):
        labels.append(f"{cell_label(r['cell'], short=True)}  {r['head']}")
        ys.append(i)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.8, 6.6),
                                 gridspec_kw={"width_ratios": [1, 1]})

    complete = all(int(r["n_seeds"]) == len(SEEDS) for r in rows)
    # Pooled over both head budgets: the rows plotted here are bb100k only,
    # and the study's largest measured range is at bb40k.
    band = noise_band.pooled_band()

    # --- left: where the three seeds land ----------------------------------
    for y, r in zip(ys, rows):
        vals = [f(r, s) for s in SEEDS]
        have = [v for v in vals if v is not None]
        colour = RESOLVED if r["resolved"] == "yes" else UNRESOLVED
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
    ax.invert_yaxis()
    ax.set_xlabel("GM-Relative MASE  (lower is better)")
    ax.set_title("Where the three head seeds land at bb100k\n"
                 "bar = min to max across seeds, hollow diamond = bb40k",
                 fontsize=10)
    ax.grid(alpha=0.22, axis="x")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- right: the change, with the spread as its error bar ---------------
    bx.axvspan(-band, band, color=INK_SOFT, alpha=0.13, zorder=0)
    for edge in (-band, band):
        bx.axvline(edge, color=INK_SOFT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    bx.axvline(0, color=INK, lw=1.2, zorder=1)
    for y, r in zip(ys, rows):
        bb = f(r, "bb40k")
        vals = [v for v in (f(r, s) for s in SEEDS) if v is not None]
        d_mean = f(r, "mean") - bb
        lo, hi = min(vals) - bb, max(vals) - bb
        colour = RESOLVED if r["resolved"] == "yes" else UNRESOLVED
        bx.plot([lo, hi], [y, y], "-", color=colour, lw=2, alpha=0.85,
                solid_capstyle="round", zorder=2)
        bx.plot([d_mean], [y], HEAD_MARKER[r["head"]], color=colour, ms=8,
                mec="white", mew=1.2, zorder=4)

    bx.set_yticks(ys)
    bx.set_yticklabels([])
    # The left panel already labels every row. Leaving these ticks drawn puts
    # a dash against each row that reads as a data mark in the gap.
    bx.tick_params(axis="y", left=False)
    bx.invert_yaxis()
    # Room for the band's edges to be visible as edges.
    lo_x, hi_x = bx.get_xlim()
    bx.set_xlim(min(lo_x, -band * 1.25), max(hi_x, band * 1.25))
    bx.set_xlabel("change bb40k → bb100k  (left of 0 = improved)")
    bx.set_title("superseded denominator: change bb40k \u2192 bb100k "
                 "against its bb100k spread only",
                 fontsize=10)
    bx.grid(alpha=0.22, axis="x")
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)

    n_unres = sum(1 for r in rows if r["resolved"] != "yes")
    print(f"  {n_unres} of {len(rows)} changes are inside their own spread")
    handles = [
        Line2D([], [], color=RESOLVED, marker="o", ms=7, lw=2, mec="white",
               label="change clear of the seed spread"),
        Line2D([], [], color=UNRESOLVED, marker="o", ms=7, lw=2, mec="white",
               label="seed spread covers the change"),
        # These keys carry the marker SHAPE, so they are drawn the way the
        # mean marks are drawn: filled, white edge. A hollow key here reads
        # as the hollow diamond, which is the bb40k mark and not a head.
        Line2D([], [], color=INK, marker="o", ms=7, lw=0, mec="white",
               mew=1.2, label="student encoder (filled circle)"),
        Line2D([], [], color=INK, marker="s", ms=7, lw=0, mec="white",
               mew=1.2, label="teacher encoder (filled square)"),
        Line2D([], [], marker="D", ms=6, lw=0, mfc="white", mec=INK, mew=1.3,
               label="bb40k value (hollow diamond, left panel)"),
        Line2D([], [], color=INK_SOFT, lw=7, alpha=0.35,
               label=f"\u00b1{band:.4f}, {noise_band.LABEL}"),
    ]
    # Below the panels: with twelve rows there is no corner of either axes
    # that is reliably empty, and a legend over a data row hides the answer.
    fig.legend(handles=handles, fontsize=8, ncol=3, loc="lower center",
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    head = "Head-seed spread at bb100k, three seeds per run per head"
    if not complete:
        head += "   [PARTIAL — not every seed has finished]"
    fig.suptitle(head, fontsize=11.5, y=0.985)
    fig.tight_layout(rect=(0, 0.095, 1, 0.955))
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
