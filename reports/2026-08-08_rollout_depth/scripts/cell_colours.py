#!/usr/bin/env python3
"""#373 — one colour per cell, one line style per depth, report-wide.

Every figure in this report imports from here and defines no colour of its
own, so a reader who learns a colour in one figure keeps it in all of them.

    colour       the CELL
    line style   the DEPTH:  dashed = k 0,  solid = k 3
    marker       the head:   circle = student,  square = teacher

The study's whole question is one comparison, so the depth rides on the most
legible channel after colour. A k = 0 curve and a k = 3 curve of one cell are
the same colour on purpose: they are the pair being compared.

Usage:  python3 cell_colours.py     # prints the mapping
"""
from __future__ import annotations

import sys

# The card's run order: B5 and B9 first (the two cells with f in both the
# numerator and the denominator of the main term), then A3 and A4, then A1,
# A2, then the rest of group B. Colours ride on this position.
ORDER = ["B5", "B9", "A3", "A4", "A1", "A2",
         "B1", "B2", "B3", "B4", "B6", "B7", "B8", "B10"]

PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # amber
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#7f7f7f",  # grey
    "#e377c2",  # magenta
    "#bcbd22",  # olive
    "#000000",  # black
    "#1a9988",  # teal
    "#c49c00",  # dark amber
    "#5254a3",  # indigo
]
COLOUR = dict(zip(ORDER, PALETTE))

# Line style is the depth, everywhere.
STYLE = {0: (0, (5, 2)), 3: "solid"}
HEAD_MARKER = {"student": "o", "teacher": "s"}

INK = "#222222"
INK_SOFT = "#666666"
GRID = "#dddddd"
PARITY = "#b0b0b0"      # the seasonal-naive parity line, furniture

# Cell id -> the slug the card names it by.
SLUG = {
    "A1": "arm5_combab_alignS_sched",
    "A2": "arm6_v2_nse_alignT_sched",
    "A3": "arm6_v2_combab_alignT_sched",
    "A4": "arm6_v2_combab_alignS_sched",
    "B1": "arm6_v2_combab_alignS_fix09",
    "B2": "arm6_v2_combab_alignT_fix09",
    "B3": "arm5_combab_alignS_fix09",
    "B4": "arm5_combab_alignT_fix09",
    "B5": "arm4_combab_fix09",
    "B6": "arm6_v2_ncpc_alignS_fix09",
    "B7": "arm6_v2_ncpc_alignT_fix09",
    "B8": "arm6_v2_nse_alignT_fix09",
    "B9": "arm1_nse_fix09",
    "B10": "arm6_v2_nse_alignS_fix09",
}


def colour(cell):
    return COLOUR.get(cell, INK)


def style(k):
    return STYLE.get(int(k), "dotted")


def label(cell):
    return f"{cell} {SLUG.get(cell, '?')}"


def rc():
    """The report's matplotlib defaults."""
    return {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
            "axes.edgecolor": INK_SOFT, "axes.labelcolor": INK,
            "xtick.color": INK, "ytick.color": INK,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6}


if __name__ == "__main__":
    for c in ORDER:
        print(f"{c:<4} {COLOUR[c]}  {SLUG[c]}")
    sys.exit(0)
