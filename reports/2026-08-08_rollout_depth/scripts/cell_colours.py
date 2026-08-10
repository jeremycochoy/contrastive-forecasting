#!/usr/bin/env python3
"""#373 — one colour per cell, report-wide.

Every figure in this report imports from here and defines no colour of its
own, so a reader who learns a colour in one figure keeps it in all of them.

    colour       the CELL
    line style   the DEPTH:  dashed = k 0,  solid = k > 0
    fill         the BACKBONE SEED: solid = 20260520, hollow = 20260521
    panel        the HEAD: one figure per head, named in its title

Four channels, four questions, no channel carrying two. The study's whole
question is one comparison, so a cell's k = 0 curve and its k = 3 curve are
the same colour on purpose: they are the pair being compared.

The palette is four hues, not fourteen. This study trained 4 of the card's
14 cells, and a categorical palette is validated on the pairs that actually
appear together. These four clear every gate on the all-pairs list against a
white surface — lightness band, chroma floor, CVD separation and the
normal-vision floor — so they hold up in a scatter and in small multiples,
not only in a bar chart. Aqua and orange sit below 3:1 contrast on white, so
every figure that uses them carries direct value labels.

A fifth hue for B5's second backbone seed does not exist: two steps of one
hue cannot both stay inside the lightness band and clear the normal-vision
floor against each other. That is the right answer anyway. A second seed is
the same cell measured twice, not a second entity, so it rides the fill
channel and gets its own figure.

Usage:  python3 cell_colours.py     # prints the mapping
"""
from __future__ import annotations

import sys

# The card's run order. Colours ride on position, so a cell keeps its colour
# whatever subset a figure draws.
ORDER = ["B9", "B1", "B5", "A3", "A4", "A1", "A2",
         "B2", "B3", "B4", "B6", "B7", "B8", "B10"]

# Slots 1, 2, 3 and 7 of the validated categorical theme. Verified with the
# palette validator, all-pairs, light surface #ffffff: worst CVD ΔE 9.2
# (deutan, aqua against orange), worst normal-vision ΔE 16.3 (violet against
# blue). Re-run it before adding a fifth.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
]
COLOUR = dict(zip(ORDER, PALETTE))

# Line style is the depth, everywhere.
STYLE = {0: (0, (5, 2)), 1: (0, (1, 1.6)), 3: "solid"}

# Fill is the backbone seed. The protocol's seed is filled; a second
# training of the same recipe is hollow.
PROTOCOL_SEED = 20260520
SEED_MARKER = {20260520: "o", 20260521: "o"}

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
PARITY = "#8f8e8a"      # the seasonal-naive parity line, furniture
BAND = "#d8d7d2"        # the head-seed noise band, furniture

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


def cell_of(arm):
    """`B5·s2` -> `B5`. An arm is a (cell, backbone seed) pair."""
    return arm.split("·")[0]


def colour(arm):
    return COLOUR.get(cell_of(arm), INK)


def hollow(arm):
    """True where the arm is a second training of the cell's recipe."""
    return "·s2" in arm


def face(arm):
    """Marker / bar fill: the cell's colour, or white for a second seed."""
    return "#ffffff" if hollow(arm) else colour(arm)


def style(k):
    return STYLE.get(int(k), "dotted")


def label(arm):
    cell = cell_of(arm)
    base = f"{cell} {SLUG.get(cell, '?')}"
    return f"{base}  seed 20260521" if hollow(arm) else base


def rc():
    """The report's matplotlib defaults. The white figure face is
    deliberate: the PNG carries its own surface, so it reads the same
    whichever theme the reader views the Markdown in."""
    return {"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
            "figure.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": INK_SOFT, "axes.labelcolor": INK,
            "axes.spines.top": False, "axes.spines.right": False,
            "text.color": INK, "xtick.color": INK, "ytick.color": INK,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
            "legend.frameon": False}


if __name__ == "__main__":
    for c in ORDER:
        col = COLOUR.get(c)
        print(f"{c:<4} {col or '(undrawn)':<10} {SLUG[c]}")
    sys.exit(0)
