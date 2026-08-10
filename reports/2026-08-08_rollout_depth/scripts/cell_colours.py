#!/usr/bin/env python3
"""#373 — one colour per cell, report-wide.

Every figure in this report imports from here and defines no colour of its
own, so a reader who learns a colour in one figure keeps it in all of them.

    colour       the CELL
    shade        the BACKBONE, inside a cell that trained more than one
    line style   the DEPTH:  dashed = k 0,  solid = k > 0
    fill         the BACKBONE SEED: solid = 20260520, hollow = 20260521
    hatch        the comparison CROSSES A MACHINE
    width        a thin line is a RETRACTED arm
    panel        the HEAD: one figure per head, named in its title

Seven channels, seven questions, no channel carrying two. The study's whole
question is one comparison, so a cell's k = 0 curve and its k = 3 curve are
the same colour on purpose: they are the pair being compared.

The shade channel exists because B5 trained three backbones and the curve
figures draw them together. One hue at one lightness for all three put two
dashed green `k = 0` curves at two levels with nothing to tell them apart.
`arm_colour()` keeps the hue, which names the cell, and steps the lightness,
which names the backbone. A bar chart compares a bar against a rule and
takes `colour()`; a line chart asks the reader to follow one curve through
a crossing and takes `arm_colour()`.

The hatch is the newest channel and it is not decoration. This study's
reproduction table separates on the machine and not on the seed, so a delta
whose two sides trained on two boxes carries a term the study cannot bound.
A reader has to see that on the bar, not in a footnote four sections later.

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runs as R                                          # noqa: E402

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

# Fill is the backbone seed. The protocol's seed is filled; a training at
# any other seed is hollow.
PROTOCOL_SEED = 20260520

# Hatch is the machine. A delta whose two sides trained on two boxes is
# hatched, because the study measured up to 0.117 GM-Relative MASE on the
# box and cannot separate that from the depth.
CROSSED_HATCH = "///"

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
    """`B5·s2` -> `B5`. An arm is a (cell, seed, machine) triple."""
    return arm.split("·")[0]


def colour(arm):
    return COLOUR.get(cell_of(arm), INK)


# How far each backbone of one cell steps off the cell's hue, in the order
# the registry lists them. 0 is the hue itself; a negative step darkens
# towards black and a positive one lightens towards white.
SHADE = [0.0, -0.42, 0.42, -0.70]


def _step(hexcol, amount):
    r, g, b = (int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    t = 0.0 if amount < 0 else 1.0
    w = abs(amount)
    return "#" + "".join(f"{round(255 * (c * (1 - w) + t * w)):02x}"
                         for c in (r, g, b))


def arm_colour(arm):
    """The hue of the arm's cell, shaded by which backbone of it this is.

    A cell that trained one backbone gets its hue unchanged, so every figure
    that draws one backbone per cell is untouched.
    """
    base = colour(arm)
    siblings = R.arms_of(cell_of(arm))
    if arm not in siblings or len(siblings) < 2:
        return base
    return _step(base, SHADE[siblings.index(arm) % len(SHADE)])


def retracted(arm):
    """True where the report has withdrawn this arm's depth delta."""
    return arm in R.RETRACTED


def width(arm, base=1.9):
    """Line width. A retracted arm draws thin, so a reader meets the arms
    the report stands behind first. Opacity would do the same job, but a
    faded curve of one hue lands on the shade of its sibling backbone."""
    return base * 0.62 if retracted(arm) else base


def hollow(arm):
    """True where the arm trained at something other than the protocol seed.

    Read off the registry, not off the arm's name. `B5·s3` is the study's
    third B5 backbone and carries the PROTOCOL seed, so it is filled; what
    makes it a third arm is the machine, and the machine rides the hatch.
    """
    seed = R.arm_seed(arm)
    return seed is not None and seed != PROTOCOL_SEED


def face(arm):
    """Marker / bar fill: the cell's colour, or white for a second seed."""
    return "#ffffff" if hollow(arm) else colour(arm)


def hatch(machine_held):
    """Bar hatch for a delta: none when both sides trained on one box."""
    return None if machine_held else CROSSED_HATCH


def style(k):
    return STYLE.get(int(k), "dotted")


def label(arm):
    """The cell, and — where the cell trained more than one backbone — which
    of them this is. B5 has three, and two of them carry the same seed, so
    the seed alone does not name a B5 arm.

    A bare cell id is also accepted, for a legend entry that stands for the
    colour rather than for one backbone.
    """
    cell = cell_of(arm)
    base = f"{cell} {SLUG.get(cell, '?')}"
    if arm == cell or len(R.arms_of(cell)) < 2:
        return base
    return f"{base}  seed {R.arm_seed(arm)}, {R.arm_where(arm)}"


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
