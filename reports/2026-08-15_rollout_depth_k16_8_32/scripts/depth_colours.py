#!/usr/bin/env python3
"""#401 — one colour per rollout depth, report-wide.

#373 maps a colour to a CELL, because its question is which cell wins. This
study runs ONE cell, so its entity is the DEPTH and the colour rides on that.
Everything else comes from #373's `cell_colours`: the same rcParams, the same
inks, the same greys, so a reader who learns this report's furniture keeps it
across both.

    colour       the DEPTH, k = 8 / 16 / 32. This protocol trains k = 8 and
                 k = 32; k = 16 keeps its hue because the summed comparison
                 arm drew it, and one depth must read the same in both.
    line style   solid for a depth this study trained, dashed for #373's
                 k = 3, dotted for the published k = 0
    grey         a frontier the study has to beat, never a subject

Three hues, taken from #373's four-hue theme in its own order, so no depth
of this study takes the hue #373 gave to a cell it draws beside. Re-validated
as a three-slot categorical palette against the white surface both reports
render on, all pairs:

    lightness band      PASS, all 3 inside L 0.43-0.77
    chroma floor        PASS, all 3 >= 0.1
    CVD separation      PASS, worst pair violet-blue dE 13.0 (deutan)
    normal vision       PASS, worst pair violet-blue dE 16.3
    contrast vs surface WARN, aqua at 2.82:1

The WARN is answered, not dismissed: every figure here direct-labels its
lines, and `results/scores.csv` is the table view of the same numbers. Colour
alone never carries an identity in this report.

Usage:  python3 depth_colours.py     # prints the mapping
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_SCRIPTS = HERE.parent.parent / "2026-08-08_rollout_depth" / "scripts"
sys.path.insert(0, str(PARENT_SCRIPTS))
import cell_colours as cc                                 # noqa: E402

# The arms this protocol trains, in run order. Colour rides on the depth, so
# an arm keeps its colour whatever subset a figure draws.
DEPTHS = [8, 32]

# Every depth any figure of this card draws. The summed comparison arm ran
# k = 16, so its hue stays here even though no arm trains it now.
DEPTHS_DRAWN = [8, 16, 32]

# Slots 1, 3 and 4 of #373's validated theme: blue, aqua, violet.
COLOUR = {
    16: cc.PALETTE[0],
    8: cc.PALETTE[2],
    32: cc.PALETTE[3],
}

# The two references every figure draws, and their inks. Neither is a
# subject, so neither takes a hue.
REF_K3_INK = cc.INK_SOFT        # #373's k = 3 on this same cell
REF_K0_INK = cc.PARITY          # the published k = 0 on this same cell
PRIOR_INK = "#9a9a96"           # the frontier before this study
PARITY_INK = "#0b0b0b"          # seasonal-naive parity, 1.0

STYLE_STUDY = "solid"
STYLE_K3 = (0, (5, 2))
STYLE_K0 = (0, (1, 1.6))

INK = cc.INK
INK_SOFT = cc.INK_SOFT
GRID = cc.GRID


def colour(k) -> str:
    """The hue of one depth. An undrawn depth takes ink, never a new hue."""
    return COLOUR.get(int(k), INK)


def rc() -> dict:
    """#373's matplotlib defaults, unchanged, so both reports render alike."""
    return cc.rc()


def label(k) -> str:
    return f"k = {int(k)}"


if __name__ == "__main__":
    for k in DEPTHS_DRAWN:
        print(f"k = {k:<3} {COLOUR[k]}")
    sys.exit(0)
