#!/usr/bin/env python3
"""#404 — one colour for one arm, across every figure of this study.

The momentum figure, the loss curves and the radar all draw the same arms. A
reader who learns a colour on one figure carries it to the next, so the map
lives here and not three times.

The card's four arms are named. The card also says to ADD arms when the four
scores show a direction, so an arm this file does not name still gets its own
colour, taken in the order the arms table lists it.
"""
from __future__ import annotations

import sys

# The card's four arms.
NAMED = {"a08": "#1f77b4", "a09": "#d62728",
         "s08": "#2ca02c", "s09": "#9467bd"}

# For arms added later. No entry repeats a named colour, so a new arm never
# reads as one of the four.
#
# No entry is a grey either. The momentum figure draws every reference in grey
# — the k = 3 band, #401's arm, the two 200,000-step lines — so a grey arm
# reads as a reference and not as an arm.
EXTRA = ("#ff7f0e", "#17becf", "#8c564b", "#e377c2",
         "#bcbd22", "#393b79", "#7b4173", "#8c6d31")


def colours(arms) -> dict[str, str]:
    """`{arm: colour}` for `arms`, in the order given.

    Past the fallback list the list CYCLES. It does not repeat one entry:
    that gave every further arm the same colour, and two arms in one colour
    read as one arm. A cycle still repeats after `len(EXTRA)` added arms, so
    the map says so on stderr. Name the arm in `NAMED` at that point.
    """
    out: dict[str, str] = {}
    spare = 0
    for arm in arms:
        if arm in NAMED:
            out[arm] = NAMED[arm]
            continue
        if spare == len(EXTRA):
            print(f"WARN: more than {len(EXTRA)} arms outside NAMED — the "
                  f"fallback colours repeat from '{arm}' on. Name the arms "
                  f"in {__file__}.", file=sys.stderr)
        out[arm] = EXTRA[spare % len(EXTRA)]
        spare += 1
    return out
