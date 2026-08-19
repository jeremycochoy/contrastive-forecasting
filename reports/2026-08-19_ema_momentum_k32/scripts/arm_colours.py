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

# The card's four arms.
NAMED = {"a08": "#1f77b4", "a09": "#d62728",
         "s08": "#2ca02c", "s09": "#9467bd"}

# For arms added later. No entry repeats a named colour, so a new arm never
# reads as one of the four.
EXTRA = ("#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f")


def colours(arms) -> dict[str, str]:
    """`{arm: colour}` for `arms`, in the order given."""
    out, spare = {}, list(EXTRA)
    for arm in arms:
        if arm in NAMED:
            out[arm] = NAMED[arm]
        else:
            out[arm] = spare.pop(0) if spare else EXTRA[-1]
    return out
