#!/usr/bin/env python3
"""#393 — one colour per run, the same colour in every figure of the report.

This module is the single source of the run colour mapping. Every plot script
imports `colour`, `linestyle` and `ORDER` from here and defines no colour of
its own, so a reader who learns a colour in one figure keeps it in all of
them.

Colour carries the RUN. Line style carries the encoder `L_align` targets, so
the two runs of one recipe still read as a pair:

    solid   L_align on the student
    dashed  L_align on the teacher
    dotted  no L_align term

A figure that also separates the two heads uses the marker shape for that
(`HEAD_MARKER`), never the colour and never the line style.

The palette is matplotlib `tab10` plus black, in the order the experiment
README numbers the ten runs. That order is fixed: it also gives the legend
order and the row order of every figure that lists runs down a y axis.

Usage:  python3 scripts/run_colours.py     # prints the mapping
"""
from __future__ import annotations

import sys

# Run 1 to run 10, exactly the numbering of `The ten runs` in the experiment
# README. Do not reorder: the colours ride on the position.
ORDER = [
    "arm6_v2_combab_alignS",
    "arm6_v2_combab_alignT",
    "arm5_combab_alignS",
    "arm5_combab_alignT",
    "arm6_v2_ncpc_alignS",
    "arm6_v2_ncpc_alignT",
    "arm6_v2_nse_alignS",
    "arm6_v2_nse_alignT",
    "arm4_combab",
    "arm1_nse",
]

# tab10 plus black: blue, red, green, purple, amber, cyan, brown, grey,
# magenta, black.
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
    "#000000",  # black
]

COLOUR = dict(zip(ORDER, PALETTE))

LINESTYLE = {"student": "-", "teacher": "--", "none": ":"}
HEAD_MARKER = {"student": "o", "teacher": "s"}

# The line-style key, printed as the legend title wherever runs are drawn as
# lines. One colour per run makes the colour key the legend itself.
LEGEND_KEY = ("one colour per run   |   L_align target:  "
              "solid = student,  dashed = teacher,  dotted = none")

# Figure furniture. None of these is a run, so none of them is one of the ten
# colours, and no plot script sets a colour outside this module.
INK = "#3f3f3f"        # axis text, guide lines
INK_SOFT = "#9a9a9a"   # soft guides
GRID = "#e1e0d9"       # grid lines
BAND = "#d9d9d9"       # the head-seed band
PARITY = "#c9c9c9"     # the seasonal-naive parity ring on the radars

_FALLBACK = "#555555"


def align_target(slug: str) -> str:
    """Which encoder `L_align` targets: `student`, `teacher` or `none`."""
    if slug.endswith("_alignS"):
        return "student"
    if slug.endswith("_alignT"):
        return "teacher"
    return "none"


def colour(slug: str) -> str:
    """The run's colour. Same value in every figure."""
    return COLOUR.get(slug, _FALLBACK)


def linestyle(slug: str) -> str:
    """The run's line style, which carries the `L_align` target."""
    return LINESTYLE[align_target(slug)]


def rank(slug: str) -> int:
    """Position in the fixed run order; unknown slugs sort last."""
    return ORDER.index(slug) if slug in ORDER else len(ORDER)


def in_order(slugs) -> list[str]:
    """The given slugs in the fixed run order."""
    return sorted(set(slugs), key=lambda s: (rank(s), s))


def line_style(slug: str, head: str | None = None) -> dict:
    """Keyword arguments for one run's line. `head` picks the marker."""
    kw = {"color": colour(slug), "ls": linestyle(slug)}
    if head is not None:
        kw["marker"] = HEAD_MARKER.get(head, "o")
    return kw


if __name__ == "__main__":
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from cell_label import label

    for i, slug in enumerate(ORDER, 1):
        print(f"{i:>3}  {colour(slug)}  {linestyle(slug):3s}  "
              f"{label(slug)}")
