#!/usr/bin/env python3
"""#393 — one human name per run, used by every figure in the report.

The CSV files key a run by its directory slug, e.g. `arm6_v2_combab_alignS`.
The `_alignS` / `_alignT` suffix is a filename, not a term: it records which
encoder the alignment loss `L_align` targets. Figures print the card's own
words instead.

    arm6_v2_combab_alignS -> arm6_v2 combab, L_align on student
    arm4_combab           -> arm4 combab

Usage:  python3 scripts/cell_label.py [slug ...]
"""
from __future__ import annotations

import sys

SUFFIX = {"_alignS": "L_align on student", "_alignT": "L_align on teacher"}


def label(slug: str, short: bool = False) -> str:
    """Human name for a run slug. `short` drops the `L_align` prefix."""
    for suffix, target in SUFFIX.items():
        if slug.endswith(suffix):
            base = slug[: -len(suffix)]
            tail = target.split(" on ")[-1] if short else target
            return f"{_cell(base)}, {'align ' + tail if short else tail}"
    return _cell(slug)


def _cell(base: str) -> str:
    """`arm6_v2_combab` -> `arm6_v2 combab`, the parent reports' cell name."""
    head, _, tail = base.rpartition("_")
    return f"{head} {tail}" if head else base


if __name__ == "__main__":
    args = sys.argv[1:] or list(SUFFIX) + ["arm6_v2_combab_alignS",
                                           "arm4_combab", "arm1_nse"]
    for slug in args:
        print(f"{slug:26s} {label(slug)}")
