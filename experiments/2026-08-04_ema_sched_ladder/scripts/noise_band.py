#!/usr/bin/env python3
"""#393 — the one head-seed noise band every plot and count uses.

The band pools EVERY head-seed range this study measured, at both head
budgets: the bb40k replicates (15,000-step heads) and the bb100k replicates
(30,000-step heads). `results/seed_spread.csv` holds bb100k rows only, so its
`range` column alone is a sub-sample of the study's own replicates. The bb40k
end is larger: `arm5_combab_alignS` student measures 1.2596 against 1.2980,
a range of 0.0384.

`results/paired_delta.csv` carries both ends in one row, so it is the source.
A range needs at least two seeds at that end.

Usage:  python3 scripts/noise_band.py [--paired FILE]
"""
from __future__ import annotations

import argparse
import csv
import os
import re

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
PAIRED = os.path.join(EXP_DIR, "results", "paired_delta.csv")

LABEL = "largest head-seed range measured at either head budget"


def ranges(path: str = PAIRED) -> list[tuple[str, str, str, float]]:
    """(cell, head, end, range) for every end that carries two or more seeds."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for end in ("bb40k", "bb100k"):
            vals = []
            for k, v in r.items():
                if re.fullmatch(end + r"_\d+", k) and (v or "").strip():
                    vals.append(float(v))
            if len(vals) >= 2:
                out.append((r["cell"], r["head"], end, max(vals) - min(vals)))
    return out


def pooled_band(path: str = PAIRED) -> float | None:
    rs = ranges(path)
    return max(r[-1] for r in rs) if rs else None


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--paired", default=PAIRED)
    a = p.parse_args()
    for cell, head, end, rng in sorted(ranges(a.paired), key=lambda x: -x[-1]):
        print(f"{rng:.4f}  {end:6s}  {cell}  {head}")
    print(f"band {pooled_band(a.paired):.4f}  ({LABEL})")
