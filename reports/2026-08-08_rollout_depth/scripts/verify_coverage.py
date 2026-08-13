#!/usr/bin/env python3
"""#373 — rebuild the coverage grid from the score files, and nothing else.

`results/coverage.md` is written by the collector, which reads the queue's
state files. This rebuilds the same grid from the score files alone. Two paths
to one table, so a cell that the queue calls done but that holds no number
cannot pass, and vice versa.

A1 and B3 share one student model, so the grid prints 72 deliverables over 70
distinct measurements. The check states both counts rather than picking one.

Usage:
  verify_coverage.py --results results
"""
from __future__ import annotations

import argparse
from pathlib import Path

CELLS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4",
         "B5", "B6", "B7", "B8", "B9", "B10"]
STOPS = ["40k", "100k", "200k"]
HEADS = ["student", "teacher"]
# The extend rule sent these eight cells to 200k. The other six stop at 100k.
TO_200K = {"A2", "A3", "A4", "B1", "B2", "B4", "B6", "B10"}
# A1 and B3 hold one student model, printed once per stop under each name.
SHARED_STUDENT = ("A1", "B3")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    res = Path(a.results)

    def want(cell, stop):
        return stop != "200k" or cell in TO_200K

    grid, missing, extra = {}, [], []
    for c in CELLS:
        for s in STOPS:
            for h in HEADS:
                f = res / f"score_{c}_k3_bb{s}_{h}.txt"
                if f.exists():
                    grid[(c, s, h)] = float(f.read_text().strip())
                    if not want(c, s):
                        extra.append(f"{c} bb{s} {h}")
                elif want(c, s):
                    missing.append(f"{c} bb{s} {h}")

    expected = sum(1 for c in CELLS for s in STOPS for h in HEADS if want(c, s))
    print(f"deliverables expected : {expected}")
    print(f"score files found     : {len(grid)}")
    print(f"missing               : {missing or 'none'}")
    print(f"scored but not a deliverable this round : {extra or 'none'}")
    print()

    print("| cell | " + " | ".join(f"{s} {h[0].upper()}"
                                   for s in STOPS for h in HEADS) + " |")
    print("|" + "---|" * 7)
    for c in CELLS:
        row = []
        for s in STOPS:
            for h in HEADS:
                v = grid.get((c, s, h))
                mark = "‡" if (c in SHARED_STUDENT and h == "student") else ""
                row.append(f"{v:.4f}{mark}" if v is not None else "stop")
        print(f"| {c} | " + " | ".join(row) + " |")

    print()
    a1, b3 = SHARED_STUDENT
    shared = 0
    for s in STOPS:
        x, y = grid.get((a1, s, "student")), grid.get((b3, s, "student"))
        if x is None or y is None:
            continue
        same = x == y
        print(f"{a1}/{b3} bb{s} student: {x:.4f} vs {y:.4f} -> "
              f"{'ONE measurement' if same else 'TWO — they differ'}")
        shared += 1 if same else 0
    print()
    print(f"deliverables {len(grid)}, distinct measurements "
          f"{len(grid) - shared}  "
          f"({shared} shared student row(s) counted once; the two teachers are "
          f"two models and stay counted twice)")
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    raise SystemExit(main())
