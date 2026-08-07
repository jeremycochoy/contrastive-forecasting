#!/usr/bin/env python3
"""The two tables the ladder rows support (#393).

First, the card's own question: does a forecasting head trained and evaluated
through the TEACHER encoder beat the same head through the STUDENT encoder.
That is a paired comparison, not two rankings, so the table carries a column
of differences and a count of which side won.

Second, the protocol's record of "the raw per-stop change each head made" —
each head's score against its own value at the previous stop, which is the
quantity the extend rule is applied to. It is empty until a cell has two
stops, so it prints nothing before the first bb100k score lands.

Reads the pooled `results/ladder_all.csv`, so it is only honest if the
pooling kept both heads. It did not, until the key in `merge_pooled.sh` was
fixed to name `head`; `test_merge_pooled.sh` is what keeps it true.

Rows sort by stop, then by the student score, so the table doubles as the arm
ranking at each stop. A stop with only one head scored is printed with the
missing side named, never as a blank or a zero.

Usage: python3 head_table.py [--csv results/ladder_all.csv] [--stop 40000]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(os.path.dirname(HERE), "results", "ladder_all.csv")


def load(path, want_stop=None):
    """-> OrderedDict[(cell, stop)] = {head: gm_rel_mase}."""
    pairs = OrderedDict()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            stop = int(row["stop"])
            if want_stop is not None and stop != want_stop:
                continue
            pairs.setdefault((row["cell"], stop), {})[row["head"]] = float(
                row["gm_rel_mase"]
            )
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--stop", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"no {args.csv}")
    pairs = load(args.csv, args.stop)
    if not pairs:
        print("no rows")
        return 0

    # Sort by stop, then by whichever score is present, so a half-scored stop
    # still lands next to the cells it will be compared against.
    def key(item):
        (_cell, stop), heads = item
        return (stop, min(heads.values()))

    print("| cell | stop | student | teacher | teacher - student |")
    print("|---|---|---|---|---|")
    teacher_lower = student_lower = tie = partial = 0
    for (cell, stop), heads in sorted(pairs.items(), key=key):
        s, t = heads.get("student"), heads.get("teacher")
        if s is None or t is None:
            partial += 1
            print(
                f"| {cell} | {stop} | "
                f"{'not measured' if s is None else f'{s:.4f}'} | "
                f"{'not measured' if t is None else f'{t:.4f}'} | not measured |"
            )
            continue
        d = t - s
        if d < 0:
            teacher_lower += 1
        elif d > 0:
            student_lower += 1
        else:
            tie += 1
        print(f"| {cell} | {stop} | {s:.4f} | {t:.4f} | {d:+.4f} |")

    n = teacher_lower + student_lower + tie
    print()
    print(
        f"{n} cell-stop(s) with both heads: teacher lower {teacher_lower}, "
        f"student lower {student_lower}, tie {tie}."
    )
    if partial:
        print(f"{partial} cell-stop(s) have only one head scored so far.")

    deltas(pairs)
    return 0


def deltas(pairs):
    """Each head against its own previous stop — what the extend rule reads."""
    by_cell = {}
    for (cell, stop), heads in pairs.items():
        by_cell.setdefault(cell, {})[stop] = heads

    rows = []
    for cell, stops in sorted(by_cell.items()):
        ordered = sorted(stops)
        for prev, cur in zip(ordered, ordered[1:]):
            for head in ("student", "teacher"):
                if head in stops[prev] and head in stops[cur]:
                    d = stops[cur][head] - stops[prev][head]
                    rows.append((cell, prev, cur, head, stops[prev][head],
                                 stops[cur][head], d))
    if not rows:
        print()
        print("No cell has two scored stops yet, so no per-stop change exists.")
        return

    print()
    print("| cell | stop | head | previous | current | change | direction |")
    print("|---|---|---|---|---|---|---|")
    for cell, prev, cur, head, p, c, d in rows:
        print(f"| {cell} | {prev} -> {cur} | {head} | {p:.4f} | {c:.4f} | "
              f"{d:+.4f} | {'down' if d < 0 else 'up' if d > 0 else 'flat'} |")


if __name__ == "__main__":
    sys.exit(main())
