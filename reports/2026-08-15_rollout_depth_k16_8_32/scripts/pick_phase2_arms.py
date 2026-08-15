#!/usr/bin/env python3
"""#401 — which two arms phase 2 retrains heads on.

The card: *wait until phase 1 has the GM-Relative MASE of every backbone stop
at every k, then take the 2 arms with the best results.*

Two decisions are made here, once, so the driver and the report cannot
disagree:

  best of an arm   the LOWEST GM-Relative MASE the arm reached at any of its
                   stops. An arm whose 200k stop is its best is still that
                   arm's result, and phase 2 retrains all three of its stops
                   anyway.
  a tie            broken by the study's run order, 16 then 8 then 32. Two
                   arms that score the same to four decimals are not
                   separated by this study, and a stable order is what makes
                   a re-run of the picker return the same pair.

Refuses an incomplete phase 1: a missing stop makes an arm look better than
it is, because its best is taken over fewer stops.

Usage:  pick_phase2_arms.py --scores results/scores.csv
        -> "16 8" on stdout, one line, space separated
"""

from __future__ import annotations

import argparse
import csv
import sys

# The card's depths, in the order it runs them. Ties break on this.
RUN_ORDER = (16, 8, 32)
STOPS = (40_000, 100_000, 200_000)
N_ARMS = 2


def pick_arms(rows, n: int = N_ARMS, run_order=RUN_ORDER, stops=STOPS):
    """The `n` depths with the lowest best score, in run order.

    `rows` are dicts with `k`, `stop` and `score`. Raises ValueError when a
    depth or a stop is missing.
    """
    best: dict[int, float] = {}
    seen: dict[int, set[int]] = {}
    for row in rows:
        k, stop, score = int(row["k"]), int(row["stop"]), float(row["score"])
        seen.setdefault(k, set()).add(stop)
        best[k] = min(best.get(k, score), score)

    missing = [k for k in run_order if k not in best]
    if missing:
        raise ValueError(f"phase 1 has no score for k = {missing}")
    for k in run_order:
        gaps = sorted(set(stops) - seen[k])
        if gaps:
            raise ValueError(f"phase 1 has no score for k = {k} at {gaps}")

    ranked = sorted(run_order, key=lambda k: (best[k], run_order.index(k)))
    return [k for k in run_order if k in set(ranked[:n])]


def read_scores(path: str, phase: int = 1):
    """The phase-1 rows of a `collect.sh` scores.csv."""
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if int(r["phase"]) == phase]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="results/scores.csv")
    ap.add_argument("--count", type=int, default=N_ARMS)
    args = ap.parse_args(argv)

    try:
        arms = pick_arms(read_scores(args.scores), n=args.count)
    except ValueError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    print(" ".join(str(k) for k in arms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
