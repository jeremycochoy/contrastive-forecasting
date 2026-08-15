#!/usr/bin/env python3
"""#373 — apply the card's extend rule and name the next round's work.

The rule, per head, against that head's own previous stop:

    both heads down   extend, keep both
    one head down     extend, keep that head
    neither down      stop

"Down" means the GM-Relative MASE at this stop is lower than at the stop
before it. The comparison is within one cell and one head. It never crosses
cells, and it never compares against the k = 0 baseline: the baseline
answers whether the depth helped, the extend rule answers whether the cell
is still improving.

Writes results/r2_extend.tsv, which the coverage table reads to mark a head
the rule has ended.

Usage: python3 r2_extend.py [--stop 100] [--write]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r2_coverage import CELLS, ENCS, RES, score  # noqa: E402

PREV = {100: 40, 200: 100}


def decide(cell, stop):
    """(verdict, per-head keep decision, the numbers the rule read)."""
    prev = PREV[stop]
    moves, nums = {}, {}
    for e in ENCS:
        now, before = score(cell, stop, e), score(cell, prev, e)
        nums[e] = (before, now)
        moves[e] = None if now is None or before is None else now < before
    known = [m for m in moves.values() if m is not None]
    if len(known) < len(ENCS):
        return "pending", {e: "pending" for e in ENCS}, nums
    down = [e for e in ENCS if moves[e]]
    if not down:
        return "stop", {e: "stop" for e in ENCS}, nums
    return "extend", {e: ("extend" if moves[e] else "stop") for e in ENCS}, nums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop", type=int, default=100, choices=[100, 200])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    stop, prev = args.stop, PREV[args.stop]

    rows, keep = [], []
    print(f"extend rule at bb{stop}k, each head against its own bb{prev}k")
    print(f"{'cell':<5} {'head':<8} {'bb'+str(prev)+'k':>8} {'bb'+str(stop)+'k':>8} "
          f"{'delta':>8}  verdict")
    for c in CELLS:
        verdict, per, nums = decide(c, stop)
        for e in ENCS:
            before, now = nums[e]
            fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
            d = f"{now - before:+.4f}" if (now is not None and before is not None) else "-"
            print(f"{c:<5} {e:<8} {fmt(before):>8} {fmt(now):>8} {d:>8}  {per[e]}")
            rows.append((c, prev, e, per[e]))
            if per[e] == "extend":
                keep.append((c, e))

    cells = sorted({c for c, _ in keep})
    print(f"\nextend to bb{stop + 100}k: {len(cells)} cell(s) — {' '.join(cells) or 'none'}")
    for c in cells:
        print(f"  {c}: keep {' '.join(e for cc, e in keep if cc == c)}")

    if args.write:
        p = os.path.join(RES, "r2_extend.tsv")
        with open(p, "w") as f:
            for c, pv, e, v in rows:
                f.write(f"{c}\t{stop}\t{e}\t{v}\n")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
