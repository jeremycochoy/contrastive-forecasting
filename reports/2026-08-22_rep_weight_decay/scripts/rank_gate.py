#!/usr/bin/env python3
"""Which score gaps of this card are ranks, and which are noise?

WHY THIS TABLE EXISTS. This card scored six arms. Only ONE schedule ran at
more than one seed, so that schedule's range IS this treatment's whole measured
run-to-run spread. A gap smaller than it is not a rank, and a report that
orders two arms inside it states more than the card measured.

WHAT IT WRITES. `results/rank_gate.tsv`, in two blocks.

  arm vs reference   each arm against the SAME schedule with no decay, from
                     the sweep, and against the card's target of 1.1491. This
                     is the comparison the card asks for.
  arm vs arm         every pair of scored arms. `verdict` is `rank` when the
                     gap clears the spread and `noise` when it does not.

THE SPREAD IS MEASURED, NOT BORROWED. It comes from the arms of THIS card that
share a schedule and differ in seed. `scores.csv` names them. If no schedule
here has two seeds, the script says so and refuses to gate: a spread taken
from another study would not be this treatment's.

Usage:
  rank_gate.py --scores results/scores.csv --arms scripts/arms.tsv \
      --out results/rank_gate.tsv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def read_scores(path):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["arm"]] = float(row["score"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out",
                   default=str(HERE.parent / "results" / "rank_gate.tsv"))
    args = p.parse_args(argv)

    arms = {r["arm"]: r for r in S.read_arms(args.arms)}
    scores = read_scores(args.scores)
    if not scores:
        print(f"no score in {args.scores}", file=sys.stderr)
        return 2

    # The spread, from the schedules of THIS card that ran at two or more
    # seeds. `repeat` means another row shares the schedule at another seed.
    groups = {}
    for arm, value in scores.items():
        row = arms.get(arm)
        if row and row.get("repeat"):
            groups.setdefault(S.schedule(row), []).append((arm, value))
    spreads = {k: max(v for _, v in g) - min(v for _, v in g)
               for k, g in groups.items() if len(g) > 1}
    if not spreads:
        print("no schedule of this card scored two seeds — no gate",
              file=sys.stderr)
        return 2
    key = max(spreads, key=lambda k: spreads[k])
    spread = spreads[key]
    seeds = sorted(a for a, _ in groups[key])

    lines = [
        f"# the gate: {spread:.4f}, the range of {len(seeds)} seeds of one "
        f"schedule, {' '.join(key)}",
        f"# those arms: {', '.join(seeds)}",
        "# a gap under the gate is not a rank. This card measured no other "
        "spread.",
        "block\tleft\tright\tleft_score\tright_score\tgap\tverdict",
    ]

    # Block 1: each arm against what the sweep scored for its own schedule,
    # and against the number the card asks an arm to beat.
    for arm, value in sorted(scores.items(), key=lambda kv: kv[1]):
        row = arms.get(arm)
        ref = S.SWEEP_SCORES.get(S.schedule(row)) if row else None
        if ref is None:
            lines.append(f"vs no-decay\t{arm}\tthe sweep never ran this "
                         f"schedule\t{value:.4f}\t-\t-\t-")
        else:
            gap = value - ref
            lines.append(
                f"vs no-decay\t{arm}\tthe same schedule, no decay\t"
                f"{value:.4f}\t{ref:.4f}\t{gap:+.4f}\t"
                f"{'rank' if abs(gap) > spread else 'noise'}")
        gap = value - S.SWEEP_BEST
        lines.append(
            f"vs target\t{arm}\tthe card's target\t{value:.4f}\t"
            f"{S.SWEEP_BEST:.4f}\t{gap:+.4f}\t"
            f"{'rank' if abs(gap) > spread else 'noise'}")

    # Block 2: every pair of this card's own arms.
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    for (a, va), (b, vb) in itertools.combinations(ranked, 2):
        gap = vb - va
        lines.append(f"arm vs arm\t{a}\t{b}\t{va:.4f}\t{vb:.4f}\t{gap:+.4f}\t"
                     f"{'rank' if abs(gap) > spread else 'noise'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    body = [ln for ln in lines if not ln.startswith("#")][1:]
    ranks = sum(1 for ln in body if ln.endswith("rank"))
    print(f"{args.out}: gate {spread:.4f} over {len(seeds)} seeds, "
          f"{ranks} of {len(body)} comparison(s) clear it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
