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
  arm vs arm         every pair of scored arms.

THE VERDICT IS A TWO-STEP RULE. `noise` when the gap is under the spread.
`threshold` when the gap clears the spread by less than the spread itself.
`rank` when the gap is at least twice the spread.

THE SPREAD IS MEASURED, NOT BORROWED. It comes from the arms of THIS card that
share a schedule and differ in seed. `scores.csv` names them. If no schedule
here has two seeds, the script says so and refuses to gate: a spread taken
from another study would not be this treatment's.

WHAT THE TABLE SAID ON 2026-08-27. The gate is 0.0471, the widest repeat-seed
range on this card, from the two seeds of 0.8 to 1.0 at 200k at ramp 10,000:
1.2352 and 1.2823. The three seeds of 0.9 to 1.0 at 100k span 0.0219.

Against the no-decay reference, the best arm sits 1.8 times the gate above
it, a threshold; the other nine arms sit at twice the gate or more, a rank.

Against each other, the arm pairs under one gate are noise, and the best arm
against the second, +0.0241, is one of them. So this card CANNOT order the
decay schedules.

Usage:
  rank_gate.py --scores results/scores.csv --arms scripts/arms.tsv \
      --out results/rank_gate.tsv
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def verdict(gap, spread):
    """The two-step rule. A gap under the spread is `noise`. A gap that clears
    the spread by less than the spread itself is `threshold`. A gap of at
    least twice the spread is `rank`."""
    if abs(gap) <= spread:
        return "noise"
    if abs(gap) < 2 * spread:
        return "threshold"
    return "rank"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out",
                   default=str(HERE.parent / "results" / "rank_gate.tsv"))
    args = p.parse_args(argv)

    arms = {r["arm"]: r for r in S.read_arms(args.arms)}
    scores = S.read_scores(args.scores)
    if not scores:
        print(f"no score in {args.scores}", file=sys.stderr)
        return 2

    # The spread, from the schedules of THIS card that ran at two or more
    # seeds. `repeat` means another row shares the TREATMENT, the EMA schedule
    # and the decay ramp, at another seed. Three arms move the decay ramp on
    # one schedule, and they are three treatments, not three seeds.
    groups = S.repeat_groups(arms.values(), scores)
    spreads = {k: max(v for _, v in g) - min(v for _, v in g)
               for k, g in groups.items()}
    if not spreads:
        print("no schedule of this card scored two seeds — no gate",
              file=sys.stderr)
        return 2
    key = max(spreads, key=lambda k: spreads[k])
    spread = spreads[key]
    seeds = sorted(a for a, _ in groups[key])

    lines = [
        f"# the gate: {spread:.4f}, the range of {len(seeds)} seeds of one "
        f"schedule, {' '.join(str(k) for k in key[:3])}, decay ramp {key[3]}",
        f"# those arms: {', '.join(seeds)}",
        "# a gap under the gate is not a rank. This card measured no other "
        "spread.",
        "# vs no-decay: the comparator is the sweep's run at the arm's own "
        "seed when it ran one, and `comparator_range` is the sweep's spread "
        "over its counted seeds of that schedule. A gap inside that spread is "
        "`inside the comparator range` whatever the gate says.",
        "# verdict: noise under the gate, threshold under twice the gate, "
        "rank at twice the gate or more.",
        "block\tleft\tright\tleft_score\tright_score\tgap\tverdict\tcomparator_range",
    ]

    # Block 1: each arm against what the sweep scored for its own schedule,
    # and against the number the card asks an arm to beat.
    for arm, value in sorted(scores.items(), key=lambda kv: kv[1]):
        row = arms.get(arm)
        ref = S.sweep_score(row) if row else None
        if ref is None:
            lines.append(f"vs no-decay\t{arm}\tthe sweep never ran this "
                         f"schedule\t{value:.4f}\t-\t-\t-\t-")
        else:
            gap = value - ref
            lo, hi = S.SWEEP_RANGES.get(S.schedule(row), (ref, ref))
            comp = hi - lo
            v = ("inside the comparator range" if abs(gap) <= comp
                 else verdict(gap, spread))
            lines.append(
                f"vs no-decay\t{arm}\tthe same schedule, no decay\t"
                f"{value:.4f}\t{ref:.4f}\t{gap:+.4f}\t{v}\t{comp:.4f}")
        gap = value - S.SWEEP_BEST
        lines.append(
            f"vs target\t{arm}\tthe card's target\t{value:.4f}\t"
            f"{S.SWEEP_BEST:.4f}\t{gap:+.4f}\t"
            f"{verdict(gap, spread)}")

    # Block 2: every pair of this card's own arms.
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    for (a, va), (b, vb) in itertools.combinations(ranked, 2):
        gap = vb - va
        lines.append(f"arm vs arm\t{a}\t{b}\t{va:.4f}\t{vb:.4f}\t{gap:+.4f}\t"
                     f"{verdict(gap, spread)}")

    # The narrowest pair the gate lets through, named in the header. A
    # threshold test says `rank` at one part in a thousand over the line, and a
    # reader who sees only the verdict column would report that as a result.
    margins = []
    for line in lines:
        parts = line.split("\t")
        if (len(parts) >= 7 and parts[6] in ("rank", "threshold")
                and parts[0] == "arm vs arm"):
            margins.append((abs(float(parts[5])) - spread, parts[1], parts[2],
                            float(parts[5])))
    if margins:
        over, left, right, gap = min(margins)
        lines.insert(3, f"# the narrowest pair the gate passes: {left} vs "
                        f"{right}, gap {gap:+.4f}, over the gate by {over:.4f}."
                        f" A margin under one gate is a threshold, not a rank.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    body = [ln for ln in lines if not ln.startswith("#")][1:]
    ranks = sum(1 for ln in body if ln.endswith("\trank"))
    thresholds = sum(1 for ln in body if ln.endswith("\tthreshold"))
    print(f"{args.out}: gate {spread:.4f} over {len(seeds)} seeds, "
          f"{ranks} rank(s) and {thresholds} threshold(s) of {len(body)} "
          f"comparison(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
