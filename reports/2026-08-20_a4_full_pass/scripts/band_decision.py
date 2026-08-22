#!/usr/bin/env python3
"""#407 — does the 665,000-step stop need a head-seed band?

The card scores each stop with ONE head seed. A head seed moves a score,
so a stop that carries one draw has no scale beside it. The answer, up to
now, was to draw two more seeds at every stop. That answer costs about 8
GPU-hours at the last stop.

A compute audit changed the rule for the LAST stop only. The band at
665,000 steps is no longer armed. It fires on the number the driver
measures.

The rule
--------

Fire the band when the 665,000-step STUDENT score lands inside the window

    |score - 1.0651| <= 0.01

1.0651 is the mean of the 200,000-step student band, over head seeds
20260722, 20260723 and 20260724 (`results/head_band.csv`, row
`200000,student`). That number is the comparison the card makes: did A4
keep improving after 200,000 steps?

Inside the window, one draw cannot decide the comparison. The two points
sit closer together than a few head draws can separate, so the card buys
the band.

Outside the window, one draw decides it. The measured pooled standard
deviation of this card's own bands is 0.0029, over both heads and the
three banded stops. The window is 0.01 wide, which is about 3.4 of those
standard deviations. A point outside it reads on its own, and the band
buys no information. So the card does not spend the 8 GPU-hours.

Exit codes
----------

  0   FIRE   the score is inside the window.
  10  SKIP   the score is outside the window.
  20  WAIT   that stop has no student score yet.
  2   bad input.

Usage
-----

    band_decision.py                        # read the live score
    band_decision.py --score 1.0700         # the test seam
    band_decision.py --explain              # the numbers behind the rule
    band_decision.py --write <path>         # record the verdict, once
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import full_pass  # noqa: E402

# The stop this rule covers. The earlier stops already carry bands.
BAND_STOP = 665_000

# The mean of the 200,000-step student band. `results/head_band.csv`, row
# `200000,student`, column `mean`. `test_the_center_is_the_measured_200k_mean`
# holds the two together.
BAND_CENTER = 1.0651

# Half the width of the window that buys a band. About 3.4 pooled standard
# deviations of this card's own bands.
BAND_RADIUS = 0.01

# The head the rule reads. The student head carries the card's comparison
# and #373's 1.0660.
BAND_HEAD = "student"

# Binary floats put 1.0651 + 0.01 at 1.0751000000000002, so a score exactly
# on the edge would read as outside. The edge belongs to FIRE: the cheap
# error is a band the card did not need.
EDGE_TOL = 1e-9

FIRE, SKIP, WAIT = "FIRE", "SKIP", "WAIT"
CODES = {FIRE: 0, SKIP: 10, WAIT: 20}


def decide(score, center: float = BAND_CENTER, radius: float = BAND_RADIUS):
    """The verdict for one score, and its distance from the center.

    `score` is None when the stop has no number yet. The caller then waits.
    """
    if score is None:
        return WAIT, None
    distance = abs(float(score) - center)
    return (FIRE if distance <= radius + EDGE_TOL else SKIP), distance


def pooled_std(csv_path, head: str | None = None):
    """The pooled standard deviation of this card's measured bands.

    It pools every row of `head_band.csv` that holds two draws or more:

        s_pooled = sqrt( sum (n_i - 1) s_i^2 / sum (n_i - 1) )

    `head` keeps one head only. The default pools both, which is the
    number the rule quotes.

    It returns None when no row qualifies.
    """
    num = 0.0
    den = 0
    try:
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if head is not None and row["head"] != head:
                    continue
                n = int(row["n_draws"])
                if n < 2:
                    continue
                # Recompute from the draws. The `std` column is rounded.
                draws = [float(part.split("=")[1])
                         for part in row["seeds"].split() if "=" in part]
                if len(draws) < 2:
                    continue
                mean = sum(draws) / len(draws)
                var = sum((d - mean) ** 2 for d in draws) / (len(draws) - 1)
                num += (len(draws) - 1) * var
                den += len(draws) - 1
    except (OSError, KeyError, ValueError):
        return None
    return (num / den) ** 0.5 if den else None


def live_score(stop: int, head: str, results):
    """The score the driver wrote for one (stop, head), or None."""
    return full_pass.score(stop, head, results)


def protocol_offsets(csv_path, head: str = BAND_HEAD):
    """`[(stop, mean, protocol draw, offset)]` for every banded row.

    A SKIP leaves its stop with one draw, and that draw comes from the
    protocol head seed. So the question the skip must answer is: how far
    does the protocol draw sit from the mean of a full band?

    Every banded stop of this card measures that offset directly. The rows
    with two draws or more are the evidence, and this reads them back.
    """
    out = []
    try:
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["head"] != head or int(row["n_draws"]) < 2:
                    continue
                draws = {int(part.split("=")[0]): float(part.split("=")[1])
                         for part in row["seeds"].split() if "=" in part}
                if full_pass.HEAD_SEED not in draws:
                    continue
                mean = sum(draws.values()) / len(draws)
                protocol = draws[full_pass.HEAD_SEED]
                out.append((int(row["stop"]), mean, protocol, protocol - mean))
    except (OSError, KeyError, ValueError):
        return []
    return out


def skip_context(csv_path, score, stop: int, head: str, center: float):
    """What the one draw at a skipped stop says about its unmeasured mean.

    The measured offsets bound the mean the band would have found. The
    bound is the widest offset in each direction, so it is the whole
    measured spread and not a standard deviation of it.

    Returns None when no banded row carries the protocol seed.
    """
    rows = protocol_offsets(csv_path, head)
    if not rows or score is None:
        return None
    offsets = [off for _, _, _, off in rows]
    # The draw sits `off` ABOVE the mean, so the mean sits `off` BELOW the
    # draw. The widest offset up gives the lowest mean, and the reverse.
    lo, hi = score - max(offsets), score - min(offsets)
    sigma = pooled_std(csv_path)
    return {"stop": stop, "head": head, "score": score, "rows": rows,
            "offset_lo": min(offsets), "offset_hi": max(offsets),
            "mean_lo": lo, "mean_hi": hi, "center": center,
            "rise_lo": lo - center, "rise_hi": hi - center,
            "pooled_sd": sigma,
            "sd_lo": (lo - center) / sigma if sigma else None,
            "sd_hi": (hi - center) / sigma if sigma else None}


def offsets_text(context) -> str:
    """The skip's evidence, as the report cites it."""
    if context is None:
        return "no banded row carries the protocol seed, so no bound"
    lines = [f"the protocol draw against its own band mean, "
             f"{context['head']} head",
             f"{'stop':>7}  {'band mean':>9}  {'protocol':>9}  {'offset':>8}"]
    for stop, mean, protocol, off in context["rows"]:
        lines.append(f"{stop:>7}  {mean:>9.4f}  {protocol:>9.4f}  "
                     f"{off:>+8.4f}")
    lines.append(
        f"measured offsets: {context['offset_lo']:+.4f} to "
        f"{context['offset_hi']:+.4f}")
    lines.append(
        f"stop {context['stop']} has one draw, {context['score']:.4f}. Its "
        f"band mean lands between {context['mean_lo']:.4f} and "
        f"{context['mean_hi']:.4f}.")
    lines.append(
        f"against the {context['center']:.4f} band mean at 200000 steps: "
        f"{context['rise_lo']:+.4f} to {context['rise_hi']:+.4f}")
    if context["pooled_sd"]:
        lines.append(
            f"pooled head-seed sd {context['pooled_sd']:.4f}, so the rise is "
            f"{context['sd_lo']:.1f} to {context['sd_hi']:.1f} of it")
    return "\n".join(lines)


def main(argv=None) -> int:
    study = os.path.dirname(HERE)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stop", type=int, default=BAND_STOP)
    ap.add_argument("--head", default=BAND_HEAD)
    ap.add_argument("--score", type=float, default=None,
                    help="use this number instead of the one on disk")
    ap.add_argument("--results", default=None,
                    help="#373's results directory")
    ap.add_argument("--csv", default=os.path.join(study, "results",
                                                  "head_band.csv"))
    ap.add_argument("--center", type=float, default=BAND_CENTER)
    ap.add_argument("--radius", type=float, default=BAND_RADIUS)
    ap.add_argument("--write", default=None,
                    help="append the verdict to this file, once")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--offsets", action="store_true",
                    help="what the one draw at a skipped stop bounds")
    ap.add_argument("--offsets-out", default=None,
                    help="write the offsets block to this file")
    args = ap.parse_args(argv)

    if args.radius < 0:
        print("ABORT: the radius must not be negative", file=sys.stderr)
        return 2

    score = args.score
    if score is None:
        results = args.results or full_pass.parent_results(
            os.environ.get("WT", full_pass.REPO_ROOT))
        score = live_score(args.stop, args.head, results)

    verdict, distance = decide(score, args.center, args.radius)

    if args.explain:
        sigma = pooled_std(args.csv)
        sigma_head = pooled_std(args.csv, args.head)
        print(f"stop       {args.stop}")
        print(f"head       {args.head}")
        print(f"center     {args.center:.4f}  "
              "(mean of the 200k student band)")
        print(f"radius     {args.radius:.4f}")
        print(f"window     [{args.center - args.radius:.4f}, "
              f"{args.center + args.radius:.4f}]")
        if sigma:
            print(f"pooled sd  {sigma:.4f}  (both heads, every banded stop)")
            print(f"radius     {args.radius / sigma:.1f} pooled sd")
        if sigma_head:
            print(f"pooled sd  {sigma_head:.4f}  ({args.head} head only)")

    if args.offsets or args.offsets_out:
        block = offsets_text(skip_context(args.csv, score, args.stop,
                                          args.head, args.center))
        print(block)
        if args.offsets_out:
            with open(args.offsets_out, "w") as fh:
                fh.write(block + "\n")
            print(f"wrote {args.offsets_out}")

    if verdict == WAIT:
        line = f"WAIT   stop {args.stop} {args.head}: no score yet"
    else:
        line = (f"{verdict}   stop {args.stop} {args.head} = {score:.4f}, "
                f"{distance:.4f} from {args.center:.4f}, "
                f"radius {args.radius:.4f}")
    print(line)

    if args.write and verdict != WAIT:
        with open(args.write, "a") as fh:
            fh.write(line + "\n")

    return CODES[verdict]


if __name__ == "__main__":
    sys.exit(main())
