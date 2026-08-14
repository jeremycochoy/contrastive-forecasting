#!/usr/bin/env python3
"""#373 — the published k = 0 numbers, transcribed from the three parents.

GM-Relative MASE over the 97 GIFT-Eval configs, per cell, per stop, per
head. Source, as the card's two tables give them:

  group A   reports/2026-08-04_ema_sched_ladder/ema_sched_ladder.md,
            two heads per stop (student encoder, teacher encoder).
  group B   the union of reports/2026-07-21_split_pred_rep_small/small_long.md
            and reports/2026-08-04_lalign_teacher/lalign_teacher.md. Both
            publish ONE head per row, trained on the student encoder, so
            group B has no published teacher number to compare against.

A missing entry means the parent report published none: that cell did not
reach that stop, or its head was dropped by the extend rule.

Usage:  python3 published.py      # prints the table
"""
from __future__ import annotations

import sys

# cell -> head -> {stop in thousands: GM-Relative MASE}
PUBLISHED = {
    "A1": {"student": {40: 1.2596, 100: 1.2102, 200: 1.1910},
           "teacher": {40: 1.2347, 100: 1.2407}},
    "A2": {"student": {40: 1.4238, 100: 1.3913, 200: 1.3586},
           "teacher": {40: 1.4177, 100: 1.3746, 200: 1.3459}},
    "A3": {"student": {40: 1.1895, 100: 1.1921},
           "teacher": {40: 1.1793, 100: 1.1963}},
    "A4": {"student": {40: 1.1603, 100: 1.1945},
           "teacher": {40: 1.1544, 100: 1.1837}},
    "B1": {"student": {40: 1.2025, 100: 1.1616, 200: 1.1652}},
    "B2": {"student": {40: 1.2765, 100: 1.2514, 200: 1.1850}},
    "B3": {"student": {40: 1.2868, 100: 1.2456, 200: 1.2034}},
    "B4": {"student": {40: 1.2728, 100: 1.3678}},
    "B5": {"student": {40: 1.2748, 100: 1.3219}},
    "B6": {"student": {40: 1.3623, 100: 1.2978, 200: 1.3011}},
    "B7": {"student": {40: 1.3159, 100: 1.3012, 200: 1.3325}},
    "B8": {"student": {40: 1.3074, 100: 1.3368}},
    "B9": {"student": {40: 1.5579, 100: 1.4548, 200: 1.3308}},
    "B10": {"student": {40: 1.3791, 100: 1.3914}},
}

# ema_sched_ladder.md's pooled head-seed band, the largest range it measured
# at either head budget. It bounds the HEAD SEED only.
NOISE_BAND = 0.0384

# lalign_teacher.md section 7's threshold, which the card reuses for the
# baseline validity gate. It gates a retrain that holds the BACKBONE SEED:
# same recipe, same seed, so the only thing left to reproduce is the run.
GATE = 0.0002

# The backbone seed the three parents trained at.
PUBLISHED_SEED = 20260520

# A retrain that CHANGES the seed cannot take that gate. It is not repeating
# the published run; it is drawing a second one, and the difference it should
# be held to is what a seed draw is worth.
#
# This study measures that once: `B5·s2` against `B5·s3`, one machine, one
# recipe, one code snapshot, +0.0035 with a 95% interval of
# [-0.0183, +0.0230]. The gate is the far end of that interval.
#
# It is ONE pair of runs, and the interval is a bootstrap over their 97 eval
# configs, so it bounds that pair's eval sample and not the spread of seeds.
# `tables.py` reads the live number out of `bootstrap.csv`, so a re-run of
# the bootstrap moves the gate with it; this constant is the fallback.
SEED_BAND = 0.0230

# What a retrain can be resolved to, and where the two parts of it come
# from. Neither part is a guess.
#
#   re-eval    `B5·pub` trains nothing: it takes the parent report's own
#              published B5 checkpoint, puts this study's head seed and
#              97-config eval on it, and lands 0.0003 away. Everything
#              downstream of the backbone is worth that much on its own.
#   rounding   the parents print four decimals, so each published value
#              carries +/-0.00005 and a difference against one carries
#              +/-0.0001.
#
# A |Δ| at or below their sum is a run this pipeline cannot separate from
# the published one. GATE sits below it: the card asks for more than the
# comparison can resolve.
REEVAL_FLOOR = 0.0003
PRINT_QUANT = 0.0001
RESOLUTION = REEVAL_FLOOR + PRINT_QUANT


def best_published():
    """The lowest GM-Relative MASE any of the three parents printed, and where.

    This is the frontier before this study: the best score the project had
    reached on the 97 GIFT-Eval configs with `k = 0` training. Every figure
    that draws a grey baseline draws THIS number, so the two lead figures
    cannot drift apart.

    Returns `(value, cell, head, stop_k)`.
    """
    best = None
    for cell, heads in PUBLISHED.items():
        for head, stops in heads.items():
            for stop_k, v in stops.items():
                if best is None or v < best[0]:
                    best = (v, cell, head, stop_k)
    return best


def verdict(d, same_seed=True, seed_band=SEED_BAND):
    """The verdict on `d`, which is |retrained - published|.

    TWO GATES, because the rows ask two questions. A same-seed retrain is
    repeating the published run and takes the card's 0.0002. A cross-seed
    retrain is drawing a new one and takes the seed band: calling a
    cross-seed 0.0032 a FAIL against 0.0002 would name as a failure the same
    quantity this report's own B5 table calls the seed and puts inside the
    noise.
    """
    if not same_seed:
        return "inside the seed band" if d <= seed_band else "FAIL"
    if d <= GATE:
        return "PASS"
    if d <= RESOLUTION:
        return "at the re-run floor"
    return "FAIL"


def at(cell, head, stop_k):
    return PUBLISHED.get(cell, {}).get(head, {}).get(stop_k)


if __name__ == "__main__":
    for cell, heads in PUBLISHED.items():
        for head, stops in heads.items():
            row = "  ".join(f"bb{s}k={v:.4f}" for s, v in sorted(stops.items()))
            print(f"{cell:<4} {head:<8} {row}")
    sys.exit(0)
