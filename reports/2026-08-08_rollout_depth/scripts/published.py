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
# baseline validity gate.
GATE = 0.0002


def at(cell, head, stop_k):
    return PUBLISHED.get(cell, {}).get(head, {}).get(stop_k)


if __name__ == "__main__":
    for cell, heads in PUBLISHED.items():
        for head, stops in heads.items():
            row = "  ".join(f"bb{s}k={v:.4f}" for s, v in sorted(stops.items()))
            print(f"{cell:<4} {head:<8} {row}")
    sys.exit(0)
