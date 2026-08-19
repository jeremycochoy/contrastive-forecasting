#!/usr/bin/env python3
"""#404 — the published scores this card compares its four arms against.

The card trains none of these. It states them so a reader sees where the
sweep starts and what it has to beat. They live in one file because the
momentum figure and the table both read them, and two copies would drift.

Every number is a GM-Relative MASE on the 97 GIFT-Eval configs.

  K3_BB200K        1.0660  k = 3, the best score of the project
  K3_BB40K         1.0862  k = 3 at the stop this card's arms reach
  K32_BB200K       1.1637  k = 32, mean, student — the best score at k = 32
  K32_BB40K        1.2082  the same arm at bb40k, which is where this card
                           starts
  K0_PARENT_BB40K  1.1600  the k = 0 parent of this cell

The arms of this card stop at 40,000 backbone steps, so K3_BB40K and
K32_BB40K are the fair comparison. The two 200,000-step numbers are a
reminder of the target, not a comparison.
"""
from __future__ import annotations

# --- The published numbers ---------------------------------------------------

K3_BB200K = 1.0660
K3_BB40K = 1.0862
K32_BB200K = 1.1637
K32_BB40K = 1.2082
K0_PARENT_BB40K = 1.1600

# The repeat spread #373 measured on this protocol: two runs of one cell land
# 0.6% to 1.3% apart. The band around K3_BB40K holds it, so a reader sees
# whether an arm enters the range where k = 3 itself lands.
SPREAD = (0.006, 0.013)

# What the table prints, in the card's own order and wording.
TABLE = (
    ("k = 3, bb200k, the best score of the project", K3_BB200K),
    ("k = 3, bb40k", K3_BB40K),
    ("k = 32, mean, student, bb200k", K32_BB200K),
    ("k = 32, mean, student, bb40k", K32_BB40K),
    ("the k = 0 parent of this cell, bb40k", K0_PARENT_BB40K),
)


def band_bounds(centre: float = K3_BB40K,
                spread: tuple[float, float] = SPREAD) -> tuple[float, float]:
    """The outer edges of the repeat band around `centre`.

    The wider of the two spreads sets the edges, so a score inside the band is
    a score no repeat of k = 3 would separate from k = 3.
    """
    return centre * (1 - max(spread)), centre * (1 + max(spread))


def inner_band_bounds(centre: float = K3_BB40K,
                      spread: tuple[float, float] = SPREAD) -> tuple[float, float]:
    """The edges at the NARROWER repeat spread, drawn inside the band."""
    return centre * (1 - min(spread)), centre * (1 + min(spread))


def enters_band(score: float) -> bool:
    """True when `score` lands inside the k = 3 repeat band at bb40k."""
    lo, hi = band_bounds()
    return lo <= score <= hi


def dotted_lines() -> list[tuple[str, float]]:
    """The two 200,000-step scores the figure draws as dotted lines."""
    return [("k = 3, bb200k (the project best)", K3_BB200K),
            ("k = 32, mean, student, bb200k", K32_BB200K)]
