#!/usr/bin/env python3
"""#404 — the published scores this card compares its four arms against.

The card trains none of these. It states them so a reader sees where the
sweep starts and what it has to beat. They live in one file because the
momentum figure and the table both read them, and two copies would drift.

Every number is a GM-Relative MASE on the 97 GIFT-Eval configs.

TWO OF THE FIVE ARE MEASURED AND THREE ARE NOT. `SOURCE` says which is which,
and every figure and table that prints a number prints its source with it. A
reader who cannot tell a measured baseline from a quoted one cannot falsify
the comparison it carries.

  K3_BB200K        1.0660  k = 3 at 200,000 steps, the best score of the
                           project. Measured: the rollout-depth report's
                           `results/splits.csv`, row `A4_k3_bb200k_student`.
  K3_BB40K         1.0862  k = 3 at the stop this card's arms reach.
                           Measured: the same file, row `A4_k3_bb40k_student`.
  K32_BB200K       1.1637  k = 32, mean, student, at 200,000 steps. The issue
                           card states it. No run in this repository measures
                           it.
  K32_BB40K        1.2082  the same arm at 40,000 steps. The issue card states
                           it. No run in this repository measures it.
  K0_PARENT_BB40K  1.1600  the same backbone with no rollout (k = 0), at
                           40,000 steps. The issue card states it. No run in
                           this repository measures it, and the sibling
                           report's k = 0 aggregates at the same stop are
                           1.2189, 1.2025, 1.1513 and 1.2590.

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

# The two ways a number on this page can be known. A figure or a table that
# prints a reference prints its source beside it.
MEASURED = "measured, sibling report"
STATED = "stated by the issue card"

SOURCE = {
    "K3_BB200K": MEASURED,
    "K3_BB40K": MEASURED,
    "K32_BB200K": STATED,
    "K32_BB40K": STATED,
    "K0_PARENT_BB40K": STATED,
}

# What every figure calls that line. "parent" and "cell" named no run a reader
# can look up, and "cell" already means a group of arms in the results table.
# The k = 0 line carries its source, because no run here measures it.
K0_LINE = f"k = 0, same 40,000 steps, {STATED}"
K3_LINE = "k = 3, same 40,000 steps"

# #401's k = 32 arm is not only a horizontal reference: it IS a cell of this
# sweep. It ran #373's runner at its own default momentum, alpha = 0.9 raised
# to 1.0 at step 100,000, which sits between this card's arm 2 (0.9 fixed) and
# arm 4 (0.9 raised to 1.0 at 200,000). So the figure can place it on the x
# axis as well as draw its level.
K32_BB40K_ALPHA = 0.9
K32_BB40K_RAMP = 100_000

# The repeat spread #373 measured on this protocol: two runs of one cell land
# 0.6% to 1.3% apart. The band around K3_BB40K holds it, so a reader sees
# whether an arm enters the range where k = 3 itself lands.
SPREAD = (0.006, 0.013)

# What the table prints, in the card's own order. `bb40k` and `bb200k` are
# gone: the figures spell the same stop out as "40,000 steps", and one item
# takes one name. Each row carries its source.
TABLE = (
    ("k = 3, 200,000 steps, the best score of the project", K3_BB200K,
     SOURCE["K3_BB200K"]),
    ("k = 3, same 40,000 steps", K3_BB40K, SOURCE["K3_BB40K"]),
    ("k = 32, mean, student, 200,000 steps", K32_BB200K,
     SOURCE["K32_BB200K"]),
    ("k = 32, mean, student, same 40,000 steps", K32_BB40K,
     SOURCE["K32_BB40K"]),
    ("the same backbone with no rollout (k = 0), same 40,000 steps",
     K0_PARENT_BB40K, SOURCE["K0_PARENT_BB40K"]),
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
