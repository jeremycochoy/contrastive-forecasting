#!/usr/bin/env python3
"""#404 — the repeat spread this card measures itself.

Round 1 called three arms one cluster. The three span 0.0037, and the only
number that made 0.0037 "small" was #373's repeat spread of 0.6% to 1.3%.
#373 measured a different cell: k = 3, and `L_align` against the STUDENT. This
card runs k = 32 against the TEACHER.

So the card now trains one arm twice. `s08` and `s08b` hold the same momentum,
the same schedule, the same depth and the same head seed. They differ in the
backbone seed alone, 20260520 against 20260521. The distance between their two
scores is the run-to-run spread of THIS cell, at THIS stop.

A repeat pair is any two rows of scores.csv that share (alpha, schedule, ramp).
Nothing names `s08` and `s08b` here: an arms table that adds a second repeat
gets a second pair, and the figure and the table both read it.

Two arms, so this is a range and not a standard deviation. Two draws give no
useful estimate of a standard deviation, and a range says what it is.
"""
from __future__ import annotations

from itertools import combinations


def pairs(rows) -> list[tuple[dict, dict]]:
    """Every pair of scored rows that share a momentum and a schedule."""
    out = []
    for a, b in combinations(rows, 2):
        if (a["alpha"], a["schedule"], a.get("ramp", 0)) == \
           (b["alpha"], b["schedule"], b.get("ramp", 0)):
            out.append((a, b) if a["arm"] < b["arm"] else (b, a))
    return out


def spread(a: dict, b: dict) -> tuple[float, float]:
    """`(absolute, relative)` distance between two scores of one pair."""
    d = abs(a["score"] - b["score"])
    return d, d / min(a["score"], b["score"])


def unresolved(rows, d: float) -> list[str]:
    """The arms within `d` of the best score, best first.

    An arm closer to the winner than one repeat is an arm this card does not
    separate from the winner. The test is against the BEST score alone, so
    every name in the list carries the same meaning.
    """
    if not rows:
        return []
    best = min(r["score"] for r in rows)
    near = [r for r in rows if r["score"] - best <= d]
    return [r["arm"] for r in sorted(near, key=lambda r: r["score"])]


def sentence(rows) -> str:
    """One sentence naming the measured spread, or an empty string.

    It also names the arms the spread cannot separate from the winner. Two
    arms closer than one repeat are two arms this card does not rank.
    """
    ps = pairs(rows)
    if not ps:
        return ""
    a, b = max(ps, key=lambda p: spread(*p)[0])
    d, rel = spread(a, b)
    text = (f"{a['arm']} and {b['arm']} are one arm at two backbone seeds. "
            f"They land {d:.4f} apart ({rel:.1%}), which is the repeat spread "
            f"this card measures.")
    near = unresolved(rows, d)
    if len(near) > 1:
        text += (" " + ", ".join(near) + " all sit within that spread of the "
                 "best score, so this card does not rank them.")
    return text


# --- The question the card asks of two named arms ----------------------------


def cell(rows, alpha: float, schedule: str) -> dict | None:
    """The one scored row at a momentum and a schedule, or `None`."""
    for r in rows:
        if abs(r["alpha"] - alpha) < 1e-9 and r["schedule"] == schedule:
            return r
    return None


def separation(rows, d: float, alpha_a: float, alpha_b: float,
               schedule: str = "fixed") -> str:
    """Whether a repeat spread of `d` separates two arms, in one sentence.

    The card asks this of the two fixed-momentum arms, 0.90 and 0.95. A gap
    between two scores that is smaller than one repeat is a gap this card
    cannot call. The test is a strict comparison: a gap equal to the spread
    does not separate.
    """
    a = cell(rows, alpha_a, schedule)
    b = cell(rows, alpha_b, schedule)
    if a is None or b is None:
        return ""
    gap = abs(a["score"] - b["score"])
    lo, hi = (a, b) if a["score"] < b["score"] else (b, a)
    text = (f"`{lo['arm']}` {lo['score']:.4f} and `{hi['arm']}` "
            f"{hi['score']:.4f} are {gap:.4f} apart. "
            f"The repeat spread is {d:.4f}.")
    if gap > d:
        return (f"{text} The gap is LARGER than the spread, so this card "
                f"separates the two arms. `{lo['arm']}` scores better.")
    return (f"{text} The gap is SMALLER than the spread, so this card does "
            f"NOT separate the two arms.")
