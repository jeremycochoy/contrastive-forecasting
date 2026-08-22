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

A repeat pair is any two rows of scores.csv that share (alpha, schedule, ramp,
align_w). Nothing names `s08` and `s08b` here: an arms table that adds a second
repeat gets a second pair, and the figure and the table both read it. The align
weight is in the key so an arm that MOVES the objective at one momentum is not
read as a second seed of it.

Two arms, so this is a range and not a standard deviation. Two draws give no
useful estimate of a standard deviation, and a range says what it is.
"""
from __future__ import annotations

from itertools import combinations


def pairs(rows) -> list[tuple[dict, dict]]:
    """Every pair of scored rows that differ in the backbone seed ALONE.

    The key carries the align weight as well as the momentum, the schedule and
    the ramp. `w3_s08` shares the first three with `s08` and moves the weight,
    so the two are a change to the objective and not a repeat. A pair without
    the weight in its key would report that change as run-to-run noise, under
    a sentence that says the two differ in the seed alone.
    """
    out = []
    for a, b in combinations(rows, 2):
        if (a["alpha"], a["schedule"], a.get("ramp", 0), a.get("align_w", 1.0)) == \
           (b["alpha"], b["schedule"], b.get("ramp", 0), b.get("align_w", 1.0)):
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


def cell(rows, alpha: float, schedule: str,
         align_w: float = 1.0) -> dict | None:
    """The one scored row at a momentum, a schedule and an align weight.

    The weight is in the match because an arm that moves it is a different
    objective at the same momentum, and this function answers "which arm is
    THE arm at 0.95". Two rows would make that question have two answers.
    """
    for r in rows:
        if (abs(r["alpha"] - alpha) < 1e-9 and r["schedule"] == schedule
                and abs(r.get("align_w", 1.0) - align_w) < 1e-9):
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


# --- The question round 8 asks -----------------------------------------------


def seeds_of(rows, row) -> list[dict]:
    """Every scored row of the SAME cell as `row`, ordered by backbone seed.

    One cell is one (momentum, schedule, ramp, align weight). Two rows of one
    cell differ in the backbone seed alone, so this is the arm at every seed
    the card trained it at. `row` itself is in the result.
    """
    def key(r):
        return (r["alpha"], r["schedule"], r.get("ramp", 0),
                r.get("align_w", 1.0))

    return sorted([r for r in rows if key(r) == key(row)],
                  key=lambda r: str(r.get("seed", "")))


def separation_from_level(row: dict, level: float, level_name: str,
                          d: float) -> str:
    """Whether a repeat spread of `d` separates one arm from a fixed score.

    `separation` compares two arms of THIS table. This compares one arm
    against a number the card does not train — the k = 0 parent of the cell,
    or a score another card published. The test is the same and it is strict:
    a gap equal to the spread does not separate.
    """
    gap = abs(row["score"] - level)
    side = "BELOW" if row["score"] < level else "ABOVE"
    text = (f"`{row['arm']}` {row['score']:.4f} sits {gap:.4f} {side} "
            f"{level_name} {level:.4f}. The repeat spread is {d:.4f}.")
    if gap > d:
        return (f"{text} The gap is LARGER than the spread, so this card "
                f"separates the two.")
    return (f"{text} The gap is SMALLER than the spread, so this card does "
            f"NOT separate the two.")


def seed_sentence(rows: list[dict], row: dict) -> str:
    """One arm at every seed it was trained at, in one sentence.

    Two seeds of the winner say whether the win holds. One seed says nothing
    about it, and the sentence says so instead of implying more.
    """
    fam = seeds_of(rows, row)
    if len(fam) < 2:
        return (f"`{row['arm']}` carries ONE backbone seed, so this card does "
                f"not say whether its score holds at a second seed.")
    names = ", ".join(f"`{r['arm']}` {r['score']:.4f} (seed {r.get('seed', '?')})"
                      for r in fam)
    lo = min(r["score"] for r in fam)
    hi = max(r["score"] for r in fam)
    return (f"{names}. The {len(fam)} seeds span {hi - lo:.4f}, "
            f"{lo:.4f} to {hi:.4f}.")
