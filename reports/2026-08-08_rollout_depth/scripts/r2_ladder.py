#!/usr/bin/env python3
"""#373 round 2 — the 14 cells at k = 3, their stops, and the extend rule.

One module, because the extend rule and the tables have to agree. A rule
implemented once in a driver and again in a table is a rule with two
answers.

Vocabulary:

  cell    one of the card's 14, `A1`..`A4` and `B1`..`B10`.
  stop    a backbone step count at which both heads are trained and scored:
          40k and 100k unconditionally, then 200k if the rule fires.
  head    `student` or `teacher`, the encoder the head reads. Trained
          separately, evaluated on its own encoder.
  down    this stop's GM-Relative MASE is lower than the same head's at the
          previous stop. Lower is better.

The k = 0 side is never retrained. It is the card's published tables, in
`published.py`. Group B's parents publish the student head only, so a group-B
teacher row has a k = 3 number and no baseline; it is printed as a number,
not as a delta.

Usage:  python3 r2_ladder.py [--results DIR]
"""
from __future__ import annotations

import argparse
import os
import sys

import published

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

K = 3
STOPS = [40, 100, 200]
HEADS = ["student", "teacher"]

# The card's order, and this study's run order with it.
CELLS = ["A1", "A2", "A3", "A4",
         "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]

# Every cell's arm and its L_align target, straight from the card's two
# tables. The report prints them so a reader does not have to hold 14 cell
# ids in their head.
CELL_ARM = {
    "A1": ("arm5 combab", "student", "scheduled"),
    "A2": ("arm6_v2 nse", "teacher", "scheduled"),
    "A3": ("arm6_v2 combab", "teacher", "scheduled"),
    "A4": ("arm6_v2 combab", "student", "scheduled"),
    "B1": ("arm6_v2 combab", "student", "fixed 0.9"),
    "B2": ("arm6_v2 combab", "teacher", "fixed 0.9"),
    "B3": ("arm5 combab", "student", "fixed 0.9"),
    "B4": ("arm5 combab", "teacher", "fixed 0.9"),
    "B5": ("arm4 combab", "none", "fixed 0.9"),
    "B6": ("arm6_v2 ncpc", "student", "fixed 0.9"),
    "B7": ("arm6_v2 ncpc", "teacher", "fixed 0.9"),
    "B8": ("arm6_v2 nse", "teacher", "fixed 0.9"),
    "B9": ("arm1 nse", "none", "fixed 0.9"),
    "B10": ("arm6_v2 nse", "student", "fixed 0.9"),
}

# The f-bearing term the flag touches in each cell. Rule 2 — `f` in the
# numerator AND every denominator — is exercised only by the first two rows.
CELL_TERM = {
    "B5": "pooled xshh_allt, floor subtracted",
    "B9": "split L_pred + CPC auxiliary",
    "A2": "L_align + CPC auxiliary",
    "B8": "L_align + CPC auxiliary",
    "B10": "L_align + CPC auxiliary",
}
DEFAULT_TERM = "L_align only"


def term(cell):
    return CELL_TERM.get(cell, DEFAULT_TERM)


def tag(cell, stop_k, head, k=K):
    return f"{cell}_k{k}_bb{stop_k}k_{head}"


# Round 1's four k = 3 runs at 40k, where its launcher's own tag differs
# from `<cell>_k3_bb40k_<head>`. B1 trained in round 1's review batch and
# carries that batch's `G6_` prefix. Round 2 resumes those checkpoints
# rather than repeating the 40k stop, so the score has to be found under
# the name round 1 gave it.
TAG_ALIAS = {
    ("B1", 40, 3): "G6_B1_k3_bb40k",
}


def score(cell, stop_k, head, results=RESULTS, k=K):
    """This study's GM-Relative MASE for one (cell, stop, head), or None."""
    names = [tag(cell, stop_k, head, k)]
    alias = TAG_ALIAS.get((cell, stop_k, k))
    if alias:
        names.append(f"{alias}_{head}")
    for name in names:
        try:
            with open(os.path.join(results, f"score_{name}.txt")) as fh:
                return float(fh.read().strip())
        except (OSError, ValueError):
            continue
    return None


def baseline(cell, stop_k, head):
    """The published k = 0 number for the same (cell, stop, head), or None."""
    return published.at(cell, head, stop_k)


# The horizon and per-domain figures need PER-CONFIG k = 0 numbers, and the
# three parent reports publish only the 97-config aggregate. Round 1 ran a
# same-code k = 0 for four cells, at the protocol seed, so those four carry
# a k = 0 overlay and the other ten do not.
#
# B5 has two such runs. The overlay stands in for the published baseline, so
# the rule is "the one that reproduces it": at bb40k the elisa run reads
# 1.2751 against a published 1.2748, and the rented-box run reads 1.3917.
# Reproduction against published, all at bb40k on the student head:
#   B1  1.2025 / 1.2025   0.0000
#   B9  1.5583 / 1.5579   0.0004
#   B5  1.2751 / 1.2748   0.0003   (G7_B5_k0_e)
#   A3  1.2189 / 1.1895   0.0294
K0_TAG = {
    "A3": "A3_k0",
    "B1": "G6_B1_k0",
    "B5": "G7_B5_k0_e",
    "B9": "G2_B9_k0",
}


def k0_tag(cell, stop_k, head):
    """The eval tag of this study's own k = 0 run for a cell, or None."""
    stem = K0_TAG.get(cell)
    return None if stem is None else f"{stem}_bb{stop_k}k_{head}"


def rows(results=RESULTS):
    """Every measured (cell, stop, head), with its baseline and delta."""
    out = []
    for cell in CELLS:
        for stop_k in STOPS:
            for head in HEADS:
                s = score(cell, stop_k, head, results)
                if s is None:
                    continue
                b = baseline(cell, stop_k, head)
                out.append({
                    "cell": cell, "stop": stop_k, "head": head,
                    "k3": s, "k0": b,
                    "delta": None if b is None else s - b,
                    "pct": None if b is None else 100.0 * (s - b) / b,
                })
    return out


def by_cell(results=RESULTS):
    """`{cell: {head: {stop: score}}}` over what this study measured."""
    out = {}
    for r in rows(results):
        out.setdefault(r["cell"], {}).setdefault(r["head"], {})[r["stop"]] = r["k3"]
    return out


def extend_decision(cell, stop_k, results=RESULTS):
    """The card's extend rule at one stop.

    Returns `(extend, keep, reason)`:
      extend  True if the cell trains another 100k.
      keep    the heads carried forward, in HEADS order.
      reason  one line, for the stop-reason table.

    The rule, verbatim from the card: "per head against its own previous
    stop: both heads down, extend and keep both; one head down, extend and
    keep that head; neither down, stop."

    40k has no previous stop, so it never decides anything: the card makes
    40k and 100k unconditional. 200k is the ceiling.
    """
    if stop_k == 40:
        return True, list(HEADS), "40k and 100k are unconditional"
    prev = {40: None, 100: 40, 200: 100}[stop_k]
    per = by_cell(results).get(cell, {})
    moved, missing = {}, []
    for head in HEADS:
        now, before = per.get(head, {}).get(stop_k), per.get(head, {}).get(prev)
        if now is None or before is None:
            missing.append(head)
            continue
        moved[head] = now - before
    if missing and not moved:
        return False, [], f"no comparison at bb{stop_k}k: missing {', '.join(missing)}"
    down = [h for h in HEADS if moved.get(h) is not None and moved[h] < 0]
    detail = ", ".join(f"{h[0].upper()} {moved[h]:+.4f}" for h in HEADS if h in moved)
    if stop_k >= 200:
        return False, down, f"ceiling: 200k is the card's last stop ({detail})"
    if len(down) == 2:
        return True, down, f"both heads down ({detail}) — extend, keep both"
    if len(down) == 1:
        kept = down[0]
        return True, down, f"{kept} down ({detail}) — extend, keep {kept}"
    return False, [], f"neither head down ({detail}) — stop"


def _fmt(v, w=7, nd=4):
    return " " * w if v is None else f"{v:{w}.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()

    print("per-cell per-stop per-head, k = 3 against the published k = 0")
    print(f"{'cell':<5}{'stop':>5} {'head':<8}{'k=3':>8}{'k=0':>9}{'delta':>9}{'%':>8}")
    for r in rows(a.results):
        pct = "" if r["pct"] is None else f"{r['pct']:+8.2f}"
        print(f"{r['cell']:<5}{r['stop']:>5} {r['head']:<8}{_fmt(r['k3'], 8)}"
              f"{_fmt(r['k0'], 9)}{_fmt(r['delta'], 9)}{pct:>8}")

    print("\nstop reasons")
    print(f"{'cell':<5}{'stop':>5}  {'extend':<7}{'keep':<18}reason")
    for cell in CELLS:
        for stop_k in STOPS:
            per = by_cell(a.results).get(cell, {})
            if not any(stop_k in per.get(h, {}) for h in HEADS):
                continue
            ext, keep, why = extend_decision(cell, stop_k, a.results)
            print(f"{cell:<5}{stop_k:>5}  {'yes' if ext else 'no':<7}"
                  f"{','.join(keep) or '-':<18}{why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
