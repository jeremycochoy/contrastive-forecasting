#!/usr/bin/env python3
"""#373 round 2 — the report's tables, in Markdown.

Two tables are the deliverable:

  1. per-cell per-stop per-head, `k = 3` against the card's published
     `k = 0`, at 40k, 100k and 200k where reached.
  2. the stop reason, per cell per stop, from the extend rule.

A third names what each cell exercises, so a reader can tell the five cells
that put `f` in the numerator AND every denominator from the nine that do
not.

Usage:  python3 r2_tables.py [--results DIR] [--out FILE] [--inject REPORT]
"""
from __future__ import annotations

import argparse
import os
import sys

import published
import r2_ladder as L

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_A = "<!-- BEGIN GENERATED TABLES -->"
MARK_B = "<!-- END GENERATED TABLES -->"


def _n(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def _d(v, nd=4):
    return "—" if v is None else f"{v:+.{nd}f}"


def scores_table(results):
    """Table 1. One row per (cell, stop), both heads side by side."""
    out = ["| cell | stop | S k=3 | S k=0 | S Δ | T k=3 | T k=0 | T Δ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    n = 0
    for cell in L.CELLS:
        for stop in L.STOPS:
            s3 = L.score(cell, stop, "student", results)
            t3 = L.score(cell, stop, "teacher", results)
            if s3 is None and t3 is None:
                continue
            s0 = L.baseline(cell, stop, "student")
            t0 = L.baseline(cell, stop, "teacher")
            sd = None if (s3 is None or s0 is None) else s3 - s0
            td = None if (t3 is None or t0 is None) else t3 - t0
            out.append(f"| {cell} | {stop}k | {_n(s3)} | {_n(s0)} | {_d(sd)} "
                       f"| {_n(t3)} | {_n(t0)} | {_d(td)} |")
            n += 1
    out.append("")
    out.append(f"{n} of {len(L.CELLS) * len(L.STOPS)} (cell, stop) pairs measured. "
               "`S` is the student-encoder head, `T` the teacher-encoder head. "
               "A `—` in a `k = 0` column means the parent report published no "
               "such number: group B's two parents publish the student head "
               "only, so a group-B teacher row carries a value and no delta.")
    return "\n".join(out)


def stop_table(results):
    """Table 2. What the extend rule decided, and on what numbers."""
    out = ["| cell | stop | extend | heads kept | reason |",
           "|---|---:|---|---|---|"]
    per = L.by_cell(results)
    for cell in L.CELLS:
        for stop in L.STOPS:
            if not any(stop in per.get(cell, {}).get(h, {}) for h in L.HEADS):
                continue
            ext, keep, why = L.extend_decision(cell, stop, results)
            out.append(f"| {cell} | {stop}k | {'yes' if ext else 'no'} | "
                       f"{', '.join(keep) or '—'} | {why} |")
    out.append("")
    out.append("The rule is the card's: per head against its own previous "
               "stop, both heads down extends and keeps both, one head down "
               "extends and keeps that head, neither down stops. 40k and 100k "
               "run unconditionally, so 40k decides nothing. Down means lower "
               "GM-Relative MASE.")
    return "\n".join(out)


def cells_table():
    """Table 3. What each cell is, and which term the flag touches in it."""
    out = ["| cell | arm | `L_align` | EMA | f-bearing term the depth copies |",
           "|---|---|---|---|---|"]
    for cell in L.CELLS:
        arm, align, ema = L.CELL_ARM[cell]
        out.append(f"| {cell} | `{arm}` | {align} | {ema} | {L.term(cell)} |")
    out.append("")
    out.append("Rule 2 of the card — `f` in the numerator and in every "
               "denominator — is exercised by B5, B9 and the CPC auxiliary of "
               "A2, B8 and B10. In the other nine cells the flag touches "
               "`L_align`, which has no denominator.")
    return "\n".join(out)


def render(results):
    return "\n\n".join([
        "### Per cell, per stop, per head",
        scores_table(results),
        "### Stop reasons",
        stop_table(results),
        "### What each cell is",
        cells_table(),
    ]) + "\n"


def inject(report, body):
    """Replace the marked block in the report, or say it is missing."""
    with open(report) as fh:
        text = fh.read()
    if MARK_A not in text or MARK_B not in text:
        print(f"  {report} holds no {MARK_A} block — not injecting")
        return False
    head, rest = text.split(MARK_A, 1)
    _, tail = rest.split(MARK_B, 1)
    with open(report, "w") as fh:
        fh.write(f"{head}{MARK_A}\n\n{body}\n{MARK_B}{tail}")
    print(f"  injected {len(body.splitlines())} lines into {report}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out")
    ap.add_argument("--inject")
    a = ap.parse_args()

    body = render(a.results)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(body)
        print(f"  tables -> {a.out}")
    else:
        print(body)
    if a.inject:
        inject(a.inject, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
