#!/usr/bin/env python3
"""#373 — the report's tables, from the score files and the splits.

Three tables:

  1. per cell, per head: k = 0 against k = 3 over the full 97, with the
     published k = 0 number beside them.
  2. the baseline validity gate: this study's retrained k = 0 against the
     published one, per group, with the card's 0.0002 threshold.
  3. the horizon split: short and medium+long, k = 0 against k = 3, with
     the card's success criterion evaluated per cell.

The published numbers are transcribed from the three parent reports and
live in PUBLISHED below. Group A publishes two heads per stop; group B
publishes one, trained on the student encoder.

Usage: score_table.py --results <results dir> --out <scores.md>
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# bb40k, from the card's two tables.
#   group A: ema_sched_ladder.md, per head.
#   group B: the union of small_long.md and lalign_teacher.md, student head.
PUBLISHED = {
    ("A1", "student"): 1.2596, ("A1", "teacher"): 1.2347,
    ("A2", "student"): 1.4238, ("A2", "teacher"): 1.4177,
    ("A3", "student"): 1.1895, ("A3", "teacher"): 1.1793,
    ("A4", "student"): 1.1603, ("A4", "teacher"): 1.1544,
    ("B1", "student"): 1.2025,
    ("B2", "student"): 1.2765,
    ("B3", "student"): 1.2868,
    ("B4", "student"): 1.2728,
    ("B5", "student"): 1.2748,
    ("B6", "student"): 1.3623,
    ("B7", "student"): 1.3159,
    ("B8", "student"): 1.3074,
    ("B9", "student"): 1.5579,
    ("B10", "student"): 1.3791,
}
GROUP_OF = {c: ("A" if c.startswith("A") else "B") for c, _h in PUBLISHED}
GATE = 0.0002          # the card's threshold, lalign_teacher.md section 7
NOISE_BAND = 0.0384    # ema_sched_ladder.md's pooled head-seed band
SCORE_RE = re.compile(r"^score_([AB]\d+)_k(\d+)_bb(\d+)k_(student|teacher)\.txt$")


def read_scores(results):
    out = {}
    for p in Path(results).glob("score_*.txt"):
        m = SCORE_RE.match(p.name)
        if not m:
            continue
        cell, k, stop, head = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        try:
            out[(cell, k, stop, head)] = float(p.read_text().strip())
        except ValueError:
            continue
    return out


def read_splits(results):
    path = Path(results) / "splits.csv"
    out = {}
    if not path.is_file():
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            parts = r["stop"].split("_")
            if len(parts) < 4 or not parts[1].startswith("k"):
                continue
            key = (parts[0], int(parts[1][1:]), parts[3])
            out.setdefault(key, {})[r["name"]] = float(r["gm_rel_mase"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    sc = read_scores(args.results)
    sp = read_splits(args.results)
    if not sc:
        print("no score file yet — no tables")
        return 0
    cells = sorted({c for c, _k, _s, _h in sc},
                   key=lambda c: (c[0], int(c[1:])))
    lines = []

    lines += ["### GM-Relative MASE at bb40k, 97 configs", "",
              "| cell | head | published k = 0 | this study k = 0 | k = 3 | "
              "k = 3 − k = 0 |", "|---|---|---|---|---|---|"]
    for cell in cells:
        for head in ("student", "teacher"):
            k0 = sc.get((cell, 0, 40, head))
            k3 = sc.get((cell, 3, 40, head))
            if k0 is None and k3 is None:
                continue
            pub = PUBLISHED.get((cell, head))
            base = k0 if k0 is not None else pub
            delta = (f"{k3 - base:+.4f}"
                     if k3 is not None and base is not None else "—")
            lines.append(
                f"| {cell} | {head} | "
                f"{pub if pub is not None else '—'} | "
                f"{f'{k0:.4f}' if k0 is not None else '—'} | "
                f"{f'{k3:.4f}' if k3 is not None else '—'} | {delta} |")
    lines += ["", f"Head-seed band ±{NOISE_BAND} "
              "(`ema_sched_ladder.md`, pooled). It bounds the head seed "
              "alone; the backbone-training spread is not measured.", ""]

    lines += ["### Baseline validity gate", "",
              "| group | cell | head | published | retrained k = 0 | "
              f"|Δ| | verdict (threshold {GATE}) |",
              "|---|---|---|---|---|---|---|"]
    any_gate = False
    for cell in cells:
        for head in ("student", "teacher"):
            k0, pub = sc.get((cell, 0, 40, head)), PUBLISHED.get((cell, head))
            if k0 is None or pub is None:
                continue
            any_gate = True
            d = abs(k0 - pub)
            lines.append(f"| {GROUP_OF.get(cell, '?')} | {cell} | {head} | "
                         f"{pub} | {k0:.4f} | {d:.4f} | "
                         f"{'PASS' if d <= GATE else 'FAIL'} |")
    if not any_gate:
        lines.append("| — | — | — | — | — | — | no gate has finished |")
    lines.append("")

    if sp:
        lines += ["### Horizon split", "",
                  "| cell | head | short k=0 | short k=3 | short Δ% | "
                  "med+long k=0 | med+long k=3 | med+long Δ% | criterion |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cell in cells:
            for head in ("student", "teacher"):
                a, b = sp.get((cell, 0, head)), sp.get((cell, 3, head))
                if not a or not b:
                    continue
                s0, s3 = a.get("short"), b.get("short")
                m0, m3 = a.get("medium_long"), b.get("medium_long")
                if None in (s0, s3, m0, m3):
                    continue
                ds = 100.0 * (s3 / s0 - 1.0)
                dm = 100.0 * (m3 / m0 - 1.0)
                ok = "MET" if (dm <= -5.0 and ds < 2.0) else "not met"
                lines.append(f"| {cell} | {head} | {s0:.4f} | {s3:.4f} | "
                             f"{ds:+.1f}% | {m0:.4f} | {m3:.4f} | {dm:+.1f}% | "
                             f"{ok} |")
        lines += ["", "Criterion, from the card: medium+long (42 configs) at "
                  "least 5% better, short (55 configs) losing less than 2%.", ""]

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(cells)} cell(s) scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
