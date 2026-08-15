#!/usr/bin/env python3
"""#373 — check every eval divides by the same seasonal-naive denominator.

Each score is a geometric mean of per-config `Relative` = MASE / SN_MASE. Two
cells are comparable only if their SN_MASE column holds the same numbers on
the same configs. The harness recomputes that column per eval, so a panel that
changed, a config that dropped out, or a re-run against a different split
would move a score without touching the model.

The other checks read a score against its own eval. This one reads across
evals: it takes the (config -> SN_MASE) map out of every summary.txt and
requires one map for the whole study.

SN_MASE prints to 4 decimals, so the check compares the printed strings. Any
difference at the 4th decimal is a real difference in the denominator.

Usage:
  verify_denominator.py --results results
"""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

CONFIGS = 97          # every eval in this study runs the same panel

ROW_RE = re.compile(r"^(\S+/\S+/\S+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
                    re.M)


def denominator(summary: Path):
    """Return {config: SN_MASE-as-printed} for one eval."""
    rows = ROW_RE.findall(summary.read_text())
    return {cfg: sn for cfg, _mase, sn, _rel in rows}


def fingerprint(den: dict) -> str:
    blob = "\n".join(f"{k} {den[k]}" for k in sorted(den))
    return hashlib.md5(blob.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()

    evals = sorted(p for p in (args.results / "eval").iterdir() if p.is_dir())
    seen = defaultdict(list)      # fingerprint -> [eval name]
    short = []                    # evals whose panel is not 97 configs
    absent = []                   # evals with no summary.txt
    orphan = []                   # a score file with no eval summary behind it

    for d in evals:
        summary = d / "summary.txt"
        scored = (args.results / f"score_{d.name}.txt").exists()
        if not summary.exists():
            absent.append((d.name, scored))
            if scored:
                orphan.append(d.name)
            continue
        den = denominator(summary)
        if len(den) != CONFIGS:
            short.append((d.name, len(den)))
        seen[fingerprint(den)].append(d.name)

    scores = sorted(p.stem[len("score_"):]
                    for p in args.results.glob("score_*.txt"))
    measured = sorted(n for names in seen.values() for n in names)

    print(f"eval directories        : {len(evals)}")
    print(f"carry a summary.txt     : {len(measured)}")
    print(f"score files             : {len(scores)}")
    print(f"distinct denominators   : {len(seen)}")

    ok = True

    if absent:
        print("\nno summary.txt:")
        for name, scored in absent:
            tag = "HAS A SCORE" if scored else "no score either — not a measurement"
            print(f"  {name}: {tag}")
    if orphan:
        print(f"\nscored without an eval summary: {', '.join(orphan)}")
        ok = False

    if measured != scores:
        print("\nthe scored set and the summarised set differ:")
        for n in sorted(set(scores) ^ set(measured)):
            side = "score only" if n in set(scores) else "summary only"
            print(f"  {n}: {side}")
        ok = False

    if short:
        print(f"\npanel not {CONFIGS} configs:")
        for name, n in short:
            print(f"  {name}: {n}")
        ok = False

    if len(seen) == 1:
        (fp, names), = seen.items()
        print(f"\nOne denominator over {len(names)} evals, md5 {fp}.")
        print("Every cell divides by the same seasonal-naive column.")
    else:
        print("\nThe evals do NOT share one denominator:")
        ref = max(seen.items(), key=lambda kv: len(kv[1]))
        ref_den = denominator(args.results / "eval" / ref[1][0] / "summary.txt")
        for fp, names in sorted(seen.items(), key=lambda kv: -len(kv[1])):
            print(f"  md5 {fp}  {len(names)} eval(s): {', '.join(names[:6])}")
            if fp == ref[0]:
                continue
            other = denominator(args.results / "eval" / names[0] / "summary.txt")
            diff = [c for c in set(ref_den) | set(other)
                    if ref_den.get(c) != other.get(c)]
            for c in sorted(diff)[:8]:
                print(f"      {c}: {ref_den.get(c, '-')} vs {other.get(c, '-')}")
        ok = False

    print("\nALL EVALS SHARE ONE DENOMINATOR" if ok else "\nDENOMINATOR CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
