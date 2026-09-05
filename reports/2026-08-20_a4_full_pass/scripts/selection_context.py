#!/usr/bin/env python3
"""#407 review gap 8 — how selected the 1.0660 target is.

The card compares its new points against 1.0660. That number is A4's
student head at 200,000 steps, and #373 chose it AFTER seeing every score
it had produced. A number picked as the best of many is biased downward, so
a new point that lands near it is not evidence of a plateau.

This puts a size on "selected". It reads #373's own score files rather than
repeating any digit, and reports:

  n            how many GM-Relative MASE scores #373 published.
  rank         where the target sits among them, best first.
  runner-up    the next best score, and the gap to it.

It draws no statistical correction. The 99 scores come from 14 different
cells, so they are not draws from one distribution and the classic
winner's-curse formula does not apply to them. What the report needs is the
plain fact that the target is an argmin over a large set, next to the
head-seed band that `head_band.py` measures.

Usage:
  selection_context.py [--parent DIR] [--json OUT]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import full_pass  # noqa: E402


def published(parent):
    """`{tag: score}` over every score file #373 wrote."""
    out = {}
    for path in sorted(glob.glob(os.path.join(str(parent), "score_*.txt"))):
        tag = os.path.basename(path)[len("score_"):-len(".txt")]
        try:
            with open(path) as fh:
                out[tag] = float(fh.read().strip())
        except (OSError, ValueError):
            continue
    return out


def context(parent, target_tag=full_pass.BEST_BEFORE_TAG) -> dict:
    scores = published(parent)
    if target_tag not in scores:
        raise SystemExit(f"ABORT: no score for {target_tag} under {parent}")
    order = sorted(scores.items(), key=lambda kv: kv[1])
    rank = [t for t, _ in order].index(target_tag) + 1
    runner = next((kv for kv in order if kv[0] != target_tag), None)
    return {
        "target_tag": target_tag,
        "target": scores[target_tag],
        "n_published": len(scores),
        "rank": rank,
        "runner_up_tag": runner[0] if runner else None,
        "runner_up": runner[1] if runner else None,
        "gap_to_runner_up": (runner[1] - scores[target_tag])
        if runner else None,
        "best_five": [[t, v] for t, v in order[:5]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default=full_pass.PARENT_RESULTS)
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    c = context(a.parent)
    print(f"target        {c['target']:.4f}  ({c['target_tag']})")
    print(f"published     {c['n_published']} GM-Relative MASE scores in #373")
    print(f"rank          {c['rank']} of {c['n_published']}, best first")
    print(f"runner-up     {c['runner_up']:.4f}  ({c['runner_up_tag']}), "
          f"{c['gap_to_runner_up']:+.4f} away")
    print("best five:")
    for tag, value in c["best_five"]:
        print(f"  {value:.4f}  {tag}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(c, fh, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
