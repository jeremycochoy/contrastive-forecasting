#!/usr/bin/env python3
"""Trace every published score back to its evidence (#393).

`ladder_all.csv` is what the report is written from. Each of its rows claims a
GM-Relative MASE for one (cell, stop, head). This checks each claim against the
GIFT-Eval summary that produced it, and fails on anything that does not line up:

  * no summary for the row at all;
  * a summary aggregating something other than the 97 official configs;
  * a summary whose number disagrees with the published one;
  * a head whose encoder-source marker does not match the head it is filed
    under, which is the one crossing the protocol forbids.

A row with two independent summaries -- the cell's own tree and the broker's
copy on elisa -- is checked against both. They must agree.

Usage: python3 audit_scores.py [--results results] [--expect-configs 97]
Exit 0 if every row is backed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RES = os.path.join(os.path.dirname(HERE), "results")
AGG = re.compile(r"Aggregate GM-Relative MASE \((\d+) configs\): ([0-9.]+)")


def summaries(eval_root):
    """-> {(cell, stop_head): [(path, n_configs, value)]}, both trees."""
    found = {}
    for dirpath, _dirs, files in os.walk(eval_root):
        if "summary.txt" not in files:
            continue
        # <eval>/<cell>/eval/<stop_head>/gift/summary.txt   (a cell's own tree)
        # <eval>/_broker/<box>/<cell>/<stop_head>/gift/summary.txt
        parts = dirpath[len(eval_root) :].strip(os.sep).split(os.sep)
        if len(parts) < 3 or parts[-1] != "gift":
            continue
        stop_head, cell = parts[-2], parts[-3]
        if cell == "eval":
            cell = parts[-4]
        if not stop_head.startswith("bb"):
            continue
        path = os.path.join(dirpath, "summary.txt")
        with open(path) as fh:
            m = AGG.search(fh.read())
        if m:
            found.setdefault((cell, stop_head), []).append(
                (path, int(m.group(1)), float(m.group(2)))
            )
    return found


def markers(eval_root):
    """-> list of (path, head_it_is_filed_under, encoder_it_says)."""
    out = []
    for dirpath, _dirs, files in os.walk(eval_root):
        for fn in files:
            if not fn.endswith("_encoder_source.txt"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p) as fh:
                out.append((p, os.path.basename(dirpath).split("_")[-1], fh.read().strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RES)
    ap.add_argument("--expect-configs", type=int, default=97)
    ap.add_argument("--tol", type=float, default=5e-5)
    args = ap.parse_args()

    ladder = os.path.join(args.results, "ladder_all.csv")
    eval_root = os.path.join(args.results, "eval")
    if not os.path.exists(ladder):
        sys.exit(f"no {ladder}")

    found, problems, checked = summaries(eval_root), [], 0

    for path, filed_under, says in markers(eval_root):
        if filed_under != says:
            problems.append(f"CROSSED  {path}: filed under {filed_under}, marker says {says}")

    with open(ladder, newline="") as fh:
        for row in csv.DictReader(fh):
            cell, stop, head = row["cell"], int(row["stop"]), row["head"]
            published = float(row["gm_rel_mase"])
            key = (cell, f"bb{stop // 1000}k_{head}")
            hits = found.get(key, [])
            if not hits:
                problems.append(f"NO EVIDENCE  {cell} {stop} {head} = {published}")
                continue
            for path, n, value in hits:
                checked += 1
                if n != args.expect_configs:
                    problems.append(
                        f"CONFIGS  {cell} {stop} {head}: {n} configs, expected "
                        f"{args.expect_configs} ({path})"
                    )
                if abs(value - published) > args.tol:
                    problems.append(
                        f"MISMATCH  {cell} {stop} {head}: published {published}, "
                        f"summary {value} ({path})"
                    )
            print(f"  ok  {cell:<24} {stop:>7} {head:<8} {published:.4f}  "
                  f"({len(hits)} {'summaries' if len(hits) != 1 else 'summary'})")

    print()
    if problems:
        for p in problems:
            print(p)
        print(f"\n{len(problems)} problem(s) over {checked} summary check(s)")
        return 1
    print(f"every published row backed: {checked} summary check(s), all "
          f"{args.expect_configs} configs, no crossed head/encoder pair")
    return 0


if __name__ == "__main__":
    sys.exit(main())
