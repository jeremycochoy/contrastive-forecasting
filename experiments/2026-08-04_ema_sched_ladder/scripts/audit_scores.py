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

A replicate head seed files its artefacts under `bb<N>k_<enc>_s<seed>`
rather than `bb<N>k_<enc>`, and a rerun that changed something else — the
GPU control — adds a further `_<tag>`. A ladder-shaped CSV naming a
`head_seed` and an optional `head_tag` column is checked against those
directories. The encoder is read as the second field of the directory name
rather than the last, or every suffixed marker would be reported as crossed
against a head called `s20260723` or `hw4090`.

Usage: python3 audit_scores.py [--results results] [--ladder FILE]
                               [--expect-configs 97]
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
# The protocol head seed. Only a replicate carries a suffix, so the
# seed-20260722 paths are exactly what they were before replicates existed.
PROTOCOL_SEED = "20260722"


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
            # The directory is bb<N>k_<enc>[_s<seed>][_<tag>], so the encoder
            # is the SECOND field, read from the front. Reading the last
            # field instead made every replicate marker look crossed against
            # a head called `s20260723`, and would do the same to the GPU
            # control's `hw4090`.
            parts = os.path.basename(dirpath).split("_")
            with open(p) as fh:
                out.append((p, parts[1] if len(parts) > 1 else "",
                            fh.read().strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RES)
    ap.add_argument("--ladder", default=None,
                    help="default: <results>/ladder_all.csv")
    ap.add_argument("--expect-configs", type=int, default=97)
    ap.add_argument("--tol", type=float, default=5e-5)
    args = ap.parse_args()

    ladder = args.ladder or os.path.join(args.results, "ladder_all.csv")
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
            seed = (row.get("head_seed") or PROTOCOL_SEED).strip()
            sfx = "" if seed == PROTOCOL_SEED else f"_s{seed}"
            # `head_tag` is how a rerun that changed something other than the
            # seed names its subtree — the GPU control is seed 20260722 on
            # the other card, so it carries no seed suffix and would collide
            # with the original row without this.
            tag = (row.get("head_tag") or "").strip()
            if tag:
                sfx += f"_{tag}"
            key = (cell, f"bb{stop // 1000}k_{head}{sfx}")
            hits = found.get(key, [])
            if not hits:
                problems.append(
                    f"NO EVIDENCE  {cell} {stop} {head} s{seed} = {published}")
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
            print(f"  ok  {cell:<24} {stop:>7} {head:<8} s{seed}"
                  f"{'/' + tag if tag else '':<8} {published:.4f}  "
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
