#!/usr/bin/env python3
"""Head-seed spread, and the gaps it has to be smaller than.

The bootstrap in `eval_bootstrap.py` resamples the eval set. It cannot see
the head: every cell in this report is one q-head, trained once, from one
seed. Review item 4 retrains the head under two more seeds on four frozen
backbones and re-runs the full 97-config eval, which turns a point into a
spread.

That spread is the bar. A teacher-vs-student difference smaller than the
range a single cell moves under nothing but a head seed is not a finding.
So this prints both: the per-cell seed spread, and the teacher/student
ratios from `eval_bootstrap_ci.csv` expressed in the same units.

Usage:
    python3 seed_spread.py --table <gm_relative_mase.csv> \
        [--ci <eval_bootstrap_ci.csv>] --out seed_spread.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--ci", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.table, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Group the teacher cells by (arm, backbone step); the members of a group
    # differ only in head_seed.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["align_target"] == "teacher":
            groups[(r["arm_slug"], r["bb_steps"])].append(r)

    out_rows = []
    for (arm, bb), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        vals = sorted(float(m["gm_rel_mase"]) for m in members)
        seeds = sorted(m["head_seed"] for m in members)
        rel_range = (vals[-1] - vals[0]) / vals[0]
        out_rows.append({
            "arm_slug": arm, "bb_steps": bb,
            "head_steps": members[0]["head_steps"],
            "n_seeds": len(vals), "seeds": " ".join(seeds),
            "gm_min": f"{vals[0]:.4f}", "gm_max": f"{vals[-1]:.4f}",
            "gm_mean": f"{st.mean(vals):.4f}",
            "gm_sd": f"{st.stdev(vals):.4f}" if len(vals) > 2 else "",
            "range": f"{vals[-1] - vals[0]:.4f}",
            "range_rel": f"{rel_range:.4f}",
            "values": " ".join(f"{v:.4f}" for v in vals),
        })

    if not out_rows:
        print("no cell has more than one head seed yet — nothing to report")
        return 1

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    print(f"{len(out_rows)} cells with replicate head seeds\n")
    for r in out_rows:
        print(f"  {r['arm_slug']:16s} bb={int(r['bb_steps'])//1000:>3}k  "
              f"{r['values']}   range {r['range']} "
              f"({float(r['range_rel']) * 100:.1f}% of the smallest)")

    ranges = [float(r["range_rel"]) for r in out_rows]
    bar = max(ranges)
    print(f"\nseed spread: median {st.median(ranges) * 100:.1f}%, "
          f"largest {bar * 100:.1f}% of the cell's own value")

    if args.ci:
        with open(args.ci, newline="") as fh:
            ci = list(csv.DictReader(fh))
        gaps = [(abs(math.log(float(c["ratio_teacher_over_student"]))), c)
                for c in ci]
        clears = [c for g, c in gaps if g > math.log(1 + bar)]
        print(f"\nteacher-vs-student cells whose |log ratio| exceeds the "
              f"largest seed spread: {len(clears)}/{len(ci)}")
        for g, c in sorted(gaps, reverse=True)[:6]:
            mark = "clears" if g > math.log(1 + bar) else "inside"
            print(f"  {c['cell']:34s} ratio "
                  f"{c['ratio_teacher_over_student']}  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
