#!/usr/bin/env python3
"""Head-seed spread, and the gaps it has to be smaller than.

The bootstrap in `eval_bootstrap.py` resamples the eval set. It cannot see
the head: every cell in this report is one q-head, trained once, from one
seed. So a teacher-vs-student difference smaller than the range a single cell
moves under nothing but a head seed is not a finding.

The first pass measured that range on four TEACHER backbones at 100k and
200k, and it came out between 0.0063 and 0.0908 — a factor of fourteen, so no
single number from those four carries across. The controlled comparison lives
somewhere else entirely: backbone 40 000, and both sides of it, teacher and
student, are separate backbones. A bar measured at 200k on the teacher side
says nothing about a 40k student cell.

So the bar is measured where the comparison is. `run_head_seeds.sh` with
`SIDES="teacher student"` retrains the head under two extra seeds on the 40k
backbones of one cell that moved (`arm5 base`) and one that did not
(`arm5 combab`), on both sides, and re-runs the full 97-config eval each
time. This script groups every cell that ended up with more than one head
seed and prints the spread per backbone step, so the 40k bar is read off 40k
cells.

Groups are keyed on (arm, backbone step, align target, code snapshot): the
members of a group differ in nothing but `--seed` on the head.

Usage:
    python3 seed_spread.py --table <gm_relative_mase.csv> \
        [--ci <eval_bootstrap_ci.csv>] \
        [--controlled <controlled_delta_40k.csv>] --out seed_spread.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict

CONTROLLED_BB = 40000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--ci", default=None)
    ap.add_argument("--controlled", default=None,
                    help="controlled_delta_40k.csv, judged against the 40k bar")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.table, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Group by everything except the head seed. `align_target` is in the key
    # because at 40k the teacher and the student cell of one arm are two
    # different backbones, and `code_snapshot` because a copied #379 cell was
    # never re-run here.
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r["align_target"] == "none":
            continue
        groups[(r["arm_slug"], r["bb_steps"], r["align_target"],
                r["code_snapshot"])].append(r)

    out_rows = []
    for (arm, bb, target, snap), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        vals = sorted(float(m["gm_rel_mase"]) for m in members)
        seeds = sorted(m["head_seed"] for m in members)
        rel_range = (vals[-1] - vals[0]) / vals[0]
        out_rows.append({
            "arm_slug": arm, "bb_steps": bb,
            "align_target": target, "code_snapshot": snap,
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
        print(f"  {r['arm_slug']:16s} bb={int(r['bb_steps']) // 1000:>3}k "
              f"{r['align_target']:7s} {r['values']}   range {r['range']} "
              f"({float(r['range_rel']) * 100:.1f}% of the smallest)")

    ranges = [float(r["range_rel"]) for r in out_rows]
    print(f"\nall cells: median {st.median(ranges) * 100:.1f}%, "
          f"largest {max(ranges) * 100:.1f}% of the cell's own value")

    # The bar that matters for section 1. The controlled deltas are all at
    # backbone 40 000, so they are judged against 40k cells only.
    at40 = [r for r in out_rows if int(r["bb_steps"]) == CONTROLLED_BB]
    bar = None
    if not at40:
        print(f"\nNO 40k CELL CARRIES A REPLICATE HEAD SEED — the controlled "
              f"deltas at backbone {CONTROLLED_BB} have no measured bar, and "
              f"the 100k/200k spreads above cannot be carried across.")
    else:
        abs_ranges = [float(r["range"]) for r in at40]
        bar = max(abs_ranges)
        print(f"\nbackbone {CONTROLLED_BB}, where every controlled delta "
              f"lives: {len(at40)} cells, "
              f"{len({r['align_target'] for r in at40})} target(s)")
        for r in at40:
            print(f"  {r['arm_slug']:16s} {r['align_target']:7s} "
                  f"{r['values']}   range {r['range']} "
                  f"({float(r['range_rel']) * 100:.1f}%)")
        print(f"  largest 40k head-seed range: {bar:.4f} "
              f"({max(float(r['range_rel']) for r in at40) * 100:.1f}%)")

    if args.controlled and bar is not None:
        with open(args.controlled, newline="") as fh:
            ctl = list(csv.DictReader(fh))
        col = next((c for c in ("delta_controlled", "delta", "gm_delta")
                    if ctl and c in ctl[0]), None)
        if col is None:
            print("\n(controlled table carries no delta column I recognise: "
                  + ", ".join(ctl[0]) + ")")
        else:
            clears = [c for c in ctl if abs(float(c[col])) > bar]
            print(f"\ncontrolled deltas larger than the 40k head-seed bar "
                  f"({bar:.4f}): {len(clears)}/{len(ctl)}")
            for c in sorted(ctl, key=lambda c: -abs(float(c[col]))):
                mark = "clears" if abs(float(c[col])) > bar else "inside"
                print(f"  {c['arm_slug']:16s} {float(c[col]):+.4f}  {mark}")

    if args.ci and bar is not None:
        with open(args.ci, newline="") as fh:
            ci = list(csv.DictReader(fh))
        rel_bar = max(float(r["range_rel"]) for r in at40)
        gaps = [(abs(math.log(float(c["ratio_teacher_over_student"]))), c)
                for c in ci]
        clears = [c for g, c in gaps if g > math.log(1 + rel_bar)]
        print(f"\nteacher-vs-earlier-sweep cells whose |log ratio| exceeds the "
              f"largest 40k seed spread: {len(clears)}/{len(ci)}")
        for g, c in sorted(gaps, reverse=True)[:6]:
            mark = "clears" if g > math.log(1 + rel_bar) else "inside"
            print(f"  {c['cell']:34s} ratio "
                  f"{c['ratio_teacher_over_student']}  {mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
