#!/usr/bin/env python3
"""Head-seed spread, per cell, and the gaps each cell's own spread has to clear.

The bootstrap in `eval_bootstrap.py` resamples the eval set. It cannot see
the head: every cell in this report is one q-head, trained once, from one
seed. So a teacher-vs-student difference smaller than the range that same
cell moves under nothing but a head seed is not a finding.

**There is no single bar.** The first pass measured the range on four
TEACHER backbones at 100k and 200k and got 0.0063 to 0.0908, a factor of
fourteen. The 40k measurement — the four cells where the controlled
comparison actually lives — spans 0.0018 to 0.0747, a factor of forty. A
number that clears the smallest of those is inside the largest. So this
script reports the range **per cell** and judges each controlled delta
against its own arm's measurement, and it refuses to reduce the four 40k
cells to one number.

`run_head_seeds.sh` with `SIDES="teacher student"` retrains the head under
two extra seeds on the 40k backbones of one cell that moved (`arm5 base`)
and one that did not (`arm5 combab`), on both sides, and re-runs the full
97-config eval each time. Eight of the ten controlled arms were never
replicated, so they have no measured spread, and this script says so rather
than lending them arm5's.

Groups are keyed on (arm, backbone step, align target, code snapshot): the
members of a group differ in nothing but `--seed` on the head. `align_target`
is in the key because at 40k the teacher and the student cell of one arm are
two different backbones, so a teacher arm and a student arm must never merge
into one row; `code_snapshot` because a copied #379 cell was never re-run
here.

Usage:
    python3 seed_spread.py --table <gm_relative_mase.csv> \
        [--controlled <controlled_delta_40k.csv>] --out seed_spread.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict

CONTROLLED_BB = 40000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--controlled", default=None,
                    help="controlled_delta_40k.csv, judged per arm against "
                         "that arm's own 40k spread")
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
    for (arm, bb, target, snap), members in sorted(
            groups.items(), key=lambda kv: (kv[0][0], int(kv[0][1]), kv[0][2])):
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

    # --- the 40k cells, where every controlled delta lives -------------------
    at40 = [r for r in out_rows if int(r["bb_steps"]) == CONTROLLED_BB]
    # (arm, target) -> absolute range, so a delta is judged by its own arm.
    bar_by_cell = {(r["arm_slug"], r["align_target"]): float(r["range"])
                   for r in at40}
    if not at40:
        print(f"\nNO 40k CELL CARRIES A REPLICATE HEAD SEED — the controlled "
              f"deltas at backbone {CONTROLLED_BB} have no measured bar, and "
              f"the 100k/200k spreads above cannot be carried across.")
    else:
        abs_ranges = [float(r["range"]) for r in at40]
        print(f"\nbackbone {CONTROLLED_BB}, where every controlled delta "
              f"lives: {len(at40)} cells, "
              f"{len({r['align_target'] for r in at40})} target(s)")
        for r in at40:
            print(f"  {r['arm_slug']:16s} {r['align_target']:7s} "
                  f"{r['values']}   range {r['range']} "
                  f"({float(r['range_rel']) * 100:.1f}%)")
        print(f"  smallest {min(abs_ranges):.4f}, largest "
              f"{max(abs_ranges):.4f} — a factor of "
              f"{max(abs_ranges) / min(abs_ranges):.0f}. No single number "
              f"stands for these four; each delta is judged by its own arm.")

    if args.controlled and bar_by_cell:
        with open(args.controlled, newline="") as fh:
            ctl = list(csv.DictReader(fh))
        col = next((c for c in ("delta_controlled", "delta", "gm_delta")
                    if ctl and c in ctl[0]), None)
        if col is None:
            print("\n(controlled table carries no delta column I recognise: "
                  + ", ".join(ctl[0]) + ")")
        else:
            judged, unmeasured = [], []
            for c in ctl:
                arm = c["arm_slug"]
                sides = [bar_by_cell[k] for k in ((arm, "teacher"),
                                                  (arm, "student"))
                         if k in bar_by_cell]
                # The delta's own per-seed range is the direct measurement of
                # how far it moves under the head seed, so it is the bar when
                # the controlled table carries it. The wider of the two side
                # ranges is the fallback: the delta inherits both.
                if c.get("delta_seed_range"):
                    judged.append((c, float(c["delta_seed_range"]), "delta"))
                elif len(sides) == 2:
                    judged.append((c, max(sides), "sides"))
                else:
                    unmeasured.append(c)
            print(f"\ncontrolled deltas judged against their OWN arm's 40k "
                  f"head-seed spread: {len(judged)}/{len(ctl)}")
            for c, bar, kind in sorted(judged,
                                       key=lambda cb: -abs(float(cb[0][col]))):
                point = float(c.get("delta_seed_mean") or c[col])
                mark = "clears" if abs(point) > bar else "inside"
                sign = c.get("delta_sign_stable", "")
                extra = f", sign stable across seeds: {sign}" if sign else ""
                print(f"  {c['arm_slug']:16s} {point:+.4f}  vs its own "
                      f"{kind} range {bar:.4f}  {mark}{extra}")
            print(f"\nno head-seed spread measured for these "
                  f"{len(unmeasured)} arms — their deltas are one head seed "
                  f"each and get no bar:")
            for c in unmeasured:
                print(f"  {c['arm_slug']:16s} {float(c[col]):+.4f}  "
                      f"(1 head seed, no spread measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
