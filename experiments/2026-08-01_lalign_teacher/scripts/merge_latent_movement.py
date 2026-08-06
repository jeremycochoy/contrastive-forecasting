#!/usr/bin/env python3
"""Merge #379's twenty un-retrained arms with #390's ten into one CSV.

Output schema is #379's exactly — ``arm_slug,label,step_later,drift_h,drift_e``
— so any plot script written against that report reads this file unchanged.
Rows are ordered by #379's canonical arm order, so the merged file reads as
one 30-cell grid rather than two appended halves.

arm5 / arm6_v2 rows come from #390 (the teacher-target retrain); every other
arm comes from #379. The two halves are on one scale because both were
measured with the same ``_latent_movement_batch.pt`` and the same
``mean_one_minus_cos``; ``--arms arm1`` under #379's own script reproduces
its committed rows bit for bit.

    merge_latent_movement.py <ref379.csv> <new390.csv> <out.csv>
"""
import csv
import sys

# #379's SLUGS order.
ORDER = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco",
         "arm1_tr1", "arm3_tr1", "arm4_tr1", "arm5_tr1", "arm6_v2_tr1", "bimoco_tr1",
         "arm1_nse", "arm3_nse", "arm4_nse", "arm5_nse", "arm6_v2_nse", "bimoco_nse",
         "arm1_ncpc", "arm3_ncpc", "arm4_ncpc", "arm5_ncpc", "arm6_v2_ncpc", "bimoco_ncpc",
         "arm1_combab", "arm3_combab", "arm4_combab", "arm5_combab", "arm6_v2_combab",
         "bimoco_combab"]
# The arms #390 retrained; their #379 rows are superseded.
RETRAINED = {s for s in ORDER if s == "arm5" or s == "arm6_v2"
             or s.startswith("arm5_") or s.startswith("arm6_v2_")}
HEADER = ["arm_slug", "label", "step_later", "drift_h", "drift_e"]


def read(path):
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        if head != HEADER:
            sys.exit(f"{path}: unexpected header {head}")
        return [row for row in r if row]


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ref, new, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [r for r in read(ref) if r[0] not in RETRAINED] + read(new)
    by_slug = {}
    for r in rows:
        by_slug.setdefault(r[0], []).append(r)

    missing = [s for s in ORDER if s not in by_slug]
    extra = [s for s in by_slug if s not in ORDER]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for slug in ORDER:
            w.writerows(sorted(by_slug.get(slug, []), key=lambda r: int(r[2])))
    print(f"wrote {out}: {len(rows)} pairs over {len(by_slug)}/{len(ORDER)} arms")
    if missing:
        print(f"MISSING arms: {' '.join(missing)}")
    if extra:
        print(f"UNPLACED arms (dropped): {' '.join(extra)}")
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
