#!/usr/bin/env python3
"""#393 — the parent half of the union table, as a checked-in file.

The two parent reports publish their own tables; this script transcribes the
rows the card's union table quotes, so every parent number in the report
traces to a file under `results/` like every other number does.

Sources, both read at the commit recorded in `source_commit`:
  prev  reports/2026-07-21_split_pred_rep_small/small_long.md
        the 30-cell sweep with L_align on the student.
  new   reports/2026-08-04_lalign_teacher/lalign_teacher.md
        the ten L_align cells retrained against the EMA teacher.

`align` selects the source: a student-align row is a `prev` row, a
teacher-align row is a `new` row, and the two rows with no L_align term
(`arm4 combab`, `arm1 nse`) carry the same numbers in both reports.

`top5_in` is the parent reports' own five-lowest placement, verbatim:
prev / new / both / none. It is a placement, not a score: a row can hold the
lowest value in its own report table and still read `none`.

Writes `results/union_parents.csv`.
"""
import csv
import os

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_COMMIT = "master@946211e"

# cell, align, source, top5_in, {stop: value}
ROWS = [
    ("arm6_v2 combab", "student", "prev", "prev",
     {40000: 1.2025, 100000: 1.1616, 200000: 1.1652}),
    ("arm6_v2 combab", "teacher", "new", "new",
     {40000: 1.2765, 100000: 1.2514, 200000: 1.1850}),
    ("arm5 combab", "student", "prev", "prev",
     {40000: 1.2868, 100000: 1.2456, 200000: 1.2034}),
    ("arm5 combab", "teacher", "new", "none",
     {40000: 1.2728, 100000: 1.3678}),
    ("arm4 combab", "n/a", "prev+new", "both",
     {40000: 1.2748, 100000: 1.3219}),
    ("arm6_v2 ncpc", "student", "prev", "prev",
     {40000: 1.3623, 100000: 1.2978, 200000: 1.3011}),
    ("arm6_v2 ncpc", "teacher", "new", "new",
     {40000: 1.3159, 100000: 1.3012, 200000: 1.3325}),
    ("arm6_v2 nse", "teacher", "new", "new",
     {40000: 1.3074, 100000: 1.3368}),
    ("arm1 nse", "n/a", "prev+new", "both",
     {40000: 1.5579, 100000: 1.4548, 200000: 1.3308}),
    ("arm6_v2 nse", "student", "prev", "none",
     {40000: 1.3791, 100000: 1.3914}),
]


def main():
    out = os.path.join(EXP_DIR, "results", "union_parents.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "align", "source_report", "source_commit",
                    "top5_in", "stop", "gm_rel_mase"])
        n = 0
        for cell, align, src, top5, stops in ROWS:
            for stop in sorted(stops):
                w.writerow([cell, align, src, SOURCE_COMMIT, top5, stop,
                            f"{stops[stop]:.4f}"])
                n += 1
    print(f"[out] {out} ({n} rows)")


if __name__ == "__main__":
    main()
