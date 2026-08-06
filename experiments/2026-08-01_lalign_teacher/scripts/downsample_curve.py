#!/usr/bin/env python3
"""Downsample a backbone `_losses.csv` to the resolution #379 committed.

The trainer writes one row per step, ~27 MB per run, and #390 has up to 30 of
them. #379's report keeps every step up to 500 and every 200th step after
that, which is 300 KB and is exactly what the plot scripts read.

    downsample_curve.py <src.csv> <dst.csv>

The last row is always kept, so the end of a wave is never cut off. Writes
atomically (.tmp then rename), so a reader never sees a half file.
"""
import csv
import os
import sys

DENSE_UNTIL = 500     # keep every step below this
STRIDE = 200          # then every 200th


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"

    with open(src, newline="") as fin, open(tmp, "w", newline="") as fout:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            os.remove(tmp)
            return 1
        writer = csv.writer(fout)
        writer.writerow(header)
        try:
            step_col = header.index("step")
        except ValueError:
            step_col = 0
        last = None
        kept_last = False
        for row in reader:
            if not row:
                continue
            try:
                step = int(float(row[step_col]))
            except (ValueError, IndexError):
                continue
            last = row
            if step < DENSE_UNTIL or step % STRIDE == 0:
                writer.writerow(row)
                kept_last = True
            else:
                kept_last = False
        # The final row is the end of the wave — never drop it.
        if last is not None and not kept_last:
            writer.writerow(last)
    os.replace(tmp, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
