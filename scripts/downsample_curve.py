#!/usr/bin/env python3
"""Downsample a per-step artefact CSV to the resolution a report commits.

The trainer writes `<run>_losses.csv` one row per step (~27 MB per run) and
`<run>_attn_amplitude.csv` nine rows every 200 steps. A report keeps a curve,
not a log: every distinct step below `--dense-until`, then every `--stride`-th
distinct step after it.

    downsample_curve.py <src.csv> <dst.csv> [--stride N] [--dense-until STEP]

The stride counts DISTINCT STEPS, not step values. A rule written on the step
value (`step % 200 == 0`) removes nothing from a source already logged every
200 steps, and removes nothing silently: that is how the 2026-08-04 report
committed 18.6 MiB of raw attention amplitude. Counting distinct steps makes
the reduction the same whatever cadence the writer used.

All rows of a kept step are kept. One amplitude step is nine rows, one per
layer and block, and a step that arrives with some of them reads as a layer
that stopped being logged.

The last step is always kept, so the end of a wave is never cut off. The write
is atomic (.tmp then rename), so a reader never sees a half file.

A run worth a second look prints a block on stderr: NO-OP when the stride
removed no step it could have removed, BARELY REDUCED when nearly every row
survived, THIN when so few did that no curve is left.

Exit codes: 0 written; 1 the source has no header; 3 written, but NO-OP.
"""
import argparse
import collections
import csv
import os
import sys

DENSE_UNTIL = 500     # keep every distinct step below this
STRIDE = 200          # then every 200th distinct step
BARELY = 0.90         # kept-row fraction that counts as barely reduced
MIN_POINTS = 10       # fewer kept steps than this is not a curve


def step_column(header: list[str]) -> int:
    """Index of the step column. Column 0 when the header does not name one."""
    try:
        return header.index("step")
    except ValueError:
        return 0


def read_steps(reader, step_col: int) -> list[int]:
    """Every row's step, in file order. Rows with no readable step are
    dropped, as they carry no place on the curve."""
    steps = []
    for row in reader:
        if not row:
            continue
        try:
            steps.append(int(float(row[step_col])))
        except (ValueError, IndexError):
            continue
    return steps


def distinct_in_order(steps: list[int]) -> list[int]:
    """The distinct steps, ordered by where each first appears."""
    return list(dict.fromkeys(steps))


def kept_steps(distinct: list[int], dense_until: int, stride: int) -> set[int]:
    """Every distinct step below `dense_until`, then every `stride`-th of
    them, counted from the first row of the file. The last step is always
    kept."""
    keep = {step for rank, step in enumerate(distinct, 1)
            if step < dense_until or rank % stride == 0}
    if distinct:
        keep.add(distinct[-1])
    return keep


def strideable(distinct: list[int], dense_until: int) -> list[int]:
    """The steps the stride could remove: at or above the dense window, and
    not the final step, which is kept whatever the stride is."""
    return [s for s in distinct[:-1] if s >= dense_until]


def infer_cadence(distinct: list[int]) -> int:
    """The most common gap between consecutive distinct steps — the cadence
    the writer logged at, read off the file rather than assumed."""
    gaps = collections.Counter(b - a for a, b in zip(distinct, distinct[1:])
                               if b > a)
    return gaps.most_common(1)[0][0] if gaps else 0


def write_kept(src: str, dst: str, step_col: int, keep: set[int]) -> int:
    """Copy the header and every row of a kept step, in the source's order.
    Returns the number of rows written."""
    tmp = dst + ".tmp"
    written = 0
    with open(src, newline="") as fin, open(tmp, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(next(reader))
        for row in reader:
            if not row:
                continue
            try:
                step = int(float(row[step_col]))
            except (ValueError, IndexError):
                continue
            if step in keep:
                writer.writerow(row)
                written += 1
    os.replace(tmp, dst)
    return written


def describe(src: str, label: str, distinct: list[int], keep: set[int],
             rows_in: int, rows_out: int, dense_until: int,
             stride: int) -> None:
    """One stderr block: what came out, and the settings that produced it read
    against the cadence inferred from the source."""
    cadence = infer_cadence(distinct)
    dense = sum(1 for s in distinct if s < dense_until)
    print(
        f"downsample_curve: {label} on {src}\n"
        f"  kept {rows_out}/{rows_in} rows, {len(keep)}/{len(distinct)} steps\n"
        f"  source cadence: {cadence} steps between logged rows\n"
        f"  --stride {stride} keeps 1 distinct step in {stride} = one point "
        f"every {cadence * stride} steps of this source\n"
        f"  --dense-until {dense_until} keeps {dense} steps whole",
        file=sys.stderr)


def loudness(candidates: list[int], keep: set[int], distinct: list[int],
             rows_in: int, rows_out: int) -> str:
    """Which silence this run is about to commit, "" for none.

    THIN is the other side of NO-OP: a stride counted in distinct steps can
    cut a coarsely logged source down to three points as quietly as the old
    value rule kept all of it.
    """
    if not candidates:
        return ""    # nothing above the dense window; there was nothing to cut
    if all(s in keep for s in candidates):
        return "NO-OP"
    if rows_out >= BARELY * rows_in:
        return "BARELY REDUCED"
    if len(keep) < MIN_POINTS <= len(distinct) / 4:
        return "THIN"
    return ""


def report(src: str, distinct: list[int], keep: set[int], rows_in: int,
           rows_out: int, dense_until: int, stride: int) -> int:
    """Say on stderr what the run did, when what it did is worth a look.

    Silence is the failure this guards: the next agent applies the same call
    to the same shape of file, reads a clean log, and commits the whole thing.
    """
    label = loudness(strideable(distinct, dense_until), keep, distinct,
                     rows_in, rows_out)
    if label:
        describe(src, label, distinct, keep, rows_in, rows_out, dense_until,
                 stride)
    return 3 if label == "NO-OP" else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--stride", type=int, default=STRIDE,
                   help=f"keep every Nth distinct step (default {STRIDE})")
    p.add_argument("--dense-until", type=int, default=DENSE_UNTIL,
                   help=f"keep every step below this (default {DENSE_UNTIL})")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        sys.exit("downsample_curve: --stride must be 1 or more")
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)

    with open(args.src, newline="") as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            print(f"downsample_curve: {args.src} has no header", file=sys.stderr)
            return 1
        step_col = step_column(header)
        steps = read_steps(reader, step_col)

    distinct = distinct_in_order(steps)
    keep = kept_steps(distinct, args.dense_until, args.stride)
    rows_out = write_kept(args.src, args.dst, step_col, keep)
    return report(args.src, distinct, keep, len(steps), rows_out,
                  args.dense_until, args.stride)


if __name__ == "__main__":
    sys.exit(main())
