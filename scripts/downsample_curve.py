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

The HIGHEST step is always kept, so the end of a wave is never cut off. That
is what "last" means here, and it is not the same as the last row or the last
step to first appear: a resume re-logs earlier steps, so the three agree only
while the steps increase. Two committed amplitude files already carry the
re-log. Row order is the source's, so the last row of the output is whichever
kept step came last in the file — the guarantee is about the highest step
being present, not about which row ends the file.

The write is atomic (.tmp then rename), so a reader never sees a half file.

The SOURCE MUST NOT BE WRITTEN while this runs. It is read twice — once for
the steps to keep, once to copy their rows — so a file still growing gives the
second pass rows the first never saw, and they are dropped for not being in
the keep set. Collect is a post-run step, so the precondition holds; the
script checks the size and the mtime across the two passes rather than trust
it, and writes nothing when they moved.

A run worth a second look prints a block on stderr: NO-OP when the stride
removed no step it could have removed, BARELY REDUCED when nearly every row
survived, THIN when so few did that no curve is left.

Exit codes. The file is on disk for 0, 3 and 4, and for those only:

    0  written
    1  the source has no header
    2  bad arguments, --stride below 1 among them
    3  written, and NO-OP: the stride removed nothing it could have removed
    4  written, and THIN: what came out is not a curve
    5  the source could not be read
    6  the source changed while it was read
"""
import argparse
import collections
import csv
import os
import sys

DENSE_UNTIL = 500     # keep every distinct step below this
STRIDE = 200          # then every 200th distinct step
BARELY = 0.90         # kept-row fraction that counts as barely reduced
MIN_POINTS = 10       # fewer kept steps than this is not a curve, at any
                      # source length: latent drift is 14 points for a run

EXIT_OK = 0
EXIT_NO_HEADER = 1
EXIT_USAGE = 2        # what argparse exits on any other bad argument
EXIT_NO_OP = 3
EXIT_THIN = 4
EXIT_UNREADABLE = 5
EXIT_SOURCE_MOVED = 6

# The two labels the caller acts on. BARELY REDUCED is a remark, not an
# outcome: the run did what it was asked and the source was short.
EXIT_OF_LABEL = {"NO-OP": EXIT_NO_OP, "THIN": EXIT_THIN}


class SourceMoved(Exception):
    """The source changed between the two passes over it."""


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


def final_step(distinct: list[int]) -> int | None:
    """The step the guarantee protects: the highest one in the file, which is
    how far the wave got. Not `distinct[-1]`, which is the last step to first
    appear and drops below the highest as soon as a resume re-logs."""
    return max(distinct) if distinct else None


def kept_steps(distinct: list[int], dense_until: int, stride: int) -> set[int]:
    """Every distinct step below `dense_until`, then every `stride`-th of
    them, counted from the first row of the file. The highest step is always
    kept."""
    keep = {step for rank, step in enumerate(distinct, 1)
            if step < dense_until or rank % stride == 0}
    last = final_step(distinct)
    if last is not None:
        keep.add(last)
    return keep


def strideable(distinct: list[int], dense_until: int) -> list[int]:
    """The steps the stride could remove: at or above the dense window, and
    not the highest step, which is kept whatever the stride is."""
    last = final_step(distinct)
    return [s for s in distinct if s >= dense_until and s != last]


def infer_cadence(distinct: list[int]) -> int:
    """The most common gap between consecutive distinct steps — the cadence
    the writer logged at, read off the file rather than assumed."""
    gaps = collections.Counter(b - a for a, b in zip(distinct, distinct[1:])
                               if b > a)
    return gaps.most_common(1)[0][0] if gaps else 0


def fingerprint(src: str) -> tuple[int, int] | None:
    """What the source has to still be at the second pass: the same size,
    written at the same time. None when it cannot be read, which for a file
    that was there a moment ago says the same thing as a different size."""
    try:
        stat = os.stat(src)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def copy_kept_rows(reader, writer, step_col: int, keep: set[int]) -> int:
    """Copy every row of a kept step, in the source's order. Returns the
    number of rows written."""
    written = 0
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
    return written


def write_kept(src: str, dst: str, step_col: int, keep: set[int],
               before: tuple[int, int] | None) -> int:
    """Second pass: copy the header and every row of a kept step to `dst`.

    `before` is the source as the keep set saw it. A source that moved makes
    the keep set describe a file that is no longer there, so this raises
    SourceMoved on either side of the pass and leaves `dst` alone, rather than
    writing a curve whose end it cannot vouch for. The check comes first as
    well as last, so a source that went away says so instead of raising.
    """
    tmp = dst + ".tmp"
    if fingerprint(src) != before:
        raise SourceMoved(src)
    with open(src, newline="") as fin, open(tmp, "w", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(next(reader))
        written = copy_kept_rows(reader, writer, step_col, keep)
    if fingerprint(src) != before:
        os.unlink(tmp)
        raise SourceMoved(src)
    os.replace(tmp, dst)
    return written


def cadence_lines(cadence: int, stride: int) -> str:
    """What the source was logged at, and what one stride means on it.

    `infer_cadence` counts the gaps that increase, so a file whose steps only
    ever go down leaves it nothing to count and returns 0. Multiplying by it
    then prints "one point every 0 steps", which reads as a measurement.
    """
    if not cadence:
        return (f"  source cadence: none, no two distinct steps increase\n"
                f"  --stride {stride} keeps 1 distinct step in {stride}")
    return (f"  source cadence: {cadence} steps between logged rows\n"
            f"  --stride {stride} keeps 1 distinct step in {stride} = one "
            f"point every {cadence * stride} steps of this source")


def describe(src: str, label: str, distinct: list[int], keep: set[int],
             rows_in: int, rows_out: int, dense_until: int,
             stride: int) -> None:
    """One stderr block: what came out, and the settings that produced it read
    against the cadence inferred from the source."""
    dense = sum(1 for s in distinct if s < dense_until)
    print(
        f"downsample_curve: {label} on {src}\n"
        f"  kept {rows_out}/{rows_in} rows, {len(keep)}/{len(distinct)} steps\n"
        f"{cadence_lines(infer_cadence(distinct), stride)}\n"
        f"  --dense-until {dense_until} keeps {dense} steps whole",
        file=sys.stderr)


def loudness(candidates: list[int], keep: set[int], distinct: list[int],
             rows_in: int, rows_out: int) -> str:
    """Which silence this run is about to commit, "" for none.

    THIN is the other side of NO-OP: a stride counted in distinct steps can
    cut a coarsely logged source down to three points as quietly as the old
    value rule kept all of it. It fires whenever a source that HAD a curve
    comes out without one. The only source it stays quiet on is one that
    never had `MIN_POINTS` steps to begin with, because no stride can be
    blamed for that.
    """
    if not candidates:
        return ""    # nothing above the dense window; there was nothing to cut
    if all(s in keep for s in candidates):
        return "NO-OP"
    if rows_out >= BARELY * rows_in:
        return "BARELY REDUCED"
    if len(keep) < MIN_POINTS <= len(distinct):
        return "THIN"
    return ""


def report(src: str, distinct: list[int], keep: set[int], rows_in: int,
           rows_out: int, dense_until: int, stride: int) -> int:
    """Say on stderr what the run did, when what it did is worth a look, and
    give the caller the exit code that says the same thing.

    Silence is the failure this guards: the next agent applies the same call
    to the same shape of file, reads a clean log, and commits the whole thing.
    A warning that only reaches stderr is that silence at one remove — the
    caller counts exit codes, and a log holds the stderr of every run at once.
    """
    label = loudness(strideable(distinct, dense_until), keep, distinct,
                     rows_in, rows_out)
    if label:
        describe(src, label, distinct, keep, rows_in, rows_out, dense_until,
                 stride)
    return EXIT_OF_LABEL.get(label, EXIT_OK)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--stride", type=int, default=STRIDE,
                   help=f"keep every Nth distinct step (default {STRIDE})")
    p.add_argument("--dense-until", type=int, default=DENSE_UNTIL,
                   help=f"keep every step below this (default {DENSE_UNTIL})")
    args = p.parse_args(argv)
    if args.stride < 1:
        p.exit(EXIT_USAGE, "downsample_curve: --stride must be 1 or more\n")
    return args


def read_source(src: str) -> tuple[int, list[int]] | None:
    """First pass: the step column, and every row's step in file order. None
    when the file has no header, which is a file with nothing in it."""
    with open(src, newline="") as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            return None
        step_col = step_column(header)
        return step_col, read_steps(reader, step_col)


def main() -> int:
    args = parse_args()
    try:
        before = fingerprint(args.src)
        source = read_source(args.src)
    except OSError as err:
        print(f"downsample_curve: cannot read {args.src}: {err.strerror}",
              file=sys.stderr)
        return EXIT_UNREADABLE
    if source is None:
        print(f"downsample_curve: {args.src} has no header", file=sys.stderr)
        return EXIT_NO_HEADER

    step_col, steps = source
    distinct = distinct_in_order(steps)
    keep = kept_steps(distinct, args.dense_until, args.stride)
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    try:
        rows_out = write_kept(args.src, args.dst, step_col, keep, before)
    except SourceMoved:
        print(f"downsample_curve: {args.src} changed while it was read, "
              f"nothing written", file=sys.stderr)
        return EXIT_SOURCE_MOVED
    return report(args.src, distinct, keep, len(steps), rows_out,
                  args.dense_until, args.stride)


if __name__ == "__main__":
    sys.exit(main())
