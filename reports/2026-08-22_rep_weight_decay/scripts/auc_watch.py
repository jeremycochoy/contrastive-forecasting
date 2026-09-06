#!/usr/bin/env python3
"""Say whether a run lost the contrastive task, and at which step.

WHY THIS SCRIPT EXISTS. L_rep carries the negatives of this objective. At
weight 0.0 nothing pushes the representations apart, and both SIGReg terms
are already near 0.0002 by step 1,000. So the card asks one question of every
run: did it keep the contrastive task, and if not, when did it stop?

The trainer writes the answer to the `auc` column of `<run>_losses.csv` every
step: the area under the ROC curve of the contrastive task, on the training
stream. A healthy run of this cell holds 0.95 to 0.98. A run at 0.5 tells a
positive from a negative no better than a coin.

THE READING. One step is noisy, so the verdict reads a rolling MEDIAN over
`--window` steps. The run "lost the task" at the first step where that median
falls under `--threshold` and never comes back above it. A dip that recovers
is not a loss, and the script reports the deepest one instead.

The EMA momentum sweep, `reports/2026-08-19_ema_momentum_k32/`, holds one
backbone of this cell that lost the task. It read 0.5745 at the 40,000-step
stop against 0.93 to 0.98 for every other arm, and it scored 1.5459 against
1.1782 for the same arm at another seed. So 0.55 is the default threshold.

THE WARMUP. The AUC of a fresh run starts near 0.5 and climbs, so the first
steps of every healthy run read as a loss. `--warmup N` drops every row at
step N or below, and the verdict then reads what is left. `auc_guard.sh` needs
this: without it the gate would stop every arm in its first minute. The default
is 0, which reads the whole run.

THE SKIP. One arm can hold rows from more than one leg: a re-fired leg appends
to the CSV of the leg that crashed. `--skip-rows N` drops the first N rows in
FILE order, so the verdict reads the rows of one leg alone. `auc_guard.sh`
counts N before the leg starts.

TWO USES. The report reads the verdict of every arm. `auc_guard.sh` reads the
exit code and stops an arm that has nothing left to train:

    exit 0   the run holds the task
    exit 1   the run lost it. Line 1 of stdout names the step
    exit 2   the CSV is missing, empty, or holds no `auc` column, or the
             warmup left it with no row to read

Usage:
  auc_watch.py <losses.csv> [--window 500] [--threshold 0.55] [--warmup 0]
  auc_watch.py results/*_losses.csv --tsv > results/auc_verdicts.tsv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

# The one backbone of this cell that lost the task fell under this value and
# stayed there. See the docstring.
DEFAULT_THRESHOLD = 0.55
# Wide enough that one bad batch cannot fire the verdict, short enough that a
# 10,000-step ramp holds twenty windows.
DEFAULT_WINDOW = 500


def read_auc(path, warmup=0, skip_rows=0):
    """`[(step, auc), ...]` for one run, in file order.

    Three kinds of row are dropped:

      * a blank or unreadable `auc`. The column is blank on a run whose
        trainer wrote no diagnostic that step.
      * an `auc` that is not finite. `float("nan")` succeeds, and
        `statistics.median` of a list that holds a NaN gives an arbitrary
        element. A NaN median is under no threshold, so one NaN row would
        read a dead arm as healthy.
      * a row at step `warmup` or below, or in the first `skip_rows` rows of
        the file.
    """
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "auc" not in reader.fieldnames:
            raise ValueError(f"{path}: no `auc` column")
        out = []
        for n, row in enumerate(reader):
            if n < skip_rows:
                continue
            try:
                step, auc = int(row["step"]), float(row["auc"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(auc) and step > warmup:
                out.append((step, auc))
    if not out:
        where = []
        if warmup:
            where.append(f"above step {warmup}")
        if skip_rows:
            where.append(f"after row {skip_rows}")
        raise ValueError(f"{path}: no readable `auc` row"
                         + (" " + " and ".join(where) if where else ""))
    return out


def rolling_median(series, window):
    """`[(step, median of the last `window` values), ...]`.

    The step is the LAST step of the window, so a verdict names a step the run
    reached, not one it was heading toward.
    """
    values, out = [], []
    for step, value in series:
        values.append(value)
        if len(values) > window:
            values.pop(0)
        out.append((step, statistics.median(values)))
    return out


def verdict(series, window=DEFAULT_WINDOW, threshold=DEFAULT_THRESHOLD):
    """`{lost, step, floor, floor_step, last, n}` for one run.

    `lost` is True when the rolling median ends under the threshold. `step` is
    then the first step of the FINAL run of under-threshold windows, which is
    where the task went and did not come back. `floor` is the lowest rolling
    median anywhere, which is what a run that dipped and recovered reports.
    """
    smoothed = rolling_median(series, window)
    floor_step, floor = min(smoothed, key=lambda p: p[1])
    last_step, last = smoothed[-1]
    lost_at = None
    if last < threshold:
        lost_at = last_step
        for step, value in reversed(smoothed):
            if value >= threshold:
                break
            lost_at = step
    return {
        "lost": lost_at is not None,
        "step": lost_at,
        "floor": floor,
        "floor_step": floor_step,
        "last": last,
        "last_step": last_step,
        "n": len(series),
    }


def format_line(path, v):
    where = f"at step {v['step']}" if v["lost"] else "held"
    return (f"{Path(path).name}\t{'lost' if v['lost'] else 'held'}\t"
            f"{v['step'] if v['lost'] else '-'}\t{v['floor']:.4f}\t"
            f"{v['floor_step']}\t{v['last']:.4f}\t{v['last_step']}\t{where}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("csv", nargs="+", help="one or more <run>_losses.csv")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"rolling median width, in rows. Default {DEFAULT_WINDOW}")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"AUC floor of a healthy run. Default {DEFAULT_THRESHOLD}")
    p.add_argument("--warmup", type=int, default=0,
                   help="drop every row at this step or below. The AUC of a "
                        "fresh run starts near 0.5 and climbs. Default 0")
    p.add_argument("--skip-rows", type=int, default=0,
                   help="drop this many rows from the START of the file. A "
                        "re-fired leg appends to the CSV of the leg that "
                        "crashed. Default 0")
    p.add_argument("--tsv", action="store_true",
                   help="write a header row, for a table the report reads")
    p.add_argument("--quiet", action="store_true", help="exit code only")
    args = p.parse_args(argv)

    if args.tsv and not args.quiet:
        print("run\tverdict\tlost_at\tfloor\tfloor_step\tlast\tlast_step\tnote")
    any_lost, any_error = False, False
    for path in args.csv:
        try:
            v = verdict(read_auc(path, args.warmup, args.skip_rows),
                        args.window, args.threshold)
        except (OSError, ValueError) as e:
            any_error = True
            if not args.quiet:
                print(f"{Path(path).name}\terror\t-\t-\t-\t-\t-\t{e}")
            continue
        any_lost = any_lost or v["lost"]
        if not args.quiet:
            print(format_line(path, v))
    if any_error:
        return 2
    return 1 if any_lost else 0


if __name__ == "__main__":
    sys.exit(main())
