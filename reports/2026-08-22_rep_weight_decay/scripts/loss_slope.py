#!/usr/bin/env python3
"""Does any arm still learn at the 40,000-step stop?

WHY THIS SCRIPT EXISTS. The card's second question asks for "a backbone that
can improve more with longer training". No arm of this card measures a score
past 40,000 steps, so the only evidence the card holds is the slope of a term
at the stop. `results/loss_terms_at_stop.csv` gives that slope as one number,
the change between two 1,000-step windows. One number cannot say whether it is
a trend or one spike.

WHAT THIS MEASURES. Over a window, it cuts the raw series into 1,000-step
blocks, takes the mean of each block, and fits a line to those block means.

  slope     the change per 10,000 steps. Negative is a term still falling,
            which is headroom.
  scatter   the standard deviation of the block means around that line. Two
            arms closer than this are not ranked by it.

Blocks, not raw steps: one step of this trainer is one batch, so the raw
column is noisy and each step is correlated with the last. A fit on raw steps
would report a standard error far smaller than the run-to-run spread.

TWO WINDOWS, ON PURPOSE. Steps 30,000 to 40,000 is the window
`notes/SECOND_ANSWER.md` names. Steps 20,000 to 40,000 doubles it, so a reader
sees which arms hold their sign when the window grows.

Usage:
  loss_slope.py --root /home/jupyter/checkpoints_backup/cf-409 \
      --arms scripts/arms.tsv --out results/loss_slope.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

DEPTH = 32
STOP = 40000
BLOCK = 1000
WINDOWS = [(30000, STOP), (20000, STOP)]
COLUMNS = ["loss"] + [f"cos_err_d{j}" for j in range(DEPTH + 1)]


def block_means(series, lo, hi, width=BLOCK):
    """The mean of each `width`-step block of `series` inside `(lo, hi]`."""
    out = []
    for start in range(lo, hi, width):
        values = [v for s, v in series if start < s <= start + width]
        if values:
            out.append((start + width, sum(values) / len(values)))
    return out


def fit(points):
    """The slope per 10,000 steps, and the scatter around the fit."""
    if len(points) < 4:
        return None, None
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx == 0:
        return None, None
    slope = sum((p[0] - mx) * (p[1] - my) for p in points) / sxx
    intercept = my - slope * mx
    residuals = [p[1] - (intercept + slope * p[0]) for p in points]
    return slope * 10000, statistics.pstdev(residuals)


def mean_cos_err(run, depth=DEPTH):
    by_step = {}
    for j in range(depth + 1):
        for step, value in run.get(f"cos_err_d{j}") or []:
            by_step.setdefault(step, []).append(value)
    return [(s, sum(v) / len(v)) for s, v in sorted(by_step.items())
            if len(v) == depth + 1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="/home/jupyter/checkpoints_backup/cf-409")
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out",
                   default=str(HERE.parent / "results" / "loss_slope.csv"))
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    paths = S.study_paths(args.root, arms)
    rows = []
    for row in arms:
        files = paths.get(row["arm"], [])
        if not files:
            continue
        run = S.read_run(files, COLUMNS, 1)
        terms = {"loss": run.get("loss") or [], "cos_err": mean_cos_err(run)}
        del run
        reached = max((s[-1][0] for s in terms.values() if s), default=0)
        if reached < STOP:
            # A slope over steps the arm never ran is a made-up number.
            print(f"skipped {row['arm']}: reached step {reached}, "
                  f"short of the {STOP}-step stop")
            continue
        for term, series in terms.items():
            for lo, hi in WINDOWS:
                slope, scatter = fit(block_means(series, lo, hi))
                if slope is None:
                    continue
                rows.append({
                    "arm": row["arm"], "ema": S.schedule_label(row),
                    "ema_at_stop": f"{S.momentum_at(row):.3f}",
                    "seed": row["seed"], "term": term,
                    "window": f"{lo}-{hi}", "blocks": BLOCK,
                    "slope_per_10k": f"{slope:+.4f}",
                    "scatter": f"{scatter:.4f}",
                })
    if not rows:
        print(f"no arm reached step {STOP} under {args.root}", file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{args.out}: {len(rows)} row(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
