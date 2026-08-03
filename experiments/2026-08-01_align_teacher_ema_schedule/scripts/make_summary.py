#!/usr/bin/env python3
"""Per-run early / late / slope summary of the adjacent-pair drift curve.

Same three statistics as PR #387's table, one row per (run, latent):
early = mean drift_cos over 10k-25k (the post-hoc probe's first usable
interval is 5k->10k, so it starts one point later than the in-training
curve); late = mean over 80k-100k; slope = least-squares fit of
drift_cos against log10(step) over the full 10k-100k range.

Reads results/drift.csv, writes results/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict


def slope(steps, values):
    """Least-squares slope of `values` against log10(steps)."""
    xs = [math.log10(s) for s in steps]
    mx = sum(xs) / len(xs)
    my = sum(values) / len(values)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


def mean_window(points, lo, hi):
    vals = [v for s, v in points if lo <= s <= hi]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    args = ap.parse_args()
    res = os.path.join(args.exp_dir, "results")

    series, meta, order = defaultdict(list), {}, []
    with open(os.path.join(res, "drift.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["kind"] != "adjacent":
                continue
            key = (r["run"], r["latent"])
            if key not in meta:
                meta[key] = (r["arm"], r["alpha"])
                order.append(key)
            series[key].append((int(r["step"]), float(r["drift_cos"])))
    for key in series:
        series[key].sort()

    out = os.path.join(res, "summary.csv")
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["run", "arm", "alpha", "latent", "early_10k_25k",
                     "late_80k_100k", "slope_per_decade"])
        for key in order:
            run, latent = key
            arm, alpha = meta[key]
            pts = series[key]
            wr.writerow([
                run, arm, alpha, latent,
                f"{mean_window(pts, 10_000, 25_000):.4f}",
                f"{mean_window(pts, 80_000, 100_000):.4f}",
                f"{slope([p[0] for p in pts], [p[1] for p in pts]):+.4f}",
            ])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
