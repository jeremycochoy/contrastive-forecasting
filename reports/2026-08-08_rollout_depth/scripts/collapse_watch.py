#!/usr/bin/env python3
"""#373 — the collapse watch the card names, as numbers.

The card asks to watch three things while the f side of the loss carries
four times its baseline weight:

  per-depth `ff`      logged as `cos_err_dj` = `1 − cos(f^(j)_t, h_{t+1+j})`.
                      A `k = 3` run's `cos_err_d0` and its `1 − ff` agree to
                      every printed digit, so `cos_err_dj` IS `ff` at depth
                      `j`. `cos_err_depth.png` draws it.
  `u_batchtime`       on `h_t` and on `e_t`. `dim_usage_per_arm.png` draws
                      both.
  `qk_logit_maxabs`   NOT LOGGED by this trainer. No run in this study, at
                      any depth, writes that column. This script reports it
                      as absent rather than leaving the reader to assume it
                      was checked.

Collapse is the latent falling onto few directions. Every latent then points
one way, so `u_batchtime` runs toward zero WHILE `ff` runs toward 1. It is
the pair that separates a collapsed model from a model that forecasts: `ff`
near 1 on its own is what a collapse looks like and what a perfect forecast
looks like.

The table prints the end-of-run level of each watched quantity and its
lowest value over the SECOND HALF of the run. A dip that recovers is still a
dip, and an end-of-run number alone hides it. The second half, not the whole
run, because every one of these curves starts near zero at step 1 and a
minimum over the whole run only ever reports that start.

Usage: collapse_watch.py --out results/collapse_watch.csv \\
           --run <arm>:<k>=<losses.csv> [--run ...]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from losses_csv import read_by_step, series             # noqa: E402

# The card's watch list. `qk_logit_maxabs` is here on purpose: the script
# looks for it in every run and reports that it is absent.
WATCH = ["ff", "cos_err_d0", "cos_err_d1", "cos_err_d2", "cos_err_d3",
         "u_batchtime", "u_batchtime_e", "qk_logit_maxabs"]
TAIL = 0.10          # the end-of-run window, as a fraction of logged steps


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   metavar="ARM:K=CSV")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows, absent = [], set(WATCH)
    for spec in args.run:
        key, path = spec.split("=", 1)
        arm, k = key.rsplit(":", 1)
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        data = read_by_step(path, WATCH)
        for col in WATCH:
            _xs, ys = series(data, col)
            if not ys:
                continue
            absent.discard(col)
            n = max(1, int(len(ys) * TAIL))
            half = len(ys) // 2
            lo = min(ys[half:]) if ys[half:] else min(ys)
            rows.append({"arm": arm, "k": int(k), "metric": col,
                         "end_of_run": f"{sum(ys[-n:]) / n:.4f}",
                         "min_last_half": f"{lo:.4f}",
                         "min_at_step": int(_xs[ys.index(lo)]),
                         "n_points": len(ys)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["arm", "k", "metric", "end_of_run",
                                "min_last_half", "min_at_step", "n_points"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} row(s))")
    if absent:
        print(f"  not logged by any run: {', '.join(sorted(absent))}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
