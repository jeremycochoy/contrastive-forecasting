#!/usr/bin/env python3
"""#373 — the depth-0 diagnostic, as a number rather than as a figure.

`cos_err_depth.png` draws it and the report reads a sign off the drawing:
does the `k = 3` run's own depth-0 error sit BELOW its `k = 0` run's, so
depth 0 got better rather than paying for the deeper depths?

A sign read off two smoothed lines is not auditable. This writes the gap

    (k = 3 run's `cos_err_d0`) − (k = 0 run's `1 − ff`)

over four end-of-run windows. Both sides are the same quantity: a `k = 3`
run's `1 − ff` and its `cos_err_d0` agree to every printed digit, and a
`k = 0` run writes no `cos_err_d*` column at all.

Four windows and not one, because the window is a choice and an arm whose
sign depends on it has no sign. `stable` is the column that says so, and it
is the column a claim about the diagnostic has to survive.

Usage: depth0_gap.py --out results/depth0_gap.csv \\
           --run <arm>:<k>=<losses.csv> [--run ...]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as R                                       # noqa: E402
from losses_csv import read_by_step, series             # noqa: E402

# Fractions of the run's logged steps, counted back from the end. The last
# is small enough to land on the final step alone.
WINDOWS = [("last_50pct", 0.50), ("last_25pct", 0.25),
           ("last_10pct", 0.10), ("final_step", 0.0)]


def tail(path, col, frac):
    """The mean of `col` over the last `frac` of the run, or None."""
    _xs, ys = series(read_by_step(path, [col]), col)
    if not ys:
        return None
    return sum(ys[-max(1, int(len(ys) * frac)):]) / max(1, int(len(ys) * frac))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   metavar="ARM:K=CSV")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    paths = {}
    for spec in args.run:
        key, path = spec.split("=", 1)
        arm, k = key.rsplit(":", 1)
        paths[(arm, int(k))] = path

    rows = []
    for arm in R.ARM_ORDER:
        base = paths.get((arm, 0))
        if base is None:
            continue
        for (a, k), deep in sorted(paths.items()):
            if a != arm or k == 0:
                continue
            gaps = {}
            for name, frac in WINDOWS:
                b0 = tail(base, "ff", frac)
                d0 = tail(deep, "cos_err_d0", frac)
                gaps[name] = None if b0 is None or d0 is None else d0 - (1 - b0)
            got = [v for v in gaps.values() if v is not None]
            if not got:
                continue
            stable = ("—" if len(got) < len(WINDOWS) else
                      "yes" if all(v < 0 for v in got) or all(v > 0 for v in got)
                      else "NO")
            rows.append({
                "arm": arm, "k": k,
                "retracted": "yes" if arm in R.RETRACTED else "",
                "k0_depth0_err": f"{1 - tail(base, 'ff', 0.25):.4f}",
                "deep_depth0_err": f"{tail(deep, 'cos_err_d0', 0.25):.4f}",
                **{n: ("" if gaps[n] is None else f"{gaps[n]:+.4f}")
                   for n, _f in WINDOWS},
                "sign_stable_across_windows": stable,
            })

    if not rows:
        print("no arm has both a k = 0 and a deeper run — no depth-0 gap table")
        return 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    wide = max(len(r["arm"]) for r in rows)
    print(f"{'arm':<{wide}} {'k':>2} " +
          " ".join(f"{n:>11}" for n, _f in WINDOWS) + "  stable")
    for r in rows:
        print(f"{r['arm']:<{wide}} {r['k']:>2} " +
              " ".join(f"{r[n]:>11}" for n, _f in WINDOWS) +
              f"  {r['sign_stable_across_windows']}")
    unstable = [r["arm"] for r in rows
                if r["sign_stable_across_windows"] == "NO"]
    if unstable:
        print(f"NOTE: the gap changes sign with the window on "
              f"{', '.join(unstable)}. The diagnostic gives those arms no "
              f"sign; only an arm marked `yes` has one.")
    print(f"wrote {args.out} ({len(rows)} arm(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
