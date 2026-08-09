#!/usr/bin/env python3
"""#373 — step time of the real 40k runs, from their own trainer logs.

`steptime.sh` measured the overhead on a 600-step probe, on a 4090 shared
with another session. The production runs measure it again, over tens of
thousands of steps, each on a card of its own — so this is the number to
carry, and the probe is the cross-check.

train.py prints `timing: data=… fwd=… bwd=… total=…` every `--log-every`
steps, averaged over that window. `data` is the input pipeline: it is
network-bound on a rented box and swings by an order of magnitude between
windows, so the depth overhead is read off `fwd + bwd`, and `data` is
reported beside it rather than folded in.

The first window of a run is warm-up and is dropped.

Usage: steptime_from_logs.py --out results/steptime_runs.csv \\
           --log <cell>:<k>:<gpu>=<run log> [--log ...]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

TIMING = re.compile(
    r"timing:\s*data=([\d.]+)ms\s+fwd=([\d.]+)ms\s+bwd=([\d.]+)ms\s+"
    r"total=([\d.]+)ms")


def windows(path):
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = TIMING.search(line)
            if m:
                out.append(tuple(float(g) for g in m.groups()))
    return out[1:]          # drop warm-up


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", required=True,
                   metavar="CELL:K:GPU=LOG")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = []
    for spec in args.log:
        head, path = spec.split("=", 1)
        cell, ktxt, gpu = head.split(":")
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        w = windows(path)
        if not w:
            print(f"  skip {spec}: no timing window", file=sys.stderr)
            continue
        med = [statistics.median(col) for col in zip(*w)]
        rows.append({"cell": cell, "k": int(ktxt), "gpu": gpu, "windows": len(w),
                     "data_ms": f"{med[0]:.1f}", "fwd_ms": f"{med[1]:.1f}",
                     "bwd_ms": f"{med[2]:.1f}",
                     "compute_ms": f"{med[1] + med[2]:.1f}",
                     "total_ms": f"{med[3]:.1f}"})
        print(f"{cell:<4} k={ktxt} {gpu:<10} n={len(w):4d}  "
              f"data={med[0]:6.1f}  fwd+bwd={med[1] + med[2]:6.1f}  "
              f"total={med[3]:6.1f} ms")

    if not rows:
        raise SystemExit("ABORT: no log produced a window")

    # Pair the two depths of a cell when both ran on the same GPU model.
    by = {(r["cell"], r["k"], r["gpu"]): r for r in rows}
    print()
    for cell, k, gpu in sorted(by):
        if k != 0:
            continue
        b = by.get((cell, 3, gpu))
        if not b:
            continue
        a = by[(cell, 0, gpu)]
        c0, c3 = float(a["compute_ms"]), float(b["compute_ms"])
        print(f"{cell} on {gpu}: fwd+bwd {c0:.1f} -> {c3:.1f} ms "
              f"({c3 / c0 - 1:+.0%})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cell", "k", "gpu", "windows",
                                           "data_ms", "fwd_ms", "bwd_ms",
                                           "compute_ms", "total_ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} run(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
