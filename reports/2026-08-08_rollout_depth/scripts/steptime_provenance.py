#!/usr/bin/env python3
"""#373 — separate the solo windows of a production run from the contended ones.

`steptime_from_logs.py` takes the median over every `timing:` window a run
log holds. That is wrong for one of the five runs. `CSVLogger` and the
trainer log both open in append mode, so when two identical A3 processes
shared box b, BOTH wrote their windows into one file: the log carries 272
windows for a 200-window run, and its median mixes the clone's windows with
the survivor's and the contended steps with the solo ones.

The step number separates them. Each process prints one window per
`--log-every` steps, and two processes on one file produce each step number
twice. Everything after the last duplicated step number is the survivor,
alone. That is the number the cost table wants.

This also states, per run, how much of it was contended — the provenance the
cost table was missing.

Usage:
    steptime_provenance.py --out results/steptime_solo.csv \\
        --log B5:0:RTX5090=path/to/run.log [--log ...]
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

STEP = re.compile(r"^\[\s*(\d+)\]")
TIMING = re.compile(
    r"timing:\s*data=([\d.]+)ms\s+fwd=([\d.]+)ms\s+bwd=([\d.]+)ms\s+"
    r"total=([\d.]+)ms")


def windows(path):
    """[(step, data, fwd, bwd, total)] in file order.

    The `timing:` line is printed under the step line it belongs to, so the
    most recent step number is the window's.
    """
    out, step = [], None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = STEP.match(line)
            if m:
                step = int(m.group(1))
                continue
            m = TIMING.search(line)
            if m and step is not None:
                out.append((step, *(float(g) for g in m.groups())))
    return out


def split_solo(rows):
    """(solo, contended). Solo = after the last step number written twice."""
    counts = Counter(r[0] for r in rows)
    dup = [s for s, n in counts.items() if n > 1]
    if not dup:
        return rows, []
    cut = max(dup)
    return [r for r in rows if r[0] > cut], [r for r in rows if r[0] <= cut]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", required=True,
                   metavar="CELL:K:GPU=LOG")
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=1,
                   help="leading windows to drop from the solo stretch")
    args = p.parse_args(argv)

    out_rows = []
    for spec in args.log:
        head, path = spec.split("=", 1)
        cell, ktxt, gpu = head.split(":")
        if not Path(path).is_file():
            print(f"  skip {spec}: no {path}", file=sys.stderr)
            continue
        rows = windows(path)
        if not rows:
            print(f"  skip {spec}: no timing window", file=sys.stderr)
            continue
        solo, cont = split_solo(rows)
        solo = solo[args.warmup:] if not cont else solo
        if not solo:
            print(f"  skip {spec}: no solo window", file=sys.stderr)
            continue
        med = [statistics.median(col) for col in zip(*solo)][1:]
        med_c = ([statistics.median(col) for col in zip(*cont)][1:]
                 if cont else None)
        out_rows.append({
            "cell": cell, "k": int(ktxt), "gpu": gpu,
            "windows_total": len(rows), "windows_solo": len(solo),
            "windows_contended": len(cont),
            "first_solo_step": solo[0][0],
            "data_ms": f"{med[0]:.1f}", "fwd_ms": f"{med[1]:.1f}",
            "bwd_ms": f"{med[2]:.1f}", "compute_ms": f"{med[1] + med[2]:.1f}",
            "total_ms": f"{med[3]:.1f}",
            "compute_ms_contended": (f"{med_c[1] + med_c[2]:.1f}" if med_c else ""),
        })
        note = (f"  ({len(cont)} contended window(s) dropped, "
                f"fwd+bwd {med_c[1] + med_c[2]:.1f} ms there)" if med_c else
                "  (ran alone throughout)")
        print(f"{cell:<4} k={ktxt} {gpu:<10} solo n={len(solo):4d}  "
              f"fwd+bwd={med[1] + med[2]:6.1f} ms{note}")

    if not out_rows:
        raise SystemExit("ABORT: no log produced a solo window")

    by = {(r["cell"], r["k"]): r for r in out_rows}
    print()
    for (cell, k) in sorted(by):
        if k != 0 or (cell, 3) not in by:
            continue
        c0 = float(by[(cell, 0)]["compute_ms"])
        c3 = float(by[(cell, 3)]["compute_ms"])
        same = by[(cell, 0)]["gpu"] == by[(cell, 3)]["gpu"]
        print(f"{cell}: fwd+bwd {c0:.1f} -> {c3:.1f} ms ({c3 / c0 - 1:+.0%})"
              f"{'' if same else '  [DIFFERENT GPU MODELS]'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out} ({len(out_rows)} run(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
