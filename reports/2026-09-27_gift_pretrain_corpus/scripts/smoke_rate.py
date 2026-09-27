#!/usr/bin/env python3
"""Steady-state speed of a trainer log (#419): steps per second between the
first and the last log line after --skip steps, and the mean of the
per-window step timings the trainer prints (data, forward, backward, total).

Usage: smoke_rate.py <run.log> [--skip 50]
"""

import argparse
import re

STEP = re.compile(r"^\[\s*(\d+)\].*?([\d.]+) sps")
TIMING = re.compile(r"timing: data=([\d.]+)ms\s+fwd=([\d.]+)ms\s+"
                    r"bwd=([\d.]+)ms\s+total=([\d.]+)ms")


def parse(path):
    """``[(step, timings)]`` for every log window of the run."""
    rows, step = [], None
    for line in open(path):
        m = STEP.search(line)
        if m:
            step = int(m.group(1))
        t = TIMING.search(line)
        if t and step is not None:
            rows.append((step, [float(v) for v in t.groups()]))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--skip", type=int, default=50)
    args = p.parse_args()
    rows = [r for r in parse(args.log) if r[0] > args.skip]
    means = [sum(r[1][i] for r in rows) / len(rows) for i in range(4)]
    print(f"windows={len(rows)} steps={rows[0][0]}..{rows[-1][0]} "
          f"data={means[0]:.1f}ms fwd={means[1]:.1f}ms bwd={means[2]:.1f}ms "
          f"total={means[3]:.1f}ms -> {1000 / means[3]:.2f} steps/s")


if __name__ == "__main__":
    main()
