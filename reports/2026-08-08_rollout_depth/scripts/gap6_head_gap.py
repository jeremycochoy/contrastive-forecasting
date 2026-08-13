#!/usr/bin/env python3
"""#373 review gap 6 — how far apart the two heads of one backbone sit.

The two heads of a cell read the same backbone file. One reads the student
encoder and one reads its EMA copy, and both train under the same recipe,
the same budget and the same head seed. So the gap between them is small
everywhere in this study, except once.

This prints every |student - teacher| gap in the grid, largest first, and the
largest reversal in a three-stop trajectory. It is the evidence behind the
report's claim that A3's bb200k student is an outlier rather than a
measurement like the others.

Usage: gap6_head_gap.py --results <results dir> [--out <tsv>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

CELLS = ["A1", "A2", "A3", "A4"] + [f"B{i}" for i in range(1, 11)]
STOPS = (40, 100, 200)


def score(res: Path, cell: str, stop: int, head: str):
    p = res / f"score_{cell}_k3_bb{stop}k_{head}.txt"
    try:
        return float(p.read_text().strip())
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = Path(a.results)

    gaps = []
    for cell in CELLS:
        for stop in STOPS:
            s, t = score(res, cell, stop, "student"), score(res, cell, stop,
                                                            "teacher")
            if s is not None and t is not None:
                gaps.append((abs(s - t), cell, stop, s, t))
    gaps.sort(reverse=True)

    lines = ["cell\tstop\tstudent\tteacher\tabs_gap\tgroup"]
    for g, cell, stop, s, t in gaps:
        lines.append(f"{cell}\tbb{stop}k\t{s:.4f}\t{t:.4f}\t{g:.4f}\t"
                     f"{'A' if cell.startswith('A') else 'B'}")

    out = "\n".join(lines)
    print(out)
    top = gaps[0]
    rest = [g for g, *_ in gaps[1:]]
    grpA = [g for g, c, *_ in gaps[1:] if c.startswith("A")]
    print(f"\nlargest gap      {top[1]} bb{top[2]}k  {top[0]:.4f}")
    print(f"next largest     {max(rest):.4f}  ({top[0] / max(rest):.1f}x smaller)")
    print(f"next in group A  {max(grpA):.4f}  ({top[0] / max(grpA):.1f}x smaller)")

    # The reversal: a three-stop trajectory that turns round at bb200k.
    print("\nthree-stop trajectories, student head:")
    worst = (0.0, "")
    for cell in CELLS:
        v = [score(res, cell, s, "student") for s in STOPS]
        if any(x is None for x in v):
            continue
        rev = v[2] - v[1]
        mono = v[0] >= v[1] >= v[2] or v[0] <= v[1] <= v[2]
        print(f"  {cell}: {v[0]:.4f} -> {v[1]:.4f} -> {v[2]:.4f}  "
              f"200k-100k {rev:+.4f}  {'monotone' if mono else 'turns round'}")
        if rev > worst[0]:
            worst = (rev, cell)
    print(f"\nlargest reversal {worst[1]} {worst[0]:+.4f}")

    if a.out:
        Path(a.out).write_text(out + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
