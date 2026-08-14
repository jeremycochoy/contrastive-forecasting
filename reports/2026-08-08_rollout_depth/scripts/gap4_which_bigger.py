#!/usr/bin/env python3
"""#373 review item 3 — is the depth segment bigger than the weight segment?

The review set a one-sided test: if the x4 re-weight alone reproduces MOST of
B1's -0.1175, the win is the weight; if not, the win is the depth. The report
publishes both segments with their own intervals, but not the interval on the
DIFFERENCE between them, which is what that test actually asks for.

    weight segment  w  - k0
    depth segment   k3 - w
    difference      (k3 - w) - (w - k0)  =  k3 - 2w + k0

Negative means the depth segment is the larger of the two. An interval that
covers zero means this cell cannot rank them, whatever the point estimate
says. Both segments come off the same 97 configs and one shared draw, so the
pairing survives into the difference.

Same resampling unit as every other interval in this study: a draw takes
DATASETS with replacement and every config of each drawn dataset.

Usage: gap4_which_bigger.py [--iters 10000]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap4_2x2 as G                                   # noqa: E402

ARMS = ("k0", "w", "k3")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--out", default=str(G.EXP / "results"
                                        / "gap4_which_bigger.csv"))
    args = p.parse_args(argv)

    sn = G.read_mase(G.SN_REF)
    rows = []

    for head in ("student", "teacher"):
        cells = {a: G.CORNERS[head][a] for a in ARMS}
        arms = {a: G.read_mase(G.EVAL / c / "all_results.csv")
                for a, c in cells.items()}
        common = set(sn)
        for v in arms.values():
            common &= set(v)
        common = sorted(common)
        if len(common) != 97:
            raise SystemExit(f"ABORT: {head} shares {len(common)} configs")

        lr = {a: {ds: math.log(v[ds] / sn[ds]) for ds in common}
              for a, v in arms.items()}
        clusters = {}
        for ds in common:
            clusters.setdefault(ds.rsplit("/", 1)[0], []).append(ds)
        keys = sorted(clusters)

        g = {a: G.gm([lr[a][d] for d in common]) for a in ARMS}
        weight = g["w"] - g["k0"]
        depth = g["k3"] - g["w"]
        diff = depth - weight

        draws = []
        rng = random.Random(args.seed)
        for _ in range(args.iters):
            pick = [clusters[keys[rng.randrange(len(keys))]]
                    for _ in range(len(keys))]
            sel = [d for grp in pick for d in grp]
            gb = {a: G.gm([lr[a][d] for d in sel]) for a in ARMS}
            draws.append((gb["k3"] - gb["w"]) - (gb["w"] - gb["k0"]))
        draws.sort()
        lo = draws[int(0.025 * len(draws))]
        hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
        share = sum(1 for x in draws if x < 0) / len(draws)
        covers = lo <= 0.0 <= hi

        print(f"== {head} ==")
        print(f"  weight segment {weight:+.4f}   depth segment {depth:+.4f}")
        print(f"  depth minus weight {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"depth is larger in {share * 100:.1f}% of resamples")
        print("  this cell CANNOT rank the two segments"
              if covers else
              "  the depth segment is the larger of the two")
        rows.append({"head": head, "weight_segment": f"{weight:+.4f}",
                     "depth_segment": f"{depth:+.4f}",
                     "depth_minus_weight": f"{diff:+.4f}",
                     "ci_lo": f"{lo:+.4f}", "ci_hi": f"{hi:+.4f}",
                     "p_depth_larger": f"{share:.3f}",
                     "ranks": "no" if covers else "yes"})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
