#!/usr/bin/env python3
"""#373 figure 4 — rollout fidelity against depth.

`cos(rollout_d, h_{T0+d})` for d = 1..16 on one fixed batch, k = 3 against
k = 0. This measures the composed operator directly, with no quantile head
in the way — the thing the training objective was changed to improve.

Reads results/rollout_fidelity.csv, written by rollout_fidelity.py.

Usage: plot_rollout_fidelity.py --csv results/rollout_fidelity.csv \\
           --out plots/rollout_fidelity.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402

plt.rcParams.update(cc.rc())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    curves = defaultdict(list)
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            curves[r["run"]].append((int(r["d"]), float(r["cos"])))
    if not curves:
        raise SystemExit(f"ABORT: no curve in {args.csv}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    pairs = defaultdict(dict)
    for run, pts in curves.items():
        # run labels are `<cell>_k<k>`
        cell, _, ktxt = run.rpartition("_k")
        k = int(ktxt) if ktxt.isdigit() else 0
        pts.sort()
        xs = [d for d, _v in pts]
        ys = [v for _d, v in pts]
        axL.plot(xs, ys, color=cc.colour(cell), linestyle=cc.style(k),
                 linewidth=1.9, marker="o", markersize=3)
        pairs[cell][k] = (xs, ys)

    for cell, byk in pairs.items():
        if 0 in byk and 3 in byk:
            xs = byk[3][0]
            axR.plot(xs, [b - a for a, b in zip(byk[0][1], byk[3][1])],
                     color=cc.colour(cell), linewidth=1.9, marker="o",
                     markersize=3, label=cc.label(cell))

    axL.set_xlabel("rollout depth d (tokens)")
    axL.set_ylabel("cos(rollout$_d$, h$_{T_0+d}$)")
    axL.set_title("Fidelity of the composed forecaster")
    axL.legend(handles=[Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(0),
                               label="k = 0"),
                        Line2D([], [], color=cc.INK_SOFT, linestyle=cc.style(3),
                               label="k = 3")],
               frameon=False, fontsize=9)

    axR.axhline(0.0, color=cc.INK_SOFT, linewidth=1.0)
    axR.set_xlabel("rollout depth d (tokens)")
    axR.set_ylabel("k = 3 minus k = 0  (positive is better)")
    axR.set_title("Change")
    axR.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(curves)} curve(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
