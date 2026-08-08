#!/usr/bin/env python3
"""#373 figure 2 — GM-Relative MASE by horizon term, k = 3 against k = 0.

This is the figure the hypothesis lives on. The rollout deficit is a
horizon effect: #327 reports short 0.976, medium 1.41, long 1.37, so a
per-step gain that compounds should land on medium and long and leave
short alone. A gain spread evenly over all three is a better model, not
the mechanism this card proposes.

Left panel: the level, one group per horizon term. Right panel: the change,
k = 3 minus k = 0 as a percentage, with the card's success criterion drawn
on it (medium+long at least 5% better, short losing less than 2%).

Reads results/splits.csv, written by split_scores.py.

Usage: plot_horizon_split.py --splits results/splits.csv \\
           --out plots/horizon_split.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402

TERMS = ["short", "medium", "long"]
plt.rcParams.update(cc.rc())


def load(path):
    """`{(cell, k, head): {term: value}}` plus the `all` value."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            # stop labels are `<cell>_k<k>_bb<N>k_<head>`
            parts = r["stop"].split("_")
            if len(parts) < 4 or not parts[1].startswith("k"):
                continue
            key = (parts[0], int(parts[1][1:]), parts[3])
            if r["split"] in ("term", "all"):
                out.setdefault(key, {})[r["name"]] = float(r["gm_rel_mase"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--head", default="student",
                   help="which head to draw; the parents publish student")
    args = p.parse_args(argv)

    data = load(args.splits)
    cells = [c for c in cc.ORDER
             if (c, 3, args.head) in data and (c, 0, args.head) in data]
    if not cells:
        raise SystemExit("ABORT: no cell has both a k=0 and a k=3 stop "
                         f"on the {args.head} head in {args.splits}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    w = 0.8 / (2 * len(cells))

    for i, cell in enumerate(cells):
        col = cc.colour(cell)
        for j, k in enumerate((0, 3)):
            vals = [data[(cell, k, args.head)].get(t) for t in TERMS]
            xs = [t_i + (2 * i + j) * w - 0.4 + w / 2
                  for t_i in range(len(TERMS))]
            axL.bar(xs, vals, width=w * 0.92, color=col,
                    alpha=1.0 if k == 3 else 0.42,
                    edgecolor=col, linewidth=0.8)

        deltas = [100.0 * (data[(cell, 3, args.head)][t]
                           / data[(cell, 0, args.head)][t] - 1.0)
                  for t in TERMS]
        axR.plot(range(len(TERMS)), deltas, marker="o", color=col,
                 linewidth=2.0, label=cc.label(cell))

    axL.axhline(1.0, color=cc.PARITY, linewidth=1.2, zorder=0)
    axL.set_xticks(range(len(TERMS)))
    axL.set_xticklabels([f"{t}\n({n} configs)"
                         for t, n in zip(TERMS, (55, 21, 21))])
    axL.set_ylabel("GM-Relative MASE  (lower is better)")
    axL.set_title(f"Level, {args.head} head")
    axL.legend(handles=[Patch(facecolor=cc.INK_SOFT, alpha=0.42, label="k = 0"),
                        Patch(facecolor=cc.INK_SOFT, label="k = 3")],
               loc="upper left", frameon=False, fontsize=9)

    axR.axhline(0.0, color=cc.INK_SOFT, linewidth=1.0)
    axR.axhspan(-100, -5, color="#2ca02c", alpha=0.08, zorder=0)
    axR.axhline(-5.0, color="#2ca02c", linewidth=1.0, linestyle=(0, (4, 3)))
    axR.axhline(2.0, color="#d62728", linewidth=1.0, linestyle=(0, (4, 3)))
    axR.set_xticks(range(len(TERMS)))
    axR.set_xticklabels(TERMS)
    axR.set_ylabel("k = 3 against k = 0  (%, negative is better)")
    axR.set_title("Change, with the card's criterion")
    axR.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(cells)} cell(s), {args.head} head)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
