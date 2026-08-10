#!/usr/bin/env python3
"""#373 figure 3 — B5 trained twice, and what the two seeds disagree about.

Same cell, same code, same recipe, same head seed, same 97-config eval. The
backbone seed is the only difference between the two lines.

The figure exists because the two seeds give opposite answers about the
depth, and the reason is visible in one look: they land on top of each other
at k = 3 and far apart at k = 0. The disagreement is a k = 0 disagreement.
The dashed rule is #379's published value for this cell, which one k = 0
sits on and the other does not.

Reads results/splits.csv (`all` rows).

Usage: plot_seed_spread.py --splits results/splits.csv \\
           --out plots/seed_spread.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
from published import PUBLISHED                        # noqa: E402

PUB = PUBLISHED["B5"]["student"][40]
SEEDS = [("seed 20260520", "B5_k{k}_bb40k_{h}", False),
         ("seed 20260521", "G5_B5_s2_k{k}_bb40k_{h}", True)]
KS = [0, 3]
plt.rcParams.update(cc.rc())


def load(path):
    return {r["stop"]: float(r["gm_rel_mase"])
            for r in csv.DictReader(open(path)) if r["split"] == "all"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = load(args.splits)
    col = cc.COLOUR["B5"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    for ax, head in zip(axes, ("student", "teacher")):
        # The published rule belongs on the student panel only: group B's
        # parent reports publish one head, trained on the student encoder,
        # so there is no published teacher number to draw against.
        if head == "student":
            ax.axhline(PUB, color=cc.PARITY, linewidth=1.2,
                       linestyle=(0, (4, 3)))
        for name, tpl, hollow in SEEDS:
            ys = [data.get(tpl.format(k=k, h=head)) for k in KS]
            if any(v is None for v in ys):
                continue
            ax.plot(KS, ys, color=col, linewidth=2.2,
                    linestyle=(0, (5, 2)) if hollow else "solid",
                    marker="o", markersize=10,
                    markerfacecolor="#ffffff" if hollow else col,
                    markeredgecolor=col, markeredgewidth=2.0,
                    label=name, zorder=3)
            for k, y in zip(KS, ys):
                ax.annotate(f"{y:.4f}", (k, y), textcoords="offset points",
                            xytext=(0, 13 if hollow else -20), ha="center",
                            fontsize=8.5, color=cc.INK)
        ax.set_xticks(KS)
        ax.set_xticklabels([f"k = {k}" for k in KS])
        ax.set_xlim(-0.55, 3.55)
        ax.set_title(f"{head} head", loc="left")
        ax.set_xlabel("training rollout depth")
    axes[0].set_ylabel("GM-Relative MASE, 97 configs  (lower is better)")
    axes[0].annotate(f"#379 publishes {PUB:.4f} for this cell",
                     (3.45, PUB), fontsize=8, color=cc.INK_SOFT,
                     ha="right", va="bottom",
                     bbox=dict(fc="#ffffff", ec="none", pad=0.8))
    axes[1].legend(loc="upper right", fontsize=9,
                   handles=[Line2D([], [], color=col, linewidth=2.2,
                                   marker="o", markersize=9,
                                   markerfacecolor=col, label=SEEDS[0][0]),
                            Line2D([], [], color=col, linewidth=2.2,
                                   linestyle=(0, (5, 2)), marker="o",
                                   markersize=9, markerfacecolor="#ffffff",
                                   markeredgewidth=2.0, label=SEEDS[1][0])])
    fig.suptitle("B5 arm4_combab_fix09, two backbone seeds", x=0.005,
                 ha="left", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
