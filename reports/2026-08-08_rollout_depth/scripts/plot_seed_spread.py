#!/usr/bin/env python3
"""#373 figure 3 — B5 trained three times, and what the three disagree about.

One cell, one recipe, one head seed, one 97-config eval. Three backbones:

    B5·s1   seed 20260520, a rented RTX 5090
    B5·s2   seed 20260521, elisa
    B5·s3   seed 20260520, elisa

The first two disagree about the depth, and the first two differ by SEED and
by MACHINE at once, so neither of them can say which it was. B5·s3 is the
third corner: it holds the seed of s1 and the machine of s2.

    s1 against s3   same seed, two machines   -> the machine
    s2 against s3   same machine, two seeds   -> the seed

The figure carries the two channels the study confounded, one each:

    marker shape   the backbone seed
    marker fill    the machine — filled is elisa, hollow is a rented box

The dashed rule is #379's published value for this cell.

Reads results/splits.csv (`all` rows) and the registry.

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
import runs as R                                       # noqa: E402
from published import PUBLISHED                        # noqa: E402

PUB = PUBLISHED["B5"]["student"][40]
KS = [0, 3]
SHAPE = {20260520: "o", 20260521: "s"}
plt.rcParams.update(cc.rc())


def load(path):
    return {r["stop"]: float(r["gm_rel_mase"])
            for r in csv.DictReader(open(path)) if r["split"] == "all"}


def arm_points(data, arm, head):
    """[(k, value)] for one backbone, over the depths it has a score at."""
    out = []
    for k in KS:
        run = R.find_run(arm, k, "depth") or R.find_run(arm, k, "control")
        if run is None:
            continue
        v = data.get(f"{run.stem}_bb40k_{head}")
        if v is not None:
            out.append((k, v))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = load(args.splits)
    col = cc.COLOUR["B5"]

    # Every B5 backbone the registry knows, minus the control that swaps in
    # a published checkpoint: that one measures the head and the eval, not a
    # training of this cell.
    arms = [a for a in R.arms_of("B5") if a != "B5·pub"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    drawn, have = 0, []
    # Alternate the value labels above and below the line, one offset per
    # backbone, so three lines that meet at k = 3 do not stack their text.
    offsets = [-20, 13, 24, -31]
    for ax, head in zip(axes, ("student", "teacher")):
        # The published rule belongs on the student panel only: group B's
        # parent reports publish one head, trained on the student encoder,
        # so there is no published teacher number to draw against.
        if head == "student":
            ax.axhline(PUB, color=cc.PARITY, linewidth=1.2,
                       linestyle=(0, (4, 3)))
        for i, arm in enumerate(arms):
            pts = arm_points(data, arm, head)
            if not pts:
                continue
            drawn += 1
            if arm not in have:
                have.append(arm)
            seed = R.arm_seed(arm)
            elisa = R.arm_where(arm) == "elisa"
            xs = [k for k, _v in pts]
            ys = [v for _k, v in pts]
            ax.plot(xs, ys, color=col, linewidth=2.2,
                    linestyle="solid" if elisa else (0, (5, 2)),
                    marker=SHAPE.get(seed, "^"), markersize=10,
                    markerfacecolor=col if elisa else "#ffffff",
                    markeredgecolor=col, markeredgewidth=2.0, zorder=3)
            off = offsets[i % len(offsets)]
            for k, y in pts:
                ax.annotate(f"{y:.4f}", (k, y), textcoords="offset points",
                            xytext=(0, off), ha="center", fontsize=8.5,
                            color=cc.INK)
        ax.set_xticks(KS)
        ax.set_xticklabels([f"k = {k}" for k in KS])
        ax.set_xlim(-0.55, 3.55)
        ax.set_title(f"{head} head", loc="left")
        ax.set_xlabel("training rollout depth")
    if not drawn:
        raise SystemExit(f"ABORT: no B5 backbone in {args.splits}")

    axes[0].set_ylabel("GM-Relative MASE, 97 configs  (lower is better)")
    axes[0].annotate(f"#379 publishes {PUB:.4f} for this cell",
                     (3.45, PUB), fontsize=8, color=cc.INK_SOFT,
                     ha="right", va="bottom",
                     bbox=dict(fc="#ffffff", ec="none", pad=0.8))

    handles = []
    for arm in have:
        seed, elisa = R.arm_seed(arm), R.arm_where(arm) == "elisa"
        handles.append(Line2D([], [], color=col, linewidth=2.2,
                              linestyle="solid" if elisa else (0, (5, 2)),
                              marker=SHAPE.get(seed, "^"), markersize=9,
                              markerfacecolor=col if elisa else "#ffffff",
                              markeredgecolor=col, markeredgewidth=2.0,
                              label=f"{arm}  seed {seed}, {R.arm_where(arm)}"))
    axes[1].legend(handles=handles, loc="upper right", fontsize=8.5)
    fig.suptitle("B5 arm4_combab_fix09 — one seed change, one machine change",
                 x=0.005, ha="left", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({drawn} line(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
