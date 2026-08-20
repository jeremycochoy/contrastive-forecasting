#!/usr/bin/env python3
"""The contrastive AUC of every arm against the backbone step.

The AUC says whether the backbone still tells a true future from a false one.
A value near 0.5 is chance, and a backbone at chance has learned nothing.

The figure exists because one arm fell to chance while it trained. A score
table alone cannot show that, because a collapsed backbone still produces a
score.

Usage:
  plot_backbone_health.py --sync-root /home/jupyter/cf404_sync --out plots/backbone_health.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The arms of this card, and the momentum each one holds.
ARMS = (
    ("a08", "0.8 constant"),
    ("a09", "0.9 constant"),
    ("a095", "0.95 constant"),
    ("s08", "0.8 rising, seed 20260520"),
    ("s09", "0.9 rising"),
    ("s08b", "0.8 rising, seed 20260521"),
)
COLLAPSED = "s08b"

# Red belongs to the collapsed arm alone. The stable arms take colours that
# no reader confuses with it.
STABLE_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf")


def series(sync_root: Path, arm: str):
    """`(steps, auc)` of one arm, from its backbone losses CSV."""
    hits = list(sync_root.glob(f"*/sync/{arm}/*/leg_40k/*_losses.csv"))
    if not hits:
        return [], []
    steps, auc = [], []
    with open(hits[0], newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("auc"):
                steps.append(int(r["step"]))
                auc.append(float(r["auc"]))
    return steps, auc


def thin(steps, auc, every=200):
    """Every n-th row. The CSV holds one row per step, which over-draws."""
    return steps[::every], auc[::every]


def draw(sync_root: Path, out: str):
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(400, 0.512, "chance", fontsize=9, color="0.25")
    drawn = 0
    for arm, label in ARMS:
        steps, auc = series(sync_root, arm)
        if not steps:
            continue
        steps, auc = thin(steps, auc)
        collapsed = arm == COLLAPSED
        colour = "#d62728" if collapsed \
            else STABLE_COLOURS[drawn % len(STABLE_COLOURS)]
        ax.plot(steps, auc, label=label, color=colour,
                linewidth=2.4 if collapsed else 1.4,
                zorder=3 if collapsed else 2,
                alpha=1.0 if collapsed else 0.75)
        drawn += 1
    ax.set_xlabel("backbone step")
    ax.set_ylabel("contrastive AUC, higher is better")
    ax.set_title("One backbone falls to chance while it trains")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {drawn} arm(s)")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sync-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    root = Path(args.sync_root)
    if not root.is_dir():
        raise SystemExit(f"ABORT: no sync root at {root}")
    draw(root, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
