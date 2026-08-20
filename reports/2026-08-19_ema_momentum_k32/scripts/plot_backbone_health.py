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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_report import auc_series  # noqa: E402
from seed_report import collapsed as arm_collapsed  # noqa: E402

# The arms of this card, and the momentum each one holds.
ARMS = (
    ("a08", "0.8 constant"),
    ("a09", "0.9 constant"),
    ("a095", "0.95 constant"),
    ("s08", "0.8 rising, seed 20260520"),
    ("s09", "0.9 rising"),
    ("s08b", "0.8 rising, seed 20260521"),
    ("s08c", "0.8 rising, seed 20260522"),
    ("s08d", "0.8 rising, seed 20260523"),
)

# Red belongs to a COLLAPSED arm alone, and the data decides which arm that is.
# The name is not hard-coded: three of these arms hold one momentum at three
# more seeds, and any of them can fall. `seed_report.collapsed` is the study's
# one definition, so this figure and the report cannot disagree.
#
# The stable arms take colours that no reader confuses with red. There is one
# colour per arm, so no two curves share one.
COLLAPSED_COLOUR = "#d62728"
STABLE_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf",
                  "#8c564b", "#7f7f7f")


def series(sync_root: Path, arm: str):
    """`(steps, auc)` of one arm, from its backbone losses CSV.

    `seed_report.auc_series` is the study's one reader of these curves. This
    figure and the report it sits beside then cannot read two different files
    for one arm — an arm trained on a rented box has a copy in the box's sync
    tree and a copy in the canonical tree.
    """
    return auc_series(sync_root, arm)


def thin(steps, auc, every=200):
    """Every n-th row. The CSV holds one row per step, which over-draws."""
    return steps[::every], auc[::every]


def draw(sync_root: Path, out: str):
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(400, 0.512, "chance", fontsize=9, color="0.25")
    drawn = 0
    fell = 0
    for arm, label in ARMS:
        steps, auc = series(sync_root, arm)
        if not steps:
            continue
        # Classified BEFORE the thinning, so the verdict reads the AUC at the
        # stop and not the AUC at the last row the thinning kept.
        collapse = arm_collapsed(auc[-1])
        steps, auc = thin(steps, auc)
        colour = COLLAPSED_COLOUR if collapse \
            else STABLE_COLOURS[drawn % len(STABLE_COLOURS)]
        ax.plot(steps, auc, label=label, color=colour,
                linewidth=2.4 if collapse else 1.4,
                zorder=3 if collapse else 2,
                alpha=1.0 if collapse else 0.75)
        drawn += 1
        fell += 1 if collapse else 0
    ax.set_xlabel("backbone step")
    ax.set_ylabel("contrastive AUC, higher is better")
    ax.set_title(f"{fell} of {drawn} backbones fall to chance while they train"
                 if fell != 1 else
                 "One backbone falls to chance while it trains")
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
