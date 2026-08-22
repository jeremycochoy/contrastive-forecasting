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

ARMS_TSV = Path(__file__).resolve().parent / "arms.tsv"


def arm_rows(tsv: Path = ARMS_TSV):
    """Every row of `arms.tsv`, typed. The study's one reader of that file."""
    rows = []
    for line in tsv.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 4:
            continue
        rows.append({
            "arm": f[0],
            "alpha": float(f[1]),
            "ramp": 0 if f[3] == "-" else int(f[3]),
            "seed": f[4] if len(f) > 4 else "",
            "align_w": float(f[5]) if len(f) > 5 and f[5] != "-" else 1.0,
        })
    return rows


def arms(tsv: Path = ARMS_TSV):
    """Every arm of the card, with a label that names what makes it unique.

    THE LIST IS NOT WRITTEN HERE. A hard-coded tuple held eight names, and
    round 6 added three arms that this figure then dropped without a word: it
    printed "8 arm(s)" beside a table of eleven. `arms.tsv` is the study's one
    place for its arms, and every other script reads it.

    The label carries a field only when that field separates this arm from
    another. So `a08` stays "0.8 constant", and `w3_s08` has to name its seed
    and its L_align weight, because `s08` shares its momentum and its ramp.
    """
    rows = arm_rows(tsv)
    shape = [(r["alpha"], r["ramp"]) for r in rows]
    out = []
    for r in rows:
        if r["ramp"]:
            label = f"{r['alpha']:g} rising to 1.0 at {r['ramp'] // 1000}k"
        else:
            label = f"{r['alpha']:g} constant"
        if shape.count((r["alpha"], r["ramp"])) > 1 and r["seed"]:
            label += f", seed {r['seed']}"
        if r["align_w"] != 1.0:
            label += f", L_align x{r['align_w']:g}"
        out.append((r["arm"], label))
    return tuple(out)


ARMS = arms()

# Red belongs to a COLLAPSED arm alone, and the data decides which arm that is.
# The name is not hard-coded: three of these arms hold one momentum at three
# more seeds, and any of them can fall. `seed_report.collapsed` is the study's
# one definition, so this figure and the report cannot disagree.
#
# EVERY STABLE ARM TAKES ONE GREY. Thirteen arms in thirteen colours needed a
# thirteen-row legend in a palette that repeated blue, orange and green three
# times over, and no reader could map a curve to a row. This figure asks one
# question — did the backbone hold the contrastive task — and the answer is
# the same for the thirteen. The score figures rank the arms.
COLLAPSED_COLOUR = "#d62728"
STABLE_COLOUR = "#8a8a8a"


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
    # On the right, because the legend now sits at the lower left.
    ax.text(0.99, 0.512, "chance", fontsize=9, color="0.25",
            transform=ax.get_yaxis_transform(), ha="right")
    drawn = 0
    fell = 0
    for arm, _ in ARMS:
        steps, auc = series(sync_root, arm)
        if not steps:
            continue
        # Classified BEFORE the thinning, so the verdict reads the AUC at the
        # stop and not the AUC at the last row the thinning kept.
        collapse = arm_collapsed(auc[-1])
        steps, auc = thin(steps, auc)
        ax.plot(steps, auc,
                color=COLLAPSED_COLOUR if collapse else STABLE_COLOUR,
                linewidth=2.4 if collapse else 1.1,
                zorder=3 if collapse else 2,
                alpha=1.0 if collapse else 0.55)
        drawn += 1
        fell += 1 if collapse else 0
    ax.set_xlabel("backbone step")
    ax.set_ylabel("contrastive AUC, higher is better")
    ax.set_title("Contrastive AUC against backbone step")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    # Two rows, because the figure draws two kinds of curve.
    handles = [plt.Line2D([], [], color=STABLE_COLOUR, lw=1.6,
                          label=f"{drawn - fell} backbones that held"),
               plt.Line2D([], [], color=COLLAPSED_COLOUR, lw=2.4,
                          label=f"{fell} that fell to chance")]
    ax.legend(handles=handles, fontsize=9, loc="lower left", framealpha=0.9)
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
