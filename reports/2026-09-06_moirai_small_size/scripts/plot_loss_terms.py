#!/usr/bin/env python3
"""The training loss by term, for each arm of the card.

WHY THIS FIGURE EXISTS. The card asks for the loss by term. `L_rep` holds 92
to 93 percent of the total on this cell, so the total alone says almost
nothing about `L_align`, and a decay arm's whole treatment is the weight on
`L_rep`.

THE TERMS THIS CELL HAS. The cell is `cosine_similarity_batch_rep_only`, so
`L_pred` DOES NOT EXIST: the trainer writes the column and leaves every cell
of it blank (`src/loss.py`). The objective is `L_rep` plus `L_align`, and this
figure draws both, with the live `L_rep` weight under them.

Each panel shares one x axis, and each carries one curve for each arm. The
rollout depth k orders the curves, so it takes one hue light to dark.

Usage:  plot_loss_terms.py [--root <checkpoint root>] [--out plots/loss.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_style as S  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = "/home/jupyter/checkpoints_backup/cf-412"
PANELS = (("l_rep", "L_rep, the term that carries the negatives"),
          ("l_align", "L_align, on the EMA teacher"),
          ("rep_w", "the live weight on L_rep"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--verdicts",
                   default=str(STUDY / "results" / "auc_verdicts.tsv"))
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--smooth", type=int, default=20)
    p.add_argument("--out", default=str(STUDY / "plots" / "loss_terms.png"))
    args = p.parse_args()

    verdicts = S.read_verdicts(args.verdicts)
    runs = []
    for row in S.read_arms(args.arms):
        paths = S.losses_csvs(args.root, row["arm"], row["k"])
        if not paths:
            continue
        data = S.read_run(paths, [c for c, _ in PANELS], args.every)
        if not any(data.values()):
            continue
        is_lost = S.lost(verdicts, paths)
        colour = S.LOST if is_lost else S.depth_colour(row["k"])
        runs.append((row["arm"], colour, data, int(row["k"]), is_lost))
    if not runs:
        raise SystemExit(f"no losses CSV under {args.root}")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.4, 8.4), sharex=True)
    fig.patch.set_facecolor(S.SURFACE)
    for ax, (column, title) in zip(axes, PANELS):
        ax.set_facecolor(S.SURFACE)
        items = []
        for arm, colour, data, _, _ in runs:
            series = data.get(column) or []
            if column != "rep_w":
                series = S.smooth(series, args.smooth)
            if not series:
                continue
            ax.plot([s for s, _ in series], [v for _, v in series],
                    color=colour, linewidth=1.6, zorder=3)
            items.append((series, arm, colour))
        ax.set_title(title, color=S.INK, fontsize=10, loc="left", pad=8)
        S.tidy(ax)
        ax._items = items
    axes[-1].set_xlabel("backbone steps")

    for k in sorted({r[3] for r in runs}):
        axes[0].plot([], [], color=S.depth_colour(k), linewidth=2.0,
                     label=f"rollout depth k = {k}")
    if any(r[4] for r in runs):
        axes[0].plot([], [], color=S.LOST, linewidth=2.0,
                     label="lost the task")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=S.INK,
                   loc="upper right")
    fig.tight_layout()
    fig.canvas.draw()
    for ax in axes:
        S.label_right(ax, ax._items)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"wrote {args.out} ({len(runs)} run(s))")


if __name__ == "__main__":
    main()
