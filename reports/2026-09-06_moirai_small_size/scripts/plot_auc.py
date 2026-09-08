#!/usr/bin/env python3
"""The contrastive AUC of every run, against the gate that reads it.

WHY THIS FIGURE EXISTS. The card asks for the AUC of every run, because #404
saw one backbone of this cell lose the contrastive task. A run whose AUC falls
to chance trains a representation with no negatives, and its score is bad for
a reason no score table shows.

WHAT IT DRAWS. One curve for each arm, over the steps it trained. The rollout
depth k orders the curves, so it takes one hue light to dark. A run the gate
stopped takes the alarm color. The 0.55 threshold and the 1,000-step warm-up
are the gate's own, and both are drawn.

Usage:  plot_auc.py [--root <checkpoint root>] [--out plots/auc.png]
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
THRESHOLD = 0.55
WARMUP = 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--verdicts",
                   default=str(STUDY / "results" / "auc_verdicts.tsv"))
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--smooth", type=int, default=20)
    p.add_argument("--out", default=str(STUDY / "plots" / "auc.png"))
    args = p.parse_args()

    verdicts = S.read_verdicts(args.verdicts)
    items, n, drawn, any_lost = [], 0, set(), False
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    fig.patch.set_facecolor(S.SURFACE)
    ax.set_facecolor(S.SURFACE)

    ax.axhspan(0.0, THRESHOLD, color=S.LOST, alpha=0.06, zorder=0)
    ax.axhline(THRESHOLD, color=S.LOST, linewidth=1.2, linestyle="--",
               zorder=1)
    ax.annotate(f"{THRESHOLD}, the gate",
                (0.005, THRESHOLD), xycoords=("axes fraction", "data"),
                xytext=(0, -12), textcoords="offset points", fontsize=8,
                color=S.LOST)
    ax.axvline(WARMUP, color=S.GRID, linewidth=1.0, zorder=1)
    ax.annotate(f"{WARMUP:,}-step warm-up", (WARMUP, 0.02),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=S.MUTED)

    for row in S.read_arms(args.arms):
        paths = S.losses_csvs(args.root, row["arm"], row["k"])
        if not paths:
            continue
        series = S.smooth(S.read_run(paths, ["auc"], args.every)["auc"],
                          args.smooth)
        if not series:
            continue
        is_lost = S.lost(verdicts, paths)
        colour = S.LOST if is_lost else S.depth_colour(row["k"])
        ax.plot([s for s, _ in series], [v for _, v in series], color=colour,
                linewidth=1.8, zorder=3)
        items.append((series, row["arm"], colour))
        drawn.add(int(row["k"]))
        any_lost = any_lost or is_lost
        n += 1
    if not n:
        raise SystemExit(f"no losses CSV under {args.root}")

    ax.set_xlabel("backbone steps")
    ax.set_ylabel("contrastive AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("the contrastive AUC of every run of this card",
                 color=S.INK, fontsize=12, loc="left", pad=12)
    S.tidy(ax)
    for k in sorted(drawn):
        ax.plot([], [], color=S.depth_colour(k), linewidth=2.0,
                label=f"rollout depth k = {k}")
    if any_lost:
        ax.plot([], [], color=S.LOST, linewidth=2.0, label="lost the task")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, loc="lower right")
    fig.canvas.draw()
    S.label_right(ax, items)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"wrote {args.out} ({n} run(s))")


if __name__ == "__main__":
    main()
