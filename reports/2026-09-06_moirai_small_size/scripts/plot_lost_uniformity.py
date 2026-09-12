#!/usr/bin/env python3
"""The dimension usage of the stopped arms, step by step.

WHY THIS FIGURE EXISTS. The arms the guard stopped do not share one
mechanism, and that claim is a direction of travel: `k32_r200_08` spreads its
latents while `k32_r100_09_dec` collapses them. Four end-state numbers hide
the trajectory, so this figure draws it.

WHAT IT DRAWS. `u_temporal` and `u_batch` against the backbone step, one
panel per statistic, one curve per stopped arm, as the 200-row rolling mean
`results/lost_arm_terms.txt` prints. A dashed line marks the step where the
guard fired on each arm. Each arm takes its own categorical hue and a direct
label.

`k3_r100_09_mean` is the third curve. It is the pass-2 arm that moves the
reduction alone, and the guard stopped it too, so the figure asks whether a
k = 3 arm under `mean` falls the way a k = 32 arm does.

Usage:  plot_lost_uniformity.py [--root <checkpoint root>] [--out <png>]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import plot_style as S  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = "/home/jupyter/checkpoints_backup/cf-412"
# `(arm, k, colour)`. The k picks the arm's losses CSV under the root.
ARMS = (("k32_r200_08", "32", "#2a78d6"),
        ("k32_r100_09_dec", "32", "#c98a2d"),
        ("k3_r100_09_mean", "3", "#7a4fa3"))
PANELS = (("u_temporal", "u_temporal, dimension usage across time"),
          ("u_batch", "u_batch, dimension usage across the batch"))
WINDOW = 200


def guard_steps(path):
    """`{run stem: lost step}` from `results/auc_verdicts.tsv`."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("verdict") == "lost":
                out[row["run"]] = int(row["lost_at"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--verdicts",
                   default=str(STUDY / "results" / "auc_verdicts.tsv"))
    p.add_argument("--out",
                   default=str(STUDY / "plots" / "lost_uniformity.png"))
    args = p.parse_args()

    lost_at = guard_steps(args.verdicts)
    runs = []
    for arm, k, colour in ARMS:
        paths = S.losses_csvs(args.root, arm, k)
        if not paths:
            continue
        df = pd.read_csv(paths[0])
        fired = next((s for r, s in lost_at.items()
                      if Path(r).name == paths[0].name), None)
        runs.append((arm, colour, df, fired))
    if not runs:
        raise SystemExit(f"no stopped-arm losses CSV under {args.root}")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8.4, 6.0), sharex=True)
    fig.patch.set_facecolor(S.SURFACE)
    for ax, (column, title) in zip(axes, PANELS):
        ax.set_facecolor(S.SURFACE)
        for arm, colour, df, fired in runs:
            smooth = df[column].rolling(WINDOW, min_periods=1).mean()
            ax.plot(df.step, smooth, color=colour, linewidth=1.8, zorder=3)
            ax.annotate(arm, (df.step.iloc[-1], smooth.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=colour, va="center",
                        annotation_clip=False)
            if fired is not None:
                ax.axvline(fired, color=S.LOST, linewidth=1.0,
                           linestyle="--", zorder=2)
        ax.set_title(title, color=S.INK, fontsize=10, loc="left", pad=8)
        S.tidy(ax)
    # The lost-step labels sit on one line and two of them collided. Stack
    # them by the step they mark, so a later arm prints under an earlier one.
    for rank, (arm, colour, df, fired) in enumerate(
            sorted((r for r in runs if r[3] is not None), key=lambda r: r[3])):
        axes[0].annotate(f"lost, {fired:,}", (fired, 1.0),
                         xycoords=("data", "axes fraction"),
                         xytext=(3, 2 + 11 * (len(runs) - 1 - rank)),
                         textcoords="offset points", fontsize=8,
                         color=S.LOST, annotation_clip=False)
    axes[-1].set_xlabel("backbone steps")
    # NO ARM KEY. Every curve carries a direct label at its right end, and
    # the key sat over one of them. The dashed rule is the only entry left.
    axes[0].plot([], [], color=S.LOST, linewidth=1.0, linestyle="--",
                 label="lost at this step")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=S.INK,
                   loc="lower right")
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"wrote {args.out} ({len(runs)} arm(s))")


if __name__ == "__main__":
    main()
