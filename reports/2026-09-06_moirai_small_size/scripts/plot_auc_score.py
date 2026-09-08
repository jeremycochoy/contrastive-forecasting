#!/usr/bin/env python3
"""The score of every scored leg against its contrastive AUC at the stop.

WHY THIS FIGURE EXISTS. A low AUC predicts a bad score, and a high AUC does
not order the good ones. That claim needs the joint picture: one point per
scored leg, the guard's AUC at the stop on x, GM-Relative MASE on y, and the
two stopped arms on the top axis line, at their last AUC, with no score.
The top line is the high-MASE edge, so a stopped arm can not read as a good
score.

Sources: `results/scores.csv` joined to `results/auc_verdicts.tsv` on the
run name and the stop.

Usage:  plot_auc_score.py [--out plots/auc_score.png]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import plot_style as S  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def read_verdicts(path):
    """`{(run stem, last step): (last AUC, verdict)}`."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[(Path(row["run"]).name, int(row["last_step"]))] = (
                float(row["last"]), row["verdict"])
    return out


def stack(ax, points, fontsize=8, pad=1.3):
    """Direct labels left of each point, stacked in pixel space."""
    dpi = ax.figure.dpi
    gap = fontsize * pad * dpi / 72.0
    rows = sorted(((ax.transData.transform((x, y))[1], x, y, t, c)
                   for x, y, t, c in points), key=lambda r: r[0])
    placed = []
    for y, *_ in rows:
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
    for (y0, x, y, text, colour), yy in zip(rows, placed):
        ax.annotate(text, (x, y), xytext=(-9, (yy - y0) * 72.0 / dpi),
                    textcoords="offset points", fontsize=fontsize,
                    color=colour, va="center", ha="right")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(STUDY / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--verdicts",
                   default=str(STUDY / "results" / "auc_verdicts.tsv"))
    p.add_argument("--out", default=str(STUDY / "plots" / "auc_score.png"))
    args = p.parse_args()

    arms = {r["arm"]: r for r in S.read_arms(args.arms)}
    verdicts = read_verdicts(args.verdicts)
    scored = S.read_scores(args.scores)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fig.patch.set_facecolor(S.SURFACE)
    ax.set_facecolor(S.SURFACE)

    labels = []
    for (arm, stop), score in sorted(scored.items()):
        name = f"{S.run_name(arm, arms[arm]['k'])}_losses.csv"
        hit = verdicts.get((name, stop))
        if hit is None:
            continue
        colour = S.depth_colour(arms[arm]["k"])
        ax.plot(hit[0], score, "o", markersize=8, color=colour, zorder=3)
        text = arm if stop == S.STOP else f"{arm} at {stop // 1000}k"
        labels.append((hit[0], score, text, S.INK))

    floor = [((Path(run).name, step), auc)
             for (run, step), (auc, verdict) in verdicts.items()
             if verdict == "lost"]
    for (run, step), auc in floor:
        ax.plot(auc, 1.0, "x", markersize=9, markeredgewidth=2.2,
                color=S.LOST, clip_on=False, zorder=4,
                transform=ax.get_xaxis_transform())
    if floor:
        ax.annotate("stopped by the guard: no score",
                    (min(a for _, a in floor), 0.97),
                    xycoords=("data", "axes fraction"), xytext=(0, -8),
                    textcoords="offset points", fontsize=8, color=S.LOST,
                    va="top")

    ax.set_xlabel("contrastive AUC at the stop, the guard's 500-row median")
    ax.set_ylabel("GM-Relative MASE — lower is better")
    ax.set_xlim(0.45, 1.03)
    ax.set_title("the score against the AUC, one point per scored leg",
                 color=S.INK, fontsize=12, loc="left", pad=12)
    S.tidy(ax)
    handles = [Line2D([], [], color=S.depth_colour(k), marker="o",
                      linestyle="none", markersize=8,
                      label=f"rollout depth k = {k}")
               for k in sorted({int(arms[a]["k"]) for a, _ in scored})]
    handles.append(Line2D([], [], color=S.LOST, marker="x", linestyle="none",
                          markersize=9, markeredgewidth=2.2,
                          label="stopped by the guard"))
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=S.INK,
              loc="lower left")
    fig.canvas.draw()
    stack(ax, labels)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"wrote {args.out} ({len(labels)} scored leg(s), "
          f"{len(floor)} stopped arm(s))")


if __name__ == "__main__":
    main()
