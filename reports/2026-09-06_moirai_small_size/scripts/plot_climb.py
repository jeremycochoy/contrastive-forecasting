#!/usr/bin/env python3
"""The climb: one arm at 11.4M against the same arm at 1.1M, stop by stop.

WHY THIS FIGURE EXISTS. The card asks whether the objective improves with
capacity. One stop cannot answer that, because the two sizes need not improve
at the same rate. This figure draws the same configuration at both sizes over
the stops both reached, so a reader sees the size gap AND how it moves.

WHAT IT DRAWS. The 11.4M track takes the series color. Its 1.1M twin takes the
muted reference ink. A stop where the two head budgets differ carries a hollow
marker, because that pair is not head-matched. The band rides on each 11.4M
point: this card's own two seeds where it measured them, else #409's floor,
whichever is wider.

Lower is better, so a line that falls is a model that improves.

Usage:  plot_climb.py [--arm k3_r100_09] [--out plots/climb.png]
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
HEAD_STEPS = 30000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(STUDY / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--arm", action="append", default=None,
                   help="repeatable; every arm with two or more stops by default")
    p.add_argument("--out", default=str(STUDY / "plots" / "climb.png"))
    args = p.parse_args()

    arms = {r["arm"]: r for r in S.read_arms(args.arms)}
    rows = S.read_scores(args.scores)
    tracks = {}
    for arm in {a for a, _ in rows}:
        track = sorted((s, v) for (a, s), v in rows.items() if a == arm)
        if track:
            tracks[arm] = track
    if args.arm:
        tracks = {a: t for a, t in tracks.items() if a in args.arm}
    else:
        climbed = {a: t for a, t in tracks.items() if len(t) > 1}
        tracks = climbed or tracks
    if not tracks:
        raise SystemExit("no arm holds a score")

    band, _ = S.effective_band(S.read_scores(args.scores, stop=S.STOP))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.patch.set_facecolor(S.SURFACE)
    ax.set_facecolor(S.SURFACE)

    ax.axhline(S.PROJECT_BEST, color=S.REFERENCE, linewidth=1.2,
               linestyle="--", zorder=1)
    ax.annotate(f"{S.PROJECT_BEST:.4f}  the project best "
                f"({S.PROJECT_BEST_LABEL})",
                (0.005, S.PROJECT_BEST), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8,
                color=S.MUTED)

    unmatched = False
    for arm, track in sorted(tracks.items()):
        colour = S.depth_colour(arms[arm]["k"]) if arm in arms else S.SERIES
        # The bar rides the 11.4M track only, because the band is an 11.4M
        # measurement. The legend entry below names its size and its source.
        ax.errorbar([s for s, _ in track], [v for _, v in track],
                    yerr=band / 2, color=colour, linewidth=2.2, marker="o",
                    markersize=8, capsize=3, elinewidth=1.2, zorder=4,
                    label=f"{arm}, 11.4M, 30,000-step head")
        for s, v in track:
            ax.annotate(f"{v:.4f}", (s, v), xytext=(8, -12),
                        textcoords="offset points", ha="left", fontsize=8,
                        color=S.INK)

        ref = [(s, S.reference(arm, s)) for s, _ in track]
        ref = [(s, r) for s, r in ref if r]
        if not ref:
            continue
        ax.plot([s for s, _ in ref], [r[0] for _, r in ref],
                color=S.REFERENCE, linewidth=1.8, alpha=0.8, zorder=3,
                label=f"{arm}, 1.1M, the parent study")
        for s, r in ref:
            matched = r[1] == HEAD_STEPS
            unmatched = unmatched or not matched
            ax.plot([s], [r[0]], marker="o", markersize=8,
                    color=S.REFERENCE if matched else S.SURFACE,
                    markeredgecolor=S.REFERENCE, markeredgewidth=1.6,
                    zorder=4)
            ax.annotate(f"{r[0]:.4f}", (s, r[0]), xytext=(8, 8),
                        textcoords="offset points", ha="left", fontsize=8,
                        color=S.MUTED)

    ax.set_xlabel("backbone steps")
    ax.set_ylabel("GM-Relative MASE — lower is better")
    ax.set_title("the same configuration at both sizes, stop by stop",
                 color=S.INK, fontsize=12, loc="left", pad=12)
    S.tidy(ax)
    ax.errorbar([float("nan")], [float("nan")], yerr=band / 2, color=S.INK,
                linewidth=0, elinewidth=1.2, capsize=3,
                label=f"the bar: ±{band / 2:.4f}, half the two-seed "
                      f"band, measured at 40,000 steps")
    if unmatched:
        ax.plot([], [], color=S.SURFACE, marker="o", markersize=8,
                markeredgecolor=S.REFERENCE, markeredgewidth=1.6,
                linestyle="none",
                label="a 1.1M stop whose head budget differs")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, loc="best")
    fig.canvas.draw()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"wrote {args.out} ({len(tracks)} arm(s), "
          f"{sum(len(t) for t in tracks.values())} stop(s))")


if __name__ == "__main__":
    main()
