#!/usr/bin/env python3
"""Pass 4: does the L_rep decay let a longer stop win, at 5.6e-4?

WHY THIS FIGURE EXISTS. Every arm of this card so far gets WORSE with a longer
stop. Pass 4 tests the one combination the card never ran: the L_rep weight
decays to zero EARLY, at the rate that fits the width, and the run goes LONG.

WHAT IT DRAWS. Three tracks at 5.6e-4, over the stops each one reached. The
reference is `k3_r100_09_lr56`, the same cell with no decay. Each decay track
carries a dashed line at its OWN 40,000-step score, so a point under that line
is the result the card asks for. The bars are the seed band at 5.6e-4.

Lower is better, so a line that falls is a model that improves.

Usage:  plot_pass4.py [--out plots/pass4.png]
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

# The three tracks, in draw order, with their ink and their label.
TRACKS = [
    ("k3_r100_09_lr56", S.REFERENCE, "no decay (the reference)"),
    ("k3_r100_09_lr56_dec", S.SERIES, "L_rep to zero by step 2,000"),
    ("k3_r100_09_lr56_dec10k", S.DEPTH_RAMP[8], "L_rep to zero by step 10,000"),
]
STOPS = (40000, 100000, 200000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=str(STUDY / "results" / "scores.csv"))
    p.add_argument("--out", default=str(STUDY / "plots" / "pass4.png"))
    args = p.parse_args()

    rows = S.read_scores(args.scores)
    drawn = [(a, c, lab, [(s, rows[(a, s)]) for s in STOPS if (a, s) in rows])
             for a, c, lab in TRACKS]
    drawn = [d for d in drawn if d[3]]
    if not drawn:
        raise SystemExit("no track holds a score")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.patch.set_facecolor(S.SURFACE)
    ax.set_facecolor(S.SURFACE)

    for arm, colour, label, track in drawn:
        xs = [s / 1000 for s, _ in track]
        ys = [v for _, v in track]
        is_ref = arm == "k3_r100_09_lr56"
        ax.errorbar(xs, ys, yerr=None if is_ref else S.BAND / 2,
                    color=colour, linewidth=2.0 if not is_ref else 1.4,
                    linestyle="--" if is_ref else "-",
                    marker="o", markersize=6, capsize=4, zorder=3,
                    label=label)
        # The arm's own 40,000-step score. A point under this line is the
        # result the card asks for.
        if not is_ref and track[0][0] == 40000:
            ax.axhline(track[0][1], color=colour, linewidth=0.9,
                       linestyle=":", alpha=0.7, zorder=1)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=colour)

    ax.set_xscale("log")
    # The log scale adds its own minor labels ("6 x 10^1"), and this axis has
    # three values the reader knows by name.
    ax.set_xticks([], minor=True)
    ax.set_xticks([s / 1000 for s in STOPS])
    ax.set_xticklabels([f"{s // 1000}k" for s in STOPS])
    ax.set_xlabel("backbone steps")
    ax.set_ylabel("GM-Relative MASE (lower is better)")
    ax.set_title("Pass 4 — the L_rep decay at 5.6e-4, over a long stop\n"
                 "dotted line: the arm's own 40,000-step score. "
                 f"bars: the seed band, {S.BAND:.4f}.",
                 fontsize=10, loc="left")
    ax.grid(True, color=S.GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
