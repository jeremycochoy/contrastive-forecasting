#!/usr/bin/env python3
"""The score of the carried arms against the backbone stop.

WHY THIS FIGURE EXISTS. The card asks whether the best decay arm improves
with longer training. The direct measurement is its score at 40,000, 80,000
and 200,000 steps, and only a curve against the stop shows its direction.

WHAT IT SHOWS. One line per arm carried past 40,000 steps, over every stop
`results/scores.csv` holds for it. The best arm keeps the series colour, the
other carried arm steps back to the held grey. The x axis is linear in steps,
so the distance to 200,000 reads as it is. Each stop names the momentum the
EMA schedule holds there. The reference and the seed-range band repeat the
other score figures.

Usage:
  plot_stops.py --scores results/scores.csv --arms scripts/arms.tsv \
      --out plots/stops.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "stops.png"))
    args = p.parse_args(argv)

    arms = {row["arm"]: row for row in S.read_arms(args.arms)}
    # Every (arm, stop) score, keyed by the arm. `read_scores` filters to one
    # stop, so this reads the CSV itself.
    curves = {}
    with open(args.scores, newline="") as fh:
        for rec in csv.DictReader(fh):
            try:
                stop, score = int(rec["stop"]), float(rec["score"])
            except (KeyError, TypeError, ValueError):
                continue
            curves.setdefault(rec["arm"], []).append((stop, score))
    curves = {a: sorted(v) for a, v in curves.items() if len(v) > 1}
    if not curves:
        print(f"no arm with two stops in {args.scores}", file=sys.stderr)
        return 2

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(S.SWEEP_BEST, color=S.REFERENCE, linestyle="--", linewidth=1.1)
    ax.axhspan(S.SWEEP_BEST, S.SWEEP_BEST + 0.0471, color=S.SERIES,
               alpha=0.10, linewidth=0)
    items = []
    for arm, points in sorted(curves.items()):
        colour = S.SERIES if arm == S.HIGHLIGHT_ARM else S.HELD
        ax.plot([s for s, _ in points], [v for _, v in points], marker="o",
                markersize=7, color=colour, linewidth=1.6,
                markeredgecolor=S.SURFACE, markeredgewidth=1.2)
        # The two curves nearly touch at 40,000 and 80,000 steps, so the best
        # arm labels above its points and the other below.
        above = arm == S.HIGHLIGHT_ARM
        for stop, value in points:
            ax.annotate(f"{value:.4f}", (stop, value),
                        xytext=(0, 9 if above else -11),
                        textcoords="offset points", fontsize=7.5,
                        color=S.INK, ha="center",
                        va="bottom" if above else "top")
        items.append((points, arm, colour))
    stops = sorted({s for v in curves.values() for s, _ in v})
    ax.set_xticks(stops)
    ax.set_xticklabels([
        f"{s:,}\nmomentum "
        f"{S.momentum_at(arms[S.HIGHLIGHT_ARM], step=s):.3f}" for s in stops])
    ax.set_xlabel("backbone stop, steps")
    ax.set_ylabel("GM-Relative MASE (lower is better)")
    ax.set_title("Score of the carried arms at each backbone stop",
                 color=S.INK, fontsize=10, loc="left")
    S.tidy(ax)
    ax.margins(x=0.14)
    lo = min(S.SWEEP_BEST, min(v for c in curves.values() for _, v in c))
    hi = max(v for c in curves.values() for _, v in c)
    ax.set_ylim(lo - 0.015, hi + 0.025)
    fig.canvas.draw()
    S.label_right(ax, items)
    ax.plot([], [], color=S.REFERENCE, linestyle="--", linewidth=1.4,
            label=f"reference, no decay, 40,000 steps, {S.SWEEP_BEST:.4f}")
    ax.fill_between([], [], [], color=S.SERIES, alpha=0.10,
                    label="one seed range above the reference, 0.0471")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=1,
              loc="lower center", bbox_to_anchor=(0.5, -0.42))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: {len(curves)} carried arm(s) over {len(stops)} stops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
