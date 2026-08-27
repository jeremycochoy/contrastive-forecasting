#!/usr/bin/env python3
"""The score against each of the card's two axes: the EMA momentum at the
stop, and the decay ramp.

WHY THIS FIGURE EXISTS. `plot_scores.py` sorts the arms by score, so neither
axis is readable there. This figure puts each axis on x, so a reader sees
where the best value of each sits inside the tested range, and how far every
point is from the reference.

Left: x is the momentum the arm holds at the 40,000-step stop, at the card's
own decay ramp of 10,000 steps. A schedule with several seeds draws each seed
as a small dot and their mean as the large one. Right: x is the decay ramp,
at the one schedule the card varied it on.

Both panels carry the reference (the best schedule with no decay) as a dashed
line, and a band of the seed range above it: a point inside the band is
within the scatter of one treatment, not a rank.

Usage:
  plot_axes.py --scores results/scores.csv --arms scripts/arms.tsv \
      --out plots/axes.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def read_scores(path):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                out[r["arm"]] = float(r["score"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def panel(ax, points, x_label, band, x_ticks=None):
    """`points` is `{x: [(arm, score), ...]}`."""
    ax.axhline(S.SWEEP_BEST, color=S.REFERENCE, linestyle="--", linewidth=1.1)
    ax.axhspan(S.SWEEP_BEST, S.SWEEP_BEST + band, color=S.SERIES, alpha=0.10,
               linewidth=0)
    xs = sorted(points)
    means = []
    for x in xs:
        values = [v for _, v in points[x]]
        if len(values) > 1:
            ax.plot([x] * len(values), values, marker="o", markersize=4,
                    color=S.SERIES, alpha=0.5, linestyle="none", zorder=2)
        mean = statistics.fmean(values)
        means.append(mean)
        ax.plot([x], [mean], marker="o", markersize=8,
                color=S.SERIES, markeredgecolor=S.SURFACE, markeredgewidth=1.4,
                linestyle="none", zorder=3)
        names = ", ".join(a for a, _ in points[x])
        # A seed group carries its label under the point, so it does not
        # collide with the label of the next point above it.
        many = len(values) > 1
        ax.annotate(f"mean {mean:.4f}\n{names}" if many
                    else f"{mean:.4f}\n{names}",
                    (x, min(values) if many else mean),
                    xytext=(0, -12 if many else 10), textcoords="offset points",
                    fontsize=6.5, color=S.INK, ha="center",
                    va="top" if many else "bottom")
    ax.plot(xs, means, color=S.SERIES, linewidth=1.2, alpha=0.6, zorder=1)
    ax.margins(x=0.18)
    if x_ticks:
        ax.set_xticks(x_ticks)
    ax.set_xlabel(x_label)
    S.tidy(ax)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "axes.png"))
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    scored = read_scores(args.scores)
    if not scored:
        print(f"no score in {args.scores}", file=sys.stderr)
        return 2
    groups = S.repeat_groups(arms, scored)
    band = max((max(v for _, v in g) - min(v for _, v in g)
                for g in groups.values()), default=0.0)

    by_momentum, by_ramp = {}, {}
    ramp_schedule = None
    for row in arms:
        if row["arm"] not in scored:
            continue
        if S.decay_ramp(row) == S.DECAY_RAMP_DEFAULT:
            by_momentum.setdefault(round(S.momentum_at(row), 3), []).append(
                (row["arm"], scored[row["arm"]]))
        elif ramp_schedule is None:
            ramp_schedule = S.schedule(row)
    for row in arms:
        if row["arm"] in scored and S.schedule(row) == ramp_schedule:
            by_ramp.setdefault(S.decay_ramp(row), []).append(
                (row["arm"], scored[row["arm"]]))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    panel(left, by_momentum, "EMA momentum at the stop, decay ramp 10,000",
          band, x_ticks=sorted(by_momentum))
    ramp_row = next(r for r in arms if S.schedule(r) == ramp_schedule)
    panel(right, by_ramp, f"decay ramp, steps, EMA {S.schedule_label(ramp_row)}",
          band, x_ticks=sorted(by_ramp))
    left.set_ylabel("GM-Relative MASE (lower is better)")
    lo = min(S.SWEEP_BEST, min(scored.values()))
    hi = max(scored.values())
    left.set_ylim(lo - 0.02, hi + 0.06)
    left.set_title("Score against the EMA momentum", color=S.INK, fontsize=10,
                   loc="left")
    right.set_title("Score against the decay ramp", color=S.INK, fontsize=10,
                    loc="left")
    right.plot([], [], color=S.REFERENCE, linestyle="--", linewidth=1.4,
               label=f"reference, no decay, {S.SWEEP_BEST:.4f}")
    right.fill_between([], [], [], color=S.SERIES, alpha=0.10,
                       label=f"seed range, {band:.4f}")
    right.plot([], [], marker="o", linestyle="none", color=S.SERIES,
               markersize=7, label="with the decay")
    fig.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(wspace=0.08)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: {len(by_momentum)} momentum value(s), "
          f"{len(by_ramp)} ramp(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
