#!/usr/bin/env python3
"""The score against each of the card's two axes: the EMA momentum at the
stop, and the decay ramp.

WHY THIS FIGURE EXISTS. `plot_scores.py` sorts the arms by score, so neither
axis is readable there. This figure puts each axis on x, so a reader sees
where the best value of each sits inside the tested range, and how far every
point is from the reference.

Left: x is the momentum the arm holds at the 40,000-step stop, at the card's
own decay ramp of 10,000 steps. A schedule with several seeds draws each seed
as a small dot and their mean as the large one. The line joins four different
schedules, so it takes categorical slot 3 of the data-viz standard and its own
legend entry, never the colour of a schedule family. Right: x is the decay
ramp, one line per schedule the card varied it on. The two families take the
first two categorical hues, and each carries its momentum as a legend entry,
so identity is never the colour alone.

Both panels carry the reference (the best schedule with no decay) as a dashed
line, and a band of the seed range above it. The seed range is the widest
range over the repeat-seed groups of this card, 0.0471 from the two seeds of
0.8 to 1.0 at 200k at ramp 10,000; the reference's own range is 0.0016. A
point inside the band is within the seed range, not a rank.

Usage:
  plot_axes.py --scores results/scores.csv --arms scripts/arms.tsv \
      --out plots/axes.png
"""
from __future__ import annotations

import argparse
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


# The second ramp family, at momentum 0.940. Categorical slot 2 of the
# data-viz standard, beside `S.SERIES`, which is slot 1.
FAMILY_2 = "#eb6834"
# The momentum panel. Its one line joins arms of four schedules, so it wears
# categorical slot 3, a colour no schedule family claims in the legend.
MOMENTUM = "#1baf7a"


def frame(ax, x_label, band, x_ticks=None):
    """The reference, the seed-range band, the ticks and the tidy frame."""
    ax.axhline(S.SWEEP_BEST, color=S.REFERENCE, linestyle="--", linewidth=1.1)
    ax.axhspan(S.SWEEP_BEST, S.SWEEP_BEST + band, color=S.SERIES, alpha=0.10,
               linewidth=0)
    ax.margins(x=0.18)
    if x_ticks:
        ax.set_xticks(x_ticks)
    ax.set_xlabel(x_label)
    S.tidy(ax)


def series(ax, points, colour=S.SERIES, label=None, side=None, at=None,
           short=False):
    """One line of means. `points` is `{x: [(arm, score), ...]}`.

    `side` is "above" or "below" for every label of the family, so two
    families on one panel never write on each other. None keeps the default:
    a seed group under its point, a single arm above it. `at` maps an x value
    to its plotted position, for a categorical x axis. `short` labels a point by
    its score alone, and a seed group by its count, where the panel has no
    room for arm names.
    """
    xs = sorted(points)
    means, pos = [], []
    for x in xs:
        values = [v for _, v in points[x]]
        px = at[x] if at else x
        pos.append(px)
        if len(values) > 1:
            ax.plot([px] * len(values), values, marker="o", markersize=4,
                    color=colour, alpha=0.5, linestyle="none", zorder=2)
        mean = statistics.fmean(values)
        means.append(mean)
        ax.plot([px], [mean], marker="o", markersize=8,
                color=colour, markeredgecolor=S.SURFACE, markeredgewidth=1.4,
                linestyle="none", zorder=3)
        # A short label is the score alone, or the seed count of a group.
        # The arm names of that panel are in the grid figure and the tables.
        names = ((f"{len(values)} seeds" if len(values) > 1 else "") if short
                 else ", ".join(a for a, _ in points[x]))
        # A seed group carries its label under the point, so it does not
        # collide with the label of the next point above it.
        below = (len(values) > 1) if side is None else (side == "below")
        anchor = (min(values) if below else max(values)) if len(values) > 1 \
            else mean
        ax.annotate((f"mean {mean:.4f}\n{names}" if len(values) > 1
                     else f"{mean:.4f}\n{names}").rstrip(),
                    (px, anchor),
                    xytext=(0, -12 if below else 10),
                    textcoords="offset points",
                    fontsize=6.5, color=S.INK, ha="center",
                    va="top" if below else "bottom")
    ax.plot(pos, means, color=colour, linewidth=1.2, alpha=0.6, zorder=1,
            label=label)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "axes.png"))
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    scored = S.read_scores(args.scores)
    if not scored:
        print(f"no score in {args.scores}", file=sys.stderr)
        return 2
    groups = S.repeat_groups(arms, scored)
    band = max((max(v for _, v in g) - min(v for _, v in g)
                for g in groups.values()), default=0.0)

    by_momentum = {}
    # The ramp families: every schedule with a scored arm off the default
    # ramp, each as `{ramp: [(arm, score), ...]}` over all its scored arms.
    families = {}
    for row in arms:
        if row["arm"] not in scored:
            continue
        if S.decay_ramp(row) == S.DECAY_RAMP_DEFAULT:
            by_momentum.setdefault(round(S.momentum_at(row), 3), []).append(
                (row["arm"], scored[row["arm"]]))
        else:
            families.setdefault(S.schedule(row), {})
    for row in arms:
        if row["arm"] in scored and S.schedule(row) in families:
            families[S.schedule(row)].setdefault(S.decay_ramp(row), []).append(
                (row["arm"], scored[row["arm"]]))
    # A line against the ramp needs a schedule the card varied the ramp ON.
    # The three fixed-momentum arms hold one ramp each, so they are cells of
    # the grid figure, not a family here.
    families = {s: f for s, f in families.items() if len(f) > 1}

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    frame(left, "EMA momentum at the stop, decay ramp 10,000", band,
          x_ticks=sorted(by_momentum))
    series(left, by_momentum, colour=MOMENTUM,
           label="four EMA schedules, one point per momentum")
    # The ramps sit at equal spacing: 1,000 to 30,000 on a linear axis
    # squashes four of the six into one fifth of the panel.
    ramp_ticks = sorted({r for f in families.values() for r in f})
    at = {r: n for n, r in enumerate(ramp_ticks)}
    frame(right, "decay ramp, steps", band, x_ticks=list(at.values()))
    right.set_xticklabels([f"{r:,}" for r in ramp_ticks])
    colours = [S.SERIES, FAMILY_2]
    for n, (sched, points) in enumerate(sorted(families.items())):
        row = next(r for r in arms if S.schedule(r) == sched)
        series(right, points, colour=colours[n % len(colours)],
               label=f"EMA {S.schedule_label(row)}, momentum "
                     f"{S.momentum_at(row):.3f} at the stop",
               side="above" if n == 0 else "below", at=at, short=True)
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
                       label=f"one seed range above the reference, {band:.4f}")
    fig.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(wspace=0.08)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: {len(by_momentum)} momentum value(s), "
          f"{len(ramp_ticks)} ramp(s) over {len(families)} famil(ies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
