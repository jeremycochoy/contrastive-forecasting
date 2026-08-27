#!/usr/bin/env python3
"""The GM-Relative MASE of every EMA schedule, with and without the decay.

WHY THIS FIGURE EXISTS. This is the card's first answer. One row is one EMA
schedule. The filled dot is that schedule WITH the decay, which this card
measured. The open dot is the SAME schedule with NO decay, which the EMA
momentum sweep published at the same cell, the same 40,000-step stop and the
same 30,000-step head. The line between them is the decay's effect on that
schedule.

This card measures NO control, so every open dot comes from
`reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md`. One schedule, 0.99
fixed, has no open dot: the sweep never ran it.

THE BAR. One treatment (an EMA schedule at one decay ramp) ran at three
backbone seeds. Its spread is the only repeat this card holds, and it is what
says whether a gap between two schedules is a rank or noise. The figure draws
it as one bar. `arm_style.repeat_groups` picks that treatment, and
`rank_gate.py` reads the same one.

A run the AUC gate stopped has no score. The figure names it under the axis
rather than leaving the reader to count the missing rows.

Usage:
  plot_scores.py --scores results/scores.csv --arms scripts/arms.tsv \
      --out plots/scores.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def read_scores(path):
    """`{arm: score}` from `results/scores.csv`."""
    out = {}
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    out[r["arm"]] = float(r["score"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "scores.png"))
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    scored = read_scores(args.scores)
    if not scored:
        print(f"no score in {args.scores}", file=sys.stderr)
        return 2

    # Best first, so the reader's eye starts where the answer is.
    ranked = sorted(((r, scored[r["arm"]]) for r in arms
                     if r["arm"] in scored), key=lambda t: t[1])
    missing = [S.arm_label(r) for r in arms if r["arm"] not in scored]
    # The one repeat this card holds: arm 1 at three seeds.
    # The seed spread: the treatment with the widest range over its scored
    # seeds. Only one treatment of this card scored more than one seed.
    groups = S.repeat_groups(arms, scored)
    spread_key = max(groups, key=lambda k: max(v for _, v in groups[k])
                     - min(v for _, v in groups[k])) if groups else None
    repeats = [v for _, v in groups[spread_key]] if spread_key else []
    spread_row = next(r for r in arms if r["arm"] == groups[spread_key][0][0]) \
        if spread_key else None

    fig, ax = plt.subplots(figsize=(8.6, 2.8 + 0.42 * (len(ranked) + 1)))
    # The number the card asks an arm to beat: the best the sweep measured on
    # this cell with no decay.
    ax.axvline(S.SWEEP_BEST, color=S.REFERENCE, linestyle="--", linewidth=1.1)

    ticks, labels = [], []
    for y, (row, value) in enumerate(reversed(ranked)):
        ticks.append(y)
        labels.append(S.arm_label(row))
        ref = S.SWEEP_SCORES.get(S.schedule(row))
        if ref is not None:
            # The pair, and the move between them. A slope reads as one fact
            # where two separate dots read as two.
            ax.plot([ref, value], [y, y], color=S.MUTED, linewidth=1.2,
                    alpha=0.55, solid_capstyle="round", zorder=1)
            ax.plot([ref], [y], marker="o", markersize=7, color=S.SURFACE,
                    markeredgecolor=S.REFERENCE, markeredgewidth=1.4,
                    linestyle="none", zorder=2)
        ax.plot([value], [y], marker="o", markersize=8, color=S.SERIES,
                markeredgecolor=S.SURFACE, markeredgewidth=1.4,
                linestyle="none", zorder=3)
        ax.annotate(f"{value:.4f}", (value, y), xytext=(0, -14),
                    textcoords="offset points", fontsize=7,
                    color=S.INK, ha="center")

    # The spread row. A gap between two schedules smaller than this bar is not
    # a rank.
    if len(repeats) > 1:
        y = len(ranked)
        ticks.append(y)
        labels.append(f"{S.schedule_label(spread_row)}, {len(repeats)} seeds")
        ax.plot([min(repeats), max(repeats)], [y, y], color=S.SERIES,
                linewidth=2.0, alpha=0.45, solid_capstyle="round")
        mean = statistics.fmean(repeats)
        ax.plot([mean], [y], marker="D", markersize=9, color=S.SERIES,
                markeredgecolor=S.SURFACE, markeredgewidth=1.4,
                linestyle="none")
        ax.annotate(f"mean {mean:.4f}, range {max(repeats) - min(repeats):.4f}",
                    (mean, y), xytext=(0, 9), textcoords="offset points",
                    fontsize=7, color=S.INK, ha="center")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9, color=S.INK)
    ax.set_ylim(-0.6, len(ticks) - 0.4)
    ax.set_xlabel(
        "GM-Relative MASE over the 97 GIFT-Eval configs (lower is better)")
    ax.set_title("Does the L_rep decay improve the score, at any EMA schedule?",
                 color=S.INK, fontsize=11, loc="left")
    # Inside the axes, at the top, on whichever side of the line has room. A
    # label above the axes would sit on the title.
    x_lo, x_hi = ax.get_xlim()
    right = (S.SWEEP_BEST - x_lo) / max(x_hi - x_lo, 1e-12) > 0.6
    ax.annotate(f"the sweep's best on this cell, {S.SWEEP_BEST:.4f}",
                (S.SWEEP_BEST, 0.995), xycoords=("data", "axes fraction"),
                xytext=(-6 if right else 6, 0),
                textcoords="offset points", fontsize=7.5, color=S.REFERENCE,
                ha="right" if right else "left", va="top")
    S.tidy(ax)
    ax.grid(axis="y", visible=False)
    # Three marks, three meanings. The schedule is the row label.
    ax.plot([], [], marker="o", linestyle="none", color=S.SERIES,
            markersize=7, label="with the decay")
    ax.plot([], [], marker="o", linestyle="none", color=S.SURFACE,
            markeredgecolor=S.REFERENCE, markeredgewidth=1.4, markersize=7,
            label="the same schedule, no decay (the sweep)")
    ax.plot([], [], color=S.REFERENCE, linestyle="--", linewidth=1.4,
            label="the sweep's best on this cell")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.20))
    if missing:
        # Under the legend, in axes fractions. A figure-coordinate note lands
        # on the x label when the panel is short, and this card can hold eight
        # rows with no score at once.
        ax.annotate(textwrap.fill("no score: " + ", ".join(missing), 96),
                    (0.5, -0.30), xycoords="axes fraction", fontsize=7,
                    color=S.LOST, ha="center", va="top",
                    annotation_clip=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: {len(ranked)} score(s), {len(missing)} without one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
