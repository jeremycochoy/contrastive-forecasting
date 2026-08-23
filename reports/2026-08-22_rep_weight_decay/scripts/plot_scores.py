#!/usr/bin/env python3
"""The GM-Relative MASE of every run, against the two published references.

WHY THIS FIGURE EXISTS. This is the card's first answer. One dot is one run of
the decay arm, at its own backbone seed. Every run is the same treatment, so
the set of dots is the decay arm's own spread, and its mean is the number that
answers the card.

THE REFERENCES. Two vertical lines carry the numbers a run is read against.
This card measures NO control: the EMA momentum sweep already scored this same
cell, at the same 40,000-step stop and the same 30,000-step head, at two
backbone seeds:

  1.1507   seed 20260520
  1.1491   seed 20260524, the best score of that sweep

They come from `reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md`. Their
range, 0.0016, is the bar on the reference: a decay run inside it is not a
rank.

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
    missing = [r["seed"] for r in arms if r["arm"] not in scored]

    fig, ax = plt.subplots(figsize=(8.0, 2.8 + 0.40 * (len(ranked) + 1)))
    # The reference is a RANGE, not a point: the sweep scored this cell at two
    # seeds. A band says so, where two lines 0.0016 apart would read as one.
    ref_lo, ref_hi = min(S.SWEEP_SCORES.values()), max(S.SWEEP_SCORES.values())
    ax.axvspan(ref_lo, ref_hi, color=S.REFERENCE, alpha=0.16, linewidth=0)
    for value in (ref_lo, ref_hi):
        ax.axvline(value, color=S.REFERENCE, linestyle="--", linewidth=1.1)

    ticks, labels = [], []
    for y, (row, value) in enumerate(reversed(ranked)):
        ticks.append(y)
        labels.append(S.seed_label(row))
        ax.plot([value], [y], marker="o", markersize=8, color=S.SERIES,
                markeredgecolor=S.SURFACE, markeredgewidth=1.4,
                linestyle="none")
        ax.annotate(f"{value:.4f}", (value, y), xytext=(0, -14),
                    textcoords="offset points", fontsize=7,
                    color=S.INK, ha="center")

    # The summary row. The card asks whether the decay beats 1.1491, and one
    # run cannot say so: the sweep measured a seed range of 0.1432 on one arm.
    if len(ranked) > 1:
        values = [v for _, v in ranked]
        y = len(ranked)
        ticks.append(y)
        labels.append(f"the decay, {len(values)} seeds")
        ax.plot([min(values), max(values)], [y, y], color=S.SERIES,
                linewidth=2.0, alpha=0.45, solid_capstyle="round")
        mean = statistics.fmean(values)
        ax.plot([mean], [y], marker="D", markersize=9, color=S.SERIES,
                markeredgecolor=S.SURFACE, markeredgewidth=1.4,
                linestyle="none")
        ax.annotate(f"mean {mean:.4f}, range {max(values) - min(values):.4f}",
                    (mean, y), xytext=(0, 9), textcoords="offset points",
                    fontsize=7, color=S.INK, ha="center")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9, color=S.INK)
    ax.set_ylim(-0.6, len(ticks) - 0.4)
    ax.set_xlabel(
        "GM-Relative MASE over the 97 GIFT-Eval configs (lower is better)")
    ax.set_title("Does the L_rep decay improve the score?",
                 color=S.INK, fontsize=11, loc="left")
    # Inside the axes, at the top, on whichever side of the band has room. A
    # label above the axes would sit on the title.
    x_lo, x_hi = ax.get_xlim()
    right = (ref_hi - x_lo) / max(x_hi - x_lo, 1e-12) > 0.6
    ax.annotate(f"the sweep on this cell, {ref_lo:.4f} to {ref_hi:.4f}",
                (ref_lo if right else ref_hi, 0.995),
                xycoords=("data", "axes fraction"),
                xytext=(-6 if right else 6, 0),
                textcoords="offset points", fontsize=7.5, color=S.REFERENCE,
                ha="right" if right else "left", va="top")
    S.tidy(ax)
    ax.grid(axis="y", visible=False)
    # Two marks, two meanings. The seed is the row label.
    ax.plot([], [], marker="o", linestyle="none", color=S.SERIES,
            markersize=7, label="one run of the decay arm")
    ax.plot([], [], color=S.REFERENCE, linestyle="--", linewidth=1.4,
            label="the sweep's two published scores on this cell")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.20))
    if missing:
        fig.text(0.01, 0.005, "no score: " + ", ".join(missing),
                 fontsize=7, color=S.LOST)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=S.SURFACE)
    print(f"{args.out}: {len(ranked)} score(s), {len(missing)} without one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
