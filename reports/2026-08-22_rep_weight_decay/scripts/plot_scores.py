#!/usr/bin/env python3
"""The GM-Relative MASE of every arm, with the seed spread beside it.

WHY THIS FIGURE EXISTS. This is the card's first answer. One dot is one run.
One row is one L_rep floor. Three floors carry a repeat at a second backbone
seed, and a bar joins that pair: it is this cell's measured run-to-run spread,
and a difference smaller than the largest of those bars is not a rank.

THE REFERENCES. Two vertical lines carry the numbers a reader compares
against:

  1.0862   the published k = 3 cell at the same 40,000-step stop, from
           `reports/2026-08-08_rollout_depth`, row A4, student encoder. It ran
           a 15,000-step head, and this card runs a 30,000-step head, so it is
           a reference and not a control.
  1.0660   the same cell at 200,000 steps, which is the project's best.

An arm the AUC gate stopped has no score. The figure names it under the axis
rather than leaving the reader to count the missing rows.

Usage:
  plot_scores.py --scores results/scores.csv --arms scripts/arms.tsv \
      --out plots/scores.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("arm_style", HERE / "arm_style.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

# The two published numbers of this same cell. See the docstring.
REF_40K = 1.0862
REF_200K = 1.0660


def row_name(rep_end, target):
    if target == "teacher":
        return "floor 0.0, teacher target"
    if rep_end in ("", "-", None):
        return "no decay (the control)"
    return f"floor {float(rep_end):.1f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scores",
                   default=str(HERE.parent / "results" / "scores.csv"))
    p.add_argument("--arms", default=str(HERE / "arms.tsv"))
    p.add_argument("--out", default=str(HERE.parent / "plots" / "scores.png"))
    args = p.parse_args(argv)

    arms = S.read_arms(args.arms)
    by_arm = {r["arm"]: r for r in arms}
    scored = {}
    try:
        with open(args.scores, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    scored[r["arm"]] = float(r["score"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    if not scored:
        print(f"no score in {args.scores}", file=sys.stderr)
        return 2

    # One row per (floor, align target), in the card's order.
    groups, order = defaultdict(list), []
    for row in arms:
        key = (row["rep_end"], row["align_target"])
        if key not in groups:
            order.append(key)
        groups[key].append(row)

    fig, ax = plt.subplots(figsize=(8.0, 3.2 + 0.42 * len(order)))
    ax.axvline(REF_40K, color=S.MUTED, linestyle="--", linewidth=1.1)
    ax.axvline(REF_200K, color=S.MUTED, linestyle=":", linewidth=1.1)

    missing = []
    ticks, labels = [], []
    for y, key in enumerate(reversed(order)):
        rows = groups[key]
        values = [(r, scored[r["arm"]]) for r in rows if r["arm"] in scored]
        for r in rows:
            if r["arm"] not in scored:
                missing.append(r["arm"])
        ticks.append(y)
        labels.append(row_name(*key))
        if len(values) > 1:
            lo = min(v for _, v in values)
            hi = max(v for _, v in values)
            ax.plot([lo, hi], [y, y], color=S.arm_colour(rows[0]),
                    linewidth=2.0, alpha=0.45, solid_capstyle="round")
            ax.annotate(f"spread {hi - lo:.4f}", ((lo + hi) / 2, y),
                        xytext=(0, 7), textcoords="offset points",
                        fontsize=7, color=S.MUTED, ha="center")
        for r, v in values:
            ax.plot([v], [y], marker="o" if r["seed"] == "20260520" else "s",
                    markersize=8, color=S.arm_colour(r),
                    markeredgecolor="#fcfcfb", markeredgewidth=1.4,
                    linestyle="none")
            ax.annotate(f"{v:.4f}", (v, y), xytext=(0, -14),
                        textcoords="offset points", fontsize=7,
                        color=S.INK, ha="center")

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9, color=S.INK)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("GM-Relative MASE over the 97 GIFT-Eval configs (lower is better)")
    ax.set_title("Does the L_rep decay improve the score?",
                 color=S.INK, fontsize=11, loc="left")
    top = ax.get_ylim()[1]
    ax.annotate("the published k = 3 cell at 40,000 steps", (REF_40K, top),
                xytext=(3, -10), textcoords="offset points",
                fontsize=7, color=S.MUTED, rotation=90, va="top")
    ax.annotate("the same cell at 200,000 steps", (REF_200K, top),
                xytext=(3, -10), textcoords="offset points",
                fontsize=7, color=S.MUTED, rotation=90, va="top")
    S.tidy(ax)
    ax.grid(axis="y", visible=False)
    # Identity is never colour alone: the marker says which seed.
    ax.plot([], [], marker="o", linestyle="none", color=S.MUTED,
            markersize=7, label="seed 20260520")
    ax.plot([], [], marker="s", linestyle="none", color=S.MUTED,
            markersize=7, label="seed 20260524")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.INK, loc="lower right")
    if missing:
        fig.text(0.01, 0.005,
                 "no score: " + ", ".join(
                     S.arm_label(by_arm[a]) for a in missing),
                 fontsize=7, color=S.ALARM)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"{args.out}: {len(scored)} score(s), {len(missing)} without one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
