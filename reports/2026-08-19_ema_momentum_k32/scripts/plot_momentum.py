#!/usr/bin/env python3
"""The score of every arm against its EMA momentum.

The x axis is the EMA momentum at step 0. Every marker sits at its own
momentum. Two markers that share a momentum sit on the same tick, and the
schedule tells them apart: a circle holds the momentum for the whole run, a
square raises it to 1.0 at 200,000 steps.

A line joins the arms of one schedule, because the momentum is a continuous
axis and the reader follows the direction.

Two or more arms that share a momentum and a schedule are a repeat family. The
figure draws their mean, and a vertical bar over their scores. That bar is the
run-to-run spread this card measures for itself.

A COLLAPSED ARM IS NOT PART OF THAT BAR. One backbone of this card fell to
chance while it trained, and its score says what a dead backbone scores, not
what its momentum is worth. Inside the mean it would move a marker, and inside
the bar it would stretch the spread over every other arm. So it takes its own
red marker, off the line, and `--sync-root` is what tells the two apart.
`seed_report.py` holds the study's one definition of a collapse.

The reference lines carry their own text inside the axes. The legend then
holds the arms alone.

Usage:
  plot_momentum.py --scores results/scores.csv --out plots/momentum.png
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


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _load("cf404_refs", "references.py")
REPEAT = _load("cf404_repeat", "repeat_spread.py")
SEEDS = _load("cf404_seeds", "seed_report.py")

# One colour and one marker per schedule. Two schedules, so two of each.
STYLE = {
    "fixed": {"colour": "#1f77b4", "marker": "o",
              "label": "the momentum holds its value"},
    "ramp": {"colour": "#d95f02", "marker": "s",
             "label": "the momentum rises to 1.0 at 200,000 steps"},
}

# The earlier k = 32 run this card starts from. It ran one momentum on a
# shorter ramp, so it takes its own marker and its own line in the legend.
EARLIER = {"colour": "0.45", "marker": "^",
           "label": "the momentum rises to 1.0 at 100,000 steps "
                    "(the earlier run this card starts from)"}

# The arm whose backbone fell to chance. Red, off the line, and out of every
# mean and every bar.
FELL = {"colour": "#d62728", "marker": "X",
        "label": "the backbone fell to chance while it trained"}


def read_scores(path) -> list[dict]:
    """The rows of scores.csv, typed. `ramp` is 0 for a held momentum."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not r.get("score"):
                continue
            rows.append({"arm": r["arm"], "alpha": float(r["alpha"]),
                         "schedule": r.get("schedule", "fixed"),
                         "ramp": int(float(r.get("ramp") or 0)),
                         "seed": r.get("seed", ""),
                         "score": float(r["score"])})
    return sorted(rows, key=lambda r: (r["schedule"], r["alpha"]))


def points_of(rows, schedule) -> list[tuple[float, float, float, float]]:
    """`(alpha, mean score, low, high)` per momentum, for one schedule.

    A momentum that one arm holds gives low equal to high. A repeat pair
    gives the two scores as the bar's ends.
    """
    by_alpha = {}
    for r in rows:
        if r["schedule"] == schedule:
            by_alpha.setdefault(r["alpha"], []).append(r["score"])
    out = []
    for alpha in sorted(by_alpha):
        s = by_alpha[alpha]
        out.append((alpha, sum(s) / len(s), min(s), max(s)))
    return out


def draw_series(ax, rows, schedule):
    """One line, one marker per momentum, and a bar over a repeat pair."""
    pts = points_of(rows, schedule)
    if not pts:
        return 0
    style = STYLE[schedule]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo = [p[1] - p[2] for p in pts]
    hi = [p[3] - p[1] for p in pts]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=style["colour"],
                marker=style["marker"], markersize=8, linewidth=1.8,
                capsize=5, elinewidth=1.6, zorder=3, label=style["label"])
    return len(pts)


def draw_references(ax, x_lo, x_hi):
    """The published scores, each labelled on its own line inside the axes.

    Two lines can sit within a hair of one another, so a line takes a side.
    The two 40,000-step lines label on the left, and the two 200,000-step
    lines label on the right. No two labels then share a corner.
    """
    low, high = REF.band_bounds()
    ax.axhspan(low, high, color="0.85", zorder=0)
    lines = (
        (REF.K3_BB40K, "-", "0.35", "left",
         f"k = 3, same 40,000 steps ({REF.K3_BB40K:.4f})"),
        (REF.K0_PARENT_BB40K, "--", "0.35", "left",
         f"the k = 0 parent, same 40,000 steps ({REF.K0_PARENT_BB40K:.4f})"),
        (REF.K3_BB200K, ":", "0.45", "right",
         f"k = 3 at 200,000 steps ({REF.K3_BB200K:.4f})"),
        (REF.K32_BB200K, ":", "0.45", "right",
         f"k = 32 at 200,000 steps ({REF.K32_BB200K:.4f})"),
    )
    for y, dash, colour, side, text in lines:
        ax.axhline(y, linestyle=dash, color=colour, linewidth=1.2, zorder=1)
        x = x_lo + 0.004 if side == "left" else x_hi - 0.004
        ax.text(x, y, text, fontsize=8, color="0.20", va="bottom",
                ha="left" if side == "left" else "right", zorder=4)
    ax.text(x_lo + 0.004, high, "the repeat spread of k = 3",
            fontsize=8, color="0.45", va="bottom", ha="left", zorder=4)


def y_range(rows, pad=0.06):
    """The y limits, from the arms that trained and the reference lines.

    A collapsed arm is not in `rows`, so it cannot stretch the axis.
    """
    band_lo, band_hi = REF.band_bounds()
    values = [r["score"] for r in rows] + [
        REF.K3_BB40K, REF.K0_PARENT_BB40K, REF.K3_BB200K, REF.K32_BB200K,
        REF.K32_BB40K, band_lo, band_hi]
    lo, hi = min(values), max(values)
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def draw(rows, out, fell=()):
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    alphas = [r["alpha"] for r in rows] + [r["alpha"] for r in fell] \
        + [REF.K32_BB40K_ALPHA]
    x_lo, x_hi = min(alphas) - 0.03, max(alphas) + 0.03
    draw_references(ax, x_lo, x_hi)
    for schedule in ("fixed", "ramp"):
        draw_series(ax, rows, schedule)
    # The y range covers the arms that trained, and the reference lines. A
    # collapsed arm scores far above every other point, and its true position
    # would squeeze every healthy arm into a band too thin to read. So the
    # axis keeps the healthy range, and a collapsed arm sits on the top edge
    # with its score in text. The reader sees that it is off the scale, and
    # sees by how much.
    y_lo, y_hi = y_range(rows)
    if fell:
        top = y_hi - 0.02 * (y_hi - y_lo)
        ax.plot([r["alpha"] for r in fell], [min(r["score"], top) for r in fell],
                linestyle="none", marker=FELL["marker"], markersize=11,
                color=FELL["colour"], zorder=4, label=FELL["label"])
        for r in fell:
            if r["score"] > top:
                ax.annotate(f"{r['score']:.4f}", (r["alpha"], top),
                            textcoords="offset points", xytext=(12, -2),
                            fontsize=8, color=FELL["colour"], va="center")
    ax.plot([REF.K32_BB40K_ALPHA], [REF.K32_BB40K], linestyle="none",
            marker=EARLIER["marker"], markersize=9, color=EARLIER["colour"],
            zorder=3, label=EARLIER["label"])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks(sorted(set(alphas)))
    ax.set_xlabel("EMA momentum at step 0")
    ax.set_ylabel("GM-Relative MASE over 97 configs, lower is better")
    ax.set_title("The EMA momentum against the score, at rollout depth 32,\n"
                 "with the align loss on the teacher, at 40,000 steps")
    ax.grid(True, alpha=0.3)
    # The arms first, then the earlier run. matplotlib orders by draw call,
    # which puts the reference marker on top.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: labels[i] == EARLIER["label"])
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=1, framealpha=0.9)
    pair_note(ax, rows)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(rows)} arm(s), {len(fell)} collapsed")
    return fig, ax


def pair_note(ax, rows):
    """One line naming the measured repeat spread, when a family exists.

    The number is the range of the bar the reader sees, so it is measured over
    the same rows the bar is drawn from.
    """
    fam = SEEDS.family(rows)
    d = SEEDS.spread(fam)
    if d is None:
        return
    ax.text(0.02, 0.03,
            f"the bar is one arm trained {len(fam)} times at {len(fam)} "
            f"backbone seeds, a range of {d:.4f}",
            transform=ax.transAxes, fontsize=8, color="0.20", va="bottom")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sync-root",
                   help="the sync tree, to read each arm's contrastive AUC")
    p.add_argument("--stop", type=int, default=40000)
    args = p.parse_args(argv)
    if not Path(args.scores).is_file():
        raise SystemExit(f"ABORT: no scores table at {args.scores}")
    rows = read_scores(args.scores)
    fell = []
    if args.sync_root:
        root = Path(args.sync_root).expanduser()
        alive = []
        for r in rows:
            auc = SEEDS.auc_at(root, r["arm"], args.stop)
            (fell if SEEDS.collapsed(auc) else alive).append(r)
        rows = alive
    draw(rows, args.out, fell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
