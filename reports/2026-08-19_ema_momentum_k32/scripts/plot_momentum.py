#!/usr/bin/env python3
"""The score of every arm against its EMA momentum.

The x axis is the EMA momentum at step 0. Every marker sits at its own
momentum. Two markers that share a momentum sit on the same tick, and the
SERIES tells them apart. A series is a schedule, a ramp length and an L_align
weight, all three together:
a circle holds the momentum for the whole run, and each ramp length takes its
own colour and its own marker.

The ramp length is part of the series because it changes the momentum the arm
reaches. `s08` and `r100_08` both start at 0.8, and at 40,000 steps they hold
0.840 and 0.880. `momentum_at_stop.png` puts that reached value on the x axis.

A line joins the arms of one series, because the momentum is a continuous
axis and the reader follows the direction.

Two or more arms that share a momentum, a schedule and a ramp are a repeat
family. The figure draws their mean, and a vertical bar over their scores.
That bar is the run-to-run spread this card measures for itself.

A COLLAPSED ARM IS NOT PART OF THAT BAR. One backbone of this card lost the
contrastive task while it trained, and its score says what a dead backbone
scores, not
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

# One colour and one marker per SERIES. A series is a schedule, a ramp length
# and an L_align WEIGHT, all three together.
#
# The ramp length has to be in the key. A ramp arm is named by the momentum it
# starts at, and two arms that start at one value reach different values by the
# stop: `s08` and `r100_08` both start at 0.8, and at 40,000 steps they hold
# 0.840 and 0.880. One series for both would average those two scores into one
# marker and draw the distance between them as a repeat spread, which is what a
# vertical bar means everywhere else on this figure.
#
# The align weight has to be in the key for the same reason, and it is the
# tighter case: `w3_s08` is `s08` with that one flag at 3.0 instead of 1.0, so
# the two share a momentum, a schedule, a ramp length AND a backbone seed. A
# key without the weight would put both on one marker and report the distance
# between two different objectives as this cell's run-to-run spread.
DEFAULT_ALIGN_W = 1.0

FIXED_STYLE = {"colour": "#1f77b4", "marker": "o",
               "label": "the momentum holds its value"}

# Taken in order of ramp length, shortest first. No entry is a grey: the
# references are grey.
RAMP_STYLES = ({"colour": "#d95f02", "marker": "s"},
               {"colour": "#7570b3", "marker": "D"},
               {"colour": "#66a61e", "marker": "v"},
               {"colour": "#e7298a", "marker": "P"})


def label_of(schedule: str, ramp: int, align_w: float) -> str:
    """The legend line of one series."""
    if schedule == "fixed":
        text = "the momentum holds its value"
    else:
        text = f"the momentum rises to 1.0 at {ramp:,} steps"
    if align_w != DEFAULT_ALIGN_W:
        text += f", L_align weight {align_w:g}"
    return text


def series_of(rows) -> list[tuple[str, int, float, dict]]:
    """`(schedule, ramp, align_w, style)` for every series present, in order.

    The fixed arms first, then the ramps by length, shortest first, and within
    one length by align weight. A series the style list does not cover reuses
    a style and says so on stderr, because two series in one style read as one
    series.
    """
    keys = sorted({(r["schedule"], r["ramp"], r["align_w"]) for r in rows},
                  key=lambda k: (k[0] != "fixed", k[1], k[2]))
    out, spare = [], 0
    for schedule, ramp, align_w in keys:
        if schedule == "fixed" and align_w == DEFAULT_ALIGN_W:
            style = dict(FIXED_STYLE)
        else:
            if spare >= len(RAMP_STYLES):
                print(f"WARN: more than {len(RAMP_STYLES)} series outside the "
                      f"held momentum — the styles repeat from "
                      f"({schedule}, {ramp}, {align_w:g}) on.", file=sys.stderr)
            style = dict(RAMP_STYLES[spare % len(RAMP_STYLES)])
            spare += 1
        style["label"] = label_of(schedule, ramp, align_w)
        out.append((schedule, ramp, align_w, style))
    return out

# The published run this card starts from. It holds the same depth and the
# same 100,000-step ramp as one arm here, and it differs in the align target:
# it pulls toward the student latent, and every arm of this card pulls toward
# the teacher. The label says that, and not an issue number, because a number
# tells a reader nothing about what the marker is.
EARLIER = {"colour": "0.45", "marker": "^",
           "label": "published, same depth, the align loss on the student"}

# The arm whose backbone lost the contrastive task. Red, off the line, and out
# of every mean and every bar.
#
# THE LABEL NAMES THE MEASURED AUC, NOT "CHANCE". Chance is 0.50 and no run of
# this study reached it: the one that fell ends at 0.5745, and the study's own
# line for a collapse is 0.80. `fell_label` reads the number off the run.
FELL = {"colour": "#d62728", "marker": "X",
        "label": "the contrastive AUC fell while the backbone trained"}


def fell_label(rows=()) -> str:
    """What a legend calls the runs in `rows`, with the AUC they reached.

    Each row carries its own `auc`, read at the stop. With no AUC on any row
    the generic label stands, because a number no file backs is worse than no
    number.
    """
    aucs = [r["auc"] for r in rows if r.get("auc") is not None]
    if not aucs:
        return FELL["label"]
    return (f"the contrastive AUC fell to {min(aucs):.2f} "
            f"while the backbone trained")


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
                         "align_w": float(r.get("align_w") or DEFAULT_ALIGN_W),
                         "score": float(r["score"])})
    return sorted(rows, key=lambda r: (r["schedule"], r["alpha"]))


def points_of(rows, schedule, ramp,
              align_w=DEFAULT_ALIGN_W) -> list[tuple[float, float, float, float]]:
    """`(alpha, mean score, low, high)` per momentum, for ONE series.

    A series is a schedule, a ramp length and an align weight together, so the
    rows this averages differ only in their backbone seed. A momentum that one
    arm holds gives low equal to high. A repeat pair gives the two scores as
    the bar's ends.
    """
    by_alpha = {}
    for r in rows:
        if (r["schedule"] == schedule and r["align_w"] == align_w
                and (schedule == "fixed" or r["ramp"] == ramp)):
            by_alpha.setdefault(r["alpha"], []).append(r["score"])
    out = []
    for alpha in sorted(by_alpha):
        s = by_alpha[alpha]
        out.append((alpha, sum(s) / len(s), min(s), max(s)))
    return out


def draw_series(ax, rows, schedule, ramp, align_w, style):
    """One line, one marker per momentum, and a bar over a repeat pair."""
    pts = points_of(rows, schedule, ramp, align_w)
    if not pts:
        return 0
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
    for schedule, ramp, align_w, style in series_of(rows):
        draw_series(ax, rows, schedule, ramp, align_w, style)
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
