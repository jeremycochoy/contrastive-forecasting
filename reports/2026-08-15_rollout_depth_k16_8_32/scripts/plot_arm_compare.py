#!/usr/bin/env python3
"""#401 — the figure that joins the mean arm to the summed arm.

The card's question in one picture: the same cell, the same depths, the same
stops, the two reductions over the k + 1 rollout-depth copies. GM-Relative MASE
against backbone train step, lower better.

  solid    this protocol, the MEAN over the copies.
  dashed   the stopped comparison arm, the SUM. Its k = 16 has no partner,
           and it keeps its hue.

Colour carries the DEPTH, the line style carries the ARM, and every line end is
direct-labelled — so no identity rides on colour alone (see depth_colours.py,
where the three hues are validated as a categorical palette against this
surface).

The vertical axis is LOGARITHMIC, and both panels share it. The two arms sit
an order of magnitude apart: the summed cells span 1.79 to 12.48 and the mean
cells span 1.16 to 1.33. On a linear axis every mean cell falls into one flat
line at the bottom, and the 0.16 that separates k = 8 mean from k = 32 mean is
1% of the axis. On a log axis equal vertical distances are equal ratios, so
the mean arm keeps its own shape beside an arm ten times worse.

It draws whatever each arm has scored. The summed arm holds 8 cells; the mean
arm fills its own in over days, and a panel with one arm on it is the normal
first view. Only two empty tables are refused.

Reads the two `scores.csv` tables `collect.sh` writes, one per arm.

Usage: plot_arm_compare.py [--sum results/scores.csv] \\
           [--mean results/mean/scores.csv] \\
           --out plots/mean/arm_compare.png
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.lines import Line2D                         # noqa: E402
from matplotlib.ticker import (FuncFormatter, LogLocator,   # noqa: E402
                               NullLocator)

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))
import depth_colours as D                                   # noqa: E402

HEAD = "student"
STOPS_K = [40, 100, 200]
PHASE_TITLE = {1: "phase 1 — head at 30k steps on every stop",
               2: "phase 2 — head budget = backbone steps"}

# The arms in drawing order: the comparison arm first, this protocol on top.
ARMS_ORDER = ("sum", "mean")

# The line style of each arm. Solid is this protocol, the same rule the ladder
# figure follows: solid for what this study trains.
ARM_STYLE = {"mean": D.STYLE_STUDY, "sum": (0, (5, 2))}
ARM_LABEL = {"mean": "mean over the k + 1 copies (this protocol)",
             "sum": "sum over the k + 1 copies (the stopped arm)"}

# The ticks of the log axis, as multiples of a decade. A plain log axis labels
# 1 and 10 and nothing between, and every mean cell of this figure sits
# between 1.16 and 1.33 — inside one unlabelled gap.
LOG_SUBS = (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0)


def read_scores(path):
    """`{phase: {k: {stop_k: score}}}` from one arm's table.

    A missing file is an empty arm: the mean arm's table does not exist until
    its first head lands.
    """
    out = {}
    if not path or not Path(path).is_file():
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["encoder"] != HEAD:
                continue
            stop = int(r["stop"])
            stop_k = stop // 1000 if stop % 1000 == 0 else stop
            (out.setdefault(int(r["phase"]), {})
                .setdefault(int(r["k"]), {})[stop_k]) = float(r["score"])
    return out


def axis_stops(arms):
    """Every stop either arm holds, in order. The card's three when it has them."""
    seen = sorted({s for scores in arms.values()
                   for panels in scores.values()
                   for pts in panels.values() for s in pts})
    return [s for s in STOPS_K if s in seen] or seen


def draw_panel(ax, phase, arms, xs, stops_k):
    """One phase. Both arms, one line per (arm, depth) the table holds."""
    values, ends = [], []
    for arm in ARMS_ORDER:
        panel = arms[arm].get(phase, {})
        for k in D.DEPTHS_DRAWN:
            pts = {s: v for s, v in panel.get(k, {}).items() if s in xs}
            if not pts:
                continue
            ss = sorted(pts)
            col = D.colour(k)
            ax.plot([xs[s] for s in ss], [pts[s] for s in ss], color=col,
                    linestyle=ARM_STYLE[arm], lw=2.0, marker="o", ms=5.0,
                    mec="white", mew=0.9, zorder=4 if arm == "mean" else 3)
            values += list(pts.values())
            ends.append((xs[ss[-1]], pts[ss[-1]], k, arm, col))

    ax.set_title(PHASE_TITLE[phase], loc="left", fontsize=10.5, color=D.INK)
    ax.set_xticks([xs[s] for s in stops_k])
    ax.set_xticklabels([f"{s}k" if s in STOPS_K else str(s) for s in stops_k])
    ax.set_xlim(-0.14, len(stops_k) - 1 + 0.72)
    ax.set_xlabel("backbone train step")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=LOG_SUBS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", color=D.GRID, lw=0.8)
    ax.set_axisbelow(True)
    return values, ends


def spread(ys, gap):
    """Push overlapping end labels apart, keeping their order."""
    orders = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for n, i in enumerate(orders[1:], start=1):
        prev = out[orders[n - 1]]
        if out[i] - prev < gap:
            out[i] = prev + gap
    return out


def label_ends(ax, ends, lo, hi):
    """`k = N mean` / `k = N sum` at each line end, in ink, on a coloured
    leader. Two arms share a hue, so the label names the arm too.

    The axis is logarithmic, so the labels are spread in LOG space. Spread in
    linear space, the fixed gap that separates two labels at 12 would leave
    the four labels between 1.16 and 1.33 on top of each other.
    """
    gap = (math.log10(hi) - math.log10(lo)) * 0.040
    ys = [10 ** y for y in spread([math.log10(e[1]) for e in ends], gap)]
    for (x, y, k, arm, col), yy in zip(ends, ys):
        ax.annotate(f"{D.label(k)} {arm}", (x, y), xytext=(x + 0.13, yy),
                    textcoords="data", fontsize=8.5, color=D.INK,
                    va="center", ha="left",
                    fontweight="bold" if arm == "mean" else "normal",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.4,
                                    shrinkA=2, shrinkB=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sum", default=str(STUDY / "results" / "scores.csv"))
    ap.add_argument("--mean",
                    default=str(STUDY / "results" / "mean" / "scores.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    arms = {"sum": read_scores(a.sum), "mean": read_scores(a.mean)}
    if not any(arms.values()):
        raise SystemExit(
            f"ABORT: no {HEAD}-head score in either arm ({a.sum}, {a.mean})")

    plt.rcParams.update(D.rc())
    stops_k = axis_stops(arms)
    xs = {s: i for i, s in enumerate(stops_k)}
    phases = [p for p in (1, 2) if any(arms[arm].get(p) for arm in ARMS_ORDER)]

    fig, axes = plt.subplots(1, len(phases), figsize=(7.0 * len(phases), 5.6),
                             squeeze=False)
    axes = axes[0]
    values, per_panel = [], []
    for ax, phase in zip(axes, phases):
        v, ends = draw_panel(ax, phase, arms, xs, stops_k)
        values += v
        per_panel.append((ax, ends))

    # Room on a log axis is a RATIO, so the pad is taken in log space.
    llo, lhi = math.log10(min(values)), math.log10(max(values))
    pad = (lhi - llo) * 0.08 + 0.004
    bot, top = 10 ** (llo - pad), 10 ** (lhi + pad * 1.3)
    for ax, ends in per_panel:
        ax.set_ylim(bot, top)
        label_ends(ax, ends, bot, top)
    axes[0].set_ylabel("GM-Relative MASE, 97 GIFT-Eval configs "
                       "(log scale, lower is better)")

    # The legend names the arms this figure DREW. An entry for an arm with no
    # scored cell yet would read as a line the reader cannot find.
    drawn_arms = [arm for arm in ARMS_ORDER
                  if any(e[3] == arm for _, ends in per_panel for e in ends)]
    fig.legend(handles=[Line2D([], [], color=D.INK, linestyle=ARM_STYLE[arm],
                               lw=2.0, label=ARM_LABEL[arm])
                        for arm in drawn_arms],
               loc="lower center", ncol=len(drawn_arms), frameon=False,
               fontsize=9)
    fig.suptitle("Sum against mean over the k + 1 rollout-depth copies",
                 fontsize=12.5, color=D.INK)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(a.out)
    drawn = sum(len(e) for _, e in per_panel)
    print(f"wrote {a.out}  ({drawn} line(s) over {len(phases)} phase(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
