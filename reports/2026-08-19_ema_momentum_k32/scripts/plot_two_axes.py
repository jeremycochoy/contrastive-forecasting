#!/usr/bin/env python3
"""The two axes of the EMA schedule, one figure for each.

The schedule has two numbers: the momentum it starts at, and the number of
steps it takes to reach 1.0. The card moved both. Neither moves the score in
one direction.

`--by start` holds the ramp length and moves the start.
`--by ramp` holds the start and moves the ramp length.

A point is the mean over the backbone seeds of one arm. A bar joins its lowest
and its highest seed. An arm with one seed has no bar.

A collapsed backbone is not in any mean or any bar. Its score says what a dead
backbone scores.

The arm with align weight 3 is not here. It moves a third number, so it belongs
to neither axis.

Usage:
  plot_two_axes.py --by start --scores results/scores.csv --out plots/by_start.png
  plot_two_axes.py --by ramp  --scores results/scores.csv --out plots/by_ramp.png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import defaultdict
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
SEEDS = _load("cf404_seeds", "seed_report.py")
MOM = _load("cf404_momentum", "plot_momentum.py")

# Ramp lengths, fastest first. 0 means the momentum never rises.
RAMP_ORDER = (60000, 100000, 200000, 0)
RAMP_NAME = {60000: "over 60k steps", 100000: "over 100k steps",
             200000: "over 200k steps", 0: "never rises"}
COLOURS = ("#1f77b4", "#d95f02", "#7570b3", "#2ca02c", "#e7298a")


def cells(rows):
    """`{(start, ramp): [score, ...]}` over the seeds of each arm."""
    out = defaultdict(list)
    for r in rows:
        if float(r.get("align_w", 1.0)) != 1.0:
            continue
        ramp = r["ramp"] if r["schedule"] == "ramp" else 0
        out[(r["alpha"], ramp)].append(r["score"])
    return out


def draw(rows, out, by):
    grid = cells(rows)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))

    if by == "start":
        xs = sorted({k[0] for k in grid})
        series = [r for r in RAMP_ORDER if any(k[1] == r for k in grid)]
        key = lambda x, s: (x, s)  # noqa: E731
        name = lambda s: ("the momentum never rises" if s == 0 else
                          f"the momentum rises to 1.0 {RAMP_NAME[s]}")  # noqa: E731
        xlabel = "the momentum the schedule starts at"
        title = ("The start of the schedule against the score\n"
                 "one line per ramp length, at 40,000 backbone steps")
        ticks, tick_labels = xs, [f"{x:g}" for x in xs]
    else:
        xs = [r for r in RAMP_ORDER if any(k[1] == r for k in grid)]
        series = sorted({k[0] for k in grid})
        key = lambda x, s: (s, x)  # noqa: E731
        name = lambda s: f"the momentum starts at {s:g}"  # noqa: E731
        xlabel = "the steps the momentum takes to reach 1.0"
        title = ("The length of the ramp against the score\n"
                 "one line per start value, at 40,000 backbone steps")
        ticks = list(range(len(xs)))
        tick_labels = [RAMP_NAME[r].replace("over ", "").replace(" steps", "")
                       for r in xs]

    for i, s in enumerate(series):
        px, py, lo, hi = [], [], [], []
        for j, x in enumerate(xs):
            vals = grid.get(key(x, s))
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            px.append(x if by == "start" else j)
            py.append(mean)
            lo.append(mean - min(vals))
            hi.append(max(vals) - mean)
        if not px:
            continue
        ax.errorbar(px, py, yerr=[lo, hi], marker="o", markersize=8,
                    linewidth=1.8, capsize=5, elinewidth=1.6,
                    color=COLOURS[i % len(COLOURS)], zorder=3, label=name(s))
        # Several series can meet on one x, and one offset then prints
        # every label over the one before it. Each series takes its own.
        dy = (9, -15, 21, -27, 33)[i % 5]
        for x, y in zip(px, py):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, dy), fontsize=7.5,
                        color=COLOURS[i % len(COLOURS)], ha="center")

    ax.axhline(REF.K3_BB40K, color="0.35", linewidth=1.3, zorder=1)
    ax.axhline(REF.K0_PARENT_BB40K, color="0.35", linestyle="--",
               linewidth=1.3, zorder=1)
    x_text = ax.get_xlim()[0]
    ax.text(x_text, REF.K3_BB40K, f" k = 3, same 40,000 steps "
            f"({REF.K3_BB40K:.4f})", fontsize=8, color="0.20", va="bottom")
    ax.text(x_text, REF.K0_PARENT_BB40K, f" the k = 0 parent of this cell "
            f"({REF.K0_PARENT_BB40K:.4f})", fontsize=8, color="0.20",
            va="bottom")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("GM-Relative MASE over 97 configs, lower is better")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=2, framealpha=0.9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(series)} series")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--by", choices=("start", "ramp"), required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sync-root")
    p.add_argument("--stop", type=int, default=40000)
    args = p.parse_args(argv)
    if not Path(args.scores).is_file():
        raise SystemExit(f"ABORT: no scores table at {args.scores}")
    rows = MOM.read_scores(args.scores)
    if args.sync_root:
        root = Path(args.sync_root).expanduser()
        rows = [r for r in rows
                if not SEEDS.collapsed(SEEDS.auc_at(root, r["arm"], args.stop))]
    draw(rows, args.out, args.by)
    return 0


if __name__ == "__main__":
    sys.exit(main())
