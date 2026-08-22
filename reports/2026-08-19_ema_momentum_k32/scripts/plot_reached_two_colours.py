#!/usr/bin/env python3
"""The score against the momentum reached at the stop, in two colours.

One colour holds the momentum for the whole run. The other colour rises toward
1.0. The ramp length is not in the colour, so a reader compares the two kinds
of schedule and nothing else.

The x axis is the momentum the backbone trains against at the stop. An arm that
holds 0.9 and an arm that rises to 0.9 sit on the same x, and their colours
tell them apart.

A point is the mean over the backbone seeds of one arm. A bar joins its lowest
and its highest seed.

The arm with align weight 3 is not here. It moves a third number, so it is
neither of the two colours.

A collapsed backbone takes a red mark on the top edge, out of every mean and
every bar.

Usage:
  plot_reached_two_colours.py --scores results/scores.csv \
      --out plots/reached_two_colours.png --sync-root ~/cf404_sync
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
AT = _load("cf404_at_stop", "plot_momentum_at_stop.py")

KIND = {
    "fixed": {"colour": "#1f77b4", "marker": "o",
              "label": "the momentum holds one value"},
    "ramp": {"colour": "#d95f02", "marker": "s",
             "label": "the momentum rises toward 1.0"},
}


def cells(rows, stop):
    """`{(kind, reached): [score, ...]}`, seeds of one arm together."""
    out = defaultdict(list)
    for r in rows:
        if float(r.get("align_w", 1.0)) != 1.0:
            continue
        x = AT.momentum_at(r["alpha"], r["schedule"], r["ramp"], stop)
        out[(r["schedule"], round(x, 4))].append(r["score"])
    return out


def require_cells(grid):
    """Stop with a message when no arm of the cell's own weight is scored.

    `make_plots.sh` redraws every 30 minutes, from the first hour of the
    study, and it prints the LAST line of a failed draw as its SKIP line. An
    empty grid otherwise reaches `min()` and that line reads
    `ValueError: min() arg is an empty sequence`.
    """
    if not grid:
        raise SystemExit(
            "ABORT: no arm at L_align weight 1.0 is scored yet — "
            "nothing to draw")
    return grid


def draw_vertical(rows, out, fell=(), stop=40000):
    """The reached momentum on the y axis, at its own value.

    The horizontal orientation of `draw` puts the momentum on x.
    This one puts it on y, so a reader reads the momentum down the
    side and the score across. The y axis stays numeric, so 0.967
    and 0.970 sit as close together as their values are.
    """
    grid = require_cells(cells(rows, stop))
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    for kind, style in KIND.items():
        pts = sorted((k[1], v) for k, v in grid.items()
                     if k[0] == kind)
        if not pts:
            continue
        py = [p[0] for p in pts]
        px = [sum(p[1]) / len(p[1]) for p in pts]
        lo = [m - min(p[1]) for p, m in zip(pts, px)]
        hi = [max(p[1]) - m for p, m in zip(pts, px)]
        ax.errorbar(px, py, xerr=[lo, hi], color=style["colour"],
                    marker=style["marker"], markersize=9,
                    linewidth=2.0, capsize=5, elinewidth=1.6,
                    zorder=3, label=style["label"])
        for x, y in zip(px, py):
            # The white box keeps the two reference lines out of the digits.
            ax.annotate(f"{x:.4f}", (x, y),
                        textcoords="offset points",
                        xytext=(0, 10), fontsize=8,
                        color=style["colour"], ha="center", zorder=6,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=0.9, alpha=0.85))
    for r in fell:
        y = AT.momentum_at(r["alpha"], r["schedule"], r["ramp"], stop)
        ax.plot([r["score"]], [y], linestyle="none",
                marker=MOM.FELL["marker"], markersize=11,
                color=MOM.FELL["colour"], zorder=4,
                label=MOM.FELL["label"])
    ax.axvline(REF.K3_BB40K, color="0.35", linewidth=1.3, zorder=1)
    ax.axvline(REF.K0_PARENT_BB40K, color="0.35", linestyle="--",
               linewidth=1.3, zorder=1)
    # The two lines sit 0.074 apart. Both labels at the top printed the k = 0
    # one straight through the best arm's own value, which is the number this
    # figure exists to show. So one labels at the top and one at the bottom.
    y_top = max([k[1] for k in grid]) + 0.012
    y_bottom = min([k[1] for k in grid]) - 0.012
    ax.text(REF.K3_BB40K, y_top,
            f" {REF.K3_LINE} ({REF.K3_BB40K:.4f})",
            fontsize=8, color="0.20", va="top", rotation=90)
    ax.text(REF.K0_PARENT_BB40K, y_bottom,
            f" {REF.K0_LINE} ({REF.K0_PARENT_BB40K:.4f})",
            fontsize=8, color="0.20", va="bottom", rotation=90)
    ax.set_yticks(sorted({k[1] for k in grid}))
    ax.set_yticklabels([f"{v:.3f}" for v in
                        sorted({k[1] for k in grid})], fontsize=9)
    ax.set_ylabel(f"the momentum the backbone trains against at "
                  f"{stop:,} steps")
    ax.set_xlabel("GM-Relative MASE over 97 configs, "
                  "lower is better")
    ax.set_title("A held momentum and a rising momentum,\n"
                 f"against the value each reaches at {stop:,} steps")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(grid)} point(s), {len(fell)} collapsed")
    return fig, ax


def draw(rows, out, fell=(), stop=40000):
    grid = require_cells(cells(rows, stop))
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    xs_all = [k[1] for k in grid] + [
        AT.momentum_at(r["alpha"], r["schedule"], r["ramp"], stop)
        for r in fell]
    x_lo, x_hi = min(xs_all) - 0.015, max(xs_all) + 0.015
    MOM.draw_references(ax, x_lo, x_hi)

    for kind, style in KIND.items():
        pts = sorted((k[1], v) for k, v in grid.items() if k[0] == kind)
        if not pts:
            continue
        px = [p[0] for p in pts]
        py = [sum(p[1]) / len(p[1]) for p in pts]
        lo = [m - min(p[1]) for p, m in zip(pts, py)]
        hi = [max(p[1]) - m for p, m in zip(pts, py)]
        ax.errorbar(px, py, yerr=[lo, hi], color=style["colour"],
                    marker=style["marker"], markersize=9, linewidth=2.0,
                    capsize=5, elinewidth=1.6, zorder=3, label=style["label"])
        dy = 11 if kind == "fixed" else -17
        for x, y in zip(px, py):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, dy), fontsize=8, color=style["colour"],
                        ha="center")

    y_lo, y_hi = MOM.y_range(rows)
    if fell:
        top = y_hi - 0.02 * (y_hi - y_lo)
        fx = [AT.momentum_at(r["alpha"], r["schedule"], r["ramp"], stop)
              for r in fell]
        ax.plot(fx, [min(r["score"], top) for r in fell], linestyle="none",
                marker=MOM.FELL["marker"], markersize=11,
                color=MOM.FELL["colour"], zorder=4, label=MOM.FELL["label"])

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_lo, x_hi)
    # One tick per reached value, but 0.967 and 0.970 print over each other.
    # A tick nearer than `gap` to the tick before it is dropped. The point and
    # its score label stay, and the axis says which side of the neighbour the
    # point is on.
    ticks, gap = [], 0.006
    for v in sorted({round(v, 3) for v in xs_all}):
        if ticks and v - ticks[-1] < gap:
            continue
        ticks.append(v)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.3f}" for v in ticks], fontsize=8)
    ax.set_xlabel(f"the momentum the backbone trains against at {stop:,} steps")
    ax.set_ylabel("GM-Relative MASE over 97 configs, lower is better")
    ax.set_title("A held momentum and a rising momentum, against the value\n"
                 f"each one reaches at {stop:,} steps")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=1, framealpha=0.9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(grid)} point(s), {len(fell)} collapsed")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sync-root")
    p.add_argument("--stop", type=int, default=40000)
    p.add_argument("--vertical", action="store_true",
                   help="the reached momentum on the y axis")
    args = p.parse_args(argv)
    if not Path(args.scores).is_file():
        raise SystemExit(f"ABORT: no scores table at {args.scores}")
    rows = MOM.read_scores(args.scores)
    fell = []
    if args.sync_root:
        root = Path(args.sync_root).expanduser()
        alive = []
        for r in rows:
            auc = SEEDS.auc_at(root, r["arm"], args.stop)
            (fell if SEEDS.collapsed(auc) else alive).append(r)
        rows = alive
    (draw_vertical if args.vertical else draw)(
        rows, args.out, fell, args.stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
