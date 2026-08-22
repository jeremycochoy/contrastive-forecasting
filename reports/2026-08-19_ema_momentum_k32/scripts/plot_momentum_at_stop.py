#!/usr/bin/env python3
"""The score of every arm against the momentum it HOLDS at the stop.

WHY THIS FIGURE EXISTS. `momentum.png` puts the momentum at step 0 on the x
axis. That axis separated every arm while every ramp ran 200,000 steps. It
stopped separating them when the card added a 100,000-step ramp: `s08` and
`r100_08` both start at 0.8, and `s09` and `r100_09` both start at 0.9. Two
arms then sit on one tick.

The momentum they REACH does separate them. At the 40,000-step stop:

  a08   0.800    s08     0.840    r100_08  0.880
  a09   0.900    s09     0.920    r100_09  0.940    a095  0.950

So this figure asks the card's question directly: of the momentum values a
backbone actually trains against at the stop, which one scores best? A reader
follows one row of x values, left to right, and reads the score off each.

The marker still names the schedule, so a reader sees whether a value was
reached by holding it or by rising to it. Two arms can reach one value by two
routes, and the figure does not merge them.

A collapsed arm keeps the treatment `momentum.png` gives it: a red X on the
top edge with its score in text, out of every mean and every line.
`seed_report.py` holds the study's one definition of a collapse.

Usage:
  plot_momentum_at_stop.py --scores results/scores.csv \
      --out plots/momentum_at_stop.png --sync-root ~/cf404_sync/box_a/sync
"""
from __future__ import annotations

import argparse
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
SEEDS = _load("cf404_seeds", "seed_report.py")
MOM = _load("cf404_momentum", "plot_momentum.py")


def momentum_at(alpha: float, schedule: str, ramp: int, step: int) -> float:
    """The momentum an arm holds at `step`.

    The same formula as `src.models.ema_tau_at_step` and as
    `cf404_momentum_at` in `study.sh`: linear over the ramp, clamped at both
    ends. `scripts/test_momentum_at.sh` holds the shell copy against the
    trainer's. This copy exists so a figure needs no import of `src`.
    """
    if schedule != "ramp" or not ramp:
        return float(alpha)
    frac = min(max(step / ramp, 0.0), 1.0)
    return float(alpha) + frac * (1.0 - float(alpha))


def place(rows, stop):
    """Each row with its momentum at the stop, as `x`."""
    for r in rows:
        r["x"] = momentum_at(r["alpha"], r["schedule"], r["ramp"], stop)
    return rows


def points_of(rows, schedule, ramp, align_w=MOM.DEFAULT_ALIGN_W):
    """`(x, mean, low, high)` per reached momentum, for one series.

    Rows that share a series AND a reached momentum differ only in their
    backbone seed, so their range is a repeat spread and the bar means the
    same thing it means on `momentum.png`. The align weight is in the series
    key, so `s08` and `w3_s08` reach one momentum and stay two points.
    """
    by_x = {}
    for r in rows:
        if (r["schedule"] == schedule and r["align_w"] == align_w
                and (schedule == "fixed" or r["ramp"] == ramp)):
            by_x.setdefault(r["x"], []).append(r["score"])
    return [(x, sum(s) / len(s), min(s), max(s))
            for x, s in sorted(by_x.items())]


def draw(rows, out, fell=(), stop=40000):
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    xs_all = [r["x"] for r in rows] + [r["x"] for r in fell]
    x_lo, x_hi = min(xs_all) - 0.02, max(xs_all) + 0.02
    MOM.draw_references(ax, x_lo, x_hi)

    for schedule, ramp, align_w, style in MOM.series_of(rows):
        pts = points_of(rows, schedule, ramp, align_w)
        if not pts:
            continue
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[[p[1] - p[2] for p in pts],
                          [p[3] - p[1] for p in pts]],
                    color=style["colour"], marker=style["marker"],
                    markersize=8, linewidth=1.8, capsize=5, elinewidth=1.6,
                    zorder=3, label=style["label"])

    y_lo, y_hi = MOM.y_range(rows)
    if fell:
        top = y_hi - 0.02 * (y_hi - y_lo)
        ax.plot([r["x"] for r in fell], [min(r["score"], top) for r in fell],
                linestyle="none", marker=MOM.FELL["marker"], markersize=11,
                color=MOM.FELL["colour"], zorder=4, label=MOM.FELL["label"])
        for r in fell:
            if r["score"] > top:
                ax.annotate(f"{r['score']:.4f}", (r["x"], top),
                            textcoords="offset points", xytext=(12, -2),
                            fontsize=8, color=MOM.FELL["colour"], va="center")

    # Every arm carries its score, not its internal name. The x axis already
    # gives the momentum the arm holds, and the legend already gives the
    # schedule. A name like `r100_09` adds nothing a reader can use.
    for r in sorted(rows, key=lambda r: r["x"]):
        ax.annotate(f"{r['score']:.4f}", (r["x"], r["score"]),
                    textcoords="offset points", xytext=(0, 9),
                    fontsize=7.5, color="0.25", ha="center")

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_lo, x_hi)
    # One tick per arm, but a tick that sits on its neighbour is unreadable.
    # `r60_09` holds 0.967 and `r100_095` holds 0.970, and the two labels
    # printed over each other. So a tick that is nearer than `tick_gap` to the
    # tick before it is dropped: the point stays, its score label stays, and
    # the axis says which side of the neighbour it is on.
    ticks, tick_gap = [], 0.006
    for x in sorted({round(v, 3) for v in xs_all}):
        if ticks and x - ticks[-1] < tick_gap:
            continue
        ticks.append(x)
    ax.set_xticks(ticks)
    ax.set_xlabel(f"EMA momentum the arm holds at {stop:,} steps")
    ax.set_ylabel("GM-Relative MASE over 97 configs, lower is better")
    ax.set_title("The score against the momentum the backbone actually\n"
                 f"trains against at the {stop:,}-step stop, at rollout "
                 "depth 32")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.09), ncol=1, framealpha=0.9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(rows)} arm(s), {len(fell)} collapsed")
    return fig, ax


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
    rows = place(MOM.read_scores(args.scores), args.stop)
    fell = []
    if args.sync_root:
        root = Path(args.sync_root).expanduser()
        alive = []
        for r in rows:
            auc = SEEDS.auc_at(root, r["arm"], args.stop)
            (fell if SEEDS.collapsed(auc) else alive).append(r)
        rows = alive
    draw(rows, args.out, fell, args.stop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
