#!/usr/bin/env python3
"""Every run of the card, one row per arm, ordered by score.

WHY THIS FIGURE EXISTS. The momentum figures put the momentum on the x axis.
That axis stacks four runs on one tick when several arms reach one momentum by
different routes, and a reader cannot then find one arm.

This figure asks the simpler question: which arm scores best, and how far do
its own seeds move? One row is one arm. One dot is one run. A bar joins the
lowest and the highest seed of an arm that holds more than one.

An arm whose backbone fell to chance keeps its dot, in red, because a reader
must see that the run happened and what a dead backbone scores.

Usage:
  plot_arm_ranking.py --scores results/scores.csv --out plots/arm_ranking.png \
      --sync-root ~/cf404_sync
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

# One colour per kind of schedule, the same two the reached-value figure
# uses. The ramp length is not in the colour.
KIND_COLOUR = {"fixed": "#1f77b4", "ramp": "#d95f02"}
KIND_NAME = {"fixed": "the momentum holds one value",
             "ramp": "the momentum rises toward 1.0"}


def arm_label(r) -> str:
    """What the reader sees for one arm. No internal name, no issue number."""
    if r["schedule"] != "ramp" or not r["ramp"]:
        base = f"{r['alpha']:g} held"
    else:
        base = f"{r['alpha']:g} rises to 1.0 over {r['ramp'] // 1000:g}k"
    if float(r.get("align_w", 1.0)) != 1.0:
        base += f", align weight {float(r['align_w']):g}"
    reached = AT.momentum_at(r["alpha"], r["schedule"], r["ramp"], 40000)
    return f"{base}  (reaches {reached:.3f})"


def group(rows):
    """`{label: [row, ...]}`, one entry per arm, seeds together."""
    out = defaultdict(list)
    for r in rows:
        out[arm_label(r)].append(r)
    return out


def draw(rows, out, fell=()):
    arms = group(list(rows) + list(fell))
    # The best seed of an arm sets its place, so a reader reads top to bottom.
    order = sorted(arms, key=lambda k: min(r["score"] for r in arms[k]))
    fig, ax = plt.subplots(figsize=(10.0, 0.52 * len(order) + 2.6))

    # The two reference values sit 0.074 apart, and their labels overlap at
    # one height. So one labels at the top of the axes and one at the bottom.
    ax.axvline(REF.K3_BB40K, color="0.35", linewidth=1.3, zorder=1)
    ax.text(REF.K3_BB40K, len(order) - 0.35,
            f" k = 3, same 40,000 steps ({REF.K3_BB40K:.4f})",
            fontsize=8, color="0.20", va="top")
    ax.axvline(REF.K0_PARENT_BB40K, color="0.35", linestyle="--",
               linewidth=1.3, zorder=1)
    ax.text(REF.K0_PARENT_BB40K, -0.6,
            f" the k = 0 parent of this cell ({REF.K0_PARENT_BB40K:.4f})",
            fontsize=8, color="0.20", va="bottom")

    fell_ids = {id(r) for r in fell}
    for y, label in enumerate(reversed(order)):
        runs = arms[label]
        alive = [r for r in runs if id(r) not in fell_ids]
        dead = [r for r in runs if id(r) in fell_ids]
        if len(alive) > 1:
            lo, hi = min(r["score"] for r in alive), max(r["score"] for r in alive)
            colour = KIND_COLOUR[runs[0]["schedule"]]
            ax.plot([lo, hi], [y, y], color=colour, linewidth=2.4,
                    zorder=2)
            ax.text(hi + 0.006, y, f"range {hi - lo:.4f}", fontsize=7.5,
                    color=colour, va="center")
        for r in alive:
            ax.plot([r["score"]], [y], marker="o", markersize=7,
                    color=KIND_COLOUR[r["schedule"]], zorder=3)
        for r in dead:
            ax.plot([r["score"]], [y], marker="X", markersize=9,
                    color="#d62728", zorder=3)
        best = min(r["score"] for r in alive) if alive else None
        if best is not None:
            ax.text(best - 0.006, y, f"{best:.4f}", fontsize=7.5,
                    color="0.25", va="center", ha="right")

    handles = [plt.Line2D([], [], marker="o", linestyle="-",
                          color=KIND_COLOUR[k], label=KIND_NAME[k])
               for k in ("fixed", "ramp")]
    ax.legend(handles=handles, fontsize=8, loc="lower right",
              framealpha=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=9)
    ax.set_xlabel("GM-Relative MASE over 97 configs, lower is better")
    ax.set_title("Every run of the card, by arm\n"
                 "one dot is one backbone seed, the bar is the seed range")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.7, len(order) - 0.2)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out} — {len(order)} arm(s), {len(fell)} collapsed")
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sync-root")
    p.add_argument("--stop", type=int, default=40000)
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
    draw(rows, args.out, fell)
    return 0


if __name__ == "__main__":
    sys.exit(main())
