#!/usr/bin/env python3
"""#373 — the EMA momentum α that each group actually trained under.

The card asks for the parent report's `alpha_schedule.png`, kept for the
group-A cells, with a statement that group B holds α at 0.9. The parent drew
the schedule it configured. This draws the `ema_tau` column every backbone
run logged, so the figure is a measurement and not a restatement of the
launcher.

Group membership comes from the launcher recipe in the file name, not from
the curve: `cf393_*` is the scheduled launcher, `bb_small_*` the fixed one.
The curve then says what each launcher did.

Every backbone leg is drawn. The legs of one cell resume each other, so a
cell's α is a single line across its stops.

Head CSVs carry no `ema_tau` and are skipped.

Usage: plot_alpha_schedule.py --curves curves --curves sync \\
           --out plots/alpha_schedule.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402

plt.rcParams.update(cc.rc())

STOPS = [40_000, 100_000, 200_000]
# Launcher recipe -> group. `cells.tsv` names the same split per cell.
GROUPS = [("cf393", "A", "group A: α scheduled 0.9 → 1.0 by step 100k",
           cc.PALETTE[0]),
          ("bb_small", "B", "group B: α held at 0.9", cc.PALETTE[1])]


def group_of(name):
    for token, g, _lab, _col in GROUPS:
        if token in name:
            return g
    return None


def read_alpha(path):
    """`(steps, alphas)` from one backbone losses CSV, or `([], [])`."""
    xs, ys = [], []
    with open(path) as fh:
        r = csv.DictReader(fh)
        if "ema_tau" not in (r.fieldnames or []):
            return [], []
        for row in r:
            try:
                xs.append(int(float(row["step"])))
                ys.append(float(row["ema_tau"]))
            except (TypeError, ValueError):
                continue
    return xs, ys


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--curves", action="append", required=True,
                   help="a directory to walk for *_losses.csv")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    tracks = {"A": [], "B": []}
    for root in args.curves:
        for f in sorted(Path(root).rglob("*_losses.csv")):
            if "qhead" in f.name or f.name.startswith("eval__"):
                continue
            g = group_of(f.name)
            if g is None:
                continue
            xs, ys = read_alpha(f)
            if xs:
                tracks[g].append((xs, ys))
    if not any(tracks.values()):
        raise SystemExit("ABORT: no backbone losses CSV carries ema_tau")

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for _token, g, lab, col in GROUPS:
        for xs, ys in tracks[g]:
            ax.plot(xs, ys, color=col, lw=1.8, alpha=0.75, solid_capstyle="butt")
    for s in STOPS:
        ax.axvline(s, color=cc.INK_SOFT, lw=0.9, ls=(0, (4, 3)), zorder=0)
        ax.text(s, 1.004, f"bb{s // 1000}k", ha="center", va="bottom",
                fontsize=8, color=cc.INK_SOFT)

    ax.set_xlabel("backbone step")
    ax.set_ylabel("EMA momentum α")
    ax.set_xlim(0, 205_000)
    ax.set_ylim(0.888, 1.012)
    ax.grid(color=cc.GRID)
    ax.set_axisbelow(True)
    ax.set_title("EMA momentum against training step, as each run logged it",
                 fontsize=11)
    ax.legend(handles=[Line2D([], [], color=col, lw=1.8,
                              label=f"{lab}  ({len(tracks[g])} leg(s))")
                       for _t, g, lab, col in GROUPS],
              frameon=False, loc="center right", fontsize=9)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  {args.out}  (A: {len(tracks['A'])} leg(s), "
          f"B: {len(tracks['B'])} leg(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
