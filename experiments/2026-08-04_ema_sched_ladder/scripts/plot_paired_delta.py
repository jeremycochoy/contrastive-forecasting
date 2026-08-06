#!/usr/bin/env python3
"""#393 — the bb40k-to-bb100k delta once BOTH of its ends carry a spread.

Reads `results/paired_delta.csv` (scripts/paired_delta.py) and draws the two things the extend rule could
not see.

Only the rows that carry two or three head seeds are drawn: they are the
rows the paired comparison can test. Each carries mean ± t(df, .05)·SE, with
df printed per row, because df is 1 for the two-seed row and 2 for the
three-seed rows. Rows with one seed are measured, not tested, and are listed
in `results/paired_delta.csv` instead.

Each small marker is one head seed's own `bb100k(s) - bb40k(s)`. The extend
rule is a strict `<`, so what it reads is exactly the SIGN. An interval
crossing zero is a branch the head seed alone can flip.

Colour is the run, from `scripts/run_colours.py`, the same colour the run
carries in every other figure of the report. Head is the marker shape.
Whether the sign holds at every seed is printed in the row label, next to df.

Usage:  python3 scripts/plot_paired_delta.py [--paired FILE] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import run_colours  # noqa: E402
from cell_label import label as cell_label  # noqa: E402

INK = run_colours.INK
INK_SOFT = run_colours.INK_SOFT
HEAD_MARKER = run_colours.HEAD_MARKER
SEEDS = ["20260722", "20260723", "20260724"]


def num(row: dict, key: str):
    v = (row.get(key) or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    res = os.path.join(EXP_DIR, "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--paired", default=os.path.join(res, "paired_delta.csv"))
    p.add_argument("--out", default=os.path.join(EXP_DIR, "plots",
                                                 "paired_delta.png"))
    a = p.parse_args()

    rows = [r for r in read_csv(a.paired)
            if num(r, "delta_mean") is not None and int(r["n_seeds"]) >= 2]
    if not rows:
        print("no paired delta rows yet; nothing to plot")
        return 0
    rank = {c: i for i, c in enumerate(run_colours.ORDER)}
    rows.sort(key=lambda r: (rank.get(r["cell"], len(rank)), r["head"]))
    labels = []
    for r in rows:
        sign = "same sign at every seed" if r["sign_stable"] == "yes" \
            else "sign flips on the head seed"
        labels.append(f"{cell_label(r['cell'], short=True)}  "
                      f"{r['head']}   [df={int(num(r, 'df') or 0)}, {sign}]")
    y = list(range(len(rows)))[::-1]

    fig, axL = plt.subplots(figsize=(13.0, 0.55 * len(rows) + 3.0))

    for yy, r in zip(y, rows):
        col = run_colours.colour(r["cell"])
        mk = HEAD_MARKER.get(r["head"], "o")
        mean = num(r, "delta_mean")
        t_crit = num(r, "t_crit_05") or 0.0
        axL.errorbar(mean, yy, xerr=t_crit * (num(r, "se_paired") or 0.0),
                     color=col, elinewidth=2.0, capsize=4, capthick=1.4,
                     zorder=2)
        for s_ in SEEDS:
            v = num(r, f"delta_{s_}")
            if v is not None:
                axL.plot(v, yy, mk, ms=4.5, mfc="none", mec=col, mew=1.0,
                         alpha=0.85, zorder=3)
        axL.plot(mean, yy, mk, ms=8, color=col, mec=col, mew=1.4, zorder=4)

    axL.axvline(0.0, color=INK, lw=1.0, zorder=1)
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=9)
    for tick, r in zip(axL.get_yticklabels(), rows):
        tick.set_color(run_colours.colour(r["cell"]))
    axL.set_xlabel("paired  bb100k \u2212 bb40k   (GM-Relative MASE, lower is better)")
    axL.set_title("The delta, both ends at the same head seed\n"
                  "interval = mean \u00b1 t(df, .05)\u00b7SE, df printed per row",
                  fontsize=10, color=INK)
    axL.grid(axis="x", color=INK_SOFT, alpha=0.35, lw=0.6)
    axL.set_axisbelow(True)

    axL.set_ylim(-0.8, len(rows) - 0.2)

    handles = [
        Line2D([], [], color=INK, marker="o", lw=0, label="student head"),
        Line2D([], [], color=INK, marker="s", lw=0, label="teacher head"),
        Line2D([], [], color=INK, marker="o", lw=0, mfc="none",
               label="one head seed's own delta (hollow)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3,
                     frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005),
                     title="one colour per run, the report's colour code")
    leg.get_title().set_fontsize(9)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"  -> {a.out}  ({len(rows)} row(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
