#!/usr/bin/env python3
"""#393 — the ladder as one picture: GM-Relative MASE against backbone step.

Reads `results/ladder_all.csv`, the pooled table, so it draws every machine's
scores and redraws itself as later stops land.

One panel: every run, both heads, against the backbone step. Colour is the
run and line style is the `L_align` target (`scripts/run_colours.py`); the
head is the marker, circle for the student encoder and square for the
teacher. Seasonal naive sits at 1.0 and nothing in this study is below it.
The legend sits outside the axes so it covers no data, and the y limit is
padded so no trace is clipped.

The per-stop change the extend rule reads is a table, not a panel here
(`results/per_stop_changes.csv`).

Usage:  python3 scripts/plot_ladder.py [--ladder FILE] [--spread FILE]
                                       [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import noise_band  # noqa: E402
import run_colours  # noqa: E402
from cell_label import label as cell_label  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


def read_scores(path: str) -> dict:
    """{cell: {head: [(stop, score), ...]}} sorted by stop."""
    out: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("gm_rel_mase") or "").strip():
                continue
            out.setdefault(r["cell"], {}).setdefault(r["head"], []).append(
                (int(r["stop"]), float(r["gm_rel_mase"])))
    for heads in out.values():
        for series in heads.values():
            series.sort()
    return out


def order(scores: dict) -> list[str]:
    """Runs in the report's fixed run order, which is the legend order."""
    return run_colours.in_order(scores)


def read_band(path: str):
    """(pooled head-seed band, set of cells that carry replicates).

    The band pools both head budgets (scripts/noise_band.py); `path` only
    names the cells that carry a replicate at all.
    """
    band = noise_band.pooled_band()
    if not os.path.exists(path):
        return band, set()
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("range") or "").strip()]
    return band, {r["cell"] for r in rows}


def draw(scores: dict, path: str, band=None, replicated=frozenset()) -> None:
    del band, replicated  # the change panel moved out; kept for the CLI
    cells = order(scores)

    fig, ax = plt.subplots(figsize=(10.4, 6.0))

    lo = hi = None
    for cell in cells:
        for head, series in sorted(scores[cell].items()):
            xs = [s / 1000 for s, _ in series]
            ys = [v for _, v in series]
            lo = min([lo] * bool(lo is not None) + ys)
            hi = max([hi] * bool(hi is not None) + ys)
            ax.plot(xs, ys, ms=4, lw=1.6, alpha=0.9,
                    label=cell_label(cell, short=True)
                    if head == "student" else None,
                    **run_colours.line_style(cell, head))
    if lo <= 1.0:
        ax.axhline(1.0, color="#333", lw=1.2)
        ax.annotate("seasonal naive", xy=(0.99, 1.0),
                    xycoords=("axes fraction", "data"), ha="right",
                    va="bottom", fontsize=8, color="#333")
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("backbone step (thousands)")
    ax.set_ylabel("GM-Relative MASE  (lower is better; "
                  "1.0 = seasonal naive)")
    ax.set_title("Every run, both heads", fontsize=10)
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color=run_colours.INK, marker="o", lw=0, ms=6,
               label="circle = student encoder"),
        Line2D([], [], color=run_colours.INK, marker="s", lw=0, ms=6,
               label="square = teacher encoder"),
    ]
    leg = ax.legend(handles=handles, fontsize=8, ncol=1, loc="upper left",
                    bbox_to_anchor=(1.01, 1.0), frameon=False,
                    title=run_colours.LEGEND_KEY.replace("   |   ", "\n"))
    leg.get_title().set_fontsize(8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    res = os.path.join(EXP_DIR, "results")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--ladder", default=os.path.join(res, "ladder_all.csv"))
    p.add_argument("--spread", default=os.path.join(res, "seed_spread.csv"))
    p.add_argument("--out", default=os.path.join(EXP_DIR, "plots",
                                                 "ladder.png"))
    a = p.parse_args()

    if not os.path.exists(a.ladder):
        print(f"plot_ladder: no {a.ladder}", file=sys.stderr)
        return 1
    scores = read_scores(a.ladder)
    if not scores:
        print(f"plot_ladder: no scored row in {a.ladder}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    band, replicated = read_band(a.spread)
    draw(scores, a.out, band, replicated)
    n = sum(len(s) for h in scores.values() for s in h.values())
    print(f"[out] {a.out}  ({len(scores)} cells, {n} scores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
