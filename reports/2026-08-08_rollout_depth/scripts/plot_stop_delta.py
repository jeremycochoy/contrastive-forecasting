#!/usr/bin/env python3
"""#373 — what the second 100,000 backbone steps buys, per cell per head.

One bar per (cell, head): GM-Relative MASE at bb200k minus the SAME cell's
own bb100k, over the same 97 configs. Negative is better.

This contrast is cleaner than the depth contrast beside it. The bb200k
backbone RESUMES the bb100k checkpoint, so the two arms share every one of
the first 100,000 steps, the head recipe and the eval. Only the second
100,000 steps differ. Nothing here crosses a machine inside the shared
prefix, so no bar is hatched.

Each bar carries the 95% percentile interval of the same paired
dataset-cluster bootstrap the rest of the report uses. It bounds the eval
sample, not run-to-run variance.

Colour carries nothing in this figure. The report's palette names a cell by
hue and holds four hues on purpose; this figure draws eight cells, so the
cell rides the axis label and every bar takes one neutral ink.

Reads results/stop_bootstrap.csv (`all` rows) and the score files.

Usage: plot_stop_delta.py --results results --out plots/stop_delta.png
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
from matplotlib.patches import Patch                   # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
from published import NOISE_BAND                       # noqa: E402

plt.rcParams.update(cc.rc())

BAR = "#4c5766"          # one neutral ink; hue names no cell in this figure


def score(res, cell, stop, head):
    p = Path(res) / f"score_{cell}_k3_bb{stop}k_{head}.txt"
    try:
        return float(p.read_text().strip())
    except (OSError, ValueError):
        return None


def load(res):
    """`[(cell, head, v100, v200, delta, lo, hi), ...]`, worst delta last."""
    ci = {}
    p = Path(res) / "stop_bootstrap.csv"
    if p.is_file():
        for r in csv.DictReader(open(p)):
            if r["subset"] != "all":
                continue
            # label is `<cell>_stop200v100_<head>`
            cell, _, head = r["label"].partition("_stop200v100_")
            ci[(cell, head)] = (float(r["ci_lo"]), float(r["ci_hi"]))

    rows = []
    for cell in ("A2", "A3", "A4", "B1", "B2", "B4", "B6", "B10"):
        for head in ("student", "teacher"):
            a, b = score(res, cell, 100, head), score(res, cell, 200, head)
            if a is None or b is None:
                continue
            lo, hi = ci.get((cell, head), (float("nan"), float("nan")))
            rows.append((cell, head, a, b, b - a, lo, hi))
    rows.sort(key=lambda r: r[4])
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = load(args.results)
    if not rows:
        raise SystemExit("ABORT: no cell holds both bb100k and bb200k")

    better = sum(1 for r in rows if r[4] < 0)
    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(rows) + 2.6))

    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.BAND, zorder=0)
    ax.axvline(0, color=cc.INK, lw=1.1, zorder=3)

    y = list(range(len(rows)))[::-1]
    for yi, (cell, head, a, b, d, lo, hi) in zip(y, rows):
        ax.barh(yi, d, height=0.62, color=BAR, zorder=2)
        if lo == lo:                                   # not NaN
            ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="none",
                        ecolor=cc.INK, elinewidth=1.3, capsize=4, zorder=4)
        far = hi if d >= 0 else lo
        pad = 0.004 if d >= 0 else -0.004
        ax.text(far + pad, yi, f"{a:.4f} → {b:.4f}   ({d:+.4f})",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=9, color=cc.INK)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  [{h}]" for c, h, *_ in rows])
    ax.set_xlabel("GM-Relative MASE, bb200k minus the same cell's bb100k   "
                  f"(97 configs, negative is better)")
    ax.set_title("The second 100,000 backbone steps, against the first: "
                 f"{better} of {len(rows)} improved")
    # Each bar's value label sits outside its whisker, so the axis needs a
    # label's width of room past the extreme interval on BOTH sides.
    ends = [v for r in rows for v in (r[4], r[5], r[6]) if v == v]
    lo_end, hi_end = min(ends + [0.0]), max(ends + [0.0])
    pad = 0.62 * (hi_end - lo_end)
    ax.set_xlim(lo_end - pad, hi_end + pad)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis="x", color=cc.GRID, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=[
        Line2D([0], [0], color=cc.INK, lw=1.3,
               label="95% CI, paired dataset-cluster bootstrap"),
        Patch(facecolor=cc.BAND, label=f"head-seed band ±{NOISE_BAND}"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.055 - 4.0 / len(rows) / 14),
        ncol=2, frameon=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  {args.out}  ({len(rows)} contrast(s), {better} improved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
