#!/usr/bin/env python3
"""#373 — the screen: every cell's k = 3 against its published k = 0.

One bar per cell: this study's GM-Relative MASE at bb100k, student head,
minus the number the cell's parent report published for k = 0 at the same
stop. Negative is better. bb100k is the stop all 14 cells reached.

This is a SCREEN and not a test. The two sides of every bar trained on
different machines, and this study's one controlled measurement of the
machine is worth 0.1166. The grey band is `ema_sched_ladder.md`'s pooled head-seed
band; it bounds the HEAD seed alone, and the backbone seed is unreplicated.

A1 and B3 run one arm that aligns to the student and passes no
`--moco-rep-keys`, so the EMA regime that separates them cannot reach the
student encoder. Their two backbones hold identical student weights, so the
two bars are one model against two published baselines, and the model count
in the title counts it once.

Colour carries nothing here: the report's palette holds four hues and this
figure draws fourteen cells, so the cell rides the axis label.

Reads results/published_bootstrap.csv (`all` rows) and the score files.

Usage: plot_screen_bb100k.py --results results --out plots/screen_bb100k.png
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
import published                                       # noqa: E402
import r2_ladder as L                                  # noqa: E402
from published import NOISE_BAND                       # noqa: E402

plt.rcParams.update(cc.rc())

BAR = "#4c5766"          # one neutral ink; hue names no cell in this figure
STOP = 100
HEAD = "student"
# The one pair of cells whose student encoders are the same weights.
SHARED = ("A1", "B3")


def load(res):
    """`[(cell, k3, pub, delta, lo, hi), ...]`, best delta first."""
    ci = {}
    p = Path(res) / "published_bootstrap.csv"
    if p.is_file():
        for r in csv.DictReader(open(p)):
            if r["subset"] != "all":
                continue
            cell, sep, rest = r["label"].partition("_vs_pub_")
            if not sep or rest != f"bb{STOP}k_{HEAD}":
                continue
            ci[cell] = (float(r["ci_lo"]), float(r["ci_hi"]))

    rows = []
    for cell in L.CELLS:
        k3 = L.score(cell, STOP, HEAD, res)
        pub = published.at(cell, HEAD, STOP)
        if k3 is None or pub is None:
            continue
        lo, hi = ci.get(cell, (float("nan"), float("nan")))
        rows.append((cell, k3, pub, k3 - pub, lo, hi))
    rows.sort(key=lambda r: r[3])
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"ABORT: no cell holds both bb{STOP}k and a baseline")

    # Count MODELS, not bars: the shared-student pair is one measurement.
    shared_here = [r for r in rows if r[0] in SHARED]
    models = len(rows) - max(0, len(shared_here) - 1)

    fig, ax = plt.subplots(figsize=(11.5, 0.52 * len(rows) + 2.6))

    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.BAND, zorder=0)
    ax.axvline(0, color=cc.INK, lw=1.1, zorder=3)

    y = list(range(len(rows)))[::-1]
    for yi, (cell, k3, pub, d, lo, hi) in zip(y, rows):
        ax.barh(yi, d, height=0.62, color=BAR, zorder=2,
                hatch="///" if cell in SHARED else None,
                edgecolor="#ffffff", linewidth=0)
        if lo == lo:                                   # not NaN
            ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="none",
                        ecolor=cc.INK, elinewidth=1.3, capsize=4, zorder=4)
        far = hi if d >= 0 else lo
        pad = 0.004 if d >= 0 else -0.004
        ax.text(far + pad, yi, f"{pub:.4f} → {k3:.4f}   ({d:+.4f})",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=9, color=cc.INK)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} ‡" if c in SHARED else c for c, *_ in rows])
    ax.set_xlabel("GM-Relative MASE, this study's k = 3 minus the published "
                  "k = 0   (97 configs, negative is better)")
    ax.set_title(f"k = 3 against each cell's published k = 0, bb{STOP}k, "
                 f"{HEAD} head   ({models} distinct models in {len(rows)} cells)")
    ends = [v for r in rows for v in (r[3], r[4], r[5]) if v == v]
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
        Patch(facecolor=BAR, hatch="///", edgecolor="#ffffff",
              label="‡ one student model, two published baselines"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.055 - 4.0 / len(rows) / 14),
        ncol=3, frameon=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  {args.out}  ({len(rows)} cell(s), {models} distinct model(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
