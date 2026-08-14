#!/usr/bin/env python3
"""#373 — k = 3 minus k = 0, per cell, at each stop.

The card asks for the parent report's `schedule_vs_fixed.png` rebuilt on
this study's 14 cells. Where the parent subtracted "fixed 0.9" from
"scheduled", this subtracts each cell's published `k = 0` from its `k = 3`,
at every stop the pair holds, on both heads.

One bar per cell per panel, ranked inside the panel, so a panel reads the
same way the parent's two panels read. Negative is better.

This is a SCREEN and not a test. Every bar reads its `k = 0` side from a
parent report. The grey band is `ema_sched_ladder.md`'s pooled head-seed
band; it bounds the HEAD seed alone, and the backbone seed is unreplicated.

Group B's two parents publish the student-encoder head only, so the teacher
row draws group A alone.

A1 and B3 run one arm that aligns to the student and passes no
`--moco-rep-keys`, so the EMA regime that separates them cannot reach the
student encoder. Their two backbones hold identical student weights: the two
bars are one model against two published baselines, and ‡ marks them.

Colour carries nothing here: the report's palette holds four hues and this
figure draws fourteen cells, so the cell rides the axis label.

Reads results/published_bootstrap.csv (`all` rows) and the score files.

Usage: plot_k3_minus_k0.py --results results --out plots/k3_minus_k0.png
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
STOPS = [40, 100, 200]
HEADS = ["student", "teacher"]
# The one pair of cells whose student encoders are the same weights.
SHARED = ("A1", "B3")


def intervals(res):
    """`{(cell, stop, head): (lo, hi)}` from the published bootstrap."""
    out = {}
    p = Path(res) / "published_bootstrap.csv"
    if not p.is_file():
        return out
    for r in csv.DictReader(open(p)):
        if r["subset"] != "all":
            continue
        cell, sep, rest = r["label"].partition("_vs_pub_")
        if not sep:
            continue
        stop, _, head = rest.partition("_")
        out[(cell, int(stop[2:-1]), head)] = (float(r["ci_lo"]),
                                              float(r["ci_hi"]))
    return out


def panel_rows(res, ci, stop, head):
    """`[(cell, k3, pub, delta, lo, hi)]` for one stop and head, best first."""
    rows = []
    for cell in L.CELLS:
        k3 = L.score(cell, stop, head, res)
        pub = published.at(cell, head, stop)
        if k3 is None or pub is None:
            continue
        lo, hi = ci.get((cell, stop, head), (float("nan"), float("nan")))
        rows.append((cell, k3, pub, k3 - pub, lo, hi))
    rows.sort(key=lambda r: r[3])
    return rows


def draw(ax, rows, stop, head, xlim, tall):
    """One panel. `tall` is the row count of the fullest panel, so a bar is
    the same height in every panel and a sparse panel reads as sparse."""
    ax.axvspan(-NOISE_BAND, NOISE_BAND, color=cc.BAND, zorder=0)
    ax.axvline(0, color=cc.INK, lw=1.0, zorder=3)
    ax.set_ylim(-0.8, tall - 0.2)
    if not rows:
        ax.text(0.5, 0.5, f"no cell holds a published k = 0\nat bb{stop}k "
                          f"on the {head} head",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=cc.INK_SOFT)
        ax.set_yticks([])
    else:
        y = [tall - 1 - i for i in range(len(rows))]
        for yi, (cell, _k3, _pub, d, lo, hi) in zip(y, rows):
            ax.barh(yi, d, height=0.62, color=BAR, zorder=2,
                    hatch="///" if cell in SHARED else None,
                    edgecolor="#ffffff", linewidth=0)
            if lo == lo:                               # not NaN
                ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="none",
                            ecolor=cc.INK, elinewidth=1.1, capsize=3,
                            zorder=4)
            pad = 0.008 if d >= 0 else -0.008
            ax.text((hi if d >= 0 else lo) + pad, yi, f"{d:+.4f}",
                    va="center", ha="left" if d >= 0 else "right",
                    fontsize=8, color=cc.INK)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c} ‡" if c in SHARED else c for c, *_ in rows])
    ax.set_xlim(*xlim)
    ax.set_title(f"bb{stop}k, {head} head  ({len(rows)} cell(s))", fontsize=10)
    ax.grid(axis="x", color=cc.GRID, zorder=0)
    ax.set_axisbelow(True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    ci = intervals(args.results)
    grid = {(s, h): panel_rows(args.results, ci, s, h)
            for s in STOPS for h in HEADS}
    if not any(grid.values()):
        raise SystemExit("ABORT: no cell holds a stop and a published k = 0")

    ends = [v for rows in grid.values() for r in rows
            for v in (r[3], r[4], r[5]) if v == v]
    lo_end, hi_end = min(ends + [0.0]), max(ends + [0.0])
    span = hi_end - lo_end
    # The value label sits outside the whisker, so the left edge needs room
    # for a nine-character number and the right edge does not need as much.
    xlim = (lo_end - 0.34 * span, hi_end + 0.24 * span)

    tall = max(len(rows) for rows in grid.values())
    fig, axes = plt.subplots(len(HEADS), len(STOPS),
                             figsize=(4.4 * len(STOPS), 0.34 * tall + 2.4),
                             squeeze=False)
    for i, head in enumerate(HEADS):
        for j, stop in enumerate(STOPS):
            draw(axes[i][j], grid[(stop, head)], stop, head, xlim, tall)
        axes[i][0].set_ylabel(f"{head} head", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("GM-Relative MASE, k = 3 minus published k = 0\n"
                      "(97 configs, negative is better)", fontsize=9)

    fig.suptitle("k = 3 against each cell's published k = 0, at every stop",
                 fontsize=11)
    fig.legend(handles=[
        Line2D([0], [0], color=cc.INK, lw=1.1,
               label="95% CI, paired dataset-cluster bootstrap"),
        Patch(facecolor=cc.BAND, label=f"head-seed band ±{NOISE_BAND}"),
        Patch(facecolor=BAR, hatch="///", edgecolor="#ffffff",
              label="‡ one student model, two published baselines"),
    ], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    n = sum(len(r) for r in grid.values())
    print(f"  {args.out}  ({n} bar(s) over {len(STOPS)} stop(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
