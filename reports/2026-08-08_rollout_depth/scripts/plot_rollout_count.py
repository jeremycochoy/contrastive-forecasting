#!/usr/bin/env python3
"""#373 — the eval's own rollout depth, and the score change against it.

Training at `k = 3` teaches the model to read its own output. The eval reads
its own output too, once per patch of the horizon, so a config with a long
horizon rolls out further. If the depth pays because the model learns to run
on itself, the gain should grow with that count.

Left panel: how many rollout steps each of the 97 configs needs, one point
per config, grouped by domain. `rollout_count.py` computes the count.

Right panel: the same count against `k = 3` minus `k = 0` on that config,
one point per config. The four arms that trained their own `k = 0` carry the
depth as their only change; A4 reads the `k = 0` its parent published, and
its marker says so. The heavy line joins each arm's median change over the
three horizon terms.

Both panels use a log2 x-axis: the counts run 1 to 57 and the short configs
would otherwise sit on top of each other.

Usage: plot_rollout_count.py --results results --out plots/rollout_count.png
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                                  # noqa: E402
import paired_bootstrap as PB                              # noqa: E402
import published_bootstrap as PBOOT                        # noqa: E402
import r2_ladder as L                                      # noqa: E402

# (cell, label, k = 0 eval tag or None for the published one, k = 3 tag,
#  stop, whether this study trained the k = 0 side)
PAIRS = [
    ("B9", "B9  bb40k", "G2_B9_k0_bb40k_student", "B9_k3_bb40k_student",
     40, True),
    ("B1", "B1  bb40k", "G6_B1_k0_bb40k_student", "G6_B1_k3_bb40k_student",
     40, True),
    ("B5", "B5·s2  bb40k", "G5_B5_s2_k0_bb40k_student",
     "G5_B5_s2_k3_bb40k_student", 40, True),
    ("A3", "A3  bb40k", "A3_k0_bb40k_student", "A3_k3_bb40k_student",
     40, True),
    ("A4", "A4  bb100k", None, "A4_k3_bb100k_student", 100, False),
]
TERMS = ("short", "medium", "long")


def spearman(xs, ys):
    """Rank correlation, without a scipy dependency. Ties take mid-ranks."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = mid
            i = j + 1
        return r

    a, b = rank(xs), rank(ys)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a)
                    * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def read_counts(path):
    with open(path) as fh:
        return {r["config"]: (int(r["rollout_steps"]), r["domain"], r["term"])
                for r in csv.DictReader(fh)}


def rel(path, sn):
    """`{config: MASE / seasonal-naive MASE}`."""
    m = PB.read_mase(path)
    return {d: m[d] / sn[d] for d in m if d in sn}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=L.RESULTS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = Path(a.results)

    counts_path = res / "rollout_count.csv"
    if not counts_path.is_file():
        print("no rollout_count.csv — no rollout-count figure")
        return 0
    counts = read_counts(counts_path)
    sn = PB.read_mase(PB.SN_REF)

    series = []
    for cell, label, k0tag, k3tag, stop, own in PAIRS:
        p3 = res / "eval" / k3tag / "all_results.csv"
        p0 = (res / "eval" / k0tag / "all_results.csv") if k0tag \
            else PBOOT.parent_csv(cell, stop, "student", res)
        if not p3.is_file() or p0 is None or not Path(p0).is_file():
            continue
        r0, r3 = rel(p0, sn), rel(p3, sn)
        pts = [(counts[d][0], r3[d] - r0[d], counts[d][2])
               for d in sorted(set(r0) & set(r3) & set(counts))]
        if len(pts) == 97:
            series.append((cell, label, own, pts))
    if not series:
        print("no finished depth pair — no rollout-count figure")
        return 0

    plt.rcParams.update(cc.rc())
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14.5, 6.0), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # ---- left: how far each config rolls out ------------------------------
    doms = {}
    for step, dom, _t in counts.values():
        doms.setdefault(dom, []).append(step)
    order = sorted(doms, key=lambda d: -len(doms[d]))
    for i, dom in enumerate(order):
        v = sorted(doms[dom])
        row = len(order) - 1 - i
        # Configs collide on the same count, so one dot per config is stacked
        # rather than jittered: the height of a stack is how many configs
        # take that count.
        tally = {s: v.count(s) for s in set(v)}
        seen = {}
        for step in v:
            seen[step] = seen.get(step, 0) + 1
            off = (seen[step] - 1 - (tally[step] - 1) / 2) * 0.085
            axL.scatter([step], [row + off], s=26, color=cc.INK_SOFT,
                        alpha=0.55, edgecolors="none", zorder=3)
        med = statistics.median(v)
        axL.scatter([med], [row], s=150, marker="|",
                    color="#0b0b0b", linewidths=2.2, zorder=4)
        axL.text(88, row, f"median {med:g}   {min(v)}–{max(v)}",
                 fontsize=8.5, va="center", color=cc.INK_SOFT)
    axL.set_yticks(range(len(order)))
    axL.set_yticklabels([f"{d}\n({len(doms[d])} configs)"
                         for d in reversed(order)], fontsize=8.5)
    axL.set_xscale("log", base=2)
    axL.set_xticks([1, 2, 4, 8, 16, 32, 64])
    axL.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    axL.set_xlim(0.8, 330)
    axL.set_ylim(-0.6, len(order) - 0.2)
    axL.set_xlabel("rollout steps the eval takes on that config")
    axL.set_title("How far the eval rolls out, per config", loc="left")
    axL.grid(axis="y", visible=False)

    # ---- right: the change against that count -----------------------------
    axR.axhline(0.0, color="#0b0b0b", linewidth=1.4, zorder=4)
    legend = []
    for cell, label, own, pts in series:
        col = cc.ladder_colour(cell)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        axR.scatter(xs, ys, s=20, color=col, alpha=0.35,
                    marker="o" if own else "^", edgecolors="none", zorder=3)
        med_x, med_y = [], []
        for t in TERMS:
            grp = [p for p in pts if p[2] == t]
            med_x.append(statistics.median([p[0] for p in grp]))
            med_y.append(statistics.median([p[1] for p in grp]))
        axR.plot(med_x, med_y, color=col, linewidth=2.4, zorder=5,
                 marker="o" if own else "^", markersize=7)
        rho = spearman(xs, ys)
        legend.append(Line2D([], [], color=col, linewidth=2.4,
                             marker="o" if own else "^", markersize=7,
                             label=f"{label}   ρ = {rho:+.2f}"))
    axR.set_xscale("log", base=2)
    axR.set_xticks([1, 2, 4, 8, 16, 32, 64])
    axR.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    axR.set_xlim(0.8, 80)
    axR.set_xlabel("rollout steps the eval takes on that config")
    axR.set_ylabel("relative MASE, k = 3 minus k = 0  (negative is better)")
    axR.set_title("The change against that count, one point per config",
                  loc="left")
    ylim = 1.6
    outside = sum(1 for _c, _l, _o, pts in series for p in pts
                  if abs(p[1]) > ylim)
    axR.set_ylim(-ylim, ylim)
    axR.text(0.99, 0.02, f"{outside} of {len(series) * 97} points fall "
                         f"outside this axis", transform=axR.transAxes,
             ha="right", va="bottom", fontsize=8.5, color=cc.INK_SOFT)
    legend.append(Line2D([], [], color=cc.INK_SOFT, linewidth=0,
                         marker="^", markersize=7,
                         label="triangle: reads a published k = 0"))
    axR.legend(handles=legend, loc="lower left", fontsize=9)

    fig.suptitle("Rollout steps at eval, and the change against them "
                 "(student head, 97 configs)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    print(f"wrote {a.out} ({len(series)} pair(s))")
    for cell, label, _own, pts in series:
        print(f"  {label:<14} spearman "
              f"{spearman([p[0] for p in pts], [p[1] for p in pts]):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
