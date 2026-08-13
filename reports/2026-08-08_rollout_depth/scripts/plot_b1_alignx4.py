#!/usr/bin/env python3
"""#373 review gap 3 — B1: is the win the depth, or the weight it carries?

Summing the depths multiplies the f-bearing term's weight against the f-free
terms by k + 1. B1 carries `L_align` as its only f-bearing term, so its
`k = 3` run changes two things at once: it trains the forecaster on its own
output, and it multiplies `L_align` by 4.

A3 already carries this control. A3 is the cell where `k = 3` does the most
damage, and every column of that table trained on a different box, so it
gives a direction and not a magnitude. B1 is the study's one machine-held,
seed-held, head-budget-matched cell: its `k = 0`, its `k = 3` and this
control all trained on elisa at seed 20260520. So this figure may name the
two segments as sizes.

The x axis is the weight multiplier, the same axis `a3_depth.png` uses. The
control sits at x = 4 with no depth, beside the `k = 3` run at the same
weight. The drop from x = 1 to the control is the re-weighting alone. The
drop from the control to `k = 3` is what the depth adds once the re-weighting
is paid for.

Reads results/splits.csv (`all` rows).

Usage: plot_b1_alignx4.py --splits results/splits.csv --out plots/b1_alignx4.png
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
import runs as R                                       # noqa: E402

# (depth, weight multiplier, tag template)
LADDER = [(0, 1, "G6_B1_k0_bb40k_{h}"),
          (3, 4, "G6_B1_k3_bb40k_{h}")]
CONTROL = (4, "G_B1_k0_aw4_bb40k_{h}")
HEADS = ("student", "teacher")
plt.rcParams.update(cc.rc())


def load(path):
    return {r["stop"]: float(r["gm_rel_mase"])
            for r in csv.DictReader(open(path)) if r["split"] == "all"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = load(args.splits)
    col = cc.COLOUR["B1"]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    share = {}
    for head in HEADS:
        pts = [(w, data.get(t.format(h=head)), k) for k, w, t in LADDER]
        cw, ctag = CONTROL
        cv = data.get(ctag.format(h=head))
        if any(v is None for _w, v, _k in pts) or cv is None:
            continue
        alpha = 1.0 if head == "student" else 0.45
        ax.plot([w for w, _v, _k in pts], [v for _w, v, _k in pts],
                color=col, linewidth=2.2, alpha=alpha, marker="o",
                markersize=10, markerfacecolor=col, markeredgecolor=col,
                zorder=3)
        ax.plot(cw, cv, marker="D", markersize=11, markerfacecolor="#ffffff",
                markeredgecolor=col, markeredgewidth=2.2, alpha=alpha,
                zorder=4)

        k0, k3 = pts[0][1], pts[1][1]
        share[head] = (cv - k0, k3 - cv, k3 - k0)

        # Label the student head only. The two heads sit within 0.01 of each
        # other at every point, so a second set of numbers would overprint
        # and say nothing the table does not.
        if head != "student":
            continue
        for w, v, k in pts:
            ax.annotate(f"k = {k}\n{v:.4f}", (w, v),
                        textcoords="offset points", xytext=(0, 13),
                        ha="center", va="bottom", fontsize=9, color=cc.INK,
                        bbox=dict(fc="#ffffff", ec="none", pad=0.6))
        ax.annotate(f"L_align x4, no depth\n{cv:.4f}", (cw, cv),
                    textcoords="offset points", xytext=(-16, 0), ha="right",
                    va="center", fontsize=9, color=cc.INK,
                    bbox=dict(fc="#ffffff", ec="none", pad=0.6))

        # The two segments, named. Every point here trained on elisa at seed
        # 20260520, so the segments are sizes and not directions.
        xs = 4.95
        ax.annotate("", (xs, k0), (xs, cv), arrowprops=dict(
            arrowstyle="<->", color=cc.INK_SOFT, linewidth=1.2))
        ax.annotate(f"the re-weighting\n{cv - k0:+.4f}", (xs + 0.08,
                    (k0 + cv) / 2), fontsize=9, color=cc.INK_SOFT,
                    ha="left", va="center")
        ax.annotate("", (xs, cv), (xs, k3), arrowprops=dict(
            arrowstyle="<->", color=cc.INK_SOFT, linewidth=1.2))
        ax.annotate(f"the depth\n{k3 - cv:+.4f}", (xs + 0.08, (cv + k3) / 2),
                    fontsize=9, color=cc.INK_SOFT, ha="left", va="center")

    if not share:
        raise SystemExit("ABORT: B1's k = 0, k = 3 and x4 control are not all "
                         "scored yet")

    ax.axhline(data["G6_B1_k0_bb40k_student"], color=cc.PARITY, linewidth=1.1,
               linestyle=(0, (4, 3)), zorder=0)
    ax.annotate("B1's own k = 0, student head",
                (1.06, data["G6_B1_k0_bb40k_student"]), fontsize=8,
                color=cc.INK_SOFT, ha="left", va="bottom",
                bbox=dict(fc="#ffffff", ec="none", pad=0.8))
    ax.set_xticks([1, 4])
    ax.set_xticklabels(["x1", "x4"])
    ax.set_xlim(0.70, 6.6)
    ax.set_xlabel("weight the f-bearing term carries against the f-free terms "
                  "(k + 1, since the depths are summed)")
    ax.set_ylabel("GM-Relative MASE, 97 configs  (lower is better)")
    ax.set_title("B1: the L_align x4 control against the depth ladder, bb40k",
                 loc="left", fontsize=12, pad=17)
    where = " · ".join(
        f"{c}: {R.resolve(f'{t}_bb40k_student').machine}"
        for c, t in (("k = 0", "G6_B1_k0"), ("x4", "G_B1_k0_aw4"),
                     ("k = 3", "G6_B1_k3"))
        if R.resolve(f"{t}_bb40k_student"))
    ax.annotate(where + " · seed 20260520 throughout", (0.0, 1.005),
                xycoords="axes fraction", fontsize=8.5, color=cc.INK_SOFT,
                va="bottom")
    handles = [
        Line2D([], [], color=col, linewidth=2.2, marker="o", markersize=9,
               label="depth ladder, student head"),
        Line2D([], [], color=col, linewidth=2.2, alpha=0.45, marker="o",
               markersize=9, label="depth ladder, teacher head"),
        Line2D([], [], color=col, linestyle="none", marker="D", markersize=10,
               markerfacecolor="#ffffff", markeredgewidth=2.2,
               label="re-weighting control, no depth")]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    for head, (rw, dp, tot) in share.items():
        print(f"{head}: k=0 -> x4 {rw:+.4f}, x4 -> k=3 {dp:+.4f}, "
              f"total {tot:+.4f}, re-weighting is "
              f"{100 * rw / tot:.0f}% of the move")
    return 0


if __name__ == "__main__":
    sys.exit(main())
