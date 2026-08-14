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

Each segment carries its own 95% paired dataset-cluster interval, drawn as a
band on the segment's far end.

The left panel anchors each band to its own segment's start level, so on the
level axis the two bands sit at two heights and read as two nearly disjoint
ranges. That is an artefact of the anchoring, not a finding. The right panel
draws the same two intervals as deltas on ONE shared zero axis, where they
overlap over most of their length. The report ranks neither share above the
other, and the right panel is where a reader sees why.

Reads results/splits.csv (`all` rows) and results/bootstrap.csv.

Usage: plot_b1_alignx4.py --splits results/splits.csv \\
           --bootstrap results/bootstrap.csv --out plots/b1_alignx4.png
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
# The two segments, and the bootstrap row that bounds each.
SEG_CI = {"re-weighting": "B1_alignx4_{h}",
          "depth": "B1_alignx4_vs_k3_{h}"}
plt.rcParams.update(cc.rc())


def load(path):
    return {r["stop"]: float(r["gm_rel_mase"])
            for r in csv.DictReader(open(path)) if r["split"] == "all"}


def load_ci(path):
    """(delta, lo, hi) per bootstrap label, `all` rows only."""
    out = {}
    with open(path) as fh:
        for r in csv.reader(fh):
            if len(r) >= 6 and r[1] == "all":
                try:
                    out[r[0]] = (float(r[3]), float(r[4]), float(r[5]))
                except ValueError:
                    pass
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--bootstrap", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    data = load(args.splits)
    ci = load_ci(args.bootstrap)
    col = cc.COLOUR["B1"]

    fig, (ax, dax) = plt.subplots(
        1, 2, figsize=(13.6, 5.4),
        gridspec_kw=dict(width_ratios=[1.0, 0.46], wspace=0.30))
    share, deltas = {}, []
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

        # The right panel takes BOTH heads. The two heads are two
        # measurements of one backbone, and the report's claim — that this
        # cell ranks neither move above the other — has to hold on each.
        for name in ("re-weighting", "depth"):
            d, lo, hi = ci[SEG_CI[name].format(h=head)]
            deltas.append((head, name, d, lo, hi))

        # Label the student head only. The two heads sit within 0.01 of each
        # other at every point, so a second set of numbers would overprint
        # and say nothing the table does not.
        if head != "student":
            continue
        # Both ladder points sit at a corner of the axes: k = 0 top left,
        # k = 3 bottom right. A label above either one climbs into the title
        # or off the axes, so both hang inward and below their marker.
        for (w, v, k), off in zip(pts, ((14, -10), (-14, -10))):
            ax.annotate(f"k = {k}\n{v:.4f}", (w, v),
                        textcoords="offset points", xytext=off,
                        ha="left" if off[0] > 0 else "right", va="top",
                        fontsize=9, color=cc.INK,
                        bbox=dict(fc="#ffffff", ec="none", pad=0.6))
        ax.annotate(f"L_align x4, no depth\n{cv:.4f}", (cw, cv),
                    textcoords="offset points", xytext=(-16, 0), ha="right",
                    va="center", fontsize=9, color=cc.INK,
                    bbox=dict(fc="#ffffff", ec="none", pad=0.6))

        # The two segments, named, each with its own 95% interval. Every
        # point here trained on elisa at seed 20260520, so the segments are
        # sizes and not directions.
        #
        # Each segment gets its own x, with the interval band behind the
        # arrow that owns it. The band hangs off the segment's far end: the
        # far end could sit anywhere in `near end + [lo, hi]`. Two x
        # positions, so where the bands cover the same y the reader sees the
        # overlap side by side.
        for name, near, far, xs in (("re-weighting", k0, cv, 4.85),
                                    ("depth", cv, k3, 5.35)):
            _d, lo, hi = ci[SEG_CI[name].format(h=head)]
            ax.plot([xs] * 2, [near + lo, near + hi], color=col,
                    linewidth=13.0, alpha=0.22, solid_capstyle="butt",
                    zorder=1)
            ax.annotate("", (xs, near), (xs, far), arrowprops=dict(
                arrowstyle="<->", color=cc.INK, linewidth=1.3), zorder=3)
            ax.annotate(f"the {name}\n{far - near:+.4f}\n[{lo:+.4f}, "
                        f"{hi:+.4f}]", (5.62, (near + far) / 2),
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
    ax.set_xlim(0.70, 6.9)
    # Room under the k = 3 marker for the label that hangs below it, and for
    # the legend beside it. The interval bands already reach well below the
    # ladder, so the pad is small.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.10 * (hi - lo), hi)
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
               label="re-weighting control, no depth"),
        Line2D([], [], color=col, linewidth=6.0, alpha=0.30,
               label="95% paired dataset-cluster interval on the segment")]
    # Lower left. The ladder falls from the top-left corner to the bottom
    # right and the k = 0 rule runs along the top, so upper right is the one
    # place the legend cannot go.
    ax.legend(handles=handles, loc="lower left", fontsize=9,
              framealpha=0.95, borderpad=0.7)

    # ---- the companion: the same two moves, on one shared zero axis -------
    # On the level axis a band is anchored to its own segment's start, so the
    # two bands sit at two heights and the eye reads them as disjoint. Here
    # both are deltas against their own start, so one x = 0 serves both and
    # the overlap is what the reader sees.
    ys = list(range(len(deltas)))[::-1]
    for yi, (head, name, d, lo, hi) in zip(ys, deltas):
        alpha = 1.0 if head == "student" else 0.45
        dax.plot([lo, hi], [yi, yi], color=col, linewidth=9.0, alpha=0.25 * (
            1.0 if head == "student" else 0.62), solid_capstyle="butt",
            zorder=1)
        dax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", markersize=8,
                     color=col, alpha=alpha, ecolor=col, elinewidth=1.6,
                     capsize=5, zorder=3)
        dax.annotate(f"{d:+.4f}", (d, yi), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9, color=cc.INK)
    dax.axvline(0, color=cc.INK, linewidth=1.1, zorder=2)
    dax.set_yticks(ys)
    dax.set_yticklabels([f"{n}\n[{h}]" for h, n, *_ in deltas], fontsize=9)
    dax.set_ylim(-0.75, len(deltas) - 0.25)
    dax.set_xlabel("change against the move's own start, 97 configs\n"
                   "(negative is better)")
    dax.set_title("the same two moves, on one zero axis", loc="left",
                  fontsize=12, pad=17)
    dax.annotate("bars are 95% paired dataset-cluster intervals",
                 (0.0, 1.005), xycoords="axes fraction", fontsize=8.5,
                 color=cc.INK_SOFT, va="bottom")
    ends = [v for _h, _n, _d, lo, hi in deltas for v in (lo, hi)] + [0.0]
    pad = 0.16 * (max(ends) - min(ends))
    dax.set_xlim(min(ends) - pad, max(ends) + pad)
    dax.grid(axis="y", visible=False)

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
