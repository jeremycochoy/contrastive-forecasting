#!/usr/bin/env python3
"""#373 figure 5 — A3: is the damage the depth, or the weight it carries?

Summing the depths multiplies the f-bearing term's weight against the f-free
terms by k + 1. So a k = 3 run changes two things at once, and the obvious
worry is that the damage is the re-weighting rather than the depth.

The x axis is that weight multiplier, which puts the control exactly where
it belongs. The `L_align x4` run sits at x = 4 with no depth at all, beside
the k = 3 run at the same weight. The vertical gap between them at x = 4 is
what the depth costs once the weight is accounted for.

The ladder is not monotonic: k = 1 is below k = 0 and k = 3 is far above it.

Reads results/splits.csv (`all` rows).

Usage: plot_a3_depth.py --splits results/splits.csv --out plots/a3_depth.png
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
LADDER = [(0, 1, "A3_k0_bb40k_{h}"),
          (1, 2, "G3_A3_k1_bb40k_{h}"),
          (3, 4, "A3_k3_bb40k_{h}")]
CONTROL = (4, "G3_A3_k0_aw4_bb40k_{h}")
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
    col = cc.COLOUR["A3"]

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for head in HEADS:
        pts = [(w, data.get(t.format(h=head)), k) for k, w, t in LADDER]
        if any(v is None for _w, v, _k in pts):
            continue
        alpha = 1.0 if head == "student" else 0.45
        ax.plot([w for w, _v, _k in pts], [v for _w, v, _k in pts],
                color=col, linewidth=2.2, alpha=alpha, marker="o",
                markersize=10, markerfacecolor=col, markeredgecolor=col,
                zorder=3, label=f"depth ladder, {head} head")
        # Label the student head only. The two heads sit within 0.01 of
        # each other, so two sets of numbers would overprint and say nothing
        # the table does not.
        if head == "student":
            top = max(v for _w, v, _k in pts)
            for w, v, k in pts:
                # The highest point's label would run into the title.
                below = v == top or k == 1
                ax.annotate(
                    f"k = {k}\n{v:.4f}", (w, v),
                    textcoords="offset points",
                    xytext=(-16, -6) if below else (0, 13),
                    ha="right" if below else "center",
                    va="top" if below else "bottom",
                    fontsize=9, color=cc.INK,
                    bbox=dict(fc="#ffffff", ec="none", pad=0.6))

        cw, ctag = CONTROL
        cv = data.get(ctag.format(h=head))
        if cv is None:
            continue
        ax.plot(cw, cv, marker="D", markersize=11, markerfacecolor="#ffffff",
                markeredgecolor=col, markeredgewidth=2.2, alpha=alpha,
                zorder=4)
        if head == "student":
            ax.annotate(f"L_align x4, no depth\n{cv:.4f}", (cw, cv),
                        textcoords="offset points", xytext=(-16, 0),
                        ha="right", va="center", fontsize=9, color=cc.INK,
                        bbox=dict(fc="#ffffff", ec="none", pad=0.6))
        if head == "student":
            k3 = data[LADDER[-1][2].format(h=head)]
            # The arrow only. Its two ends trained on two boxes, so the
            # figure marks the gap and leaves its size to the body.
            ax.annotate("", xy=(cw + 0.22, k3), xytext=(cw + 0.22, cv),
                        arrowprops=dict(arrowstyle="<->", color=cc.INK,
                                        linewidth=1.4))

    ax.axhline(data["A3_k0_bb40k_student"], color=cc.PARITY, linewidth=1.1,
               linestyle=(0, (4, 3)), zorder=0)
    ax.annotate("A3's own k = 0, student head",
                (2.55, data["A3_k0_bb40k_student"]),
                fontsize=8, color=cc.INK_SOFT, ha="left", va="bottom",
                bbox=dict(fc="#ffffff", ec="none", pad=0.8))
    ax.set_xticks([1, 2, 4])
    ax.set_xticklabels(["x1", "x2", "x4"])
    ax.set_xlim(0.70, 5.9)
    ax.set_xlabel("weight the f-bearing term carries against the f-free terms "
                  "(k + 1, since the depths are summed)")
    ax.set_ylabel("GM-Relative MASE, 97 configs  (lower is better)")
    ax.set_title("A3 depth ladder against the L_align x4 control, bb40k",
                 loc="left", fontsize=12, pad=17)
    # Which box each point trained on, above the panel. Every point here
    # trained on a different box from at least one other; what that costs the
    # reading is a body sentence, not a caption.
    where = " · ".join(f"{c}: {R.resolve(f'{t}_bb40k_student').machine}"
                       for c, t in (("k = 0", "A3_k0"), ("k = 1", "G3_A3_k1"),
                                    ("x4", "G3_A3_k0_aw4"), ("k = 3", "A3_k3"))
                       if R.resolve(f"{t}_bb40k_student"))
    ax.annotate(where, (0.0, 1.005), xycoords="axes fraction", fontsize=8.5,
                color=cc.INK_SOFT, va="bottom")
    handles = [
        Line2D([], [], color=col, linewidth=2.2, marker="o", markersize=9,
               label="depth ladder, student head"),
        Line2D([], [], color=col, linewidth=2.2, alpha=0.45, marker="o",
               markersize=9, label="depth ladder, teacher head"),
        Line2D([], [], color=col, linestyle="none", marker="D", markersize=10,
               markerfacecolor="#ffffff", markeredgewidth=2.2,
               label="re-weighting control, no depth")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
