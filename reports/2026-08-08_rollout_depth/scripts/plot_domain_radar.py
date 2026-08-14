#!/usr/bin/env python3
"""#373 — per-domain GM-Relative MASE, k = 3 against k = 0.

The card's `domain_radar`, extended with the k = 0 polygon overlaid. It is
the figure the second half of the question lives on: GM-Relative MASE = 1.0
is parity with seasonal naive, and the card names the four families where
the k = 0 model loses to it — Energy, Econ/Fin, Web/CloudOps, Healthcare —
and the two where it already wins, Nature and Sales. Energy carries 32 of
the 97 configs, so its spoke carries a third of the aggregate.

The 1.0 parity ring is drawn on every panel, in black, and labelled. A
polygon inside it beats seasonal naive on that family.

The radial axis is log2(ratio), so equal multiplicative steps are equal
distances.

Six panels, chosen so the reader sees every kind of pair this study holds:

  A4 bb100k   the cell that sets the study's frontier, against the k = 0
              its parent published at the same stop.
  B1 bb40k    a pair whose two sides trained on ONE machine.
  B5·s2 bb40k the other machine-held pair, which moves the other way.
  B2 bb200k   the exact (arm, stop) the card quotes its per-family numbers
              from: `arm6_v2 combab` with L_align on the teacher, bb200k.
  B9 bb40k    the largest gain in the study. Two machines.
  A3 bb40k    the largest loss in the study. Two machines.

A pair whose two sides trained on two boxes carries a machine change worth
0.1166 on this study's one controlled measurement of it, so every panel says
which it is.

Reads `results/splits.csv` (this study's evals, written by
`split_scores.py`) and `results/splits_k0.csv` (the parents' own evals,
written by `parent_splits.py`). The two files carry colliding row keys on
purpose — `A3_k0_bb40k_student` is a rented-box retrain in one and the
published run in the other — so each panel names its file.

Usage: plot_domain_radar.py --splits results/splits.csv \\
           --splits-k0 results/splits_k0.csv --head student \\
           --out plots/domain_radar_student.png
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cell_colours as cc                              # noqa: E402
import r2_ladder as L                                  # noqa: E402

plt.rcParams.update(cc.rc())

PARITY_INK = "#0b0b0b"

# (cell, stop, k0 file, k0 key, k3 file, k3 key, what the pair holds)
# `S` is the head, filled in per figure.
PANELS = [
    ("A4", 100, "k0", "A4_k0_bb100k_{S}", "study", "A4_k3_bb100k_{S}",
     "two machines", "the study's frontier cell"),
    ("B1", 40, "study", "G6_B1_k0_bb40k_{S}", "study", "G6_B1_k3_bb40k_{S}",
     "ONE machine", "machine held"),
    ("B5", 40, "study", "G5_B5_s2_k0_bb40k_{S}", "study",
     "G5_B5_s2_k3_bb40k_{S}", "ONE machine",
     "machine held, backbone seed 20260521"),
    ("B2", 200, "k0", "B2_k0_bb200k_{S}", "study", "B2_k3_bb200k_{S}",
     "two machines", "the arm and stop the card quotes"),
    ("B9", 40, "study", "G2_B9_k0_bb40k_{S}", "study", "B9_k3_bb40k_{S}",
     "two machines", "the study's largest gain"),
    ("A3", 40, "study", "A3_k0_bb40k_{S}", "study", "A3_k3_bb40k_{S}",
     "two machines", "the study's largest loss"),
]


def load(path):
    out, counts = {}, {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "domain":
                continue
            out.setdefault(r["stop"], {})[r["name"]] = float(r["gm_rel_mase"])
            counts[r["name"]] = int(r["n"])
    return out, counts


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--splits", required=True)
    p.add_argument("--splits-k0", required=True)
    p.add_argument("--head", default="student")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    study, counts = load(args.splits)
    pub, c2 = load(args.splits_k0)
    counts.update(c2)
    src = {"study": study, "k0": pub}

    panels = []
    for cell, stop, f0, k0, f3, k3, note, why in PANELS:
        a = src[f0].get(k0.format(S=args.head))
        b = src[f3].get(k3.format(S=args.head))
        if a and b:
            panels.append((cell, stop, a, b, note, why))
    if not panels:
        raise SystemExit(f"ABORT: no panel has both sides on the {args.head} head")

    domains = sorted(counts, key=lambda d: -counts[d])
    n = len(domains)
    ang = [i / n * 2 * math.pi for i in range(n)] + [0.0]

    allv = [v for _, _, a, b, _, _ in panels for d in (a, b) for v in d.values()]
    lo, hi = min(allv), max(allv)
    ticks = [t for t in (0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0)
             if lo / 1.1 <= t <= hi * 1.1] or [round(lo, 2), round(hi, 2)]
    if ticks[0] > lo:
        ticks.insert(0, round(lo, 2))
    if ticks[-1] < hi:
        ticks.append(round(hi, 2))

    ncol = min(len(panels), 3)
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 5.7 * nrow),
                             subplot_kw={"projection": "polar"}, squeeze=False)

    for idx, (cell, stop, k0v, k3v, note, why) in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels([f"{d}\n({counts[d]})" for d in domains], fontsize=7)
        ax.set_ylim(math.log2(ticks[0]) - 0.03, math.log2(ticks[-1]) + 0.03)
        ax.set_yticks([math.log2(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
        ax.plot(ang, [0.0] * len(ang), color=PARITY_INK, linewidth=1.7,
                zorder=5)

        col = cc.ladder_colour(cell)
        for vals, ls, alpha in ((k0v, (0, (5, 2)), 0.75), (k3v, "solid", 1.0)):
            v = [math.log2(vals.get(d, 1.0)) for d in domains]
            ax.plot(ang, v + v[:1], color=col, linewidth=2.0, linestyle=ls,
                    alpha=alpha, zorder=4)

        arm, align, ema = L.CELL_ARM[cell]
        tgt = "no L_align" if align == "none" else f"L_align→{align}"
        ax.set_title(f"{cell}  {arm} · {tgt}\n"
                     f"k = 0 against k = 3 at bb{stop}k   ({note})\n{why}",
                     fontsize=9, pad=18)

    for idx in range(len(panels), nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")

    fig.legend(handles=[
        Line2D([], [], color=cc.INK_SOFT, linestyle=(0, (5, 2)),
               label="k = 0, no rollout depth in training"),
        Line2D([], [], color=cc.INK_SOFT, linestyle="solid",
               label="k = 3, the depth this study trains"),
        Line2D([], [], color=PARITY_INK, linewidth=1.7,
               label="seasonal-naive parity, 1.0. Inside it the model wins")],
        loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Per-domain GM-Relative MASE, {args.head}-encoder head "
                 "(radial axis log2, lower is better)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=4.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out} ({len(panels)} panel(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
