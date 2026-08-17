#!/usr/bin/env python3
"""#401 — does the collapse probe explain the phase-1 scores?

Two panels, from results/diag/collapse_vs_score.csv:

  left    GM-Relative MASE against the effective rank of the encoder latent
          across series. Every scored #401 cell sits at rank near 1. The
          k = 0 parent, at rank 7.2, carries the anchor score 1.1600.

  right   GM-Relative MASE against `readout_r`, the correlation between the
          input series and the one direction the collapsed encoder keeps.

Usage:  python3 plot_collapse_vs_score.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RES = STUDY / "results"
PLOTS = STUDY / "plots"

# The k = 0 anchor, measured by this study's own path: control c2.
K0_SCORE = float(
    (RES / "diag/score_c2_k0anchor_a4parent_bb40k_h30k_student.txt")
    .read_text().strip())

COLOUR = {0: "#2f6f4e", 8: "#c9772a", 16: "#a63a3a", 32: "#3a5ba6"}


def load():
    rows = []
    with (RES / "diag/collapse_vs_score.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    rows = load()
    scored = [r for r in rows if r["score"] and int(r["k"]) > 0]
    # the parent at bb40k is the checkpoint control c2 scored
    anchor = [r for r in rows if int(r["k"]) == 0 and int(r["step_k"]) == 40
              and "393" in r["label"]][0]

    PLOTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for a, xk, xlabel in (
            (ax[0], "eff_rank", "effective rank of encoder latent, "
                                "across series"),
            (ax[1], "readout_r", "readout r: |corr| of input with the "
                                 "surviving direction")):
        a.axhline(K0_SCORE, color="#888", lw=1, ls=":", zorder=1)
        a.text(0.02, K0_SCORE, f"  k = 0 anchor {K0_SCORE:.4f}",
               transform=a.get_yaxis_transform(), va="bottom",
               fontsize=8, color="#666")
        # `readout_r` reads one direction. That summarises a rank-1 encoder.
        # It does not summarise the rank-7 parent, so the parent stays off
        # the right panel and keeps only its score line.
        if xk != "readout_r":
            a.scatter([float(anchor[xk])], [K0_SCORE], s=110, marker="*",
                      color=COLOUR[0], zorder=3, label="k = 0 parent, bb40k")
        # Points at the same rank sit on top of each other, so the labels
        # step through four offsets in score order.
        order = sorted(scored, key=lambda r: (float(r[xk]),
                                              float(r["score"])))
        off = [(8, 4), (8, -12), (8, 16), (8, -24)]
        for i, r in enumerate(order):
            k = int(r["k"])
            a.scatter([float(r[xk])], [float(r["score"])], s=54,
                      color=COLOUR[k], zorder=3)
            a.annotate(f"k={k} {r['step_k']}k",
                       (float(r[xk]), float(r["score"])),
                       textcoords="offset points", xytext=off[i % 4],
                       fontsize=7.5, color=COLOUR[k])
        a.set_xlabel(xlabel, fontsize=9)
        a.set_ylabel("GM-Relative MASE (lower is better)", fontsize=9)
        a.set_yscale("log")
        a.grid(alpha=0.25, which="both")
        a.tick_params(labelsize=8)
        lo, hi = a.get_xlim()                      # room for the labels
        a.set_xlim(lo, hi + 0.18 * (hi - lo))

    ax[0].set_title("Rank separates k = 0 from k > 0. It does not order "
                    "the k > 0 scores.", fontsize=9.5)
    ax[1].set_title("Inside the collapsed set, the score follows the "
                    "readout  (Spearman -0.76, n = 8)", fontsize=9.5)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLOUR[k],
                          label=f"k = {k}") for k in (8, 16, 32)]
    handles.insert(0, plt.Line2D([], [], marker="*", ls="", color=COLOUR[0],
                                 markersize=11, label="k = 0 parent"))
    ax[0].legend(handles=handles, fontsize=8, loc="upper right")

    fig.suptitle("#401 phase 1 — every scored cell is collapsed, including "
                 "the best one", fontsize=11)
    fig.tight_layout()
    out = PLOTS / "collapse_vs_score.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
