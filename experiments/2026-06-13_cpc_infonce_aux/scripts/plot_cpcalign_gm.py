#!/usr/bin/env python3
"""#344 follow-up — GM-Relative MASE of the enc6 CPC+align/no-main arm against
the enc6 baseline (contrastive loss) and the enc6 main+CPC arm, per head ×
checkpoint. Reads results/gm_table.csv. Writes plots/cpcalign_gm.png.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-13_cpc_infonce_aux/results"
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "cpcalign_gm.png")
CLIP = 1.45


def main():
    rows = list(csv.DictReader(open(f"{RES}/gm_table.csv")))
    gm = {(r["arm"], r["head"], r["ckpt"]): (float(r["gm"]) if r["gm"] not in ("", "--") else None)
          for r in rows}
    cells = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
    series = [("enc6 baseline (contrastive)", "base_enc6", "0.6"),
              ("enc6 main + CPC", "cpc_enc6", "C2"),
              ("enc6 CPC + align, NO main loss", "cpcalign_enc6", "C3")]
    x = range(len(cells)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (label, key, color) in enumerate(series):
        vals = [gm.get((key, h, c)) for h, c in cells]
        xs = [j + (i - 1) * w for j in x]
        ax.bar(xs, [min(v, CLIP) if v else 0 for v in vals], w, label=label, color=color)
        for xj, v in zip(xs, vals):
            if v and v > CLIP:
                ax.text(xj, CLIP, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.axhline(1.0, color="k", lw=0.8, ls=":", label="seasonal-naive")
    ax.set_ylim(1.1, CLIP + 0.04)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{h} {c}" for h, c in cells])
    ax.set_ylabel("GM-Relative MASE (lower better)")
    ax.set_title("Can CPC + a separate forecaster loss replace the contrastive loss? No.\n"
                 "enc6: baseline vs main+CPC vs CPC+align/no-main (bars clipped at "
                 f"{CLIP}, true value labelled)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
