#!/usr/bin/env python3
"""Headline GM figure (3 panels), GIFT-Eval full-97 GM-Relative MASE, grouped by
head × checkpoint. Panel 1 is the encoder-depth ladder {0 = no-encoder, 3, 6}
for the base loss. Panel 2 places every + CPC arm side by side — the two
no-encoder CPC variants beside the encoder'd + CPC arms, so + CPC_All is directly
comparable to the encoder'd backbones. Panel 3 is the no-encoder loss comparison
— base vs + CPC vs + CPC_All (the full-marginal CPC variant, which exists only at
depth 0). Reads results/gm_table.csv. Writes plots/gm_summary.png.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-15_no_encoder_redo/results"
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "gm_summary.png")

CELLS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
# (panel title, legend title, [(bar label, arm key, colour)])
PANELS = [
    ("base contrastive loss", "encoder depth",
     [("no-enc", "noenc_base", "C0"), ("enc-3", "base_enc3", "0.7"), ("enc-6", "base_enc6", "0.45")]),
    ("+ CPC and + CPC_All", "arm",
     [("no-enc + CPC", "noenc_cpc", "C3"), ("no-enc + CPC_All", "noenc_cpcall", "C2"),
      ("enc-3 + CPC", "cpc_enc3", "0.7"), ("enc-6 + CPC", "cpc_enc6", "0.45")]),
    ("no encoder — three losses", "loss",
     [("base", "noenc_base", "C0"), ("+ CPC", "noenc_cpc", "C3"), ("+ CPC_All", "noenc_cpcall", "C2")]),
]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    gm = {(r["arm"], r["head"], r["ckpt"]): f(r["gm"])
          for r in csv.DictReader(open(f"{RES}/gm_table.csv"))}
    vals = [gm.get((k, h, c)) for _, _, bars in PANELS for _, k, _ in bars
            for h, c in CELLS if gm.get((k, h, c))]
    lo, hi = (min(vals), max(vals)) if vals else (1.1, 1.45)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
    x = range(len(CELLS))
    for ax, (title, leg, bars) in zip(axes, PANELS):
        w = 0.8 / len(bars)
        for j, (blabel, key, col) in enumerate(bars):
            ys = [gm.get((key, h, c)) for h, c in CELLS]
            off = (j - (len(bars) - 1) / 2) * w
            ax.bar([i + off for i in x], [y or 0 for y in ys], w, label=blabel, color=col)
            for i, y in enumerate(ys):
                if y:
                    ax.text(i + off, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{h}\n{c}" for h, c in CELLS], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.legend(title=leg, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("GM-Relative MASE (lower better)")
    axes[0].set_ylim(max(1.0, lo - 0.02), hi + 0.03)
    fig.suptitle("No-encoder backbone — GIFT-Eval full-97 GM-Relative MASE", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
