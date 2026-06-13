#!/usr/bin/env python3
"""#341 — GM-Relative MASE grouped BY ARCHITECTURE (x-axis = arm), one bar per
head×checkpoint. Complements gm_summary.png (which groups by cell): here each
architecture's four cells sit together, so the best-vs-last divergence per arm is
read at a glance — arm 4's two 'last' bars spike to ~2.2 while its 'best' bars sit
with everyone else. Env: OUT.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity"
OUT = os.environ.get("OUT", f"{EXP}/plots/gm_by_arch.png")

gm = {}
for r in csv.DictReader(open(f"{EXP}/results/gm_table.csv")):
    if r["gm"] not in ("", "None"):
        gm[(r["arm"], r["head"], r["ckpt"])] = float(r["gm"])

ARMS = [  # key, x-axis label (architecture)
    ("a1_bn_enc6",      "arm1\nbn · no-sg\n(#336)"),
    ("a2_sg_enc3_nobn", "arm2\nenc3 · full · sg\n(#339)"),
    ("a3_sg_enc6_nobn", "arm3\nenc6 · full · sg\n(new)"),
    ("a4_sg_enc6_bn",   "arm4\nenc6 · bn · sg\n(new)"),
]
# one bar per head×checkpoint — best bars (greens) left, last bars (reds) right
CELLS = [("2L", "best", "2L best", "#a6d96a"),
         ("6L", "best", "6L best", "#1a9641"),
         ("2L", "last", "2L last", "#fdae61"),
         ("6L", "last", "6L last", "#d7191c")]

CEIL = 1.5  # collapsed bars (~2.2) are clipped here so the ~1.16-1.21 differences stay legible
fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(ARMS)); w = 0.8 / len(CELLS)
for j, (h, c, lab, col) in enumerate(CELLS):
    vals = [gm.get((key, h, c), np.nan) for key, _ in ARMS]
    drawn = [min(v, CEIL) if not np.isnan(v) else np.nan for v in vals]
    bx = x + (j - (len(CELLS) - 1) / 2) * w
    ax.bar(bx, drawn, w, label=lab, color=col)
    for xi, v in zip(bx, vals):
        if np.isnan(v):
            continue
        if v > CEIL:  # clipped bar: draw to ceiling, label true value with an up-arrow
            ax.text(xi, CEIL - 0.012, f"↑{v:.3f}", ha="center", va="top", fontsize=7.5,
                    rotation=90, color="white", fontweight="bold")
        else:
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
ax.axhline(1.0, color="grey", ls=":", lw=1, label="seasonal-naive")
ax.set_xticks(x, [lab for _, lab in ARMS])
ax.set_ylabel("GM-Relative MASE (lower is better)")
ax.set_ylim(1.0, CEIL)
ax.legend(fontsize=8, ncols=5, loc="upper center")
ax.set_title("GM-Relative MASE by architecture (one bar per head × checkpoint)", fontsize=11)
ax.annotate("only arm 4's 'last' bars collapse\n(clipped at 1.5; true values labelled)",
            xy=(3.0, CEIL), xytext=(1.75, 1.40),
            fontsize=8, color="#d7191c", ha="center",
            arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.2))
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
