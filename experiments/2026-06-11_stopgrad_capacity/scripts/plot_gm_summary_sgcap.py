#!/usr/bin/env python3
"""#341 — headline figure.

Left: GM-Relative MASE per arm × cell (4 arms grouped). The stop-grad+bottleneck
arm (4) collapses at the last checkpoint (~2.2) while every arm sits ~1.16-1.21
at best-loss — the two findings in one panel.
Right: the capacity step enc3->enc6 as a paired-bootstrap Δ with 90% CI, WITHOUT
stop-grad (#336 references) vs WITH stop-grad (arms 2->3) — does stop-grad flip
the sign of the capacity knob? (No: it shrinks the penalty toward 0, never below.)
Env: OUT.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity"
OUT = os.environ.get("OUT", f"{EXP}/plots/gm_summary.png")

gm = {}
for r in csv.DictReader(open(f"{EXP}/results/gm_table.csv")):
    if r["gm"] not in ("", "None"):
        gm[(r["arm"], r["head"], r["ckpt"])] = float(r["gm"])
pair = {}
for r in csv.DictReader(open(f"{EXP}/results/pairwise_table.csv")):
    if r["delta"] not in ("", "None"):
        pair[(r["A"], r["B"], r["head"], r["ckpt"])] = (float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"]))

CELLS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
ARMS = [  # key, label, color
    ("a1_bn_enc6",      "arm1 bn·no-sg",     "#9dc3e6"),
    ("a2_sg_enc3_nobn", "arm2 enc3·full·sg", "#7f7f7f"),
    ("a3_sg_enc6_nobn", "arm3 enc6·full·sg", "#2ca02c"),
    ("a4_sg_enc6_bn",   "arm4 enc6·bn·sg",   "#d62728"),
]
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5), width_ratios=[1.25, 1])

CEIL = 1.5  # collapsed bars (~2.2) are clipped here so the ~1.16-1.21 differences stay legible
x = np.arange(len(CELLS)); w = 0.8 / len(ARMS)
for j, (key, lab, col) in enumerate(ARMS):
    vals = [gm.get((key, h, c), np.nan) for h, c in CELLS]
    drawn = [min(v, CEIL) if not np.isnan(v) else np.nan for v in vals]
    bx = x + (j - (len(ARMS) - 1) / 2) * w
    axL.bar(bx, drawn, w, label=lab, color=col)
    for xi, v in zip(bx, vals):
        if np.isnan(v):
            continue
        if v > CEIL:  # clipped bar: draw to ceiling, label true value with an up-arrow
            axL.text(xi, CEIL - 0.012, f"↑{v:.3f}", ha="center", va="top", fontsize=7,
                     rotation=90, color="white", fontweight="bold")
        else:
            axL.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
axL.axhline(1.0, color="grey", ls=":", lw=1)
axL.set_xticks(x, [f"{h} {c}" for h, c in CELLS])
axL.set_ylabel("GM-Relative MASE (lower is better)")
axL.set_ylim(1.0, CEIL)
axL.legend(fontsize=8, ncols=2, loc="upper left")
axL.set_title("Forecast error per arm × head × checkpoint", fontsize=10)
axL.annotate("stop-grad + bottleneck\ncollapses at 'last'\n(clipped at 1.5)", xy=(1.30, CEIL),
             xytext=(2.05, 1.36), fontsize=8, color="#d62728", ha="center",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

# Right: capacity step enc3->enc6, no-sg vs sg.
SERIES = [("WITHOUT stop-grad (#336)", "c_nobn_enc3", "c_nobn_enc6", "#c0504d"),
          ("WITH stop-grad (arms 2->3)", "a2_sg_enc3_nobn", "a3_sg_enc6_nobn", "#2e7d32")]
xr = np.arange(len(CELLS))
for k, (lab, A, B, col) in enumerate(SERIES):
    off = (k - 0.5) * 0.18
    for xi, (h, c) in zip(xr, CELLS):
        d = pair.get((A, B, h, c))
        if d:
            delta, lo, hi = d
            axR.errorbar(xi + off, delta, yerr=[[delta - lo], [hi - delta]], fmt="o", color=col,
                         capsize=4, lw=1.8, markersize=6, label=lab if xi == 0 else None)
axR.axhline(0.0, color="grey", lw=1)
axR.set_xticks(xr, [f"{h} {c}" for h, c in CELLS])
axR.set_ylabel("Δ GM-Relative MASE  (enc6 − enc3)")
axR.set_title("Capacity step enc3→enc6: penalty WITHOUT vs WITH stop-grad\n(Δ>0 = enc6 worse; 90% CI)", fontsize=10)
axR.legend(fontsize=8, loc="upper left")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
