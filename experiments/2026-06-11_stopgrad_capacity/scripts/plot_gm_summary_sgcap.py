#!/usr/bin/env python3
"""#341 — headline figure from results/{gm_table,pairwise_table}.csv.

Left: GM-Relative MASE per arm (4 grouped bars) per head × checkpoint.
Right: the three hypothesis contrasts (arm3−arm2: capacity under stop-grad;
arm4−arm1: stop-grad on base; arm4−arm2: bottleneck+enc6 vs the #339 arm)
as paired-bootstrap Δ with 90% CI. Env: OUT (png).
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-11_stopgrad_capacity"
OUT = os.environ.get("OUT", "/tmp/cf-341/experiments/2026-06-11_stopgrad_capacity/plots/gm_summary.png")

ARM_STYLE = [  # (key, short label, color)
    ("a1_bn_enc6",      "base+triplet (no sg)", "#9dc3e6"),
    ("a2_sg_enc3_nobn", "sg enc3+nobn (#339)",  "#7f7f7f"),
    ("a3_sg_enc6_nobn", "sg enc6+nobn NEW",     "#c0504d"),
    ("a4_sg_enc6_bn",   "sg enc6+bn NEW",       "#70ad47"),
]
CONTRASTS = [  # (A, B, label)
    ("a2_sg_enc3_nobn", "a3_sg_enc6_nobn", "capacity under sg\n(arm3 − arm2)"),
    ("a1_bn_enc6",      "a4_sg_enc6_bn",   "sg on base\n(arm4 − arm1)"),
    ("a2_sg_enc3_nobn", "a4_sg_enc6_bn",   "enc6+bn vs #339 arm\n(arm4 − arm2)"),
]
CELLS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]

gms = {}
for r in csv.DictReader(open(f"{EXP}/results/gm_table.csv")):
    if r["gm"] not in ("", "None"):
        gms[(r["arm"], r["head"], r["ckpt"])] = float(r["gm"])
pairs = {}
for r in csv.DictReader(open(f"{EXP}/results/pairwise_table.csv")):
    if r["delta"] not in ("", "None"):
        pairs[(r["A"], r["B"], r["head"], r["ckpt"])] = (
            float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), width_ratios=[1.15, 1])

x = np.arange(len(CELLS))
nb = len(ARM_STYLE)
w = 0.8 / nb
vals_all = []
for j, (key, lab, col) in enumerate(ARM_STYLE):
    vals = [gms.get((key, h, c), np.nan) for h, c in CELLS]
    vals_all += [v for v in vals if not np.isnan(v)]
    ax1.bar(x + (j - (nb - 1) / 2) * w, vals, w, label=lab, color=col)
    for xi, v in zip(x, vals):
        if not np.isnan(v):
            ax1.text(xi + (j - (nb - 1) / 2) * w, v, f"{v:.3f}", ha="center",
                     va="bottom", fontsize=6.5, rotation=90)
ax1.set_xticks(x, [f"{h} {c}" for h, c in CELLS])
ax1.set_ylabel("GM-Relative MASE (lower is better)")
if vals_all:
    ax1.set_ylim(min(vals_all) - 0.03, max(vals_all) + 0.05)
ax1.legend(fontsize=8, ncols=2)
ax1.set_title("Forecast error per head × checkpoint", fontsize=10)

xticks, xlabels = [], []
xi = 0
for A, B, lab in CONTRASTS:
    for h, c in CELLS:
        if (A, B, h, c) in pairs:
            d, lo, hi = pairs[(A, B, h, c)]
            reliable = hi < 0 or lo > 0
            col = ("#2e7d32" if hi < 0 else "#c62828") if reliable else "#777777"
            ax2.errorbar(xi, d, yerr=[[d - lo], [hi - d]], fmt="o", color=col,
                         capsize=4, lw=1.8, markersize=5)
        xticks.append(xi)
        xlabels.append(f"{h[0]}{c[0]}")  # 2b 2l 6b 6l
        xi += 1
    xi += 1  # gap between contrast groups
ax2.axhline(0.0, color="grey", lw=1)
ax2.set_xticks(xticks, xlabels, fontsize=8)
for k, (A, B, lab) in enumerate(CONTRASTS):
    ax2.text(k * 5 + 1.5, ax2.get_ylim()[1], lab, ha="center", va="top", fontsize=8)
ax2.set_ylabel("Δ GM-Relative MASE (B − A)")
ax2.set_title("Hypothesis contrasts: paired-bootstrap Δ, 90% CI (below 0 = B better)",
              fontsize=10)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
