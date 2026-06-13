#!/usr/bin/env python3
"""Headline figure: full 97-config GIFT-Eval GM-Rel MASE.

R9_E13 is recomputed from the committed full summary.txt (geomean of the
per-config Relative column). The legacy GRU baseline (#10) full-eval
number is carried from the prior #10 RESUME50k report (notes/CANDIDATES.md
records GM-MASE 1.1828) and is NOT recomputable from data committed in
this experiment dir -- it is drawn hatched and labelled as carried-over.
Leaderboard numbers are published GIFT-Eval figures.

Output: plots/headline_full_eval.png
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")


def gm_rel_from_summary(path):
    rels = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                rel, sn, mase = float(parts[-1]), float(parts[-2]), float(parts[-3])
            except ValueError:
                continue
            if sn > 0 and abs(rel - mase / sn) < 1e-3:
                rels.append(rel)
    return float(np.exp(np.mean(np.log(rels)))), len(rels)


r9, n = gm_rel_from_summary(
    os.path.join(RES, "R9_E13_xfmr12L_quant_moirai_cosine_e_then_f_60k_full",
                 "summary.txt"))
assert n == 97, f"expected 97 full configs, got {n}"

# (label, value, kind) -- kind drives colour/hatch.
BASELINE_10 = 1.183  # carried from #10 report (CANDIDATES.md 1.1828); not recomputable here
rows = [
    ("Sundial",        0.673, "leaderboard"),
    ("TimesFM",        0.680, "leaderboard"),
    ("PatchTST",       0.762, "leaderboard"),
    ("Chronos",        0.786, "leaderboard"),
    ("Moirai",         0.809, "leaderboard"),
    ("R9_E13 (ours)",  r9,    "ours"),
    ("legacy GRU #10", BASELINE_10, "carried"),
]
# sort ascending (best at top).
rows = sorted(rows, key=lambda r: r[1])
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
kinds = [r[2] for r in rows]

cmap = {"leaderboard": "#90a4ae", "ours": "#2e7d32", "carried": "#ef9a9a"}
colors = [cmap[k] for k in kinds]
hatches = ["//" if k == "carried" else "" for k in kinds]

fig, ax = plt.subplots(figsize=(9.5, 4.6))
y = np.arange(len(labels))
bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
for b, h in zip(bars, hatches):
    b.set_hatch(h)

ax.axvline(1.0, color="#c62828", lw=1.4, ls="--", zorder=2,
           label="seasonal-naive (1.000)")

for yi, (v, k) in enumerate(zip(vals, kinds)):
    suffix = "  (carried from #10)" if k == "carried" else ""
    ax.text(v + 0.008, yi, f"{v:.3f}{suffix}", va="center", fontsize=9,
            fontweight="bold" if k == "ours" else "normal")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Full 97-config GIFT-Eval GM-Rel MASE  (lower better)")
ax.set_title("Headline: causal-transformer head + e_then_f input layout\n"
             "takes the frozen backbone from 1.183 (legacy GRU) to 1.029 on full eval")
ax.set_xlim(0.0, 1.45)
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(axis="x", alpha=0.3, zorder=0)
ax.invert_yaxis()

fig.tight_layout()
out = os.path.join(ROOT, "plots", "headline_full_eval.png")
fig.savefig(out, dpi=130)
print("wrote", out)
print(f"R9_E13 full recomputed = {r9:.4f} over {n} configs")
