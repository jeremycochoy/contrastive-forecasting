#!/usr/bin/env python3
"""Round-progression bar chart for qhead-improvements.

Reads each round's committed triage summary.txt, recomputes the triage
GM-Rel MASE as the geometric mean of the per-config Relative column
(model MASE / seasonal-naive MASE) over the 11-config proxy, and plots
one bar per round against the seasonal-naive=1.0 reference line.

Output: plots/round_progression.png
All numbers come from results/<run>_triage/summary.txt (committed).
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")

# (label, results-dir, head/recipe one-liner) in forward-experiment order.
ROUNDS = [
    ("baseline", "baseline_legacy_triage", "legacy GRU-q"),
    ("R1_E1", "R1_E1_linear_quant_lr3e4_triage", "linear-q"),
    ("R3_E4", "R3_E4_xfmr6L_quant_moirai_cosine_triage", "xfmr-q 6L"),
    ("R5_E7", "R5_E7_xfmr12L_quant_moirai_cosine_60k_triage", "xfmr-q 12L 60k"),
    ("R6_E8", "R6_E8_xfmr6L_bidir_fl128_quant_moirai_cosine_triage", "bidir fl128"),
    ("R7_E9", "R7_E9_xfmr12L_quant_moirai_cosine_100k_triage", "xfmr-q 12L 100k*"),
    ("R8_E10", "R8_E10_xfmr12L_gauss_moirai_cosine_60k_triage", "gauss 12L"),
    ("R9_E13", "R9_E13_xfmr12L_quant_moirai_cosine_e_then_f_60k_triage", "e_then_f 60k"),
    ("R9_E14", "R9_E14_xfmr12L_quant_moirai_cosine_e_then_f_100k_triage", "e_then_f 100k"),
]

WINNER = "R9_E13"


def gm_rel_from_summary(path):
    """Geometric mean of the per-config Relative column in a GIFT-Eval summary.txt."""
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
            # data rows only: Relative must equal MASE / SN_MASE.
            if sn > 0 and abs(rel - mase / sn) < 1e-3:
                rels.append(rel)
    if not rels:
        raise RuntimeError(f"no config rows parsed from {path}")
    return float(np.exp(np.mean(np.log(rels)))), len(rels)


labels, vals, recipes, ns = [], [], [], []
for label, d, recipe in ROUNDS:
    g, n = gm_rel_from_summary(os.path.join(RES, d, "summary.txt"))
    labels.append(label)
    vals.append(g)
    recipes.append(recipe)
    ns.append(n)

assert all(n == 11 for n in ns), f"expected 11-config triage everywhere, got {ns}"

fig, ax = plt.subplots(figsize=(10, 5.2))
x = np.arange(len(labels))
colors = []
for label, v in zip(labels, vals):
    if label == WINNER:
        colors.append("#2e7d32")          # winner: green
    elif v < 1.0:
        colors.append("#66bb6a")          # below naive: light green
    else:
        colors.append("#9e9e9e")          # at/above naive: grey

bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6, zorder=3)

# seasonal-naive reference.
ax.axhline(1.0, color="#c62828", lw=1.4, ls="--", zorder=2,
           label="seasonal-naive (1.000)")
# Moirai full-eval reference (leaderboard, not this triage subset).
ax.axhline(0.809, color="#1565c0", lw=1.2, ls=":", zorder=2,
           label="Moirai (full eval, 0.809)")

for xi, (v, recipe, label) in enumerate(zip(vals, recipes, labels)):
    ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold" if label == WINNER else "normal")
    ax.text(xi, 0.83, recipe, ha="center", va="bottom", fontsize=7.5,
            rotation=90, color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Triage GM-Rel MASE  (11-config proxy; lower better)")
ax.set_title("Head-improvement rounds: triage GM-Rel MASE\n"
             "R9_E13 (e_then_f) is the only round below seasonal-naive")
ax.set_ylim(0.80, 1.16)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.tight_layout()
out = os.path.join(ROOT, "plots", "round_progression.png")
fig.savefig(out, dpi=130)
print("wrote", out)
for label, v in zip(labels, vals):
    print(f"  {label:9} {v:.4f}")
