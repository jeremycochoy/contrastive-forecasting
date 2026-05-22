#!/usr/bin/env python3
"""#313 follow-up — 2L vs 6L q-head: does a stronger head recover the align
backbone? Grouped bars (align vs (B)) × (2L, 6L head), full-97 + triage-11,
with the v11c reference line. Lower = better.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B"
CL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
ENC = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster"
OUT = f"{MAIN}/plots"; os.makedirs(OUT, exist_ok=True)


def gm(p):
    if not os.path.exists(p): return None
    for line in open(p):
        if "Aggregate GM-Relative MASE" in line:
            for t in reversed(line.replace(":", " ").split()):
                try: return float(t)
                except ValueError: continue
    return None


# [full_dir, triage_dir] per (arm, head)
SRC = {
    ("(B)+L_align+floor", "2L"): (f"{MAIN}/results/gift_eval_full_bb_alignfloor_50k",
                                  f"{MAIN}/results/gift_eval_triage_bb_alignfloor_50k"),
    ("(B)+L_align+floor", "6L"): (f"{MAIN}/results/gift_eval_full_bb_alignfloor_50k_6L",
                                  f"{MAIN}/results/gift_eval_triage_bb_alignfloor_50k_6L"),
    ("(B) baseline", "2L"):      (f"{CL}/results/gift_eval_full_cl_hh_50k",
                                  f"{CL}/results/gift_eval_triage_cl_hh_50k"),
    ("(B) baseline", "6L"):      (f"{MAIN}/results/gift_eval_full_B_cl_hh_50k_6L",
                                  f"{MAIN}/results/gift_eval_triage_B_cl_hh_50k_6L"),
}
arms = ["(B)+L_align+floor", "(B) baseline"]
heads = ["2L", "6L"]
v11c_full = gm(f"{ENC}/results/gift_eval_full_v11c/summary.txt")
v11c_trg = gm(f"{ENC}/results/gift_eval_triage_v11c/summary.txt")

fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
COL = {"(B)+L_align+floor": "#ff7f0e", "(B) baseline": "#7f7f7f"}
HATCH = {"2L": "", "6L": "//"}
x = np.arange(len(arms)); w = 0.36
for ax, which, vref, title in [
    (axs[0], 0, v11c_full, "full-97 GM-Relative MASE"),
    (axs[1], 1, v11c_trg, "triage-11 GM-Relative MASE")]:
    for j, h in enumerate(heads):
        vals = [gm(f"{SRC[(a, h)][which]}/summary.txt") for a in arms]
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=[COL[a] for a in arms],
                      edgecolor="k", hatch=HATCH[h], alpha=0.9,
                      label=f"{h} head")
        for xi, v in zip(x + (j - 0.5) * w, vals):
            if v: ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    if vref:
        ax.axhline(vref, color="#9467bd", ls="--", lw=2.0, label=f"v11c (2L) = {vref:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=9)
    ax.set_ylabel(title); ax.set_title(title)
    ax.set_ylim(1.25, max(1.75, (vref or 1.4)) * 1.02 if which else 1.55)
    ax.grid(True, axis="y", alpha=0.3)
# de-dup legend (color=arm via bars; hatch=head; line=v11c)
from matplotlib.patches import Patch
h2 = [Patch(facecolor="w", edgecolor="k", hatch="", label="2L head"),
      Patch(facecolor="w", edgecolor="k", hatch="//", label="6L head"),
      Patch(facecolor=COL["(B)+L_align+floor"], label="(B)+L_align+floor"),
      Patch(facecolor=COL["(B) baseline"], label="(B) baseline"),
      plt.Line2D([0], [0], color="#9467bd", ls="--", lw=2, label="v11c (2L)")]
axs[0].legend(handles=h2, loc="upper right", fontsize=8, frameon=True)
fig.suptitle("#313 follow-up — 2L vs 6L q-head: a stronger head does NOT recover the align backbone "
             "(it overfits; align stays worse, gap widens)", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/head_2L_vs_6L.png", dpi=120, bbox_inches="tight"); plt.close()
print(f"wrote {OUT}/head_2L_vs_6L.png  v11c_full={v11c_full} v11c_trg={v11c_trg}")
