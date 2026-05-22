#!/usr/bin/env python3
"""#313 headline — full-97 GM-MASE: per-arm rectangles vs the v11c line.

Bars = (B)+align+floor and (B); dashed line = v11c (the target);
dotted line = seasonal naive (1.0). Lower = better. Adapted from #309
plot_summary.py. Auto-skips arms whose summary.txt is missing.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B"
CL_ABL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
ENC = "/home/jupyter/contrastive-forecasting/experiments/2026-05-11_exp_encoder_forecaster"
OUT = f"{MAIN}/plots"; os.makedirs(OUT, exist_ok=True)

C_NEW, C_B, C_V = "#ff7f0e", "#7f7f7f", "#9467bd"
# label, full_eval_dir, colour
ARMS = [
    ("(B)+L_align+floor  bneck·fp16·τ0.1·β2.95", f"{MAIN}/results/gift_eval_full_bb_alignfloor_50k", C_NEW),
    ("(B)  bneck·fp16·τ0.1·β2.95",               f"{CL_ABL}/results/gift_eval_full_cl_hh_50k",        C_B),
]
V11C = f"{ENC}/results/gift_eval_full_v11c"


def agg_gm(sum_txt):
    if not os.path.exists(sum_txt): return None
    with open(sum_txt) as f:
        for line in f:
            if "Aggregate GM-Relative MASE" in line:
                for t in reversed(line.replace(":", " ").split()):
                    try: return float(t)
                    except ValueError: continue
    return None


rows = []
for lab, edir, col in ARMS:
    gm = agg_gm(f"{edir}/summary.txt")
    if gm is not None:
        rows.append((lab, gm, col))
    else:
        print(f"summary: skipping {lab} (no eval yet)")
v11c = agg_gm(f"{V11C}/summary.txt")

if not rows:
    print("summary: no arm GMs yet — skipping")
    raise SystemExit

rows.sort(key=lambda r: r[1], reverse=True)   # worst at top
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
fig, ax = plt.subplots(figsize=(11, 0.7 * len(rows) + 2.2))
y = range(len(rows))
ax.barh(list(y), vals, color=cols, alpha=0.85, height=0.6)
for i, v in enumerate(vals):
    ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("full-97 GM-Relative MASE (lower = better)")
xmin = min([1.27] + [v for v in [v11c] if v]) - 0.01
ax.set_xlim(xmin, max(vals) * 1.03)
handles = []
if v11c:
    ax.axvline(v11c, color=C_V, ls="--", lw=2.0)
    handles.append(plt.Line2D([0], [0], color=C_V, ls="--", lw=2.0, label=f"v11c (target) = {v11c:.4f}"))
ax.axvline(1.0, color="k", ls=":", lw=1.0, alpha=0.5)
handles.append(plt.Line2D([0], [0], color="k", ls=":", lw=1.0, alpha=0.5, label="seasonal naive = 1.0"))
ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)
ax.set_title("#313 — does L_align (+floor) on (B) close the gap to v11c?  full GIFT-Eval (97 cfg)", fontsize=12)
ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/gm_summary.png", dpi=120, bbox_inches="tight"); plt.close()
print(f"wrote {OUT}/gm_summary.png — {len(rows)} arms; v11c={v11c}")
