#!/usr/bin/env python3
"""#313 — training curves: (B)+align+floor vs (B), both floor-subtracted.

Both curves are re-based by the SAME InfoNCE uniformity floor
`log(1 + N·e^(−1/τ))` (computed from the code's own helper), so they share a
baseline (0 = floor) and are directly comparable (PR #315 review item 1). The
new arm's logged `loss` is already floor-subtracted by training; (B)'s raw
`loss` has the same constant subtracted here.

Two panels (log-log):
 1. loss − floor — total training loss on a common baseline. The new arm
    carries the extra `+λ·L_align` term, so its plateau sits above (B)'s
    pure-contrastive floor by ≈ the converged align value.
 2. loss_tau_ref — normalized-InfoNCE at a fixed reference τ (align/floor do
    not enter it) → the cleanest comparable contrastive diagnostic.

(AUC / top-1 deliberately omitted: they saturate trivially here because the
negatives used to compute them are weak, so they carry no signal — per review.)
"""
import csv, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/home/jupyter/cf-wt-align-floor")
from src.loss import infonce_floor, _effective_negative_count

MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B"
CL_ABL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
OUT = f"{MAIN}/plots"; os.makedirs(OUT, exist_ok=True)

# (B) recipe: bs256 (1-GPU), t_raw=4096, W=16 → T=256, C=1, τ=0.1, hh_negs.
B_, T_, C_, TAU = 256, 4096 // 16, 1, 0.10
FLOOR = infonce_floor(TAU, _effective_negative_count("cosine_similarity_batch_full_hh_negs", B_, T_, C_))

C_NEW, C_B = "#ff7f0e", "#7f7f7f"
# label, colour, losses_csv, floor_already_subtracted_by_training?
ARMS = [
    ("(B)+L_align+floor", C_NEW, f"{MAIN}/runs/bb_alignfloor_50k_losses.csv", True),
    ("(B) baseline",      C_B,   f"{CL_ABL}/runs/cl_hh_50k_losses.csv",        False),
]


def load_csv(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f:
        return list(csv.DictReader(f))


curves = [(lab, col, rows, sub) for lab, col, p, sub in ARMS if (rows := load_csv(p)) is not None]
if not curves:
    print("loss: no CSV yet"); raise SystemExit

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
panels = [(f"loss − InfoNCE floor (both re-based; floor={FLOOR:.3f})", "loss"),
          ("loss_tau_ref  (norm InfoNCE, fixed τ — comparable)", "loss_tau_ref")]
for ax, (title, key) in zip(axs, panels):
    for lab, c, rows, already_sub in curves:
        xs, ys = [], []
        off = 0.0 if (key != "loss" or already_sub) else FLOOR  # re-base (B)'s raw loss only
        for r in rows:
            try:
                s = int(r["step"]); y = float(r[key]) - off
            except (KeyError, ValueError):
                continue
            if y <= 0: continue
            xs.append(s); ys.append(y)
        if not xs: continue
        if len(xs) > 800:
            idx = np.linspace(0, len(xs) - 1, 800).astype(int)
            xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
        ax.plot(xs, ys, color=c, lw=2.0, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, fontsize=10); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
axs[0].legend(loc="lower left", fontsize=9)
fig.suptitle("#313 training curves — (B)+L_align+floor vs (B), both floor-subtracted "
             "(τ=0.1, β2=0.95, fp16, bneck d=128)", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/loss.png", dpi=120, bbox_inches="tight"); plt.close()
print(f"wrote {OUT}/loss.png — arms={len(curves)} floor={FLOOR:.4f}")
