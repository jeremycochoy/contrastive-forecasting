#!/usr/bin/env python3
"""#313 — training curves: (B)+align+floor vs (B).

4 panels (log-log), each (B) in grey and the new arm in orange:
 1. loss               — raw logged loss. The new arm is floor-subtracted
                         (re-based to ~0 at the InfoNCE uniformity floor)
                         AND carries the +L_align term; (B) is the raw
                         normalized InfoNCE (plateaus at the floor > 0).
                         The two are therefore NOT on a common baseline —
                         panels 2-4 are the comparable diagnostics.
 2. loss_tau_ref       — normalized-InfoNCE at a fixed reference τ,
                         computed identically regardless of align/floor →
                         directly comparable; "did L_align change the
                         contrastive convergence vs (B)?"
 3. 1 − AUC            — 1 − (separability of positive vs negative pairs).
 4. top1               — top-1 retrieval accuracy of the true future.

Robust to a missing new-arm CSV (skipped) so it validates on (B) alone.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAIN = "/home/jupyter/contrastive-forecasting/experiments/2026-05-22_align_floor_loss_B"
CL_ABL = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
OUT = f"{MAIN}/plots"; os.makedirs(OUT, exist_ok=True)

C_NEW, C_B = "#ff7f0e", "#7f7f7f"
ARMS = [   # label, colour, losses_csv
    ("(B)+L_align+floor", C_NEW, f"{MAIN}/runs/bb_alignfloor_50k_losses.csv"),
    ("(B) baseline",      C_B,   f"{CL_ABL}/runs/cl_hh_50k_losses.csv"),
]


def load_csv(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f:
        return list(csv.DictReader(f))


curves = [(lab, col, rows) for lab, col, p in ARMS if (rows := load_csv(p)) is not None]
for lab, _, p in ARMS:
    if load_csv(p) is None:
        print(f"loss: skipping {lab} (no CSV yet at {p})")
if not curves:
    print("loss: no CSV yet — skipping"); raise SystemExit

fig, axs = plt.subplots(2, 2, figsize=(13, 9))
panels = [("loss (new: floor-subtracted +L_align;  (B): raw InfoNCE)", "loss"),
          ("loss_tau_ref  (norm InfoNCE, fixed τ — comparable)", "loss_tau_ref"),
          ("1 − AUC", "auc"), ("top1 retrieval acc", "top1")]
for ax, (title, key) in zip(axs.flat, panels):
    for lab, c, rows in curves:
        xs, ys = [], []
        for r in rows:
            try:
                s = int(r["step"]); y = float(r[key])
            except (KeyError, ValueError):
                continue
            if title.startswith("1 − AUC"): y = 1.0 - y
            if key in ("loss", "loss_tau_ref", "auc") and y <= 0: continue
            xs.append(s); ys.append(y)
        if not xs: continue
        if len(xs) > 800:
            idx = np.linspace(0, len(xs) - 1, 800).astype(int)
            xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
        ax.plot(xs, ys, color=c, lw=2.0, label=lab)
    ax.set_xscale("log")
    if key != "top1":
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
axs[0, 0].legend(loc="upper right", fontsize=9)
fig.suptitle("#313 training curves — (B)+L_align+floor vs (B)  (τ=0.1, β2=0.95, fp16, bneck d=128)", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/loss.png", dpi=120, bbox_inches="tight"); plt.close()
print(f"wrote {OUT}/loss.png — arms={len(curves)}")
