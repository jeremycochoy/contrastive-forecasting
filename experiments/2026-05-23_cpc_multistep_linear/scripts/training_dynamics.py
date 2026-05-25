#!/usr/bin/env python3
"""#316 training dynamics — 2x2 grid, all arms.

  loss        (log-log)   training loss
  gap_ratio   (log-log)   forecast "pos-gap" ratio: cos(f,h_t)/cos(f,h_{t+1}),
                          -> 0 as the forecast stops resembling the current
                          latent (lower = better).
  u_batch     (log-x)     uniformity along the batch dimension
  u_temporal  (log-x)     uniformity along the time dimension
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "/home/jupyter/contrastive-forecasting/experiments"
M = f"{R}/2026-05-23_cpc_multistep_linear/runs"
OUT = "/home/jupyter/contrastive-forecasting/.claude/worktrees/exp-bottleneck-beta2-confound/experiments/2026-05-23_cpc_multistep_linear/plots/training_dynamics.png"

# (label, color, linestyle, csv)  — k=1 dashed, k=12 solid, β the black reference.
ARMS = [
    ("β  (transformer-1L / β-neg / k=1)", "#000000", "-",  f"{R}/2026-05-20_bottleneck_beta2_confound/runs/bb_beta_50k_losses.csv"),
    ("transformer-1L / β-neg / k=12",     "#1f77b4", "-",  f"{M}/bb_cpctrf_k12_s20260520_fp32_50k_losses.csv"),
    ("linear / β-neg / k=1",              "#2ca02c", "--", f"{M}/bb_linbn_k1_s20260520_fp32_50k_losses.csv"),
    ("linear / β-neg / k=12",             "#2ca02c", "-",  f"{M}/bb_linbn_k12_s20260520_fp32_50k_losses.csv"),
    ("linear / CPC-neg / k=1",            "#d62728", "--", f"{M}/bb_lincn_k1_s20260520_fp32_50k_losses.csv"),
    ("linear / CPC-neg / k=12 (seed A)",  "#d62728", "-",  f"{M}/bb_cpc_k12_s20260520_fp32_50k_losses.csv"),
    ("linear / CPC-neg / k=12 (seed B)",  "#d62728", ":",  f"{M}/bb_cpc_k12_s20260523_fp32_50k_combined_losses.csv"),
]

# (column, y-scale, title)
PANELS = [
    ("loss",       "log",    "Training loss  (log-log)"),
    ("gap_ratio",  "log",    "Forecast 'pos-gap' ratio  (log-log; lower = better, → 0)"),
    ("u_batch",    "linear", "Uniformity — batch dimension  (log-x)"),
    ("u_temporal", "linear", "Uniformity — time dimension  (log-x)"),
]

def series(path, col):
    if not os.path.exists(path): return None, None
    xs, ys = [], []
    for r in csv.DictReader(open(path)):
        try: s, y = int(r["step"]), float(r[col])
        except (KeyError, ValueError): continue
        if s <= 0 or y <= 0: continue
        xs.append(s); ys.append(y)
    if not xs: return None, None
    if len(xs) > 600:
        idx = np.linspace(0, len(xs)-1, 600).astype(int)
        xs = [xs[i] for i in idx]; ys = [ys[i] for i in idx]
    return xs, ys

fig, axs = plt.subplots(2, 2, figsize=(13, 9))
handles = None
for ax, (col, yscale, title) in zip(axs.flat, PANELS):
    for lab, color, ls, path in ARMS:
        xs, ys = series(path, col)
        if xs: ax.plot(xs, ys, color=color, ls=ls, lw=1.8, label=lab)
    ax.set_xscale("log"); ax.set_yscale(yscale)
    ax.set_xlabel("training step"); ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    if handles is None: handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("#316 training dynamics — all arms  (k=1 dashed · k=12 solid · β black · seed B dotted)", fontsize=12)
plt.tight_layout(rect=(0, 0.05, 1, 0.98))
plt.savefig(OUT, dpi=120, bbox_inches="tight"); plt.close()
print("wrote", OUT)
