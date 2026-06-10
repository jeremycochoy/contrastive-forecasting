#!/usr/bin/env python3
"""Stop-grad follow-up — training-dynamics log-log panel, stop-grad vs reference.

6 log-log panels from --start: loss, ratio gap (1-ff)/(1-fp), 1-R²_naive,
1-R²_random (top, lower=better); U_batch, U_temporal (bottom, used dimensions,
higher=better). Same panel layout and metrics as #328's disentanglement plot,
so the two reports read the same way.
Env: OUT (png), START (100), SMOOTH (120).
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
ARMS = [  # (label, color, linestyle, csv)
    ("reference (no stop-grad)", "#2f6da8", "--",
     f"{EXP}/results/reference/bb_allt08_xftrip_nobn_enc3_qk_aon_b1024_losses.csv"),
    ("stop-grad", "#d62728", "-",
     f"{EXP}/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_losses.csv"),
]
OUT = os.environ.get("OUT", "/tmp/cf-sgpos/experiments/2026-06-10_stopgrad_positive/plots/training_metrics.png")
START = int(os.environ.get("START", "100"))
SMOOTH = int(os.environ.get("SMOOTH", "120"))
EPS = 1e-3
PANELS = [
    ("loss", "contrastive loss − InfoNCE floor  (↓)", lambda v: v),
    ("gap_ratio", "ratio gap (1−ff)/(1−fp)  (↓→0)", lambda v: v),
    ("r2_naive", "1 − R²_naive  (↓→0)", lambda v: 1.0 - v),
    ("r2_random", "1 − R²_random  (↓→0)", lambda v: 1.0 - v),
    ("u_batch", "U_batch — used dimensions  (↑)", lambda v: v),
    ("u_temporal", "U_temporal — used dimensions  (↑)", lambda v: v),
]
COLS = ["step", "loss", "gap_ratio", "r2_naive", "r2_random", "u_batch", "u_temporal"]


def read(path):
    if not os.path.exists(path):
        return None
    d = {k: [] for k in COLS}
    for row in csv.DictReader(open(path)):
        if int(float(row["step"])) <= START:
            continue
        for k in COLS:
            d[k].append(float(row.get(k, "nan")))
    return {k: np.array(v) for k, v in d.items()} if d["step"] else None


def smooth(y, w):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


loaded = [(lab, c, ls, read(p)) for lab, c, ls, p in ARMS]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (key, title, tf) in zip(axes.ravel(), PANELS):
    for lab, c, ls, d in loaded:
        if d is None:
            continue
        y = np.maximum(tf(d[key]), EPS)
        sm = smooth(y, SMOOTH)
        ax.plot(d["step"][len(d["step"]) - len(sm):], sm, color=c, ls=ls, lw=1.5, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step (from %d)" % START)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
laststep = max((int(d["step"][-1]) for _, _, _, d in loaded if d is not None), default=0)
fig.suptitle("Stop-grad on the positive's encoder side vs the reference — training dynamics "
             f"(log-log; latest step {laststep}; {SMOOTH}-step MA). Top: lower=better. "
             "Bottom U_*: used dimensions, higher=better.", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for lab, _, _, d in loaded:
    if d is not None and len(d["step"]):
        print(f"  {lab:28s} step {int(d['step'][0])}..{int(d['step'][-1])}  loss {d['loss'][-1]:.3f}  "
              f"gap_ratio {d['gap_ratio'][-1]:.4f}  1-R²_naive {1-d['r2_naive'][-1]:.4f}  U_b {d['u_batch'][-1]:.3f}")
