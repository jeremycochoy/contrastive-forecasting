#!/usr/bin/env python3
"""#328 — training-dynamics panel: triplet arm vs the allt·0.8% parent.

8 panels overlaid (triplet solid, parent dashed), smoothed, from --start (log-x):
  loss (log-log), gap_ratio=(1-ff)/(1-fp) [-> 0, log-log], R²_naive, R²_random,
  AUC, Top-1, U_batch (log-log), gap=ff-fp [-> 1]. The ratio gap converging to 0
  is the convergence diagnostic (the subtraction gap -> 1 is the same fact, the
  other way round). Env (elisa defaults): TRIP_CSV, BASE_CSV, OUT, START, SMOOTH.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS_T = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet/runs"
RUNS_B = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/runs"
TRIP_CSV = os.environ.get("TRIP_CSV", f"{RUNS_T}/bb_allt08_xftrip_nobn_enc3_qk_aon_b1024_losses.csv")
BASE_CSV = os.environ.get("BASE_CSV", f"{RUNS_B}/bb_xshh_allt_forked2_qk_aon_6Lf_b1024_losses.csv")
OUT = os.environ.get("OUT", "/tmp/cf-328/experiments/2026-06-03_crossfade_triplet/plots/training_metrics.png")
START = int(os.environ.get("START", "100"))
SMOOTH = int(os.environ.get("SMOOTH", "120"))
TRIP_C, BASE_C = "#2f6da8", "#d08a3e"
EPS = 1e-3

COLS = ["step", "loss", "gap", "gap_ratio", "ff", "fp", "r2_random", "r2_naive",
        "u_batch", "auc", "top1"]


def read(path, start):
    d = {k: [] for k in COLS}
    if not os.path.exists(path):
        return None
    for row in csv.DictReader(open(path)):
        if int(float(row["step"])) < start:
            continue
        for k in COLS:
            d[k].append(float(row.get(k, "nan")))
    return {k: np.array(v) for k, v in d.items()}


def smooth(y, w):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def overlay(ax, d, key, color, ls, label):
    y = d[key]
    sm = smooth(y, SMOOTH)
    x = d["step"][len(d["step"]) - len(sm):]
    ax.plot(x, np.maximum(sm, EPS) if key in ("loss", "gap_ratio", "u_batch") else sm,
            color=color, ls=ls, lw=1.6, label=label)


trip, base = read(TRIP_CSV, START), read(BASE_CSV, START)
# panel: (column, title, xscale, yscale, ylim)
PANELS = [
    ("loss", "contrastive loss − InfoNCE floor", "log", "log", None),
    ("gap_ratio", "ratio gap (1−ff)/(1−fp)  → 0", "log", "log", None),
    ("r2_naive", "R²_naive  → 1", "log", "linear", (0, 1)),
    ("r2_random", "R²_random  → 1", "log", "linear", (0, 1)),
    ("auc", "AUC (future-vs-present retrieval)  → 1", "log", "linear", None),
    ("top1", "Top-1 retrieval  → 1", "log", "linear", None),
    ("u_batch", "U_batch (cross-series uniformity)", "log", "log", None),
    ("gap", "subtraction gap ff − fp  → 1", "log", "linear", None),
]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, (key, title, xs, ys, ylim) in zip(axes.ravel(), PANELS):
    if base is not None:
        overlay(ax, base, key, BASE_C, "--", "0.8%-fork parent")
    if trip is not None:
        overlay(ax, trip, key, TRIP_C, "-", "crossfade triplet arm")
    ax.set_xscale(xs)
    ax.set_yscale(ys)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
last = int(trip["step"][-1]) if trip is not None and len(trip["step"]) else 0
fig.suptitle(f"Training dynamics — crossfade triplet arm vs 0.8%-fork parent "
             f"(triplet at step {last}/12500; {SMOOTH}-step MA; from step {START})", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for tag, d in (("triplet", trip), ("parent", base)):
    if d is not None and len(d["step"]):
        print(f"  {tag:8s} step {int(d['step'][0])}..{int(d['step'][-1])}  "
              f"loss {d['loss'][0]:.3f}->{d['loss'][-1]:.3f}  "
              f"gap_ratio {d['gap_ratio'][0]:.3f}->{d['gap_ratio'][-1]:.3f}  "
              f"R²_naive {d['r2_naive'][0]:.3f}->{d['r2_naive'][-1]:.3f}  "
              f"AUC {d['auc'][0]:.3f}->{d['auc'][-1]:.3f}  top1 {d['top1'][0]:.3f}->{d['top1'][-1]:.3f}")
