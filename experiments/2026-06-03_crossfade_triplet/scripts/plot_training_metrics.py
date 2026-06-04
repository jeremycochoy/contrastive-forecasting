#!/usr/bin/env python3
"""#328 — training-dynamics panel: triplet arm vs the allt·0.8% parent.

6 log-log panels overlaid (triplet solid, parent dashed), smoothed, from --start.
Top row decays toward 0 (lower = better): loss (floor-subtracted),
gap_ratio=(1-ff)/(1-fp), 1-R²_naive. Then 1-R²_random. Bottom: U_batch and
U_temporal are USED-DIMENSIONS metrics — U = 1/(d·mean cos²) in [0,1] (src/
metrics.py); higher = more dimensions used (the collapse diagnostic), so these
rise, they do NOT go to 0. (AUC/Top-1 saturate ~1 and are not shown.)
Env (elisa defaults): TRIP_CSV, BASE_CSV, OUT, START (100), SMOOTH (120).
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

COLS = ["step", "loss", "gap_ratio", "r2_naive", "r2_random", "u_batch", "u_temporal"]
# panel: (column, title, transform)
PANELS = [
    ("loss", "contrastive loss − InfoNCE floor  (↓)", lambda v: v),
    ("gap_ratio", "ratio gap (1−ff)/(1−fp)  (↓ → 0)", lambda v: v),
    ("r2_naive", "1 − R²_naive  (↓ → 0)", lambda v: 1.0 - v),
    ("r2_random", "1 − R²_random  (↓ → 0)", lambda v: 1.0 - v),
    ("u_batch", "U_batch — used dimensions  (↑)", lambda v: v),
    ("u_temporal", "U_temporal — used dimensions  (↑)", lambda v: v),
]


def read(path, start):
    d = {k: [] for k in COLS}
    if not os.path.exists(path):
        return None
    for row in csv.DictReader(open(path)):
        if int(float(row["step"])) <= start:
            continue
        for k in COLS:
            d[k].append(float(row.get(k, "nan")))
    return {k: np.array(v) for k, v in d.items()} if d["step"] else None


def smooth(y, w):
    return np.convolve(y, np.ones(w) / w, mode="valid") if len(y) >= w else y


def overlay(ax, d, key, tf, color, ls, label):
    y = np.maximum(tf(d[key]), EPS)        # all panels are log-y -> clamp positive
    sm = smooth(y, SMOOTH)
    x = d["step"][len(d["step"]) - len(sm):]
    ax.plot(x, sm, color=color, ls=ls, lw=1.6, label=label)


trip, base = read(TRIP_CSV, START), read(BASE_CSV, START)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (key, title, tf) in zip(axes.ravel(), PANELS):
    if base is not None:
        overlay(ax, base, key, tf, BASE_C, "--", "0.8%-fork parent")
    if trip is not None:
        overlay(ax, trip, key, tf, TRIP_C, "-", "crossfade triplet arm")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step (from %d)" % START)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
last = int(trip["step"][-1]) if trip is not None and len(trip["step"]) else 0
fig.suptitle(f"Training dynamics (log-log) — crossfade triplet vs 0.8%-fork parent "
             f"(triplet at step {last}/12500; {SMOOTH}-step MA; from step {START}). "
             f"Top: lower=better. Bottom U_*: used dimensions, higher=better.", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
for tag, d in (("triplet", trip), ("parent", base)):
    if d is not None and len(d["step"]):
        print(f"  {tag:8s} step {int(d['step'][0])}..{int(d['step'][-1])}  "
              f"loss {d['loss'][-1]:.3f}  gap_ratio {d['gap_ratio'][-1]:.4f}  "
              f"1-R²_naive {1-d['r2_naive'][-1]:.4f}  1-R²_random {1-d['r2_random'][-1]:.4f}  "
              f"U_b {d['u_batch'][-1]:.4f}")
