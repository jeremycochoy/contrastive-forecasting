#!/usr/bin/env python3
"""The four contrastive cosines through training, 0.8%-fork base vs triplet arm.

From the training CSVs (log-x; cosines are signed so the y-axes are linear):
  ff = cos(forecast, future)   — the forecast aligned with what actually comes next
  fp = cos(forecast, present)  — the forecast aligned with the current value (the shortcut)
  tp = cos(future, present)    — how distinct the future and present states are in latent space
  cross = cos(series i, series j) — different series; near 0 means they stay distinct (no collapse)
The contrastive gap reported elsewhere is ff − fp.
Env (elisa defaults): TRIP_CSV, BASE_CSV, OUT, START (default 100).
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_T = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-03_crossfade_triplet/runs"
RUNS_B = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/runs"
TRIP_CSV = os.environ.get("TRIP_CSV", f"{RUNS_T}/bb_allt08_xftrip_nobn_enc3_qk_aon_b1024_losses.csv")
BASE_CSV = os.environ.get("BASE_CSV", f"{RUNS_B}/bb_xshh_allt_forked2_qk_aon_6Lf_b1024_losses.csv")
OUT = os.environ.get("OUT", "/tmp/cf-328/experiments/2026-06-03_crossfade_triplet/plots/cosines.png")
START = int(os.environ.get("START", "100"))
TRIP_C, BASE_C = "#2f6da8", "#d08a3e"
PANELS = [
    ("ff", "cos(forecast, future)\nhigher = forecast matches the future"),
    ("fp", "cos(forecast, present)\nlower = not just echoing the present"),
    ("tp", "cos(future, present)\nlower = future and present states more distinct"),
    ("cross_batch", "cos(series i, series j)\nflat near 0 = different series stay distinct"),
]


def read(path, start):
    cols = {k: [] for k in ("step", "ff", "fp", "tp", "cross_batch")}
    for row in csv.DictReader(open(path)):
        if int(float(row["step"])) < start:
            continue
        for k in cols:
            cols[k].append(float(row[k]))
    return cols


trip, base = read(TRIP_CSV, START), read(BASE_CSV, START)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (key, title) in zip(axes.ravel(), PANELS):
    ax.semilogx(base["step"], base[key], color=BASE_C, lw=1.6, ls="--", label="0.8%-fork base")
    ax.semilogx(trip["step"], trip[key], color=TRIP_C, lw=1.9, label="crossfade triplet arm")
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
fig.suptitle("Contrastive cosines through training (log step), 0.8%-fork base vs crossfade triplet arm",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=125)
print("wrote", OUT)
for key, _ in PANELS:
    if base[key] and trip[key]:
        print(f"  {key:12s} base {base[key][0]:+.3f}->{base[key][-1]:+.3f}   "
              f"trip {trip[key][0]:+.3f}->{trip[key][-1]:+.3f}")
