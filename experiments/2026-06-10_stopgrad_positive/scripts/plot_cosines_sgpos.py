#!/usr/bin/env python3
"""The four contrastive cosines through training, stop-grad vs reference.

Same panel as #326's cosine plot (log-x, linear y):
  ff = cos(forecast, future), fp = cos(forecast, present),
  tp = cos(future, present), cross = cos(series i, series j).
Source CSVs as in plot_training_metrics.py.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
SG_CSV = f"{EXP}/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_losses.csv"
REF_CSV = f"{EXP}/results/reference/bb_allt08_xftrip_nobn_enc3_qk_aon_b1024_losses.csv"
OUT = "/tmp/cf-sgpos/experiments/2026-06-10_stopgrad_positive/plots/cosines.png"
START = int(os.environ.get("START", "100"))
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


sg, ref = read(SG_CSV, START), read(REF_CSV, START)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (key, title) in zip(axes.ravel(), PANELS):
    ax.semilogx(ref["step"], ref[key], color="#2f6da8", lw=1.6, ls="--",
                label="reference (no stop-grad)")
    ax.semilogx(sg["step"], sg[key], color="#d62728", lw=1.9, label="stop-grad")
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
fig.suptitle("Contrastive cosines through training (log step), stop-grad vs reference", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=125)
print("wrote", OUT)
for key, _ in PANELS:
    print(f"  {key:12s} ref {ref[key][0]:+.3f}->{ref[key][-1]:+.3f}   "
          f"sg {sg[key][0]:+.3f}->{sg[key][-1]:+.3f}")
