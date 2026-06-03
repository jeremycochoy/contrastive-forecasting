#!/usr/bin/env python3
"""The four contrastive cosines through training, best recipe vs + crossfade.

From the training CSVs (log-x; cosines are signed so the y-axes are linear):
  ff = cos(forecast, future)   — the forecast aligned with what actually comes next
  fp = cos(forecast, present)  — the forecast aligned with the current value (the shortcut)
  tp = cos(future, present)    — how distinct the future and present states are in latent space
  cross = cos(series i, series j) — different series; near 0 means they stay distinct (no collapse)
The contrastive gap reported elsewhere is ff − fp.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XF_CSV = os.environ["XF_CSV"]
BASE_CSV = os.environ["BASE_CSV"]
OUT = os.environ["OUT"]
START = int(os.environ.get("START", "100"))
XF_C, BASE_C = "#2f6da8", "#d08a3e"
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


xf, base = read(XF_CSV, START), read(BASE_CSV, START)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (key, title) in zip(axes.ravel(), PANELS):
    ax.semilogx(base["step"], base[key], color=BASE_C, lw=1.6, ls="--", label="best recipe")
    ax.semilogx(xf["step"], xf[key], color=XF_C, lw=1.9, label="+ 10% regime crossfade")
    ax.axhline(0.0, color="0.7", lw=0.8, ls=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
fig.suptitle("Contrastive cosines through training (log step), best recipe vs + 10% regime crossfade",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=125)
print("wrote", OUT)
for key, _ in PANELS:
    print(f"  {key:12s} best {base[key][0]:+.3f}->{base[key][-1]:+.3f}   "
          f"xfade {xf[key][0]:+.3f}->{xf[key][-1]:+.3f}")
