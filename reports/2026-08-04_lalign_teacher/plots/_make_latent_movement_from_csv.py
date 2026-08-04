"""Latent movement per arm, replotted from the committed per-pair CSV.

`results/latent_movement_pairs.csv` already holds, for every arm and every
adjacent checkpoint pair, the mean `1 - cos` displacement of the encoder
output (`drift_h`) and of the patch embedding (`drift_e`) on one fixed
held-out batch. Both halves of that file — the twenty copied arms and the
ten retrained ones — were measured against the same batch, so this figure
needs no GPU and no checkpoint.

One panel per loss recipe; x is the later checkpoint's step on a log scale.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
PAIRS = HERE.parent / "results" / "latent_movement_pairs.csv"

ARMS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco"]
VARIANTS = ["", "_tr1", "_nse", "_ncpc", "_combab"]
ARM_COLOR = {"arm1": "#2a78d6", "arm3": "#eb6834", "arm4": "#008300",
             "arm5": "#8b1e8b", "arm6_v2": "#b8860b", "bimoco": "#00a3a3"}
ARM_LOSS = {
    "arm1": "L_pred + L_rep", "arm3": "L_pred_moco + L_rep",
    "arm4": "pooled + MoCo", "arm5": "L_align + L_rep",
    "arm6_v2": "L_align + L_rep_moco", "bimoco": "L_pred_moco + L_rep_moco",
}
VAR_SHORT = {"": "base", "_tr1": "tr1", "_nse": "nse",
             "_ncpc": "ncpc", "_combab": "combab"}
VAR_STYLE = {
    "":        {"ls": "-",  "marker": "o", "lw": 2.0, "ms": 6},
    "_tr1":    {"ls": "--", "marker": "s", "lw": 1.4, "ms": 5},
    "_nse":    {"ls": ":",  "marker": "^", "lw": 1.4, "ms": 5},
    "_ncpc":   {"ls": "-.", "marker": "D", "lw": 1.4, "ms": 5},
    "_combab": {"ls": "-",  "marker": "P", "lw": 2.0, "ms": 7},
}
RETRAINED = ("arm5", "arm6_v2")

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

series: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
with open(PAIRS, newline="") as fh:
    for r in csv.DictReader(fh):
        series[r["arm_slug"]].append(
            (int(r["step_later"]), float(r["drift_h"]), float(r["drift_e"])))
for v in series.values():
    v.sort()

fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharey=True)
for ax, arm in zip(axes.flatten(), ARMS):
    for var in VARIANTS:
        pts = series.get(f"{arm}{var}")
        if not pts:
            continue
        st = VAR_STYLE[var]
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                color=ARM_COLOR[arm], label=VAR_SHORT[var], **st)
    tag = "  (teacher target)" if arm in RETRAINED else ""
    ax.set_title(f"{arm}  ({ARM_LOSS[arm]}){tag}", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("backbone step of the later checkpoint")
    ax.grid(True, color=GRID, alpha=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=2)
for ax in axes[:, 0]:
    ax.set_ylabel("1 − cos between adjacent checkpoints  (encoder output h)")
fig.suptitle(
    "Latent movement between adjacent backbone checkpoints, fixed held-out "
    "batch (B=8, T=4096, C=1)\n"
    "arm 5 and arm 6 v2 are the teacher-target retrain; the other four "
    "recipes carry no L_align term and are the earlier sweep's runs",
    fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / "latent_movement_per_arm.png"
fig.savefig(out)
print(f"wrote {out}  ({sum(len(v) for v in series.values())} pairs, "
      f"{len(series)} arms)")
