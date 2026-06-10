#!/usr/bin/env python3
"""Stop-grad follow-up — headline figure from results/gm_table.csv.

Left: GM-Relative MASE bars, reference vs stop-grad, per head × checkpoint.
Right: the paired-bootstrap Δ (stop-grad − reference) with its 90% CI —
the uncertainty that decides the verdict, shown visually.
Env: OUT (png).
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-10_stopgrad_positive"
OUT = os.environ.get("OUT", "/tmp/cf-sgpos/experiments/2026-06-10_stopgrad_positive/plots/gm_summary.png")

rows = list(csv.DictReader(open(f"{EXP}/results/gm_table.csv")))
cells = [(r["head"], r["ckpt"], float(r["ref_gm"]), float(r["sg_gm"]),
          float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"]))
         for r in rows if r["sg_gm"] not in ("", "None")]
labels = [f"{h} {c}" for h, c, *_ in cells]
x = np.arange(len(cells))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

w = 0.38
ax1.bar(x - w / 2, [c[2] for c in cells], w, label="reference (no stop-grad)",
        color="#9dc3e6")
ax1.bar(x + w / 2, [c[3] for c in cells], w, label="stop-grad", color="#c0504d")
for xi, c in zip(x, cells):
    ax1.text(xi - w / 2, c[2], f"{c[2]:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.text(xi + w / 2, c[3], f"{c[3]:.3f}", ha="center", va="bottom", fontsize=8)
ax1.axhline(1.0, color="grey", ls=":", lw=1, label="seasonal-naive")
ax1.set_xticks(x, labels)
ax1.set_ylabel("GM-Relative MASE (lower is better)")
lo = min(min(c[2], c[3]) for c in cells)
hi = max(max(c[2], c[3]) for c in cells)
ax1.set_ylim(min(0.98, lo - 0.03), hi + 0.03)
ax1.legend(fontsize=8)
ax1.set_title("Forecast error per head × checkpoint", fontsize=10)

ax2.axhline(0.0, color="grey", lw=1)
for xi, c in zip(x, cells):
    _, _, _, _, d, lo_, hi_ = c
    reliable = hi_ < 0 or lo_ > 0
    col = ("#2e7d32" if hi_ < 0 else "#c62828") if reliable else "#777777"
    ax2.errorbar(xi, d, yerr=[[d - lo_], [hi_ - d]], fmt="o", color=col,
                 capsize=5, lw=2, markersize=6)
    ax2.text(xi + 0.07, d, f"{d:+.3f}", fontsize=8, va="center")
ax2.set_xticks(x, labels)
ax2.set_xlim(-0.5, len(cells) - 0.5)
ax2.set_ylabel("Δ GM-Relative MASE (stop-grad − reference)")
ax2.set_title("Paired-bootstrap Δ with 90% CI (below 0 = stop-grad better)",
              fontsize=10)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
