#!/usr/bin/env python3
"""Plateau figure: floor-subtracted contrastive loss for all five arms, log-log, from step 100.
Each arm's loss stalls or ticks up partway through (the temporary plateau) before resuming its
fall. The marked dots are the mid-plateau checkpoints scored against the final model in the
section's table."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-05-29_forked_6Lf_b1024/runs"
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/forked-6Lf-b1024/"
       "experiments/2026-05-29_forked_6Lf_b1024/plots/plateau.png")

# (label, loss-csv basename, colour, mid-plateau step scored | None)
ARMS = [
    ("β·10%",     "bb_beta_forked10pct_qk_aon_6Lf_b1024",      "#1f77b4", None),
    ("β·0.8%",    "bb_beta_forked2_qk_aon_6Lf_b1024",          "#17becf", None),
    ("allt·50%",  "bb_xshh_allt_forked_qk_aon_6Lf_b1024",      "#d62728", 1000),
    ("allt·10%",  "bb_xshh_allt_forked10pct_qk_aon_6Lf_b1024", "#ff7f0e", 2500),
    ("allt·0.8%", "bb_xshh_allt_forked2_qk_aon_6Lf_b1024",     "#9467bd", 2500),
]


def curve(name, start=100):
    s, l = [], []
    for r in csv.DictReader(open(f"{RUNS}/{name}_losses.csv")):
        st = int(float(r["step"]))
        if st >= start:
            s.append(st); l.append(float(r["loss"]))
    return s, l


fig, ax = plt.subplots(figsize=(10, 5.6))
for label, name, c, mid in ARMS:
    try:
        s, l = curve(name)
    except FileNotFoundError:
        continue
    ax.loglog(s, l, color=c, lw=1.6, label=label, alpha=0.9)
    if mid is not None:
        i = min(range(len(s)), key=lambda i: abs(s[i] - mid))
        ax.scatter([s[i]], [l[i]], color=c, s=60, zorder=5, edgecolor="k", lw=0.7)

ax.set_xlabel("training step (log)")
ax.set_ylabel("contrastive loss above its floor (log)")
ax.set_title("Every arm's contrastive loss has a temporary plateau before it falls again")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=10, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
