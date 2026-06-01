#!/usr/bin/env python3
"""#322 — log-log training curves of the batch-1024 backbones trained so far.

Left:  contrastive loss MINUS its InfoNCE floor (the CSV `loss` column — every arm
       runs with --subtract-contrastive-floor, so 0 = the run's own uniformity floor,
       making the β and all-time arms comparable despite very different negative pools).
       Log-log: the healthy decay shows as a clean downward line.
Right: the contrastive gap = cos(forecast, future) - cos(forecast, present), log-x;
       climbs toward ~1 as the backbone learns to encode the future, not the present.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ("/home/jupyter/workspaces/contrastive-forecasting/"
        "experiments/2026-05-29_forked_6Lf_b1024/runs")
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
       "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots/"
       "trained_curves_loglog.png")

# (label, csv basename, color) — all five trained arms, full length
ARMS = [
    ("β·0.8%   (mix 0.0078)", "bb_beta_forked2_qk_aon_6Lf_b1024_losses.csv",          "#1f77b4"),
    ("β·10%    (mix 0.10)",   "bb_beta_forked10pct_qk_aon_6Lf_b1024_losses.csv",      "#17becf"),
    ("allt·50%  (mix 0.50)",  "bb_xshh_allt_forked_qk_aon_6Lf_b1024_losses.csv",      "#d62728"),
    ("allt·10%  (mix 0.10)",  "bb_xshh_allt_forked10pct_qk_aon_6Lf_b1024_losses.csv", "#ff7f0e"),
    ("allt·0.8% (mix 0.0078)","bb_xshh_allt_forked2_qk_aon_6Lf_b1024_losses.csv",     "#9467bd"),
]
EPS = 1e-3  # floor for log axis when loss-floor dips to ~0 near convergence


def read(path):
    s, l, g = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            s.append(int(float(row["step"])))
            l.append(float(row["loss"]))   # already floor-subtracted
            g.append(float(row["gap"]))
    return s, l, g


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for label, name, c in ARMS:
    try:
        s, l, g = read(f"{RUNS}/{name}")
    except FileNotFoundError:
        continue
    if not s:
        continue
    done = "FINAL" if s[-1] >= 12400 else f"@{s[-1]//1000}k"
    ax1.loglog(s, [max(x, EPS) for x in l], color=c, lw=1.6,
               label=f"{label}  [{done}]")
    ax2.semilogx(s, g, color=c, lw=1.6, label=label)

ax1.set_xlabel("training step")
ax1.set_ylabel("contrastive loss − InfoNCE floor")
ax1.set_title("Training loss (floor-subtracted), log-log")
ax1.grid(True, which="both", alpha=0.25)
ax1.legend(fontsize=9, framealpha=0.9)

ax2.axhline(1.0, color="grey", ls=":", lw=0.8)
ax2.set_xlabel("training step")
ax2.set_ylabel("gap = cos(fcst, future) − cos(fcst, present)")
ax2.set_title("Contrastive gap (log-x)")
ax2.grid(True, which="both", alpha=0.25)
ax2.legend(fontsize=9, framealpha=0.9)

fig.suptitle("#322 batch-1024 backbones (qk-norm + attn-out-norm) — trained curves",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=110)
print("wrote", OUT)
