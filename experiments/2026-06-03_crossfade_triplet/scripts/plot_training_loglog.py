#!/usr/bin/env python3
"""#328 — log-log training curves: triplet arm vs #322's allt·0.8% base.

Panels (x = training step, from --start, log-x; triplet solid, allt·0.8% dashed):
  1. contrastive loss − InfoNCE floor (`loss` col, floor-subtracted), LOG-LOG.
  2. contrastive gap = cos(fcst, future) − cos(fcst, present), semilog-x.

Env (all have elisa defaults): TRIP_CSV, BASE_CSV, OUT, START (default 100).
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
OUT = os.environ.get("OUT", "/tmp/cf-328/experiments/2026-06-03_crossfade_triplet/plots/training_curves_loglog.png")
START = int(os.environ.get("START", "100"))
EPS = 1e-3


def read(path, start):
    s, loss, gap = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            st = int(float(row["step"]))
            if st < start:
                continue
            s.append(st)
            loss.append(float(row["loss"]))
            gap.append(float(row["gap"]))
    return s, loss, gap


xs, xloss, xgap = read(TRIP_CSV, START)
bs, bloss, bgap = read(BASE_CSV, START)
TRIP_C, BASE_C = "#2f6da8", "#d08a3e"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
a1.loglog(xs, [max(v, EPS) for v in xloss], color=TRIP_C, lw=1.7, label="crossfade triplet arm")
a1.loglog(bs, [max(v, EPS) for v in bloss], color=BASE_C, lw=1.5, ls="--", label="0.8%-fork base")
a1.set_xlabel("training step")
a1.set_ylabel("contrastive loss − InfoNCE floor")
a1.set_title(f"Training loss (floor-subtracted), log-log (from step {START})")
a1.grid(True, which="both", alpha=0.25)
a1.legend(fontsize=9, framealpha=0.9)

a2.semilogx(xs, xgap, color=TRIP_C, lw=1.7, label="crossfade triplet arm")
a2.semilogx(bs, bgap, color=BASE_C, lw=1.5, ls="--", label="0.8%-fork base")
a2.axhline(1.0, color="grey", ls=":", lw=0.8)
a2.set_xlabel("training step")
a2.set_ylabel("gap = cos(fcst, future) − cos(fcst, present)")
a2.set_title("Contrastive gap (log-x)")
a2.grid(True, which="both", alpha=0.25)
a2.legend(fontsize=9, framealpha=0.9)

fig.suptitle("Crossfade triplet arm vs the 0.8%-fork base — training curves", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
if xs:
    print(f"  triplet : {len(xs)} pts, step {xs[0]}..{xs[-1]}, loss {xloss[0]:.3f}->{xloss[-1]:.3f}, gap {xgap[0]:.3f}->{xgap[-1]:.3f}")
if bs:
    print(f"  base    : {len(bs)} pts, step {bs[0]}..{bs[-1]}, loss {bloss[0]:.3f}->{bloss[-1]:.3f}, gap {bgap[0]:.3f}->{bgap[-1]:.3f}")
