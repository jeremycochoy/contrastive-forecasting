#!/usr/bin/env python3
"""#325 — log-log training curves: allt·10% + crossfade·10% vs #322's allt·10%.

Panels (x = training step, from --start, log-x; curves overlaid: this run solid,
#322 allt·10% baseline dashed):
  1. contrastive loss − InfoNCE floor (the CSV `loss` col, --subtract-contrastive-floor
     so 0 = the run's own uniformity floor), LOG-LOG — healthy decay = downward line.
  2. contrastive gap = cos(fcst, future) − cos(fcst, present), semilog-x — climbs to ~1.
  3. cross-series cosine (`cross_batch`), semilog-x — the collapse diagnostic; stays ~0
     when distinct series repel (a runaway toward ~0.5 is the b1024 collapse #322 fixed).

Env: XF_CSV, BASE_CSV, OUT, START (default 100).
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
EPS = 1e-3


def read(path, start):
    s, loss, gap, cb = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            st = int(float(row["step"]))
            if st < start:
                continue
            s.append(st)
            loss.append(float(row["loss"]))
            gap.append(float(row["gap"]))
            cb.append(float(row.get("cross_batch", "nan")))
    return s, loss, gap, cb


xs, xloss, xgap, xcb = read(XF_CSV, START)
bs, bloss, bgap, bcb = read(BASE_CSV, START)
XF_C, BASE_C = "#2f6da8", "#d08a3e"
xf_done = "FINAL" if xs and xs[-1] >= 12400 else (f"@{xs[-1]//1000}k" if xs else "—")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))

a1.loglog(xs, [max(v, EPS) for v in xloss], color=XF_C, lw=1.7,
          label="+ 10% regime crossfade")
a1.loglog(bs, [max(v, EPS) for v in bloss], color=BASE_C, lw=1.5, ls="--",
          label="best recipe so far")
a1.set_xlabel("training step")
a1.set_ylabel("contrastive loss − InfoNCE floor")
a1.set_title(f"Training loss (floor-subtracted), log-log (from step {START})")
a1.grid(True, which="both", alpha=0.25)
a1.legend(fontsize=9, framealpha=0.9)

a2.semilogx(xs, xgap, color=XF_C, lw=1.7, label="+ 10% regime crossfade")
a2.semilogx(bs, bgap, color=BASE_C, lw=1.5, ls="--", label="best recipe")
a2.axhline(1.0, color="grey", ls=":", lw=0.8)
a2.set_xlabel("training step")
a2.set_ylabel("gap = cos(fcst, future) − cos(fcst, present)")
a2.set_title("Contrastive gap (log-x)")
a2.grid(True, which="both", alpha=0.25)
a2.legend(fontsize=9, framealpha=0.9)

fig.suptitle("Regime crossfade vs the best recipe — training curves", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=120)
print("wrote", OUT)
print(f"  crossfade: {len(xs)} pts, step {xs[0]}..{xs[-1]}, "
      f"loss {xloss[0]:.3f}->{xloss[-1]:.3f}, gap {xgap[0]:.3f}->{xgap[-1]:.3f}")
print(f"  baseline : {len(bs)} pts, step {bs[0]}..{bs[-1]}, "
      f"loss {bloss[0]:.3f}->{bloss[-1]:.3f}, gap {bgap[0]:.3f}->{bgap[-1]:.3f}")
