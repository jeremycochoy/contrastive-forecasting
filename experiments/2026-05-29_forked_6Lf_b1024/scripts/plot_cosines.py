#!/usr/bin/env python3
"""#322 — the contrastive cosine components through training, b1024 (diverges) vs
b256 (#320, stable). Helps see whether positives or negatives drive the collapse.

Definitions (src/models.py:compute_metrics, all on L2-normalised latents):
  ff          = cos(forecast_t, future-encoder_{t+1})   POSITIVE pair  (want high)
  fp          = cos(forecast_t, present-encoder_t)       present-copy shortcut (want low/neg)
  tp          = cos(future-encoder, present-encoder)     encoder autocorrelation
  cross_batch = cross-SERIES cos(future_i, forecast_j)   negative pool (want ~0)
  gap = ff - fp.

Left: ff / fp / tp vs step (log-x). Right: cross_batch vs step (log-x, log-y) — the
cross-series-negative blow-up that marks the collapse. b1024 = solid, b256 = dashed.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
B1024 = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/DIVERGED_5e-4_losses.csv"
B256 = f"{ROOT}/2026-05-26_forked_6Lf/runs/bb_beta_forked2_6Lf_50k_losses.csv"
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
       "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots/cosines_through_training.png")


def read(path):
    out = {k: [] for k in ("step", "ff", "fp", "tp", "cross_batch")}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                s = int(float(row["step"]))
            except (KeyError, ValueError):
                continue
            if s <= 0:
                continue
            out["step"].append(s)
            for k in ("ff", "fp", "tp", "cross_batch"):
                out[k].append(float(row[k]))
    return out


b1024, b256 = read(B1024), read(B256)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))

COL = {"ff": "#1f77b4", "fp": "#d62728", "tp": "#7f7f7f"}
LAB = {"ff": "ff = cos(forecast, FUTURE) [positive]",
       "fp": "fp = cos(forecast, present) [shortcut]",
       "tp": "tp = cos(future, present) [enc autocorr]"}
for k in ("ff", "fp", "tp"):
    axA.plot(b1024["step"], b1024[k], "-", color=COL[k], lw=1.8, label=f"b1024 {LAB[k]}")
    axA.plot(b256["step"], b256[k], "--", color=COL[k], lw=1.2, alpha=0.7,
             label=f"b256 {k}")
axA.axhline(0, color="k", lw=0.6, ls=":")
axA.set_xscale("log"); axA.set_xlabel("training step (log)"); axA.set_ylabel("cosine similarity")
axA.set_title("Positives (ff) vs shortcut (fp) vs enc-autocorr (tp)")
axA.grid(True, which="both", alpha=0.3); axA.legend(fontsize=7, loc="lower left")

# cross_batch: clip to small positive floor for log scale
def floor(xs, eps=1e-4):
    return [max(x, eps) for x in xs]
axB.plot(b1024["step"], floor(b1024["cross_batch"]), "-", color="#d62728", lw=2.0,
         label="b1024 cross_batch (DIVERGES)")
axB.plot(b256["step"], floor(b256["cross_batch"]), "--", color="#2ca02c", lw=1.4,
         label="b256 cross_batch (stable)")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("training step (log)"); axB.set_ylabel("cross-series negative cosine (log, |·|)")
axB.set_title("Cross-series negatives — the collapse signature")
axB.grid(True, which="both", alpha=0.3); axB.legend(fontsize=8, loc="upper left")

fig.suptitle("#322 β·0.8% — contrastive cosines through training: b1024 (diverges) vs b256 (#320, stable)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
for name, d in (("b1024", b1024), ("b256", b256)):
    if d["step"]:
        i = -1
        print(f"  {name} last (step {d['step'][i]}): ff={d['ff'][i]:.3f} fp={d['fp'][i]:.3f} "
              f"tp={d['tp'][i]:.3f} cross_batch={d['cross_batch'][i]:.4f}")
