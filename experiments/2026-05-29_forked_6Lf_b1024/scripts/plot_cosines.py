#!/usr/bin/env python3
"""#322 — contrastive cosines through training across configs.
Left: ff = cos(forecast, FUTURE) [positive, want high].
Right: cross_batch = cross-SERIES negative cosine [want ~0] — the collapse signature.
Runs: b256 (stable), b1024 diverged (5e-4), b1024 +QK-norm, b1024 τ=0.20.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
RUNS = [  # (label, losses.csv, color)
    ("b256 (#320, stable)", f"{ROOT}/2026-05-26_forked_6Lf/runs/bb_beta_forked2_6Lf_50k_losses.csv", "#2ca02c"),
    ("b1024 lr5e-4 (diverged)", f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/DIVERGED_5e-4_losses.csv", "#d62728"),
    ("b1024 lr1e-3 +QK-norm", f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/bb_beta_forked2_qknorm_6Lf_b1024_losses.csv", "#ff7f0e"),
    ("b1024 lr5e-4 τ=0.20", f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/smoke_beta_b1024_losses.csv", "#9467bd"),
    ("b1024 lr1e-3 +attn-out-norm", f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/bb_beta_forked2_aon_6Lf_b1024_losses.csv", "#17becf"),
]
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
       "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots/cosines_through_training.png")


def read(path, col):
    s, v = [], []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                st = int(float(row["step"]))
                if st > 0:
                    s.append(st); v.append(float(row[col]))
    except FileNotFoundError:
        pass
    return s, v


fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))
for label, path, c in RUNS:
    sf, ff = read(path, "ff")
    sc, cb = read(path, "cross_batch")
    if sf:
        axA.plot(sf, ff, "-", color=c, lw=1.8, label=label)
    if sc:
        axB.plot(sc, [max(x, 1e-4) for x in cb], "-", color=c, lw=1.8, label=label)
axA.set_xscale("log"); axA.axhline(1.0, color="k", lw=0.5, ls=":")
axA.set_xlabel("training step (log)"); axA.set_ylabel("ff = cos(forecast, future) [positive]")
axA.set_title("Positive alignment (ff) — holds high until the collapse")
axA.grid(True, which="both", alpha=0.3); axA.legend(fontsize=8)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("training step (log)"); axB.set_ylabel("cross-series negative cosine (log)")
axB.set_title("Cross-series negatives — the collapse signature (b256 stays ~1e-3)")
axB.grid(True, which="both", alpha=0.3); axB.legend(fontsize=8)
fig.suptitle("#322 β·0.8% — contrastive cosines: b256 stable; b1024 (diverged / QK-norm / τ=0.20) all creep up in cross_batch", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
for label, path, c in RUNS:
    sc, cb = read(path, "cross_batch")
    if sc:
        print(f"  {label}: cross_batch {cb[0]:.4f}->{cb[-1]:.4f} (step {sc[-1]})")
