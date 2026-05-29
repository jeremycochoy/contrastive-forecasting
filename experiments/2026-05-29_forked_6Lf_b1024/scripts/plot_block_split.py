#!/usr/bin/env python3
"""#322 — which block explodes: ENCODER vs FORECASTER. Splits the diverged b1024 run's
amplitude log by the `block` column ('enc' / 'fcst'), max-over-layers, log-log.
Left = attention output sa_out (directly comparable across blocks); right = residual
post-FFN (note: forecaster is the 128-d bottleneck vs encoder 384-d, so its residual
magnitude is not directly comparable — read sa_out for the clean answer).
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
A = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/DIVERGED_5e-4_attn_amplitude.csv"
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
       "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots/block_split.png")


def by_block(path, col):
    agg = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            s = int(float(row["step"])); b = row["block"]
            d = agg.setdefault(b, {})
            d[s] = max(d.get(s, 0.0), abs(float(row[col])))
    return agg


fig, (axS, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
STYLE = [("enc", "#d62728", "-", "ENCODER"), ("fcst", "#1f77b4", "--", "FORECASTER")]
for col, ax, ttl in [("sa_out_maxabs", axS, "attention output sa_out — which block's attention runs away"),
                     ("resid_post_ffn_maxabs", axR, "residual post-FFN (enc=384d, fcst=128d — not directly comparable)")]:
    agg = by_block(A, col)
    for b, c, ls, lab in STYLE:
        if b in agg:
            steps = sorted(agg[b])
            ax.plot(steps, [agg[b][s] for s in steps], ls, color=c, lw=2.0, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step (log)"); ax.set_ylabel("max-abs (log)"); ax.set_title(ttl, fontsize=10)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=10)
fig.suptitle("#322 diverged b1024 — ENCODER attention output explodes; FORECASTER attention stays flat", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
for col in ("sa_out_maxabs", "resid_post_ffn_maxabs"):
    agg = by_block(A, col)
    for b, *_ in STYLE:
        if b in agg:
            steps = sorted(agg[b])
            print(f"  {col} {b}: {agg[b][steps[0]]:.0f} -> {agg[b][steps[-1]]:.0f}")
