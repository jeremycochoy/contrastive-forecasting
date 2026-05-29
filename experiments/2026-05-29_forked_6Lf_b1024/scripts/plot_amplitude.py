#!/usr/bin/env python3
"""#322 — activation amplitudes through training: b256 (stable) vs b1024-diverged (5e-4)
vs b1024+QK-norm. Two panels (max-over-layers, log-log):
  left  = QK logits (pre-softmax)        — what QK-norm directly bounds
  right = residual stream (post-FFN)     — the activation magnitude QK-norm does NOT
                                           directly touch.
Shows QK-norm caps the logits but the residual still grows → residual is the remaining lever.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
RUNS = [  # (label, attn_amplitude.csv, color)
    ("b256 (#320, stable)", f"{ROOT}/2026-05-26_forked_6Lf/runs/bb_beta_forked2_6Lf_50k_attn_amplitude.csv", "#2ca02c"),
    ("b1024 lr5e-4 (diverged)", f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/DIVERGED_5e-4_attn_amplitude.csv", "#d62728"),
    ("b1024 lr1e-3 +QK-norm", f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/bb_beta_forked2_qknorm_6Lf_b1024_attn_amplitude.csv", "#ff7f0e"),
    ("b1024 lr1e-3 +attn-out-norm", f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/bb_beta_forked2_aon_6Lf_b1024_attn_amplitude.csv", "#17becf"),
]
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
       "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots/activation_amplitudes.png")


def max_over_layers(path, col):
    agg = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                s = int(float(row["step"]))
                agg[s] = max(agg.get(s, 0.0), abs(float(row[col])))
    except FileNotFoundError:
        return [], []
    steps = sorted(agg)
    return steps, [agg[s] for s in steps]


PANELS = [("qk_logit_maxabs", "QK logits — QK-norm BOUNDS these", "max|QK logit| (log)"),
          ("sa_out_maxabs", "attention output (value path) — the DRIVER", "max|sa_out| (log)"),
          ("resid_post_ffn_maxabs", "residual stream (post-FFN) — the symptom", "max|residual| (log)")]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
for ax, (col, ttl, yl) in zip(axes, PANELS):
    for label, path, c in RUNS:
        s, v = max_over_layers(path, col)
        if s:
            ax.plot(s, v, "-", color=c, lw=1.9, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step (log)"); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=10)
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("#322 β·0.8% — QK-norm bounds the logits, but the attention OUTPUT (value path) still runs away → residual grows", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
for label, path, c in RUNS:
    sr, r = max_over_layers(path, "resid_post_ffn_maxabs")
    sq, q = max_over_layers(path, "qk_logit_maxabs")
    if sr:
        print(f"  {label}: qk_logit {q[0]:.0f}->{q[-1]:.0f}  resid {r[0]:.0f}->{r[-1]:.0f} (step {sr[-1]})")
