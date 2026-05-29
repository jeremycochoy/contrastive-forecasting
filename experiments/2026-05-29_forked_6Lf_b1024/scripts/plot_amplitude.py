#!/usr/bin/env python3
"""#322 — activation amplitudes through training, b1024 (diverges) vs b256 (#320, stable).

From the --log-attn-amplitude CSVs (per layer/block, every 200 steps): max-over-layers of
  qk_logit_maxabs       (pre-softmax attention logits)
  sa_out_maxabs         (self-attention output)
  resid_post_ffn_maxabs (residual stream after FFN — the main activation magnitude)
Log-log. b1024 = solid, b256 = dashed. A vertical marker shows where b1024's gap collapsed.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
B1024 = f"{ROOT}/2026-05-29_forked_6Lf_b1024/runs/DIVERGED_5e-4_attn_amplitude.csv"
B256 = f"{ROOT}/2026-05-26_forked_6Lf/runs/bb_beta_forked2_6Lf_50k_attn_amplitude.csv"
OUT = ("/home/jupyter/workspaces/contrastive-forecasting/.claude/worktrees/"
       "forked-6Lf-b1024/experiments/2026-05-29_forked_6Lf_b1024/plots/activation_amplitudes.png")
METRICS = ["qk_logit_maxabs", "sa_out_maxabs", "resid_post_ffn_maxabs"]
COL = {"qk_logit_maxabs": "#d62728", "sa_out_maxabs": "#ff7f0e", "resid_post_ffn_maxabs": "#1f77b4"}
LAB = {"qk_logit_maxabs": "QK logits", "sa_out_maxabs": "SA output", "resid_post_ffn_maxabs": "residual (post-FFN)"}


def max_over_layers(path):
    agg = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                s = int(float(row["step"]))
            except (KeyError, ValueError):
                continue
            d = agg.setdefault(s, {m: 0.0 for m in METRICS})
            for m in METRICS:
                d[m] = max(d[m], abs(float(row[m])))
    steps = sorted(agg)
    return steps, {m: [agg[s][m] for s in steps] for m in METRICS}


s1, a1 = max_over_layers(B1024)
s2, a2 = max_over_layers(B256)

fig, ax = plt.subplots(figsize=(11, 6))
for m in METRICS:
    ax.plot(s1, a1[m], "-", color=COL[m], lw=2.0, label=f"b1024 {LAB[m]}")
    ax.plot(s2, a2[m], "--", color=COL[m], lw=1.2, alpha=0.7, label=f"b256 {LAB[m]}")
ax.axvline(5500, color="k", ls=":", lw=1, alpha=0.6)
ax.text(5600, ax.get_ylim()[1] if False else 5, "b1024 gap\ncollapse →", fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training step (log)"); ax.set_ylabel("max-abs activation amplitude (log)")
ax.set_title("#322 β·0.8% — activation amplitudes through training\n"
             "b1024 (solid, diverges) vs b256 (#320, dashed, stable)")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=3, loc="upper left")
fig.tight_layout(); fig.savefig(OUT, dpi=120)
print(f"wrote {OUT}")
for name, s, a in (("b1024", s1, a1), ("b256", s2, a2)):
    if s:
        print(f"  {name}: step {s[0]}..{s[-1]}  resid {a['resid_post_ffn_maxabs'][0]:.0f}->"
              f"{a['resid_post_ffn_maxabs'][-1]:.0f}  qk {a['qk_logit_maxabs'][0]:.0f}->{a['qk_logit_maxabs'][-1]:.0f}")
