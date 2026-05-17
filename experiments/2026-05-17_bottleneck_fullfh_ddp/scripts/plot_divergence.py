#!/usr/bin/env python3
"""Divergence curve for the bf16-body 2L run — log-log.

Top: training loss vs step (log-log), healthy min and collapse onset
marked. Bottom: forecaster-L1 residual + qk max-abs vs step (log-log) —
the documented bf16 residual-amplitude explosion that drives it.
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
RUN = "enc_fcst_bneck128_dk07_fullfh_norminfonce_ddp_50k"
LOSS = f"{EXP}/runs/{RUN}_losses.csv"
AMP = f"{EXP}/runs/{RUN}_attn_amplitude.csv"
OUT = f"{EXP}/plots/divergence_loglog.png"

step, loss = [], []
with open(LOSS) as f:
    for r in csv.DictReader(f):
        try:
            s, l = int(float(r["step"])), float(r["loss"])
        except (ValueError, KeyError):
            continue
        if s > 0 and l > 0 and math.isfinite(l):
            step.append(s); loss.append(l)

lo = min(loss); lo_s = step[loss.index(lo)]

ast, qk, rsa, rff = [], [], [], []
with open(AMP) as f:
    for r in csv.DictReader(f):
        if r.get("block") == "fcst" and r.get("layer_idx") == "1":
            try:
                ast.append(int(float(r["step"])))
                qk.append(float(r["qk_logit_maxabs"]))
                rsa.append(float(r["resid_post_sa_maxabs"]))
                rff.append(float(r["resid_post_ffn_maxabs"]))
            except (ValueError, KeyError):
                continue

fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

a1.loglog(step, loss, lw=1.1, color="#1f77b4")
a1.scatter([lo_s], [lo], color="green", zorder=5,
           label=f"healthy min loss={lo:.3f} @ step {lo_s}")
a1.axvspan(1100, 6000, color="red", alpha=0.10,
           label="collapse (~step 1.1k → 6k)")
a1.set_ylabel("training loss (normalized InfoNCE)")
a1.set_title("Divergence — bf16 body (attn/ffn/conv), residual fp32, 2L fcst\n"
             "full_fh_negs + pos-in-denominator, dk0.70, DDP 256, 50k")
a1.grid(True, which="both", ls=":", alpha=0.4)
a1.legend(fontsize=8, loc="lower right")

a2.loglog(ast, rff, lw=1.2, color="#d62728", label="residual post-FFN max-abs (fcst L1)")
a2.loglog(ast, rsa, lw=1.0, color="#ff7f0e", ls="--", label="residual post-SA max-abs")
a2.loglog(ast, qk, lw=1.0, color="#7f7f7f", ls=":", label="QK^T logit max-abs")
a2.axvspan(1100, 6000, color="red", alpha=0.10)
a2.set_xlabel("training step (log)")
a2.set_ylabel("max-abs amplitude (log)")
a2.grid(True, which="both", ls=":", alpha=0.4)
a2.legend(fontsize=8, loc="upper left")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
print(f"loss: {len(step)} pts, min {lo:.4f}@{lo_s}, last {loss[-1]:.3f}@{step[-1]}")
print(f"resid_post_ffn: {rff[0]:.1f}@{ast[0]} -> max {max(rff):.0f}@{ast[rff.index(max(rff))]}")
