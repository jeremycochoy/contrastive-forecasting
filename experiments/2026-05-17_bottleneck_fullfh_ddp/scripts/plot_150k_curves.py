#!/usr/bin/env python3
"""Full continuous 0→150k trajectory (3 stitched segments: 50k base +
50k→100k + 100k→150k resumes, continuous optimizer). Same 4 log-log
panels as the 50k figure: A loss+loss_tau_ref | B 1-AUC | C amplitude
(residual+attn) | D dimension-usage. Segment boundaries marked.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
R = f"{EXP}/runs"
SEGS = [
    "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k",
    "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_resume50k_100000",
    "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_resume50k_100000_resume100k_150000",
]
OUT = f"{EXP}/plots/trajectory_0_150k_loglog.png"

def load_losses(seg):
    out = {}
    p = f"{R}/{seg}_losses.csv"
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                s = int(float(r["step"]))
            except (ValueError, KeyError):
                continue
            if s <= 0 or s in out:
                continue
            out[s] = (float(r["loss"]), float(r["loss_tau_ref"]),
                      max(1.0 - float(r["auc"]), 1e-8),
                      float(r["u_temporal"]), float(r["u_batch"]))
    return out

def load_amp(seg):  # forecaster (1L → layer 0)
    out = {}
    p = f"{R}/{seg}_attn_amplitude.csv"
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("block") == "fcst" and r.get("layer_idx") == "0":
                try:
                    out[int(float(r["step"]))] = (float(r["qk_logit_maxabs"]),
                        float(r["sa_out_maxabs"]), float(r["resid_post_ffn_maxabs"]))
                except (ValueError, KeyError):
                    pass
    return out

L, A = {}, {}
for seg in SEGS:
    L.update(load_losses(seg))   # later segments' steps (>prev) extend the trajectory
    A.update(load_amp(seg))
xs = sorted(L)
loss = [L[s][0] for s in xs]; tau = [L[s][1] for s in xs]
oneauc = [L[s][2] for s in xs]; ut = [L[s][3] for s in xs]; ub = [L[s][4] for s in xs]
ax_ = sorted(A); qk = [A[s][0] for s in ax_]; so = [A[s][1] for s in ax_]; rf = [A[s][2] for s in ax_]

mn = min(loss); mns = xs[loss.index(mn)]
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
for a in ax.flat:
    for b in (50000, 100000):
        a.axvline(b, color="grey", ls="--", lw=.8, alpha=.6)

ax[0,0].loglog(xs, loss, lw=1.1, color="#1f77b4", label="train loss (normalized InfoNCE)")
ax[0,0].loglog(xs, tau, lw=1.1, color="#9467bd", label="loss_tau_ref (τ=0.10 ref)")
ax[0,0].scatter([mns], [mn], color="green", zorder=5, s=18, label=f"min loss {mn:.4f}@{mns}")
ax[0,0].set_title("A. Loss & τ-ref — 0→150k log-log"); ax[0,0].set_xlabel("step"); ax[0,0].set_ylabel("loss")
ax[0,0].grid(True, which="both", ls=":", alpha=.4); ax[0,0].legend(fontsize=8)

ax[0,1].loglog(xs, oneauc, lw=1.0, color="#d62728")
ax[0,1].set_title("B. 1 − AUC — log-log"); ax[0,1].set_xlabel("step"); ax[0,1].set_ylabel("1 − AUC")
ax[0,1].grid(True, which="both", ls=":", alpha=.4)

ax[1,0].loglog(ax_, rf, color="#d62728", lw=1.2, label="resid post-FFN (fcst)")
ax[1,0].loglog(ax_, so, color="#ff7f0e", lw=1.0, ls="--", label="SA-out (fcst)")
ax[1,0].loglog(ax_, qk, color="#7f7f7f", lw=1.0, ls=":", label="QKᵀ logit (fcst)")
ax[1,0].set_title("C. Forecaster amplitude — log-log (bounded ⇒ stable)")
ax[1,0].set_xlabel("step"); ax[1,0].set_ylabel("max-abs")
ax[1,0].grid(True, which="both", ls=":", alpha=.4); ax[1,0].legend(fontsize=8)

ax[1,1].loglog(xs, ut, lw=1.1, color="#2ca02c", label="u_temporal")
ax[1,1].loglog(xs, ub, lw=1.1, color="#17becf", label="u_batch")
ax[1,1].set_title("D. Dimension usage (higher = less collapse) — log-log")
ax[1,1].set_xlabel("step"); ax[1,1].set_ylabel("usage")
ax[1,1].grid(True, which="both", ls=":", alpha=.4); ax[1,1].legend(fontsize=8)

fig.suptitle("Continuous 0→150k (dashed = 50k/100k resume boundaries) — "
             "1L/fp16 bottleneck, full_fh_negs + norm-InfoNCE, dk0.70", fontsize=11)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT, dpi=130)
print("wrote", OUT)
print(f"steps {xs[0]}..{xs[-1]} n={len(xs)} ; loss {loss[0]:.2f}->{loss[-1]:.3f} (min {mn:.4f}@{mns})")
print(f"tau_ref {tau[0]:.2f}->{tau[-1]:.3f} ; u_temporal {ut[0]:.3f}->{ut[-1]:.3f} ; "
      f"fcst resid_post_ffn {rf[0]:.1f}->{rf[-1]:.1f} (max {max(rf):.1f})")
