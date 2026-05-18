#!/usr/bin/env python3
"""Requested log-log curves for the successful 1L/fp16 run:
A loss + loss_tau_ref | B 1-AUC | C amplitude (residual+attn) | D dim-usage.
All axes log-log. DDP writes one row per rank per step → dedup on step.
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
B = "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k"
LOSS, AMP = f"{EXP}/runs/{B}_losses.csv", f"{EXP}/runs/{B}_attn_amplitude.csv"
OUT = f"{EXP}/plots/success_curves_loglog.png"

seen, step, loss, tau, oneauc, ut, ub = set(), [], [], [], [], [], []
with open(LOSS) as f:
    for r in csv.DictReader(f):
        try:
            s = int(float(r["step"]))
        except (ValueError, KeyError):
            continue
        if s in seen or s <= 0:
            continue
        seen.add(s)
        step.append(s); loss.append(float(r["loss"])); tau.append(float(r["loss_tau_ref"]))
        oneauc.append(max(1.0 - float(r["auc"]), 1e-8))   # floor for log axis
        ut.append(float(r["u_temporal"])); ub.append(float(r["u_batch"]))

# amplitude: forecaster (1L → layer 0) and deepest encoder layer (L5)
amp = {("fcst", "0"): {}, ("enc", "5"): {}}
with open(AMP) as f:
    for r in csv.DictReader(f):
        k = (r.get("block"), r.get("layer_idx"))
        if k in amp:
            try:
                amp[k][int(float(r["step"]))] = (float(r["qk_logit_maxabs"]),
                    float(r["sa_out_maxabs"]), float(r["resid_post_ffn_maxabs"]))
            except (ValueError, KeyError):
                pass
def series(k):
    xs = sorted(amp[k]); return xs, [amp[k][x] for x in xs]

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

ax[0,0].loglog(step, loss, lw=1.2, color="#1f77b4", label="train loss (normalized InfoNCE)")
ax[0,0].loglog(step, tau, lw=1.2, color="#9467bd", label="loss_tau_ref (τ=0.10 reference)")
ax[0,0].set_title("A. Loss & τ-ref — log-log"); ax[0,0].set_xlabel("step"); ax[0,0].set_ylabel("loss")
ax[0,0].grid(True, which="both", ls=":", alpha=.4); ax[0,0].legend(fontsize=8)

ax[0,1].loglog(step, oneauc, lw=1.1, color="#d62728")
ax[0,1].set_title("B. 1 − AUC — log-log  (contrastive discrimination)")
ax[0,1].set_xlabel("step"); ax[0,1].set_ylabel("1 − AUC")
ax[0,1].grid(True, which="both", ls=":", alpha=.4)
ax[0,1].annotate("AUC→1 within ~30 steps\n(task trivially separable)", (0.30, 0.78),
                 xycoords="axes fraction", fontsize=8)

xs, v = series(("fcst","0")); xe, ve = series(("enc","5"))
ax[1,0].loglog(xs, [t[2] for t in v], color="#d62728", lw=1.3, label="resid post-FFN (fcst)")
ax[1,0].loglog(xs, [t[1] for t in v], color="#ff7f0e", lw=1.0, ls="--", label="SA-out (fcst)")
ax[1,0].loglog(xe, [t[2] for t in ve], color="#8c564b", lw=1.0, label="resid post-FFN (enc L5)")
ax[1,0].loglog(xs, [t[0] for t in v], color="#7f7f7f", lw=1.0, ls=":", label="QKᵀ logit (fcst)")
ax[1,0].set_title("C. Amplitude max-abs — log-log  (bounded ⇒ stable)")
ax[1,0].set_xlabel("step"); ax[1,0].set_ylabel("max-abs")
ax[1,0].grid(True, which="both", ls=":", alpha=.4); ax[1,0].legend(fontsize=8)

ax[1,1].loglog(step, ut, lw=1.2, color="#2ca02c", label="u_temporal")
ax[1,1].loglog(step, ub, lw=1.2, color="#17becf", label="u_batch")
ax[1,1].set_title("D. Dimension usage (uniformity) — log-log")
ax[1,1].set_xlabel("step"); ax[1,1].set_ylabel("usage")
ax[1,1].grid(True, which="both", ls=":", alpha=.4); ax[1,1].legend(fontsize=8)

fig.suptitle("Successful run — bottleneck d128/4h, 6enc/1fcst, dk0.70, "
             "full_fh_negs + norm-InfoNCE, fp16 body, DDP 256, 50k", fontsize=11)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT, dpi=130)
print("wrote", OUT)
print(f"loss {loss[0]:.2f}->{loss[-1]:.3f} (min {min(loss):.4f}) ; tau_ref {tau[0]:.2f}->{tau[-1]:.3f}")
print(f"1-AUC {oneauc[0]:.2e}->{min(oneauc):.1e} ; u_temporal {ut[0]:.3f}->{ut[-1]:.3f} ; u_batch {ub[0]:.3f}->{ub[-1]:.3f}")
fl = series(("fcst","0"))[1]
if fl:
    print(f"fcst resid_post_ffn {fl[0][2]:.1f} -> {fl[-1][2]:.1f} (max {max(t[2] for t in fl):.1f}) — bounded")
