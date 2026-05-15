#!/usr/bin/env python3
"""Training curves for the encoder-forecaster dropkey sweep.

Continuation runs are stitched from their predecessor segments so every arm
shows a full 0 -> 50k trajectory.

Two figures, each two panels:
  (top)  loss          vs step  — log-log  ("the train curve")
  (bot)  loss_tau_ref   vs step  — log-x / linear-y (goes negative)

Fig 1: dropkey sweep (v11c, v16, v17, v20, v27c).
Fig 2: same + bottleneck / forecaster-depth arms (v13, v14, v15).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CK = "/home/jupyter/contrastive-forecasting/checkpoints"
OUT = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-11_exp_encoder_forecaster/plots"
os.makedirs(OUT, exist_ok=True)

# label -> (color, [ (csv, step_lo, step_hi), ... ] stitched in order)
SWEEP = [
    ("v11c  dk0.9 (fp32 ref)", "#1f77b4", [
        ("enc_fcst_v11c_jepa_newconv_fp32_50k_losses.csv", 1, 5000),
        ("enc_fcst_v11c_cont_from5k_50k_losses.csv", 5001, 50000)]),
    ("v16   dk0.7 (fp32)", "#2ca02c", [
        ("enc_fcst_v16_jepa_enc6_fcst1_dk07_newconv_fp32_50k_losses.csv", 1, 50000)]),
    ("v17   dk0.95 (fp32)", "#d62728", [
        ("enc_fcst_v17_jepa_enc6_fcst1_dk095_newconv_fp32_50k_losses.csv", 1, 50000)]),
    ("v20   dk0.9 (warmup->fp16)", "#9467bd", [
        ("enc_fcst_v20_phaseA_fp32warmup_5k_losses.csv", 1, 5000),
        ("enc_fcst_v20_v11c_freshwarmup_fp16_50k_losses.csv", 5001, 50000)]),
    ("v27c  dk0.8 (ffn-fp16)", "#ff7f0e", [
        ("enc_fcst_v27_dk08_ffnfp16_50k_losses.csv", 1, 20000),
        ("enc_fcst_v27b_dk08_ffnfp16_resume20k_50k_losses.csv", 20001, 25000),
        ("enc_fcst_v27c_dk08_ffnfp16_resume25k_50k_losses.csv", 25001, 50000)]),
]
EXTRA = [
    ("v13   dk0.9 (fcst-bottleneck128)", "#8c564b", [
        ("enc_fcst_v13_jepa_fcstbottleneck128_newconv_fp32_50k_losses.csv", 1, 50000)]),
    ("v14   dk0.9 (fcst 6L)", "#e377c2", [
        ("enc_fcst_v14_jepa_enc6_fcst6_dk09_newconv_fp32_50k_losses.csv", 1, 50000)]),
    ("v15   dk0.9 (fcst 4L)", "#7f7f7f", [
        ("enc_fcst_v15_jepa_enc6_fcst4_dk09_newconv_fp32_50k_losses.csv", 1, 50000)]),
]


def roll(a, w=150):
    a = np.asarray(a, dtype=float)
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="valid")


def load_stitched(segs):
    parts = []
    for csv, lo, hi in segs:
        p = os.path.join(CK, csv)
        if not os.path.exists(p):
            print(f"  [skip seg] missing {csv}")
            continue
        d = np.genfromtxt(p, delimiter=",", names=True,
                          invalid_raise=False)
        d = d[~np.isnan(d["step"]) & ~np.isnan(d["loss"])]
        m = (d["step"] >= lo) & (d["step"] <= hi)
        parts.append(d[m])
    if not parts:
        return None
    d = np.concatenate(parts)
    order = np.argsort(d["step"])
    return d[order]


def make_fig(arms, title, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=True)
    for label, color, segs in arms:
        d = load_stitched(segs)
        if d is None:
            print(f"  [skip] {label}")
            continue
        step, loss, tref = d["step"], d["loss"], d["loss_tau_ref"]
        ax1.plot(step, loss, color=color, alpha=0.10, lw=0.5)
        ax1.plot(step[: len(roll(loss))], roll(loss), color=color, lw=1.8, label=label)
        ax2.plot(step, tref, color=color, alpha=0.10, lw=0.5)
        ax2.plot(step[: len(roll(tref))], roll(tref), color=color, lw=1.8, label=label)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_ylabel("contrastive loss  (log)")
    ax1.set_title(title + "  —  train loss (log-log), full 0->50k stitched")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8, ncol=2)

    ax2.set_xscale("log")
    ax2.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax2.set_xlabel("training step  (log)")
    ax2.set_ylabel("loss_tau_ref  (linear; <0 = converged)")
    ax2.set_title("tau-referenced loss  (log-x, linear-y — goes negative)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    print("Generating sweep training-curve figures (stitched)...")
    make_fig(SWEEP, "Dropkey sweep", "sweep_train_curves_loglog.png")
    make_fig(SWEEP + EXTRA, "Dropkey sweep + bottleneck/fcst-depth",
             "sweep_train_curves_loglog_with_bottleneck.png")
    print("done.")
