#!/usr/bin/env python3
"""#307 variance figure — per-arm full-97 GM with seed spread.

One column per loss arm, ordered by mean GM. Each column shows:
  • individual seed dots (the raw full-97 GM of each seed)
  • a mean marker (●)
  • a thick bar = mean ± 1 std (the "candle body"; only N≥2)
  • a thin whisker = min … max across seeds (only N≥2)
Single-seed arms (#303 + #307 of-record) show their one dot, annotated N=1.

Colour splits the two clusters the data separates into: arms WITHOUT the
all-time f↔h negative (A) vs arms WITH it. Horizontal refs: v11c-recipe
mean ± min/max band (n=3), seasonal-naive (=1.0), R9_E13 best-ever.

This is the variance-of-record view — the radar (perdomain_star.png) is
per-domain and too dense to read arm-vs-arm uncertainty off.
"""
import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A17 = "/home/jupyter/cf-encoder-forecaster-v2/experiments/2026-05-17_bottleneck_fullfh_ddp"
ART303 = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_ablation"
SY = "/home/jupyter/contrastive-forecasting/experiments/2026-05-19_crossed_loss_xbranch_ablation"
OUT = "/home/jupyter/cf-wt-crossed-loss/experiments/2026-05-19_crossed_loss_xbranch_ablation/plots"
A_NAME = "enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k"
os.makedirs(OUT, exist_ok=True)


def agg(d):
    p = f"{d}/summary.txt"
    if not os.path.exists(p):
        return None
    for line in open(p):
        if "Aggregate GM-Relative MASE" in line:
            for t in reversed(line.replace(":", " ").split()):
                try:
                    return float(t)
                except ValueError:
                    pass
    return None


# arm -> (has_A, [eval dirs over seeds])
ARMS = {
    "(A)\nfull_fh": (True, [f"{A17}/results/gift_eval_full_{A_NAME}"]),
    "(B)\nfull_hh": (False, [
        f"{ART303}/results/gift_eval_full_cl_hh_50k",
        f"{SY}/variance/hh_seed20260518/results/gift_eval_full_cl_hh_50k_s18",
        f"{SY}/variance/hh_seed20260519/results/gift_eval_full_cl_hh_50k_s19"]),
    "(C)\nfull_ff": (False, [f"{ART303}/results/gift_eval_full_cl_ff_50k"]),
    "(A)+(B)\nfull_fh_hh": (True, [f"{ART303}/results/gift_eval_full_cl_fhhh_50k"]),
    "(B)+(C)\nfull_hh_ff": (False, [f"{SY}/downstream_hhff/results/gift_eval_full_cl_hhff_50k"]),
    "(A)+(B)+(C)\nfull_fh_hh_ff": (True, [f"{SY}/downstream_fhhhff/results/gift_eval_full_cl_fhhhff_50k"]),
    "(B)-xbfree\nfull_hh_xbf": (False, [
        f"{SY}/downstream_hhxbf/results/gift_eval_full_cl_hhxbf_50k",
        f"{SY}/variance/hhxbf_seed20260518/results/gift_eval_full_cl_hhxbf_50k_s18",
        f"{SY}/variance/hhxbf_seed20260519/results/gift_eval_full_cl_hhxbf_50k_s19"]),
}

rows = []  # (label, has_A, vals[])
for lab, (has_A, dirs) in ARMS.items():
    vals = [v for v in (agg(d) for d in dirs) if v is not None]
    if vals:
        rows.append((lab, has_A, vals))
rows.sort(key=lambda r: np.mean(r[2]))  # order by mean GM ascending

# v11c-recipe references (3 seeds/precisions)
v11c = [v for v in (agg(f"{SY}/../2026-05-11_exp_encoder_forecaster/results/{d}")
                    for d in ("gift_eval_full_v11c", "gift_eval_full_v20R",
                              "gift_eval_full_dk09fp32x150k")) if v]
r9 = agg("/home/jupyter/contrastive-forecasting/experiments/"
         "2026-05-05_exp_qhead_improvements/results/"
         "R9_E13_xfmr12L_quant_moirai_cosine_e_then_f_60k_full")

fig, ax = plt.subplots(figsize=(12, 7))
C_NOA, C_A = "#1a8a3a", "#c22020"      # non-A cluster green, A-containing red
xs = np.arange(len(rows))
for x, (lab, has_A, vals) in zip(xs, rows):
    col = C_A if has_A else C_NOA
    vals = np.array(vals)
    m = vals.mean()
    if len(vals) >= 2:
        sd = vals.std(ddof=1)
        # whisker = min..max
        ax.plot([x, x], [vals.min(), vals.max()], color=col, lw=1.4, zorder=2)
        for yv in (vals.min(), vals.max()):       # whisker caps
            ax.plot([x - 0.06, x + 0.06], [yv, yv], color=col, lw=1.4, zorder=2)
        # candle body = mean ± 1 std
        ax.add_patch(plt.Rectangle((x - 0.17, m - sd), 0.34, 2 * sd,
                                   facecolor=col, alpha=.18, edgecolor=col,
                                   lw=1.0, zorder=1))
        ax.text(x + 0.22, m, f"n={len(vals)}\nμ={m:.3f}\nσ={sd:.3f}",
                fontsize=7.5, va="center", color=col)
    else:
        ax.text(x + 0.22, m, f"n=1\n{m:.3f}", fontsize=7.5, va="center",
                color=col)
    # individual seed dots (jittered)
    jit = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.12
    ax.scatter(x + jit, vals, s=42, color=col, edgecolors="white",
               linewidths=0.8, zorder=4)
    # mean marker
    ax.plot([x - 0.17, x + 0.17], [m, m], color=col, lw=2.6, zorder=5)

# reference bands. Zoom the y-axis on the arm region (≈1.28–1.47) so the
# seed candles are readable; v11c band (1.29–1.33) stays in frame; the
# far-below refs (R9_E13 1.029, seasonal-naive 1.0) are annotated as
# off-scale rather than compressing every candle.
if v11c:
    vmn, vmx, vmean = min(v11c), max(v11c), float(np.mean(v11c))
    ax.axhspan(vmn, vmx, color="#8c564b", alpha=.12, zorder=0)
    ax.axhline(vmean, color="#8c564b", ls=(0, (5, 2)), lw=1.3, zorder=0,
               label=f"v11c-recipe mean {vmean:.3f} (band {vmn:.3f}–{vmx:.3f}, n=3)")

allv = [v for _, _, vals in rows for v in vals]
ax.set_ylim(min(min(allv), (min(v11c) if v11c else 1.29)) - 0.015,
            max(allv) + 0.018)
ax.set_xticks(xs)
ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_ylabel("held-out full-97 GM-Relative MASE  (lower = better)")
ax.set_title("#307 cross-branch ablation — per-arm full-97 GM with seed spread\n"
             "candle = mean ± 1σ · whisker = min–max · dots = individual seeds · "
             "green = no all-time f↔h (A) · red = contains (A)", fontsize=10)
ax.grid(axis="y", ls=":", alpha=.4)
# off-scale references noted at the bottom of the frame
off = []
if r9:
    off.append(f"best-ever R9_E13 (#127) = {r9:.3f}")
off.append("seasonal naive = 1.000")
ax.text(0.015, 0.02, "off-scale below:  " + "   ·   ".join(off),
        transform=ax.transAxes, fontsize=8, color="#555", style="italic",
        va="bottom")
ax.legend(loc="upper left", fontsize=8, framealpha=.95)
ax.margins(x=0.06)
fig.tight_layout()
fig.savefig(f"{OUT}/variance_box.png", dpi=150)
print("variance_box arms (mean):",
      [(r[0].split(chr(10))[0], round(float(np.mean(r[2])), 4), len(r[2])) for r in rows])
