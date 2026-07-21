"""One training-loss subplot per arm run.

Reads each `runs*/bb_*_losses.csv`. Where a `_r2_losses.csv` exists (25k
prolongation), it is concatenated onto the base 12,500-step curve so the
subplot shows the full 0 → 25,000 trajectory.

Floors are the STRICT best lower bound of each recorded loss column.
For a term of the shape `LSE(exp(pos/τ), exp(neg/τ)·N) − pos/τ` the
strict min is `log(1 + N·exp(−2/τ))` (attained at cos_pos = +1,
cos_neg = −1). For a pure LSE with no positive (`L_rep`), the strict
min is `log(N·exp(−1/τ)) = log(N) − 1/τ`. `L_align = 2 − 2·cos` has
strict min 0. Sums decompose additively.

B = 512, T = 4096, C = 1, τ = 0.10 → N_pred = 262,144, N_rep ≈ 1.074×10⁹:
  L_pred (and L_pred_moco) strict min  ≈  5.4×10⁻⁴  ≈ 0
  L_rep_moco             strict min  =  log(1 + N_rep·e⁻²⁰) = 1.17
  L_rep                  strict min  =  log(N_rep) − 10 = 10.79
  L_align                strict min  =  0

Layout: 3 columns × ceil(N/3) rows. Pure log-log on `loss − strict_min`
so every subplot's y-axis is strictly positive and auto-scaled to its
own arm's dynamic range.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-10_split_pred_rep"
B, T, C, TAU = 512, 4096, 1, 0.10


def infonce_floor(tau: float, n_negatives: int) -> float:
    return math.log1p(float(n_negatives) * math.exp(-1.0 / float(tau)))


# Strict best lower bounds, decomposed additively over the term summands.
N_PRED = B * (C + (B - 1))
N_REP = B * ((C - 1) + (T - 1) + (B - 1) * T)

F_INFONCE_PRED_STRICT = math.log1p(N_PRED * math.exp(-2.0 / TAU))   # ≈ 5.4e-4
F_INFONCE_REP_STRICT = math.log1p(N_REP * math.exp(-2.0 / TAU))     # 1.17
F_REP_LSE_STRICT = math.log(N_REP) - 1.0 / TAU                       # 10.79
F_ALIGN_STRICT = 0.0

F_ARM1_STRICT = F_INFONCE_PRED_STRICT + F_REP_LSE_STRICT              # 10.79
F_ARM3_STRICT = F_INFONCE_PRED_STRICT + F_REP_LSE_STRICT              # 10.79 (moco doesn't change the shape)
F_ARM4_STRICT = 0.0                                                    # pooled loss already subtracts a floor at train time
F_ARM5_STRICT = F_ALIGN_STRICT + F_REP_LSE_STRICT                     # 10.79
F_ARM6_STRICT = F_ALIGN_STRICT + F_INFONCE_REP_STRICT                 # 1.17
F_BIMOCO_STRICT = F_INFONCE_PRED_STRICT + F_INFONCE_REP_STRICT        # 1.17

RUNS = [
    ("arm 1  (L_pred + L_rep)",
     "runs", "bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     F_ARM1_STRICT, "#2a78d6"),
    ("arm 3  (L_pred_moco + L_rep)",
     "runs", "bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     F_ARM3_STRICT, "#eb6834"),
    ("arm 4  (pooled + MoCo, floor pre-subtracted)",
     "runs_arm4", "bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090",
     F_ARM4_STRICT, "#008300"),
    ("arm 5  (L_align + L_rep)",
     "runs_arm5", "bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090",
     F_ARM5_STRICT, "#8b1e8b"),
    ("arm 6  (L_align + L_rep_moco)",
     "runs_arm6_v2", "bb_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090",
     F_ARM6_STRICT, "#b8860b"),
    ("bimoco  (L_pred_moco + L_rep_moco)",
     "runs_bimoco_v2", "bb_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     F_BIMOCO_STRICT, "#00a3a3"),
]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def load(dir_: str, name: str) -> pd.DataFrame:
    # `_losses_full.csv` (if present) contains the full 0 → 12,500 base
    # curve; the plain `_losses.csv` for arms that were resumed mid-run
    # (e.g. arm 1 resumed at step 901) only contains the post-resume tail
    # so the plot loses steps 1–900. Prefer _losses_full when it exists.
    full = EXP / dir_ / f"{name}_losses_full.csv"
    base = full if full.exists() else EXP / dir_ / f"{name}_losses.csv"
    df = pd.read_csv(base, usecols=["step", "loss"])
    r2 = EXP / dir_ / f"{name}_r2_losses.csv"
    if r2.exists() and pd.read_csv(r2, usecols=["step"])["step"].max() > 12500:
        df = pd.concat([df, pd.read_csv(r2, usecols=["step", "loss"])], ignore_index=True)
    ext = EXP / dir_ / f"{name}_ext25k_losses.csv"
    if ext.exists():
        df_ext = pd.read_csv(ext, usecols=["step", "loss"])
        df_ext["step"] += 12500
        df = pd.concat([df, df_ext], ignore_index=True)
    return df.sort_values("step").reset_index(drop=True)


N = len(RUNS)
COLS = 3
ROWS = math.ceil(N / COLS)
fig, axes = plt.subplots(ROWS, COLS, figsize=(4.6 * COLS, 3.4 * ROWS), sharex=False)
axes = axes.flatten()

for ax, (label, dir_, name, floor, colour) in zip(axes, RUNS):
    try:
        df = load(dir_, name)
    except FileNotFoundError:
        ax.set_title(f"{label}\n(no losses.csv)")
        continue
    df = df[df["step"] >= 100]
    y = df["loss"] - floor
    # With a strict-min floor y must be > 0; guard against numerical zero for log-scale.
    y = y.clip(lower=1e-6)
    ax.plot(df["step"], y, color=colour, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(100, max(df["step"].max(), 12500) * 1.05)
    # y auto-scales per subplot (matplotlib default with sharey=False).
    ax.set_title(f"{label}\nstrict-min floor {floor:.3f}   final step {df['step'].max():,}", fontsize=8.5)
    ax.grid(True, color=GRID, alpha=0.6, which="both")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("loss − strict_min  (log)")

for extra in axes[N:]:
    extra.set_visible(False)

fig.suptitle("Per-arm training-loss deviation from the strict-min floor, log-log  (B = 512, T = 4096, C = 1, τ = 0.10; y auto-scaled per arm)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "per_run_loss.png"
fig.savefig(out)
print(f"wrote {out}")
