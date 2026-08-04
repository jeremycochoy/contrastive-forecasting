"""One training-loss subplot per arm run, #379.

Reads each `runs/bb_*_losses.csv` and its `_r2_/_r3_` extensions where
present.

Floors are the STRICT best lower bound of each recorded loss column.
For a term of the shape `LSE(exp(pos/τ), exp(neg/τ)·N) − pos/τ` the
strict min is `log1p(N·exp(−2/τ))` (attained at cos_pos = +1,
cos_neg = −1). For a pure LSE with no positive (`L_rep`), the strict
min is `log(N·exp(−1/τ)) = log(N) − 1/τ`. `L_align = 2 − 2·cos` has
strict min 0. Sums decompose additively.

For the small-model sweep, B = 64, T = 4096, C = 1, τ = 0.10:
  N_pred = B·(C + (B - 1))              = 64 · 64             = 4,096
  N_rep  = B·((C-1) + (T-1) + (B-1)·T)  = 64 · (0 + 4095 + 63·4096)
                                                              = 16,777,152
  L_pred (+moco) strict min = log1p(N_pred·e^-2)              ≈ 5.5e-4
  L_rep          strict min = log(N_rep) - 10                   = 6.63
  L_rep_moco     strict min = log1p(N_rep·e^-2)                 = 6.63 (dominates)

Sums decompose additively; arm 4's pooled loss already subtracts a
floor at train time and stays at 0.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-08-01_lalign_teacher"

# --- #390 path resolution --------------------------------------------------
# Curves live in this report's results/, not in an experiments/runs dir, and
# the ten L_align runs carry the `_alignteacher` name suffix.
CURVES = Path(__file__).resolve().parent.parent / "results" / "training_curves"


def _curve(name: str, suffix: str = "") -> Path:
    for stem in (f"{name}_alignteacher{suffix}", f"{name}{suffix}"):
        p = CURVES / f"{stem}_losses.csv"
        if p.exists():
            return p
    return CURVES / f"{name}{suffix}_losses.csv"
# ---------------------------------------------------------------------------

B, T, C, TAU = 64, 4096, 1, 0.10


def infonce_floor(tau: float, n_negatives: int) -> float:
    return math.log1p(float(n_negatives) * math.exp(-1.0 / float(tau)))


N_PRED = B * (C + (B - 1))
N_REP = B * ((C - 1) + (T - 1) + (B - 1) * T)

F_INFONCE_PRED_STRICT = math.log1p(N_PRED * math.exp(-2.0 / TAU))
F_INFONCE_REP_STRICT = math.log1p(N_REP * math.exp(-2.0 / TAU))
F_REP_LSE_STRICT = math.log(N_REP) - 1.0 / TAU
F_ALIGN_STRICT = 0.0

F_ARM1_STRICT = F_INFONCE_PRED_STRICT + F_REP_LSE_STRICT
F_ARM3_STRICT = F_INFONCE_PRED_STRICT + F_REP_LSE_STRICT
F_ARM4_STRICT = 0.0  # pooled loss already subtracts its floor at train time
F_ARM5_STRICT = F_ALIGN_STRICT + F_REP_LSE_STRICT
F_ARM6_STRICT = F_ALIGN_STRICT + F_INFONCE_REP_STRICT
F_BIMOCO_STRICT = F_INFONCE_PRED_STRICT + F_INFONCE_REP_STRICT

RUNS = [
    ("arm 1  (L_pred + L_rep)",
     "bb_small_arm1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     F_ARM1_STRICT, "#2a78d6"),
    ("arm 3  (L_pred_moco + L_rep)",
     "bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     F_ARM3_STRICT, "#eb6834"),
    ("arm 4  (pooled + MoCo, floor pre-subtracted)",
     "bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     F_ARM4_STRICT, "#008300"),
    ("arm 5  (L_align + L_rep)",
     "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     F_ARM5_STRICT, "#8b1e8b"),
    ("arm 6 v2  (L_align + L_rep_moco)",
     "bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     F_ARM6_STRICT, "#b8860b"),
    ("bimoco  (L_pred_moco + L_rep_moco)",
     "bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     F_BIMOCO_STRICT, "#00a3a3"),
]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def load(name: str) -> pd.DataFrame:
    base = _curve(name)
    df = pd.read_csv(base, usecols=["step", "loss"])
    for suffix in ("_r2", "_r3"):
        alt = _curve(name, suffix)
        if alt.exists() and pd.read_csv(alt, usecols=["step"])["step"].max() > 10_000:
            df = pd.concat([df, pd.read_csv(alt, usecols=["step", "loss"])], ignore_index=True)
    return df.sort_values("step").reset_index(drop=True)


N = len(RUNS)
COLS = 3
ROWS = math.ceil(N / COLS)
fig, axes = plt.subplots(ROWS, COLS, figsize=(4.6 * COLS, 3.4 * ROWS), sharex=False)
axes = axes.flatten()

for ax, (label, name, floor, colour) in zip(axes, RUNS):
    try:
        df = load(name)
    except FileNotFoundError:
        ax.set_title(f"{label}\n(no losses.csv)")
        continue
    df = df[df["step"] >= 100]
    y = df["loss"] - floor
    y = y.clip(lower=1e-6)
    ax.plot(df["step"], y, color=colour, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(100, max(df["step"].max(), 200_000) * 1.05)
    ax.set_title(
        f"{label}\nstrict-min floor {floor:.3f}   final step {df['step'].max():,}",
        fontsize=8.5)
    ax.grid(True, color=GRID, alpha=0.6, which="both")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("loss − strict_min  (log)")

for extra in axes[N:]:
    extra.set_visible(False)

fig.suptitle(
    "Per-arm training-loss deviation from the strict-min floor, log-log  "
    f"(B = {B}, T = {T}, C = {C}, τ = {TAU}; y auto-scaled per arm)",
    fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "per_run_loss.png"
fig.savefig(out)
print(f"wrote {out}")
