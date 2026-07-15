"""Training-loss curves for all six arms, floor-subtracted so all arms read
zero at the uniformity floor. X-axis starts at step 100.

Per-arm floor rules (all at B=512, T=4096, C=1, τ=0.10):

  - split shape (arm 1, arm 3, bimoco): --subtract-contrastive-floor was
    NOT set at training time; recorded loss = L_pred + L_rep. Post-hoc
    subtract (f_pred + f_rep) computed by `src.loss._split_pred_rep_floors`:
    f_pred = infonce_floor(τ, B·(C+B−1)) ≈ 2.557
    f_rep  = log(B·((C−1)+(T−1)+(B−1)·T))  ≈ 20.794
    total ≈ 23.351.

  - rep-only + L_align (arm 5): recorded loss = L_align + L_rep, no train-
    time subtraction. L_align has min 0 (2 − 2·cos), no floor. Subtract
    log(N_rep) ≈ 20.794.

  - rep-only + L_align_moco (arm 6): recorded loss = L_align_moco + L_rep.
    L_align_moco is a per-anchor InfoNCE with (B−1) cross-batch keys; its
    floor is infonce_floor(τ, B−1) ≈ 0.023 (tiny). Subtract that + log(N_rep).

  - xshh_allt + --subtract-contrastive-floor (arm 4): recorded loss is
    ALREADY floor-subtracted at training time. Plot as-is (subtract 0).

The bimoco run also emits `loss_tau_ref` (a fixed-τ diagnostic reference at
τ=0.07 with the same shape); we only plot the training loss column here.

Output: `plots/train_loss_curves_floor_subtracted.png`.
"""
from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-10_split_pred_rep"

# Common training config.
B, T, C, TAU = 512, 4096, 1, 0.10


def infonce_floor(tau: float, n_negatives: int) -> float:
    return math.log1p(float(n_negatives) * math.exp(-1.0 / float(tau)))


def split_pred_rep_floor(tau: float, B: int, T: int, C: int) -> float:
    n_pred = B * (C + (B - 1))
    n_rep = B * ((C - 1) + (T - 1) + (B - 1) * T)
    return infonce_floor(tau, n_pred) + math.log(n_rep)


def rep_only_floor(tau: float, B: int, T: int, C: int, moco: bool) -> float:
    n_rep = B * ((C - 1) + (T - 1) + (B - 1) * T)
    align_moco_floor = infonce_floor(tau, B - 1) if moco else 0.0
    return math.log(n_rep) + align_moco_floor


def infonce_floor_rep(tau: float, B: int, T: int, C: int) -> float:
    """L_rep_moco floor: normalized InfoNCE with N_rep negatives at cos_pos = 1,
    cos_neg = 0. Distinct from `rep_only_floor` (which was the pooled-LSE form
    of L_rep, no positive)."""
    n_rep = B * ((C - 1) + (T - 1) + (B - 1) * T)
    return infonce_floor(tau, n_rep)


F_SPLIT_ORIG = split_pred_rep_floor(TAU, B, T, C)                # arm 1, arm 3: L_pred (with pos) + L_rep (LSE only)
F_REP_LSE = math.log(B * ((C - 1) + (T - 1) + (B - 1) * T))       # arm 5: L_align (min 0) + L_rep (LSE)
F_REP_MOCO_CORRECT = infonce_floor_rep(TAU, B, T, C)              # L_rep_moco as normalized InfoNCE, ≈ 10.80
F_SPLIT_BIMOCO_CORRECT = infonce_floor(TAU, B * (C + (B - 1))) + F_REP_MOCO_CORRECT  # L_pred_moco + L_rep_moco

# Correct-implementation arms only. Wrong bimoco / wrong arm 6 excluded.
ARMS = [
    ("L_pred + L_rep",
     EXP / "runs" / "bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses.csv",
     F_SPLIT_ORIG, "#2a78d6"),
    ("L_pred_moco + L_rep",
     EXP / "runs" / "bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses.csv",
     F_SPLIT_ORIG, "#eb6834"),
    ("L_pooled_moco (floor pre-subtracted)",
     EXP / "runs_arm4" / "bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090_losses.csv",
     0.0, "#008300"),
    ("L_align + L_rep",
     EXP / "runs_arm5" / "bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090_losses.csv",
     F_REP_LSE, "#8b1e8b"),
    ("L_align + L_rep_moco (arm 6 v2, correct)",
     EXP / "runs_arm6_v2" / "bb_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090_losses.csv",
     F_REP_MOCO_CORRECT, "#b8860b"),
    ("L_pred_moco + L_rep_moco (bimoco v2, correct)",
     EXP / "runs_bimoco_v2" / "bb_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses.csv",
     F_SPLIT_BIMOCO_CORRECT, "#00a3a3"),
]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# The "uniformity floor" (cos_neg ≡ 0 for cos_pos = 1) is not a strict lower
# bound: a run that pushes cos_neg < 0 can drive the recorded loss BELOW it.
# For a log-scaled y-axis we use symlog with a linear threshold around 0 so
# negatives are still readable; the linear-zero band renders as a straight
# stripe near the axis.
fig, ax = plt.subplots(figsize=(12.5, 6.5))
for label, path, floor, colour in ARMS:
    if not path.exists():
        print(f"skip {label}: {path} missing")
        continue
    df = pd.read_csv(path, usecols=["step", "loss"])
    df = df[df["step"] >= 100]
    y = df["loss"] - floor
    ax.plot(df["step"], y, color=colour, lw=1.4,
            label=f"{label}  (floor {floor:.3f})")

ax.axhline(0.0, color=MUTED, lw=0.9, ls="--")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=0.1)
ax.set_xlabel("training step  (log)")
ax.set_ylabel("training loss − uniformity floor  (symlog, linthresh 0.1)")
ax.set_xlim(100, None)
ax.grid(True, color=GRID, alpha=0.6, which="both")
ax.legend(loc="upper right", fontsize=8, frameon=False)
ax.set_title(
    "Training loss above uniformity floor (log-log), step 100 onward. "
    "B = 512, T = 4096, C = 1, τ = 0.10. "
    "Negatives = the run pushed cos⁻ below 0 (spread past the random-init reference).",
    fontsize=9, loc="left")

out = HERE / "train_loss_curves_floor_subtracted_loglog.png"
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")
print(f"floors: split_orig = {F_SPLIT_ORIG:.4f}   rep_lse = {F_REP_LSE:.4f}   "
      f"rep_moco = {F_REP_MOCO_CORRECT:.4f}   bimoco_correct = {F_SPLIT_BIMOCO_CORRECT:.4f}")
