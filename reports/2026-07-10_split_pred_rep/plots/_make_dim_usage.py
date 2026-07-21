"""Per-arm dimension usage across training steps.

Two curves per arm:
  * u_batchtime    — dim usage of h_t (main representation)
  * u_batchtime_e  — dim usage of e_t (embedding)

Both pool (batch × time) into one sample axis of size B·T, then compute
1 / (d · off-diag Gram mean), clamped to [0, 1]. 1 = every H dim carries
independent info; low = the representation is collapsing onto a subspace.
This is the exact axis SIGReg regularises, so the plot answers "is the
encoder still using its capacity, or drifting into a lower-dimensional
manifold?".

Concatenation mirrors `_make_cos_error.py` (base + r2 + arm-3-ext25k
shifted + r3 where present). y linear (naturally bounded [0,1]).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-10_split_pred_rep"

RUNS = [
    ("arm 1  (L_pred + L_rep)",
     "runs", "bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "#2a78d6"),
    ("arm 3  (L_pred_moco + L_rep)",
     "runs", "bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "#eb6834"),
    ("arm 4  (pooled + MoCo)",
     "runs_arm4", "bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090",
     "#008300"),
    ("arm 5  (L_align + L_rep)",
     "runs_arm5", "bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090",
     "#8b1e8b"),
    ("arm 6  (L_align + L_rep_moco)",
     "runs_arm6_v2", "bb_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090",
     "#b8860b"),
    ("bimoco  (L_pred_moco + L_rep_moco)",
     "runs_bimoco_v2", "bb_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     "#00a3a3"),
]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

COLS_NEEDED = ["step", "u_batchtime", "u_batchtime_e"]


def load(dir_: str, name: str) -> pd.DataFrame:
    full = EXP / dir_ / f"{name}_losses_full.csv"
    base = full if full.exists() else EXP / dir_ / f"{name}_losses.csv"
    df = pd.read_csv(base, usecols=COLS_NEEDED)
    r2 = EXP / dir_ / f"{name}_r2_losses.csv"
    if r2.exists() and pd.read_csv(r2, usecols=["step"])["step"].max() > 12500:
        df = pd.concat([df, pd.read_csv(r2, usecols=COLS_NEEDED)], ignore_index=True)
    ext = EXP / dir_ / f"{name}_ext25k_losses.csv"
    if ext.exists():
        df_ext = pd.read_csv(ext, usecols=COLS_NEEDED)
        df_ext["step"] += 12500
        df = pd.concat([df, df_ext], ignore_index=True)
    r3 = EXP / dir_ / f"{name}_r3_losses.csv"
    if r3.exists():
        df = pd.concat([df, pd.read_csv(r3, usecols=COLS_NEEDED)], ignore_index=True)
    return df.sort_values("step").reset_index(drop=True)


N = len(RUNS)
COLS = 3
ROWS = math.ceil(N / COLS)
fig, axes = plt.subplots(ROWS, COLS, figsize=(4.6 * COLS, 3.4 * ROWS), sharex=False)
axes = axes.flatten()

for ax, (label, dir_, name, colour) in zip(axes, RUNS):
    try:
        df = load(dir_, name)
    except FileNotFoundError:
        ax.set_title(f"{label}\n(no losses.csv)")
        continue
    df = df[df["step"] >= 100]
    ax.plot(df["step"], df["u_batchtime"], color=colour, lw=1.4, label="h_t (main rep)")
    ax.plot(df["step"], df["u_batchtime_e"], color=colour, lw=1.0, ls="--", alpha=0.75, label="e_t (embedding)")
    ax.set_xscale("log")
    ax.set_xlim(100, max(df["step"].max(), 12500) * 1.05)
    fh, fe = df["u_batchtime"].iloc[-1], df["u_batchtime_e"].iloc[-1]
    ax.set_title(f"{label}\nfinal step {df['step'].max():,}   h_t={fh:.3f}  e_t={fe:.3f}", fontsize=8.5)
    ax.grid(True, color=GRID, alpha=0.6, which="both")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("u_batchtime  (dim usage)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)

for extra in axes[N:]:
    extra.set_visible(False)

fig.suptitle("Dimension usage per arm  (u_batchtime = 1/(d · off-diag Gram mean), pooled over B×T; 1 = all dims used)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "dim_usage_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
