"""Per-arm dimension usage across training steps, #379.

Two curves per arm:
  * u_batchtime    — dim usage of h_t (main representation)
  * u_batchtime_e  — dim usage of e_t (embedding)

Both pool (B × T) into one sample axis of size B·T, then compute
1 / (d · off-diag Gram mean), clamped to [0, 1]. 1 = every H dim carries
independent info; low = the representation is collapsing onto a subspace.
This is the exact axis SIGReg regularises.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-21_split_pred_rep_small"

RUNS = [
    ("arm 1  (L_pred + L_rep)",
     "bb_small_arm1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3  (L_pred_moco + L_rep)",
     "bb_small_arm3_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4  (pooled + MoCo)",
     "bb_small_arm4_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5  (L_align + L_rep)",
     "bb_small_arm5_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2  (L_align + L_rep_moco)",
     "bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco  (L_pred_moco + L_rep_moco)",
     "bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
    # #379 no-sigreg-embedding reruns — paler variant of the base colour.
    ("arm 1 nse  (sigreg_e=0)",
     "bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#a6c8ee"),
    ("arm 3 nse  (sigreg_e=0)",
     "bb_small_arm3_nse_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#f5b39a"),
    ("arm 4 nse  (sigreg_e=0)",
     "bb_small_arm4_nse_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fc17f"),
    ("arm 5 nse  (sigreg_e=0)",
     "bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#c58fc5"),
    ("arm 6 v2 nse  (sigreg_e=0)",
     "bb_small_arm6_v2_nse_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#dcc385"),
    ("bimoco nse  (sigreg_e=0)",
     "bb_small_bimoco_nse_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fd1d1"),
    # #379 no-CPC reruns — same base colour, dashed via STYLE below.
    ("arm 1 ncpc  (cpc=0)",
     "bb_small_arm1_ncpc_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3 ncpc  (cpc=0)",
     "bb_small_arm3_ncpc_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4 ncpc  (cpc=0)",
     "bb_small_arm4_ncpc_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5 ncpc  (cpc=0)",
     "bb_small_arm5_ncpc_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2 ncpc  (cpc=0)",
     "bb_small_arm6_v2_ncpc_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco ncpc  (cpc=0)",
     "bb_small_bimoco_ncpc_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
]
SLUGS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco",
         "arm1_nse", "arm3_nse", "arm4_nse", "arm5_nse", "arm6_v2_nse", "bimoco_nse",
         "arm1_ncpc", "arm3_ncpc", "arm4_ncpc", "arm5_ncpc", "arm6_v2_ncpc", "bimoco_ncpc"]
# ncpc runs render dashed so a shared base colour + linestyle disambiguates
# CPC-off from base in the per-panel plot.
STYLE = {
    "arm1_ncpc": "--",
    "arm3_ncpc": "--",
    "arm4_ncpc": "--",
    "arm5_ncpc": "--",
    "arm6_v2_ncpc": "--",
    "bimoco_ncpc": "--",
}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

COLS_NEEDED = ["step", "u_batchtime", "u_batchtime_e"]


def load(name: str) -> pd.DataFrame:
    base = EXP / "runs" / f"{name}_losses.csv"
    df = pd.read_csv(base, usecols=COLS_NEEDED)
    for suffix in ("_r2", "_r3"):
        alt = EXP / "runs" / f"{name}{suffix}_losses.csv"
        if alt.exists() and pd.read_csv(alt, usecols=["step"])["step"].max() > 10_000:
            df = pd.concat([df, pd.read_csv(alt, usecols=COLS_NEEDED)], ignore_index=True)
    return df.sort_values("step").reset_index(drop=True)


N = len(RUNS)
COLS = 3
ROWS = math.ceil(N / COLS)
fig, axes = plt.subplots(ROWS, COLS, figsize=(4.6 * COLS, 3.4 * ROWS), sharex=False)
axes = axes.flatten()

for ax, (label, name, colour) in zip(axes, RUNS):
    try:
        df = load(name)
    except FileNotFoundError:
        ax.set_title(f"{label}\n(no losses.csv)")
        continue
    df = df[df["step"] >= 100]
    ax.plot(df["step"], df["u_batchtime"], color=colour, lw=1.4, label="h_t (main rep)")
    ax.plot(df["step"], df["u_batchtime_e"], color=colour, lw=1.0, ls="--", alpha=0.75,
            label="e_t (embedding)")
    ax.set_xscale("log")
    ax.set_xlim(100, max(df["step"].max(), 200_000) * 1.05)
    fh, fe = df["u_batchtime"].iloc[-1], df["u_batchtime_e"].iloc[-1]
    ax.set_title(
        f"{label}\nfinal step {df['step'].max():,}   h_t={fh:.3f}  e_t={fe:.3f}",
        fontsize=8.5)
    ax.grid(True, color=GRID, alpha=0.6, which="both")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("u_batchtime  (dim usage)")
    ax.legend(loc="lower right", fontsize=7, frameon=False)

for extra in axes[N:]:
    extra.set_visible(False)

fig.suptitle(
    "Dimension usage per arm  (u_batchtime = 1/(d · off-diag Gram mean), "
    "pooled over B×T; 1 = all dims used)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "dim_usage_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
