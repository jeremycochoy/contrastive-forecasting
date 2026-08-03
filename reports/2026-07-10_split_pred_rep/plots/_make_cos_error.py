"""Per-arm average cosine error (1 - ff) across training steps.

`ff` is mean cos(f_forecast, f_future) over L2-normalized vectors, logged
each step in `_losses.csv`. Directly comparable across arms since every
arm normalizes the same way; it does not depend on loss shape, τ, or
SIGReg weighting. Perfect alignment → ff = 1 → 1 - ff = 0.

Concatenation follows the same rule as `_make_per_run_loss.py`:
`_losses.csv` (0 → 12,500) + `_r2_losses.csv` (12,501 → 25,000) if the
r2 file's max step > 12,500; also `_ext25k_losses.csv` shifted by +12,500
for arm 3 (no optimizer state at 12,500, so extension re-warmed from
step 0 and is plotted at 12,500 → 25,000). `_r3_losses.csv` for 50k
extensions is added on top when present.

Layout: 3 columns × ceil(N/3) rows. y auto-scales per subplot; x log.
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


def load(dir_: str, name: str) -> pd.DataFrame:
    full = EXP / dir_ / f"{name}_losses_full.csv"
    base = full if full.exists() else EXP / dir_ / f"{name}_losses.csv"
    df = pd.read_csv(base, usecols=["step", "ff"])
    r2 = EXP / dir_ / f"{name}_r2_losses.csv"
    if r2.exists() and pd.read_csv(r2, usecols=["step"])["step"].max() > 12500:
        df = pd.concat([df, pd.read_csv(r2, usecols=["step", "ff"])], ignore_index=True)
    ext = EXP / dir_ / f"{name}_ext25k_losses.csv"
    if ext.exists():
        df_ext = pd.read_csv(ext, usecols=["step", "ff"])
        df_ext["step"] += 12500
        df = pd.concat([df, df_ext], ignore_index=True)
    r3 = EXP / dir_ / f"{name}_r3_losses.csv"
    if r3.exists():
        df = pd.concat([df, pd.read_csv(r3, usecols=["step", "ff"])], ignore_index=True)
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
    y = 1.0 - df["ff"]
    ax.plot(df["step"], y, color=colour, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlim(100, 52500)
    ax.set_title(f"{label}\nfinal step {df['step'].max():,}   final ff = {df['ff'].iloc[-1]:.3f}", fontsize=8.5)
    ax.grid(True, color=GRID, alpha=0.6, which="both")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("1 − ff")

for extra in axes[N:]:
    extra.set_visible(False)

fig.suptitle("Average cosine error 1 − ⟨cos(f̂, f_true)⟩ per arm  (log x, linear y — target 0)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "cos_error_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
