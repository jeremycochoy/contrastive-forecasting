"""Per-arm average cosine error (1 - ff) across training steps, #379.

`ff` = mean cos(f_forecast, f_future) over L2-normalized vectors, logged
each step in `_losses.csv`. Directly comparable across arms (does not
depend on loss shape, τ, or SIGReg weighting). Perfect alignment →
ff = 1 → 1 − ff = 0.

Concatenation supports one restart per run: base `_losses.csv` +
`_r2_losses.csv` if its `step.max() > save_every` (avoids the trivial
restart at step 0 that safe_run_name leaves behind). x-axis in log
scale; the 200k-step trajectory naturally wants log-x to keep the
first 10k readable.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-21_split_pred_rep_small"

# One entry per arm. Base run name matches run_arm.sh's per-arm case block.
RUNS = [
    ("arm 1  (L_pred + L_rep)",
     "bb_small_arm1_split_pred_rep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3  (L_pred_moco + L_rep)",
     "bb_small_arm3_split_pred_rep_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4  (pooled + MoCo)",
     "bb_small_arm4_xshh_allt_moco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5  (L_align + L_rep)",
     "bb_small_arm5_lalign_lrep_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2  (L_align + L_rep_moco)",
     "bb_small_arm6_v2_lalign_lrepmoco_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco  (L_pred_moco + L_rep_moco)",
     "bb_small_bimoco_split_pred_rep_moco_bothsides_enc3l3_b128_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def load(name: str) -> pd.DataFrame:
    base = EXP / "runs" / f"{name}_losses.csv"
    df = pd.read_csv(base, usecols=["step", "ff"])
    r2 = EXP / "runs" / f"{name}_r2_losses.csv"
    if r2.exists() and pd.read_csv(r2, usecols=["step"])["step"].max() > 10_000:
        df = pd.concat([df, pd.read_csv(r2, usecols=["step", "ff"])], ignore_index=True)
    r3 = EXP / "runs" / f"{name}_r3_losses.csv"
    if r3.exists() and pd.read_csv(r3, usecols=["step"])["step"].max() > 10_000:
        df = pd.concat([df, pd.read_csv(r3, usecols=["step", "ff"])], ignore_index=True)
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
    y = 1.0 - df["ff"]
    ax.plot(df["step"], y, color=colour, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlim(100, 210_000)
    ax.set_title(
        f"{label}\nfinal step {df['step'].max():,}   final ff = {df['ff'].iloc[-1]:.3f}",
        fontsize=8.5)
    ax.grid(True, color=GRID, alpha=0.6, which="both")
    ax.set_xlabel("training step (log)")
    ax.set_ylabel("1 − ff")

for extra in axes[N:]:
    extra.set_visible(False)

fig.suptitle(
    "Average cosine error 1 − ⟨cos(f̂, f_true)⟩ per arm  "
    "(log x, linear y — target 0)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = HERE / "cos_error_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
