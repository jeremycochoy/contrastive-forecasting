"""HEADLINE: `1 − ff` per arm vs training step, all six on one axes, #379.

`ff` = mean cos(f_forecast, f_future) over L2-normalized vectors, logged
each step in `_losses.csv`. Directly comparable across arms (does not
depend on loss shape, τ, or SIGReg weighting). Perfect alignment →
ff = 1 → 1 − ff = 0.

`1 − ff` is a form of *log perplexity* of the forecast under the
future's von-Mises-Fisher on the unit sphere (small angle ⇔ small
1 − cos). This is the deliverable of #379: how does the 1 − ff
trajectory differ across the 6 loss recipes over 200k steps.

x-axis on log scale (temporal log axis — early-training dynamics take
one decade, mid takes one, late takes one, so log-x keeps all three
readable); y-axis linear.

Concatenation supports one or two restarts per run: base `_losses.csv`
+ `_r2_losses.csv` / `_r3_losses.csv` when its `step.max() > 10_000`
(avoids the trivial restart at step 0 that safe_run_name leaves behind).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-21_split_pred_rep_small"

# One entry per arm. Base run name matches run_arm.sh's per-arm case block.
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
]

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def load(name: str) -> pd.DataFrame:
    base = EXP / "runs" / f"{name}_losses.csv"
    df = pd.read_csv(base, usecols=["step", "ff"])
    for suffix in ("_r2", "_r3"):
        alt = EXP / "runs" / f"{name}{suffix}_losses.csv"
        if alt.exists() and pd.read_csv(alt, usecols=["step"])["step"].max() > 10_000:
            df = pd.concat([df, pd.read_csv(alt, usecols=["step", "ff"])], ignore_index=True)
    return df.sort_values("step").reset_index(drop=True)


fig, ax = plt.subplots(figsize=(9, 5.5))

for label, name, colour in RUNS:
    try:
        df = load(name)
    except FileNotFoundError:
        continue
    df = df[df["step"] >= 100]
    if df.empty:
        continue
    ax.plot(df["step"], 1.0 - df["ff"], color=colour, lw=1.4, label=label)

ax.set_xscale("log")
ax.set_xlim(100, 210_000)
ax.set_xlabel("training step (log)")
ax.set_ylabel("1 − ff  (log perplexity of f̂ under future's vMF)")
ax.grid(True, color=GRID, alpha=0.6, which="both")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_title(
    "1 − ⟨cos(f̂, f_true)⟩ per arm  (log-x temporal axis, linear y)",
    fontsize=11)
fig.tight_layout()
out = HERE / "cos_error_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
