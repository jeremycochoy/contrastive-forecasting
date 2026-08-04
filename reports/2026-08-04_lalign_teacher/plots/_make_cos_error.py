"""HEADLINE: `1 − ff` per arm vs training step, all six on one axes, #379.

`ff` = mean cos(f_forecast, f_future) over L2-normalized vectors, logged
each step in `_losses.csv`. Directly comparable across arms (does not
depend on loss shape, τ, or SIGReg weighting). Perfect alignment →
ff = 1 → 1 − ff = 0.

`1 − ff` is a cosine distance in [0, 2]; smaller means the forecast
sits closer to the future latent.

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
EXP = ROOT / "experiments" / "2026-08-01_lalign_teacher"

# --- #390 path resolution --------------------------------------------------
# Curves live in this report's results/, not in an experiments/runs dir, and
# the ten L_align runs carry the `_alignteacher` name suffix.
CURVE_DIR = Path(__file__).resolve().parent.parent / "results" / "training_curves"


def _curve(name: str, suffix: str = "") -> Path:
    for stem in (f"{name}_alignteacher{suffix}", f"{name}{suffix}"):
        p = CURVE_DIR / f"{stem}_losses.csv"
        if p.exists():
            return p
    return CURVE_DIR / f"{name}{suffix}_losses.csv"
# ---------------------------------------------------------------------------


# One entry per arm. Base run name matches run_arm.sh's per-arm case block.
# Base + tau_rep + nse (paler variant of base colour) + ncpc (base colour,
# dashed via STYLE). Line style for each run defaults to solid "-"; ncpc
# runs are overridden through the STYLE lookup below to distinguish
# CPC-off runs from base without needing another colour.
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
    # #379 tau_rep=1.0 reruns — paler variant of the base colour.
    ("arm 1 tr1  (L_pred + L_rep, all τ=1.0)",
     "bb_small_arm1_tr1_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fb0e8"),
    ("arm 3 tr1  (L_pred_moco + L_rep, all τ=1.0)",
     "bb_small_arm3_tr1_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#f4a680"),
    ("arm 4 tr1  (pooled + MoCo, all τ=1.0)",
     "bb_small_arm4_tr1_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fc17f"),
    ("arm 5 tr1  (L_align + L_rep, all τ=1.0)",
     "bb_small_arm5_tr1_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#c98cc9"),
    ("arm 6 v2 tr1  (L_align + L_rep_moco, all τ=1.0)",
     "bb_small_arm6_v2_tr1_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#dcbb60"),
    ("bimoco tr1  (L_pred_moco + L_rep_moco, all τ=1.0)",
     "bb_small_bimoco_tr1_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#66c4c4"),
    # #379 no-sigreg-embedding reruns — palest variant of the base colour.
    ("arm 1 nse  (L_pred + L_rep, sigreg_e=0)",
     "bb_small_arm1_nse_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#a6c8ee"),
    ("arm 3 nse  (L_pred_moco + L_rep, sigreg_e=0)",
     "bb_small_arm3_nse_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#f5b39a"),
    ("arm 4 nse  (pooled + MoCo, sigreg_e=0)",
     "bb_small_arm4_nse_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fc17f"),
    ("arm 5 nse  (L_align + L_rep, sigreg_e=0)",
     "bb_small_arm5_nse_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#c58fc5"),
    ("arm 6 v2 nse  (L_align + L_rep_moco, sigreg_e=0)",
     "bb_small_arm6_v2_nse_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#dcc385"),
    ("bimoco nse  (L_pred_moco + L_rep_moco, sigreg_e=0)",
     "bb_small_bimoco_nse_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#7fd1d1"),
    # #379 no-CPC reruns — same base colour, dashed via STYLE below.
    ("arm 1 ncpc  (L_pred + L_rep, cpc=0)",
     "bb_small_arm1_ncpc_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3 ncpc  (L_pred_moco + L_rep, cpc=0)",
     "bb_small_arm3_ncpc_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4 ncpc  (pooled + MoCo, cpc=0)",
     "bb_small_arm4_ncpc_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5 ncpc  (L_align + L_rep, cpc=0)",
     "bb_small_arm5_ncpc_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2 ncpc  (L_align + L_rep_moco, cpc=0)",
     "bb_small_arm6_v2_ncpc_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco ncpc  (L_pred_moco + L_rep_moco, cpc=0)",
     "bb_small_bimoco_ncpc_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
    # #379 combined-ablation (combab): all τ=1.0 + cpc=0 + nse (only for
    # arm1/3/4 where nse helped; arm5/6_v2/bimoco keep sigreg_e=1.0).
    ("arm 1 combab  (τ=1.0 + cpc=0 + sigreg_e=0)",
     "bb_small_arm1_combab_split_pred_rep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#2a78d6"),
    ("arm 3 combab  (τ=1.0 + cpc=0 + sigreg_e=0)",
     "bb_small_arm3_combab_split_pred_rep_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#eb6834"),
    ("arm 4 combab  (τ=1.0 + cpc=0 + sigreg_e=0)",
     "bb_small_arm4_combab_xshh_allt_moco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#008300"),
    ("arm 5 combab  (τ_rep=1.0 + cpc=0)",
     "bb_small_arm5_combab_lalign_lrep_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#8b1e8b"),
    ("arm 6 v2 combab  (τ_rep=1.0 + cpc=0)",
     "bb_small_arm6_v2_combab_lalign_lrepmoco_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#b8860b"),
    ("bimoco combab  (all τ=1.0 + cpc=0)",
     "bb_small_bimoco_combab_split_pred_rep_moco_bothsides_enc3l3_b64_200k_sigreg_ema_qk_aon_cpc_tau090",
     "#00a3a3"),
]
# Short slug per arm (parallel to RUNS by index). Used both for legend
# grouping and to gate --arms in the latent-movement plot.
SLUGS = ["arm1", "arm3", "arm4", "arm5", "arm6_v2", "bimoco",
         "arm1_tr1", "arm3_tr1", "arm4_tr1", "arm5_tr1", "arm6_v2_tr1", "bimoco_tr1",
         "arm1_nse", "arm3_nse", "arm4_nse", "arm5_nse", "arm6_v2_nse", "bimoco_nse",
         "arm1_ncpc", "arm3_ncpc", "arm4_ncpc", "arm5_ncpc", "arm6_v2_ncpc", "bimoco_ncpc",
         "arm1_combab", "arm3_combab", "arm4_combab", "arm5_combab", "arm6_v2_combab", "bimoco_combab"]
# Per-slug linestyle. ncpc → dashed (base colour reused). Everything else
# defaults to solid via the .get() fallback.
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
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})


def load(name: str) -> pd.DataFrame:
    base = _curve(name)
    df = pd.read_csv(base, usecols=["step", "ff"])
    for suffix in ("_r2", "_r3"):
        alt = _curve(name, suffix)
        if alt.exists() and pd.read_csv(alt, usecols=["step"])["step"].max() > 10_000:
            df = pd.concat([df, pd.read_csv(alt, usecols=["step", "ff"])], ignore_index=True)
    return df.sort_values("step").reset_index(drop=True)


# 2x2 panel grid, one variant per panel. Same shared y-limits so panels
# are directly comparable at the same 1−ff altitude.
PANEL_SUFFIXES = [
    ("base  (all τ=0.10, sigreg_e=1.0, cpc=1.0)", ""),
    ("tr1  (all τ=1.00)", "_tr1"),
    ("nse  (sigreg_e=0)", "_nse"),
    ("ncpc  (cpc=0)", "_ncpc"),
    ("combab  (all τ=1.0 + cpc=0 + conditional nse)", "_combab"),
]

# Cache curves once so we can compute a shared y-limit before drawing.
CURVES: dict[str, tuple[list, list, str, str]] = {}
for (label, name, colour), slug in zip(RUNS, SLUGS):
    try:
        df = load(name)
    except FileNotFoundError:
        continue
    df = df[df["step"] >= 100]
    if df.empty:
        continue
    CURVES[slug] = (df["step"].tolist(), (1.0 - df["ff"]).tolist(),
                    colour, label)

y_max = max((max(vals) for _, vals, _, _ in CURVES.values()), default=1.0)
y_min = min((min(vals) for _, vals, _, _ in CURVES.values()), default=0.0)

ncols, nrows = len(PANEL_SUFFIXES), 1
fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 5.4),
                         sharex=True, sharey=True)
axes = axes.flatten()

for ax, (panel_title, suffix) in zip(axes, PANEL_SUFFIXES):
    for slug, (steps, vals, colour, label) in CURVES.items():
        if suffix == "":
            # Base panel: any slug WITHOUT tr1/nse/ncpc/combab suffix.
            if any(s in slug for s in ("_tr1", "_nse", "_ncpc", "_combab")):
                continue
        else:
            if not slug.endswith(suffix):
                continue
        ax.plot(steps, vals, color=colour, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xlim(100, 210_000)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, color=GRID, alpha=0.6, which="both")
    # Legend under the axes: inside the panel it covers the curves.
    ax.set_xlabel("training step (log)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5,
              frameon=False)
    ax.set_title(panel_title.split("  (")[0], fontsize=10)

axes[0].set_ylabel("1 − ff  (cosine distance)")

fig.suptitle("1 − ff against training step, one panel per setting  "
             "(shared axes)", fontsize=11)
fig.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.33,
                    wspace=0.12)
out = HERE / "cos_error_per_arm.png"
fig.savefig(out)
print(f"wrote {out}")
