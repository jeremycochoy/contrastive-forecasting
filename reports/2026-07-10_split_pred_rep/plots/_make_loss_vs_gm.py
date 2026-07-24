"""Backbone training loss per arm, aligned with that arm's GM-Relative MASE snapshots.

One small-multiple panel per arm. Left y-axis: backbone training `loss` vs `step`,
concatenated across the training segments (1-12,500, the 12,500-25,000 resume, and
the 25,000-50,000 resume where it exists). Right y-axis: the arm's evaluated
GM-Relative MASE cells at the backbone steps where an eval exists, 2L and 6L
distinguished by marker shape, with a vertical guide at each evaluated step.

`loss` is NOT comparable across arms: the arms optimise different loss shapes
(split vs pooled InfoNCE, BYOL alignment, different negative counts, and arm 4
subtracts the analytic contrastive floor). Each panel is read within itself.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-10_split_pred_rep"

C_ARM1, C_ARM3, C_ARM4 = "#2a78d6", "#eb6834", "#008300"
C_ARM5, C_ARM6, C_BIMOCO = "#8b1e8b", "#b8860b", "#00a3a3"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# (label, colour, results-dir, eval base name, best_step,
#  runs-dir, [(losses file, step offset), ...]).
ARMS = [
    ("arm 1 (split)", C_ARM1, "results",
     "gift_eval_full_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     12500, "runs", [
         ("bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses_full.csv", 0),
         ("bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_r2_losses.csv", 0),
         ("bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_r3_losses.csv", 0),
     ]),
    ("arm 3 (split + MoCo)", C_ARM3, "results",
     "gift_eval_full_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     11800, "runs", [
         ("bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses.csv", 0),
         # the two resume runs re-index their step counter from 1.
         ("bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_ext25k_losses.csv", 12500),
         ("bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_r3_losses.csv", 12500),
     ]),
    ("arm 4 (pooled + MoCo)", C_ARM4, "results_arm4",
     "gift_eval_full_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090",
     600, "runs_arm4", [
         ("bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090_losses.csv", 0),
         ("bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090_r2_losses.csv", 0),
         ("bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090_r3_losses.csv", 0),
     ]),
    ("arm 5 (L_align + L_rep)", C_ARM5, "results_arm5",
     "gift_eval_full_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090",
     11800, "runs_arm5", [
         ("bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090_losses.csv", 0),
         ("bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090_r2_losses.csv", 0),
         ("bb_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090_r3_losses.csv", 0),
     ]),
    ("arm 6 (L_align + L_rep_moco)", C_ARM6, "results_arm6_v2",
     "gift_eval_full_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090",
     8700, "runs_arm6_v2", [
         ("bb_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090_losses.csv", 0),
         ("bb_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090_r2_losses.csv", 0),
         ("bb_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090_r3_losses.csv", 0),
     ]),
    ("arm bimoco (L_pred_moco + L_rep_moco)", C_BIMOCO, "results_bimoco_v2",
     "gift_eval_full_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
     12400, "runs_bimoco_v2", [
         ("bb_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses.csv", 0),
         ("bb_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_r2_losses.csv", 0),
     ]),
]

MARKER = {"2L": "o", "6L": "^"}
LINESTYLE = {"2L": ":", "6L": "--"}


def gm(path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(r"Aggregate GM-Relative MASE.*?([0-9]+\.[0-9]+)", path.read_text())
    return float(m.group(1)) if m else None


def loss_curve(runs_dir: str, segments) -> pd.DataFrame:
    frames = []
    for name, offset in segments:
        p = EXP / runs_dir / name
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=["step", "loss"])
        if not len(d):
            continue
        d = d.dropna(subset=["loss"])
        d["step"] = d["step"] + offset
        frames.append(d)
    out = pd.concat(frames).sort_values("step").drop_duplicates("step", keep="last")
    # 100-step rolling mean: the per-step trace is dominated by batch noise.
    out["loss"] = out["loss"].rolling(100, min_periods=1).mean()
    return out


fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, (label, colour, rd, base, best_step, runs_dir, segs) in zip(axes.ravel(), ARMS):
    curve = loss_curve(runs_dir, segs)
    ax.plot(curve["step"], curve["loss"], color=colour, lw=1.2)
    ax.set_title(label, fontsize=9, color=colour)
    ax.set_ylabel("backbone loss (100-step mean)", fontsize=8, color=colour)
    ax.tick_params(axis="y", labelcolor=colour)
    ax.grid(True, color=GRID, alpha=0.6)
    ax.set_xlim(0, 51000)

    ax2 = ax.twinx()
    for HL in ("2L", "6L"):
        pts = []
        for step, suffix in ((2000, "_2k"), (best_step, ""), (12500, "_last"),
                             (25000, "_25k"), (50000, "_50k")):
            val = gm(EXP / rd / (base + suffix + f"_{HL}") / "summary.txt")
            if val is not None:
                pts.append((step, val))
        pts = sorted({s: v for s, v in pts}.items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, color=INK, lw=0.7, ls=LINESTYLE[HL], marker=MARKER[HL],
                 markersize=6, markerfacecolor="none", markeredgewidth=1.3)
        for x in xs:
            ax.axvline(x, color=MUTED, lw=0.6, ls=":", alpha=0.7)
    ax2.set_ylim(1.09, 1.35)
    ax2.set_ylabel("GM-Relative MASE", fontsize=8)

for ax in axes[1]:
    ax.set_xlabel("backbone step")

fig.legend(handles=[
    Line2D([], [], color=INK, lw=1.2, label="backbone training loss (left axis)"),
    Line2D([], [], color=INK, lw=0.7, ls=":", marker="o", markerfacecolor="none",
           markeredgewidth=1.3, label="GM-Relative MASE, 2L head (right axis)"),
    Line2D([], [], color=INK, lw=0.7, ls="--", marker="^", markerfacecolor="none",
           markeredgewidth=1.3, label="GM-Relative MASE, 6L head (right axis)"),
], loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.suptitle(
    "Backbone training loss and evaluated GM-Relative MASE snapshots, per arm\n"
    "loss is not comparable across arms (different loss shapes) — read each panel within itself",
    fontsize=9)
fig.tight_layout(rect=(0, 0.045, 1, 1))
out = HERE / "loss_vs_gm_snapshots.png"
fig.savefig(out)
print(f"wrote {out}")
