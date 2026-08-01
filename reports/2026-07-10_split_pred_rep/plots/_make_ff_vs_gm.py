"""1 - `ff` retrieval error aligned with downstream GM-Relative MASE, per arm.

For each arm, two stacked panels sharing the x-axis (backbone step):
  * top    — 1 - ff during training (100-step rolling mean of the losses
             CSV column `ff` = cos(f_hat, f_true)); lower is better
  * bottom — evaluated GM-Relative MASE snapshots (2L = circles + thin
             dotted line, 6L = triangles + thin dashed line); lower is
             better

Vertical guides at each evaluated backbone step tie top-panel training
signal to bottom-panel downstream aggregate at the same step.

6 arms → 4 rows × 3 cols of panels (each column is one arm; each pair
of rows is [1-ff, GM]).
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
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


def ff_curve(runs_dir: str, segments) -> pd.DataFrame:
    frames = []
    for name, offset in segments:
        p = EXP / runs_dir / name
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=["step", "ff"])
        d = d.dropna(subset=["ff"])
        if not len(d):
            continue
        d["step"] = d["step"] + offset
        frames.append(d)
    out = pd.concat(frames).sort_values("step").drop_duplicates("step", keep="last")
    out["one_minus_ff"] = 1.0 - out["ff"].rolling(100, min_periods=1).mean()
    return out


fig = plt.figure(figsize=(15.5, 9.0))
gs = GridSpec(4, 3, figure=fig,
              height_ratios=[3, 2, 3, 2], hspace=0.05, wspace=0.28)
axes_ff = []
axes_gm = []
for i, (label, colour, rd, base, best_step, runs_dir, segs) in enumerate(ARMS):
    row_pair, col = divmod(i, 3)
    top = fig.add_subplot(gs[row_pair * 2, col])
    bot = fig.add_subplot(gs[row_pair * 2 + 1, col], sharex=top)
    axes_ff.append(top)
    axes_gm.append(bot)

    curve = ff_curve(runs_dir, segs)
    top.plot(curve["step"], curve["one_minus_ff"], color=colour, lw=1.3)
    top.set_title(label, fontsize=9, color=colour)
    top.set_ylabel("1 − ff\n(100-step mean)", fontsize=8)
    top.grid(True, color=GRID, alpha=0.6)
    top.set_xlim(0, 51000)
    top.tick_params(axis="x", labelbottom=False)

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
        bot.plot(xs, ys, color=INK, lw=1.3, ls=LINESTYLE[HL], marker=MARKER[HL],
                 markersize=6.5, markerfacecolor="none", markeredgewidth=1.3)
        for x in xs:
            top.axvline(x, color=MUTED, lw=0.5, ls=":", alpha=0.6)
            bot.axvline(x, color=MUTED, lw=0.5, ls=":", alpha=0.6)
    bot.set_ylim(1.09, 1.35)
    bot.set_ylabel("GM-Relative MASE", fontsize=8)
    bot.grid(True, color=GRID, alpha=0.6)
    if row_pair == 1:
        bot.set_xlabel("backbone step", fontsize=9)

fig.legend(handles=[
    Line2D([], [], color=INK, lw=1.3, label="1 − ff during training (top of each pair)"),
    Line2D([], [], color=INK, lw=0.7, ls=":", marker="o", markerfacecolor="none",
           markeredgewidth=1.3, label="GM-Relative MASE, 2L head"),
    Line2D([], [], color=INK, lw=0.7, ls="--", marker="^", markerfacecolor="none",
           markeredgewidth=1.3, label="GM-Relative MASE, 6L head"),
], loc="lower center", ncol=3, fontsize=8.5, frameon=False)
fig.suptitle(
    "Training-time retrieval error (1 − ff) aligned with evaluated GM-Relative MASE snapshots, per arm\n"
    "top of each pair = training signal (lower better); bottom = downstream aggregate at that backbone step (lower better)",
    fontsize=9)
fig.tight_layout(rect=(0, 0.045, 1, 0.96))
out = HERE / "ff_vs_gm_snapshots.png"
fig.savefig(out)
print(f"wrote {out}")
