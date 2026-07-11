"""Generate the three plots for the split L_pred + L_rep report.

Reads only checked-in experiment artefacts:
  * GIFT-Eval summaries   experiments/2026-07-10_split_pred_rep/results/gift_eval_full_*/summary.txt
  * champion reference    experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv
  * gradient shares       experiments/2026-07-10_split_pred_rep/results/gradient_share_measurement.csv
  * training losses       experiments/2026-07-10_split_pred_rep/runs*/bb_*_losses*.csv

Run from the repo root:  python3 reports/2026-07-10_split_pred_rep/plots/_make_plots.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
EXP = ROOT / "experiments" / "2026-07-10_split_pred_rep"
SIGREG = ROOT / "experiments" / "2026-06-28_sigreg_lambda_tau_cross"

# Arm colours (validated categorical trio; hatch/linestyle carry identity too).
C_ARM1, C_ARM3, C_ARM4 = "#2a78d6", "#eb6834", "#008300"
# Tensor colours (validated categorical quad).
C_TENSOR = {
    "log_neg_zy": "#1baf7a",
    "log_neg_cross_batch": "#4a3aa7",
    "log_neg_hh_all": "#eda100",
    "log_neg_xs_allt": "#e87ba4",
}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

GROUPS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]
ARM_DIRS = {
    "arm1": "gift_eval_full_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
    "arm3": "gift_eval_full_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090",
}


def read_aggregate(arm: str, head: str, ckpt: str) -> float:
    suffix = f"_{head}" if ckpt == "best" else f"_last_{head}"
    text = (EXP / "results" / (ARM_DIRS[arm] + suffix) / "summary.txt").read_text()
    m = re.search(r"Aggregate GM-Relative MASE \(97 configs\): ([0-9.]+)", text)
    return float(m.group(1))


def champion_cells() -> dict:
    gm = pd.read_csv(SIGREG / "results" / "gm_table.csv")
    rows = gm[gm["arm"] == "cross_C"]
    return {(r["head"], r["ckpt"]): float(r["gm"]) for _, r in rows.iterrows()}


def headline() -> None:
    champ = champion_cells()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(GROUPS))
    width = 0.20
    C_CHAMP = "#52514e"
    for off, arm, colour, label in [(-1.5 * width, "arm1", C_ARM1, "arm 1 (split)"),
                                    (-0.5 * width, "arm3", C_ARM3, "arm 3 (split + MoCo)")]:
        vals = [read_aggregate(arm, h, c) for h, c in GROUPS]
        bars = ax.bar(x + off, np.array(vals) - 1.0, width, bottom=1.0,
                      color=colour, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                    ha="center", va="bottom", rotation=90, fontsize=8.5, color=INK)
    for gx in x:  # arm 4 slots: results pending, full-height hatched placeholder
        ax.bar(gx + 0.5 * width, 0.20, width, bottom=0.99, facecolor="none",
               edgecolor=C_ARM4, hatch="///", linewidth=0.9, alpha=0.45)
        ax.text(gx + 0.5 * width, 1.085, "pending", ha="center", va="center",
                rotation=90, fontsize=8.5, color=C_ARM4)
    champ_vals = [champ[g] for g in GROUPS]
    bars = ax.bar(x + 1.5 * width, np.array(champ_vals) - 1.0, width, bottom=1.0,
                  color=C_CHAMP, label="arm C ref (champion)")
    for b, v in zip(bars, champ_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", rotation=90, fontsize=8.5, color=INK)
    ax.bar(0, 0, facecolor="none", edgecolor=C_ARM4, hatch="///",
           label="arm 4 (pooled + MoCo) — pending")
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", label="seasonal-naive = 1.0")
    ax.set_xticks(x, [f"{h} / {c}" for h, c in GROUPS])
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0.99, 1.19)
    ax.set_ylabel("GM-Relative MASE (97 configs)\nlower is better")
    ax.set_title("Downstream GM-Relative MASE, 97 GIFT-Eval configs  (N = 1 seed; no error bars)")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
              frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "headline_relmase.png")
    plt.close(fig)


FINAL_CKPT = {"arm1": "_FINAL.pth", "arm3": "_FINAL.pth", "arm4": "_FINAL.pth"}
STACK_ORDER = ["log_neg_zy", "log_neg_cross_batch", "log_neg_hh_all", "log_neg_xs_allt"]
TENSOR_LABEL = {
    "log_neg_zy": "log_neg_zy  (adjacent f↔f)",
    "log_neg_cross_batch": "log_neg_cross_batch  (cross-batch f↔h′)",
    "log_neg_hh_all": "log_neg_hh_all  (within-series h↔h)",
    "log_neg_xs_allt": "log_neg_xs_allt  (cross-series h↔h′)",
}
BAR_SLOTS = [("arm1", "pred", "arm 1\nL_pred", 0.0), ("arm1", "rep", "arm 1\nL_rep", 0.85),
             ("arm3", "pred", "arm 3\nL_pred", 2.05), ("arm3", "rep", "arm 3\nL_rep", 2.90),
             ("arm4", "pooled", "arm 4\npooled", 4.10)]


def gradient_share_stack() -> None:
    df = pd.read_csv(EXP / "results" / "gradient_share_measurement.csv")
    df["share"] = pd.to_numeric(df.share, errors="coerce")  # "n/a" -> NaN
    df = df.dropna(subset=["share"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, batch in zip(axes, ["mixed", "periodic"]):
        for arm, denom, ticklabel, xpos in BAR_SLOTS:
            rows = df[(df.arm_name == arm) & (df.batch_type == batch)
                      & (df.denom == denom)
                      & df.ckpt_path.str.endswith(FINAL_CKPT[arm])]
            bottom = 0.0
            for tensor in STACK_ORDER:
                r = rows[rows.tensor == tensor]
                if r.empty:
                    continue
                share = float(r.share.iloc[0])
                ax.bar(xpos, share, 0.7, bottom=bottom,
                       color=C_TENSOR[tensor], edgecolor="white", linewidth=1.0)
                if share >= 0.04:
                    dark_fill = tensor == "log_neg_cross_batch"
                    ax.text(xpos, bottom + share / 2, f"{share:.3f}",
                            ha="center", va="center", fontsize=8.5,
                            color="white" if dark_fill else INK)
                if arm == "arm4" and tensor == "log_neg_cross_batch":
                    ax.annotate(f"log_neg_cross_batch\nshare = {share:.3f}",
                                xy=(xpos + 0.36, bottom + share / 2),
                                xytext=(xpos + 0.55, 0.17), fontsize=8, color=INK,
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
                bottom += share
        ax.set_xlim(-0.6, 5.9)
        ax.set_xticks([s[3] for s in BAR_SLOTS], [s[2] for s in BAR_SLOTS], fontsize=9)
        ax.set_title(f"{batch} batch", fontsize=10.5)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of the term's denominator")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_TENSOR[t]) for t in STACK_ORDER]
    fig.legend(handles, [TENSOR_LABEL[t] for t in STACK_ORDER],
               loc="upper center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Per-family denominator share at each arm's step-12,500 backbone snapshot  "
                 "(τ = 0.10, fixed batches, B = 64)", y=0.88, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(HERE / "gradient_share_stack.png")
    plt.close(fig)


LOSS_CSVS = {
    "arm 1 (split)": (
        "runs/bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses_full.csv",
        C_ARM1, "-"),
    "arm 3 (split + MoCo)": (
        "runs/bb_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses.csv",
        C_ARM3, "--"),
    "arm 4 (pooled + MoCo)": (
        "runs_arm4/bb_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090_losses.csv",
        C_ARM4, "-."),
}


def loss_curves() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for label, (rel, colour, ls) in LOSS_CSVS.items():
        df = pd.read_csv(EXP / rel, usecols=["step", "loss"]).set_index("step")
        final_level = df.loc[12401:12500, "loss"].mean()
        shifted = df.loss - final_level
        smooth = shifted.rolling(101, center=True, min_periods=1).mean()
        ax.plot(shifted.loc[100:].index, shifted.loc[100:], color=colour,
                lw=0.5, alpha=0.15)
        ax.plot(smooth.loc[100:].index, smooth.loc[100:], color=colour,
                lw=1.8, ls=ls, label=label)
    ax.axhline(0.0, color=MUTED, lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(100, 12500)
    ax.set_xticks([100, 300, 1000, 3000, 10000],
                  ["100", "300", "1,000", "3,000", "10,000"])
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("total training loss − final level (nats)")
    ax.set_title("Total training loss, shifted by each run's final level "
                 "(mean of steps 12,401–12,500)")
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "loss_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    headline()
    gradient_share_stack()
    loss_curves()
    print("wrote", *(p.name for p in sorted(HERE.glob("*.png"))))
