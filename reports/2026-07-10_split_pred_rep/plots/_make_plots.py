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
C_ARM1, C_ARM3, C_ARM4, C_ARM6, C_BIMOCO = "#2a78d6", "#eb6834", "#008300", "#b8860b", "#00a3a3"
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
ARM_SPECS = {
    "arm1": (EXP / "results", "gift_eval_full_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
    "arm3": (EXP / "results", "gift_eval_full_split_pred_rep_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
    "arm4": (EXP / "results_arm4", "gift_eval_full_allt08_moco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm4_tau090"),
    "arm5": (EXP / "results_arm5", "gift_eval_full_lalign_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm5_tau090"),
    "arm6": (EXP / "results_arm6_v2", "gift_eval_full_lalign_lrepmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6v2_tau090"),
    "bimoco": (EXP / "results_bimoco_v2", "gift_eval_full_split_pred_rep_bimoco_v2_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
    # Wrong-implementation snapshots kept for the headline plot only (semi-transparent).
    "arm6_wrong": (EXP / "results_arm6", "gift_eval_full_lalignmoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_arm6_tau090"),
    "bimoco_wrong": (EXP / "results_bimoco", "gift_eval_full_split_pred_rep_bimoco_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090"),
}


def read_aggregate(arm: str, head: str, ckpt: str):
    root, base = ARM_SPECS[arm]
    suffix = f"_{head}" if ckpt == "best" else f"_last_{head}"
    p = root / (base + suffix) / "summary.txt"
    if not p.exists(): return None
    m = re.search(r"Aggregate GM-Relative MASE \(97 configs\): ([0-9.]+)", p.read_text())
    return float(m.group(1)) if m else None


def champion_cells() -> dict:
    gm = pd.read_csv(SIGREG / "results" / "gm_table.csv")
    rows = gm[gm["arm"] == "cross_C"]
    return {(r["head"], r["ckpt"]): float(r["gm"]) for _, r in rows.iterrows()}


def cell_relatives(arm: str, head: str, ckpt: str):
    """Per-config 'Relative' column from a cell's summary.txt (table rows only)."""
    root, base = ARM_SPECS[arm]
    suffix = f"_{head}" if ckpt == "best" else f"_last_{head}"
    p = root / (base + suffix) / "summary.txt"
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    dashes = [i for i, l in enumerate(lines) if set(l.strip()) == {"-"}]
    if len(dashes) < 2:
        return []
    vals = []
    for l in lines[dashes[0] + 1:dashes[1]]:
        try:
            vals.append(float(l.split()[-1]))
        except (ValueError, IndexError):
            pass
    return vals


def bootstrap_gm_ci(vals, n_resamples=20000, seed=42):
    """95% band on the geometric mean via config bootstrap (single-seed, task-level)."""
    if not vals:
        return None
    rng = np.random.default_rng(seed)
    logs = np.log(np.asarray(vals))
    idx = rng.integers(0, len(logs), size=(n_resamples, len(logs)))
    gms = np.exp(logs[idx].mean(axis=1))
    return float(np.percentile(gms, 2.5)), float(np.percentile(gms, 97.5))


C_ARM5 = "#8b1e8b"


def headline() -> None:
    champ = champion_cells()
    fig, ax = plt.subplots(figsize=(16.5, 5.6))
    x = np.arange(len(GROUPS))
    width = 0.12
    C_CHAMP = "#52514e"
    arm_slots = [(-3.0 * width, "arm1", C_ARM1, "arm 1 (split)"),
                 (-2.0 * width, "arm3", C_ARM3, "arm 3 (split + MoCo)"),
                 (-1.0 * width, "arm4", C_ARM4, "arm 4 (pooled + MoCo)"),
                 ( 0.0 * width, "arm5", C_ARM5, "arm 5 (L_align + L_rep)"),
                 ( 1.0 * width, "arm6", C_ARM6, "arm 6 (L_align + L_rep_moco)"),
                 ( 2.0 * width, "bimoco", C_BIMOCO, "arm bimoco (L_pred_moco + L_rep_moco)"),
                 ( 3.0 * width, None,  C_CHAMP, "arm C ref (champion)")]
    for off, arm, colour, label in arm_slots:
        if arm is None:
            vals = [champ[g] for g in GROUPS]
        else:
            vals = [read_aggregate(arm, h, c) for h, c in GROUPS]
        for xi, (v, (h, c)) in zip(x, zip(vals, GROUPS)):
            ax.bar(xi + off, v - 1.0, width, bottom=1.0, color=colour)
            ax.text(xi + off, v + 0.003, f"{v:.4f}",
                    ha="center", va="bottom", rotation=90, fontsize=7.5, color=INK)
            if arm is not None:
                ci = bootstrap_gm_ci(cell_relatives(arm, h, c))
                if ci is not None:
                    ax.errorbar(xi + off, v, yerr=[[v - ci[0]], [ci[1] - v]],
                                fmt="none", ecolor=INK, elinewidth=1.0, capsize=2)
        ax.bar(0, 0, color=colour, label=label)  # legend proxy
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--", label="seasonal-naive = 1.0")
    ax.set_xticks(x, [f"{h} / {c}" for h, c in GROUPS])
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0.99, 1.38)
    ax.set_ylabel("GM-Relative MASE (97 configs)\nlower is better")
    ax.set_title("Downstream GM-Relative MASE, 97 GIFT-Eval configs  (N = 1 seed)")
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
             ("arm4", "pooled", "arm 4\npooled", 4.20)]


def gradient_share_stack() -> None:
    df = pd.read_csv(EXP / "results" / "gradient_share_measurement.csv")
    df["share"] = pd.to_numeric(df.share, errors="coerce")  # "n/a" -> NaN
    df = df.dropna(subset=["share"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharey=True)
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
                                xytext=(xpos + 1.05, 0.30), fontsize=8, color=INK,
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
                bottom += share
        ax.set_xlim(-0.6, 6.9)
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
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Per-family denominator share at each arm's FINAL.pth backbone snapshot"
                 "\n(arm 1: step 12,500; arm 3: step 11,800; arm 4: step 600  —  τ = 0.10, B = 64)",
                 y=0.86, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.78))
    fig.savefig(HERE / "gradient_share_stack.png")
    plt.close(fig)


FOREST_ROWS = [
    ("2L / last",  "arm 3 vs arm 4", "split ↔ pooled  (MoCo fixed)", "arm3", "arm4"),
    ("6L / last",  "arm 3 vs arm 4", "split ↔ pooled  (MoCo fixed)", "arm3", "arm4"),
    ("2L / last",  "arm 1 vs arm 3", "MoCo off ↔ on  (split fixed)", "arm1", "arm3"),
    ("6L / last",  "arm 1 vs arm 3", "MoCo off ↔ on  (split fixed)", "arm1", "arm3"),
    ("2L / last",  "arm 1 vs arm 4", "joint (split+no-MoCo ↔ pooled+MoCo)", "arm1", "arm4"),
    ("6L / last",  "arm 1 vs arm 4", "joint (split+no-MoCo ↔ pooled+MoCo)", "arm1", "arm4"),
    ("2L / best*", "arm 3 vs arm 4", "split — 11,200-step gap", "arm3", "arm4"),
    ("6L / best*", "arm 3 vs arm 4", "split — 11,200-step gap", "arm3", "arm4"),
    ("2L / best*", "arm 1 vs arm 3", "MoCo — ckpt-selection confound", "arm1", "arm3"),
    ("6L / best*", "arm 1 vs arm 3", "MoCo — ckpt-selection confound", "arm1", "arm3"),
    ("2L / best*", "arm 1 vs arm 4", "joint — ckpt-selection confound", "arm1", "arm4"),
    ("6L / best*", "arm 1 vs arm 4", "joint — ckpt-selection confound", "arm1", "arm4"),
    ("2L / last",  "arm 5 vs arm 1", "L_align+L_rep ↔ split", "arm5", "arm1"),
    ("6L / last",  "arm 5 vs arm 1", "L_align+L_rep ↔ split", "arm5", "arm1"),
    ("2L / last",  "arm 1 vs arm 6", "split ↔ L_align+L_rep_moco", "arm1", "arm6"),
    ("6L / last",  "arm 1 vs arm 6", "split ↔ L_align+L_rep_moco", "arm1", "arm6"),
    ("2L / last",  "arm 3 vs arm 6", "split+MoCo ↔ L_align+L_rep_moco", "arm3", "arm6"),
    ("6L / last",  "arm 3 vs arm 6", "split+MoCo ↔ L_align+L_rep_moco", "arm3", "arm6"),
    ("2L / last",  "arm 4 vs arm 6", "pooled+MoCo ↔ L_align+L_rep_moco", "arm4", "arm6"),
    ("6L / last",  "arm 4 vs arm 6", "pooled+MoCo ↔ L_align+L_rep_moco", "arm4", "arm6"),
    ("2L / last",  "arm 5 vs arm 6", "L_rep ↔ L_rep_moco (L_align fixed)", "arm5", "arm6"),
    ("6L / last",  "arm 5 vs arm 6", "L_rep ↔ L_rep_moco (L_align fixed)", "arm5", "arm6"),
    ("2L / last",  "arm 1 vs bimoco",  "split ↔ split + moco (both L_pred and L_rep)", "arm1", "bimoco"),
    ("6L / last",  "arm 1 vs bimoco",  "split ↔ split + moco (both L_pred and L_rep)", "arm1", "bimoco"),
    ("2L / last",  "arm 3 vs bimoco",  "split+MoCo(pred) ↔ split+MoCo(pred+rep)", "arm3", "bimoco"),
    ("6L / last",  "arm 3 vs bimoco",  "split+MoCo(pred) ↔ split+MoCo(pred+rep)", "arm3", "bimoco"),
    ("2L / last",  "arm 4 vs bimoco",  "pooled+MoCo ↔ split + MoCo both terms", "arm4", "bimoco"),
    ("6L / last",  "arm 4 vs bimoco",  "pooled+MoCo ↔ split + MoCo both terms", "arm4", "bimoco"),
    ("2L / best*", "arm 1 vs bimoco",  "split ↔ split + moco (both terms)", "arm1", "bimoco"),
    ("6L / best*", "arm 1 vs bimoco",  "split ↔ split + moco (both terms)", "arm1", "bimoco"),
    ("2L / best*", "arm 3 vs bimoco",  "split+MoCo(pred) ↔ split+MoCo(pred+rep)", "arm3", "bimoco"),
    ("6L / best*", "arm 3 vs bimoco",  "split+MoCo(pred) ↔ split+MoCo(pred+rep)", "arm3", "bimoco"),
    ("2L / best*", "arm 4 vs bimoco",  "pooled+MoCo ↔ split + MoCo both terms", "arm4", "bimoco"),
    ("6L / best*", "arm 4 vs bimoco",  "pooled+MoCo ↔ split + MoCo both terms", "arm4", "bimoco"),
]


def _lookup(df, cell, arm_a, arm_b):
    head, ck = cell.split(" / ")
    ck = ck.rstrip("*")
    r = df[(df["head"] == head) & (df["ckpt"] == ck) &
           (df["arm_a"] == arm_a) & (df["arm_b"] == arm_b)]
    if r.empty:
        r = df[(df["head"] == head) & (df["ckpt"] == ck) &
               (df["arm_a"] == arm_b) & (df["arm_b"] == arm_a)]
        row = r.iloc[0]
        return 1.0/row["ratio_a_over_b"], 1.0/row["ci_hi"], 1.0/row["ci_lo"]
    row = r.iloc[0]
    return row["ratio_a_over_b"], row["ci_lo"], row["ci_hi"]


def ci_forest() -> None:
    task = pd.read_csv(EXP / "results" / "pairwise_bootstrap_ci.csv")
    clu  = pd.read_csv(EXP / "results" / "pairwise_bootstrap_ci_clustered.csv")
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    n = len(FOREST_ROWS)
    for i, (cell, contrast, ax_label, a, b) in enumerate(FOREST_ROWS):
        yt = n - 1 - i
        rt, lot, hit = _lookup(task, cell, a, b)
        rc, loc_, hic = _lookup(clu,  cell, a, b)
        col = ("#00a3a3" if a == "bimoco" or b == "bimoco" else
               "#b8860b" if a == "arm6" or b == "arm6" else
               "#8b1e8b" if a == "arm5" or b == "arm5" else INK)
        ax.plot([lot, hit], [yt + 0.12, yt + 0.12], color=col, lw=1.6)
        ax.plot(rt, yt + 0.12, "o", color=col, markersize=6)
        ax.plot([loc_, hic], [yt - 0.12, yt - 0.12], color=col, lw=1.0, alpha=0.55)
        ax.plot(rc, yt - 0.12, "s", color=col, markersize=5, alpha=0.55)
        ax.text(-0.02, yt, f"{cell:<10s}  {contrast:<16s}  {ax_label}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9, family="monospace")
    ax.axvline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.set_yticks([])
    ax.set_xlim(0.82, 1.20)
    ax.set_ylim(-1.2, n + 0.6)
    ax.set_xlabel("ratio A / B  (ratio < 1 → A better; ratio > 1 → A worse)")
    ax.text(0.5, n + 0.2,
            "Paired-bootstrap 95 % CIs on GM-Relative MASE ratios",
            transform=ax.get_yaxis_transform(), ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    marker_handles = [
        plt.Line2D([], [], color=INK, marker="o", markersize=6, lw=1.6,
                   label="task-level bootstrap (97 configs)"),
        plt.Line2D([], [], color=INK, marker="s", markersize=5, lw=1.0,
                   alpha=0.55, label="28-dataset-clustered bootstrap"),
    ]
    ax.legend(handles=marker_handles, loc="upper right", fontsize=8.5,
              frameon=False)
    ax.text(1.005, -0.06, "cells marked * are checkpoint-selection or step-confounded — see report.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(HERE / "ci_forest.png")
    plt.close(fig)


if __name__ == "__main__":
    headline()
    gradient_share_stack()
    ci_forest()
    print("wrote", *(p.name for p in sorted(HERE.glob("*.png"))))
