"""Generate plots for the SIGReg λ × EMA-τ cross report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EXP = HERE.parent.parent.parent / "experiments" / "2026-06-28_sigreg_lambda_tau_cross"
GM = pd.read_csv(EXP / "results" / "gm_table.csv")


def cell(arm: str, head: str, ckpt: str, col: str = "gm") -> float:
    row = GM[(GM["arm"] == arm) & (GM["head"] == head) & (GM["ckpt"] == ckpt)]
    return float(row[col].iloc[0])


CROSS_A = "cross_A"
CROSS_B = "cross_B"
A363_E100 = "anchor_363_emb100_enc10"
A363_E10000 = "anchor_363_emb10000_enc10"
A357 = "anchor_357_tau090"

ARMS = [
    (CROSS_A, r"cross_A  $\lambda_e{=}10,\ \tau{=}0.90$", "#1f77b4"),
    (CROSS_B, r"cross_B  $\lambda_e{=}1000,\ \tau{=}0.90$", "#ff7f0e"),
    (A363_E100, r"anchor_363  $\lambda_e{=}10,\ \tau{=}0.99$", "#9ecae1"),
    (A363_E10000, r"anchor_363  $\lambda_e{=}1000,\ \tau{=}0.99$", "#c6dbef"),
    (A357, r"anchor_357  $\lambda_e{=}\lambda_h{=}0.1,\ \tau{=}0.90$", "#bdbdbd"),
]

GROUPS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]


def headline():
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(GROUPS))
    width = 0.16
    for i, (arm, label, colour) in enumerate(ARMS):
        vals = [cell(arm, h, c) for h, c in GROUPS]
        offset = (i - (len(ARMS) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=colour)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7.5, rotation=90)
    sn_line = ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} / {c}" for h, c in GROUPS])
    ax.set_ylabel("GM-Relative MASE  (lower is better)")
    ax.set_ylim(1.10, 1.21)
    ax.set_title("Two crosses vs the three single-axis anchors  (N=1 seed; no error bars)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(sn_line)
    labels.append("seasonal-naive  (GM-Rel MASE = 1.0)")
    ax.legend(handles, labels, fontsize=8, loc="upper right", ncol=1)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / "headline_relmase.png", dpi=140)
    plt.close(fig)


def four_metric():
    metrics = [
        ("gm", "GM-Relative MASE"),
        ("gm_mase", "GM-MASE  (raw)"),
        ("gm_mape_sn", "GM-MAPE / SN_MAPE"),
        ("gm_crps_sn", "GM-CRPS / SN_CRPS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    x = np.arange(len(GROUPS))
    width = 0.16
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for i, (arm, label, colour) in enumerate(ARMS):
            vals = [cell(arm, h, c, col=col) for h, c in GROUPS]
            offset = (i - (len(ARMS) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=label, color=colour)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h} / {c}" for h, c in GROUPS], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[0, 0].set_ylim(1.10, 1.21)
    axes[0, 1].set_ylim(1.55, 1.68)
    axes[1, 0].set_ylim(1.04, 1.23)
    axes[1, 1].set_ylim(0.83, 0.92)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Four GM aggregates — two crosses vs three anchors", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(HERE / "four_aggregates.png", dpi=140)
    plt.close(fig)


def training_curves():
    cols = ["loss", "loss_tau_ref", "u_temporal", "u_batch", "sigreg_e", "sigreg_h"]
    titles = {
        "loss": "loss  (objective at run τ)",
        "loss_tau_ref": "loss_tau_ref  (fixed-τ=0.07 diagnostic)",
        "u_temporal": "U_temporal  (temporal-axis usage)",
        "u_batch": "U_batch  (batch-axis usage)",
        "sigreg_e": "sigreg_e  (per-step embedding regulariser)",
        "sigreg_h": "sigreg_h  (per-step encoding regulariser)",
    }
    files = {
        "cross_A": EXP / "runs" / "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lA_emb100_enc10_tau090_losses.csv",
        "cross_B": EXP / "runs" / "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_lB_emb10000_enc10_tau090_losses.csv",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8))
    colours = {"cross_A": "#1f77b4", "cross_B": "#ff7f0e"}
    for arm, path in files.items():
        df = pd.read_csv(path)
        for ax, col in zip(axes.ravel(), cols):
            y = df[col].rolling(100, min_periods=1).mean()
            ax.plot(df["step"], y, color=colours[arm], label=arm, linewidth=1.1)
    for ax, col in zip(axes.ravel(), cols):
        ax.set_title(titles[col], fontsize=10)
        ax.set_xlabel("step", fontsize=9)
        ax.set_xscale("log")
        ax.grid(True, linestyle=":", alpha=0.4)
        if col in {"sigreg_e", "sigreg_h"}:
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Backbone training curves  (rolling 100-step mean; log-x)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(HERE / "training_curves.png", dpi=140)
    plt.close(fig)


def cross_vs_best_anchor():
    """Per group, plot the cross value as a coloured bar and overlay the best anchor as a tick."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(GROUPS))
    width = 0.32
    cross_a_vals = [cell(CROSS_A, h, c) for h, c in GROUPS]
    cross_b_vals = [cell(CROSS_B, h, c) for h, c in GROUPS]
    best_anchor_vals = []
    best_anchor_names = []
    for h, c in GROUPS:
        anchors = [
            (A363_E100, cell(A363_E100, h, c)),
            (A363_E10000, cell(A363_E10000, h, c)),
            (A357, cell(A357, h, c)),
        ]
        best = min(anchors, key=lambda kv: kv[1])
        best_anchor_vals.append(best[1])
        best_anchor_names.append(best[0])

    b_a = ax.bar(x - width / 2, cross_a_vals, width, color="#1f77b4", label="cross_A")
    b_b = ax.bar(x + width / 2, cross_b_vals, width, color="#ff7f0e", label="cross_B")
    for xi, va, vb in zip(x, cross_a_vals, cross_b_vals):
        ax.text(xi - width / 2, va + 0.003, f"{va:.4f}", ha="center", va="bottom",
                fontsize=8, rotation=90)
        ax.text(xi + width / 2, vb + 0.003, f"{vb:.4f}", ha="center", va="bottom",
                fontsize=8, rotation=90)
    for xi, v, name in zip(x, best_anchor_vals, best_anchor_names):
        ax.hlines(v, xi - width, xi + width, color="k", linewidth=1.6,
                  label="best anchor in group" if xi == 0 else None)
        ax.text(xi, v - 0.004, f"{v:.4f}\n({name.replace('anchor_', '#')})",
                ha="center", va="top", fontsize=7.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} / {c}" for h, c in GROUPS])
    ax.set_ylabel("GM-Relative MASE")
    ax.set_ylim(1.10, 1.21)
    ax.set_title("Each cross vs the best single-axis anchor in the same group")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(HERE / "cross_vs_best_anchor.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    headline()
    cross_vs_best_anchor()
    four_metric()
    training_curves()
    print("plots written to", HERE)
