"""Generate plots for the SIGReg (λ_e, λ_h) × EMA-τ report."""
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


def cell_opt(arm: str, head: str, ckpt: str, col: str = "gm") -> float | None:
    row = GM[(GM["arm"] == arm) & (GM["head"] == head) & (GM["ckpt"] == ckpt)]
    if row.empty:
        return None
    return float(row[col].iloc[0])


CROSS_A = "cross_A"
CROSS_B = "cross_B"
CROSS_C = "cross_C"
CROSS_D = "cross_D"
CROSS_E = "cross_E"
CROSS_F = "cross_F"
CROSS_G = "cross_G"
CROSS_H = "cross_H"
CROSS_I = "cross_I"
A363_E100 = "anchor_363_emb100_enc10"
A363_E10000 = "anchor_363_emb10000_enc10"
A357 = "anchor_357_tau090"

ARMS = [
    (CROSS_A,    r"$\lambda_e{=}10,\ \lambda_h{=}1,\ \tau{=}0.90$",     "#1f77b4"),
    (CROSS_B,    r"$\lambda_e{=}1000,\ \lambda_h{=}1,\ \tau{=}0.90$",   "#ff7f0e"),
    (CROSS_C,    r"$\lambda_e{=}1,\ \lambda_h{=}1,\ \tau{=}0.90$",      "#2ca02c"),
    (CROSS_H,    r"$\lambda_e{=}1,\ \lambda_h{=}10,\ \tau{=}0.90$",     "#e377c2"),
    (A363_E100,  r"$\lambda_e{=}10,\ \lambda_h{=}1,\ \tau{=}0.99$",     "#9ecae1"),
    (A363_E10000,r"$\lambda_e{=}1000,\ \lambda_h{=}1,\ \tau{=}0.99$",   "#c6dbef"),
    (A357,       r"$\lambda_e{=}\lambda_h{=}0.1,\ \tau{=}0.90$",        "#bdbdbd"),
]

GROUPS = [("2L", "best"), ("2L", "last"), ("6L", "best"), ("6L", "last")]


def headline():
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(GROUPS))
    width = 0.12
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
    ax.set_title("GM-Relative MASE, seven of the twelve arms  (N=1 seed; no error bars)")
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
    width = 0.12
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
    fig.suptitle("Four GM aggregates, same seven arms  (N=1 seed; no error bars)", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(HERE / "four_aggregates.png", dpi=140)
    plt.close(fig)


def lambda_grid_tau090():
    """(λ_e, λ_h) heatmap of GM-Relative MASE over the τ=0.90 arms,
    one panel per (head depth, checkpoint). Cells not run are hatched.
    """
    lam_e = [0.1, 1.0, 10.0, 100.0, 1000.0]
    lam_h = [0.1, 1.0, 10.0, 100.0, 1000.0]
    cells_tau090 = {
        (0.1,   0.1):    A357,
        (1.0,   1.0):    CROSS_C,
        (1.0,   10.0):   CROSS_H,
        (10.0,  1.0):    CROSS_A,
        (10.0,  10.0):   CROSS_D,
        (100.0, 1.0):    CROSS_I,
        (100.0, 10.0):   CROSS_G,
        (100.0, 100.0):  CROSS_E,
        (1000.0, 1.0):   CROSS_B,
        (1000.0, 1000.0): CROSS_F,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fig.subplots_adjust(hspace=0.35)
    vmin, vmax = 1.12, 1.19
    cmap = plt.get_cmap("RdBu_r")
    for ax, (head, ckpt) in zip(axes.ravel(), GROUPS):
        grid = np.full((len(lam_h), len(lam_e)), np.nan)
        for (le, lh), arm in cells_tau090.items():
            v = cell_opt(arm, head, ckpt)
            if v is None:
                continue
            i = lam_h.index(lh); j = lam_e.index(le)
            grid[i, j] = v
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="lower", aspect="auto")
        for i in range(len(lam_h)):
            for j in range(len(lam_e)):
                v = grid[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, hatch="///",
                                               edgecolor="#888", linewidth=0))
                else:
                    ax.text(j, i, f"{v:.4f}",
                            ha="center", va="center", fontsize=9.5,
                            fontweight="bold", color="black")
        ax.set_xticks(range(len(lam_e)))
        ax.set_xticklabels([f"{v:g}" for v in lam_e])
        ax.set_yticks(range(len(lam_h)))
        ax.set_yticklabels([f"{v:g}" for v in lam_h])
        ax.set_xlabel(r"$\lambda_e$  (log)")
        ax.set_ylabel(r"$\lambda_h$  (log)")
        ax.set_title(f"{head} q-head / {ckpt}-ckpt")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.85, pad=0.02, label="GM-Relative MASE")
    fig.suptitle(
        r"GM-Relative MASE per $(\lambda_e, \lambda_h)$ at $\tau{=}0.90$"
        "  (hatched = not run; blue = better, red = worse)",
        y=0.99,
    )
    fig.savefig(HERE / "lambda_grid_tau090.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def lambda_grid_last_minus_best_tau090():
    """(last − best) GM-Relative MASE per (λ_e, λ_h) cell at τ=0.90,
    one panel per head depth. Negative = last checkpoint better than best.
    """
    lam_e = [0.1, 1.0, 10.0, 100.0, 1000.0]
    lam_h = [0.1, 1.0, 10.0, 100.0, 1000.0]
    cells_tau090 = {
        (0.1,   0.1):    A357,
        (1.0,   1.0):    CROSS_C,
        (1.0,   10.0):   CROSS_H,
        (10.0,  1.0):    CROSS_A,
        (10.0,  10.0):   CROSS_D,
        (100.0, 1.0):    CROSS_I,
        (100.0, 10.0):   CROSS_G,
        (100.0, 100.0):  CROSS_E,
        (1000.0, 1.0):   CROSS_B,
        (1000.0, 1000.0): CROSS_F,
    }
    heads = ["2L", "6L"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    vmax = 0.045
    cmap = plt.get_cmap("RdBu_r")
    for ax, head in zip(axes.ravel(), heads):
        grid = np.full((len(lam_h), len(lam_e)), np.nan)
        for (le, lh), arm in cells_tau090.items():
            v_best = cell_opt(arm, head, "best")
            v_last = cell_opt(arm, head, "last")
            if v_best is None or v_last is None:
                continue
            i = lam_h.index(lh); j = lam_e.index(le)
            grid[i, j] = v_last - v_best
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax,
                       origin="lower", aspect="auto")
        for i in range(len(lam_h)):
            for j in range(len(lam_e)):
                v = grid[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, hatch="///",
                                               edgecolor="#888", linewidth=0))
                else:
                    ax.text(j, i, f"{v:+.4f}",
                            ha="center", va="center", fontsize=9.5,
                            fontweight="bold", color="black")
        ax.set_xticks(range(len(lam_e)))
        ax.set_xticklabels([f"{v:g}" for v in lam_e])
        ax.set_yticks(range(len(lam_h)))
        ax.set_yticklabels([f"{v:g}" for v in lam_h])
        ax.set_xlabel(r"$\lambda_e$  (log)")
        ax.set_ylabel(r"$\lambda_h$  (log)")
        ax.set_title(f"{head} q-head")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02,
                 label="GM-Rel MASE  (last − best)")
    fig.suptitle(
        r"GM-Relative MASE, last − best checkpoint, per $(\lambda_e, \lambda_h)$ at $\tau{=}0.90$"
        "  (blue = last is better; red = last is worse)",
        y=1.00,
    )
    fig.savefig(HERE / "lambda_grid_last_minus_best_tau090.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    headline()
    four_metric()
    lambda_grid_tau090()
    lambda_grid_last_minus_best_tau090()
    print("plots written to", HERE)
