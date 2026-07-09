"""Plots for the B=1024 retrain of the τ=0.90 winner (#369).

Reads:
  ../../experiments/2026-07-03_b1024_traj_ckpts/results/gm_table.csv
  ../../experiments/2026-07-03_b1024_traj_ckpts/runs/bb_..._losses.csv (+ _r2 + _r3)

Emits into this directory:
  gm_vs_step.png       — GM-Rel MASE per head vs backbone step, parent lines
  backbone_loss.png    — contrastive loss (10-step ma) vs backbone step
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EXP = HERE.parent.parent.parent / "experiments" / "2026-07-03_b1024_traj_ckpts"
GM = pd.read_csv(EXP / "results" / "gm_table.csv")
RUNS = EXP / "runs"
TAG = "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_l_emb10_enc10_tau090_b1024"

PARENT_BUDGET_STEP = 12500
PARENT_BEST_LOSS_STEP = 500
COLOR_2L = "#1f77b4"
COLOR_6L = "#d62728"


def _step_of(ckpt: str) -> int:
    return int(ckpt.replace("step", ""))


def _trajectory(head: str) -> tuple[list[int], list[float]]:
    rows = GM[(GM["source"] == "retrain") & (GM["head"] == head)].copy()
    rows["step"] = rows["ckpt"].map(_step_of)
    rows = rows.sort_values("step")
    return rows["step"].tolist(), rows["gm"].tolist()


def _parent(head: str, ckpt: str) -> float:
    row = GM[(GM["source"] == "parent_366") & (GM["head"] == head) & (GM["ckpt"] == ckpt)]
    return float(row["gm"].iloc[0])


def gm_vs_step() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for head, colour in [("2L", COLOR_2L), ("6L", COLOR_6L)]:
        steps, gms = _trajectory(head)
        ax.plot(steps, gms, "o-", color=colour, label=f"B=1024 retrain, {head} head", linewidth=1.6, markersize=5)
        for s, g in zip(steps, gms):
            ax.text(s, g + 0.0022, f"{g:.4f}", ha="center", va="bottom", fontsize=7, color=colour, rotation=90)
        parent_last = _parent(head, "last")
        ax.axhline(parent_last, color=colour, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(1500, parent_last - 0.0018, f"parent B=512 last  {parent_last:.4f}",
                ha="left", va="top", fontsize=8, color=colour)

    ax.axvline(PARENT_BUDGET_STEP, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(PARENT_BUDGET_STEP, 1.198, "  spec budget (12,500 steps)", fontsize=8, va="top")
    ax.axvspan(PARENT_BUDGET_STEP, 38000, color="grey", alpha=0.06)
    ax.text(25000, 1.198, "extension (out of issue scope)",
            ha="center", va="top", fontsize=8, style="italic", color="dimgrey")

    ax.set_xlabel("Backbone step (B=1024)")
    ax.set_ylabel("GM-Relative MASE  (lower is better)")
    ax.set_xlim(0, 39000)
    ax.set_ylim(1.115, 1.20)
    ax.set_xticks([500, 12500, 15000, 20000, 25000, 30000, 35000, 37500])
    ax.set_xticklabels(["500", "12500", "15k", "20k", "25k", "30k", "35k", "37.5k"], fontsize=8)
    ax.set_title("GM-Rel MASE vs backbone step — B=1024 retrain vs B=512 parent (arm C, τ=0.90)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "gm_vs_step.png", dpi=140)
    plt.close(fig)


def _read_losses() -> pd.DataFrame:
    frames = []
    for suffix in ["", "_r2", "_r3"]:
        path = RUNS / f"{TAG}{suffix}_losses.csv"
        if path.exists():
            frames.append(pd.read_csv(path)[["step", "loss"]])
    return pd.concat(frames, ignore_index=True).sort_values("step")


def backbone_loss() -> None:
    df = _read_losses()
    df["loss_ma"] = df["loss"].rolling(200, min_periods=1).mean()
    df = df[df["step"] >= 200]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.plot(df["step"], df["loss_ma"], color="#333", linewidth=1.2, label="contrastive loss (200-step MA)")
    ax.set_xlim(0, df["step"].max())
    ax.set_ylim(3.5, 4.4)
    ax.axvline(PARENT_BEST_LOSS_STEP, color=COLOR_2L, linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(PARENT_BEST_LOSS_STEP + 400, 4.35, "parent best-loss step (500)",
            fontsize=8, va="top", color=COLOR_2L)
    ax.axvline(PARENT_BUDGET_STEP, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(PARENT_BUDGET_STEP + 400, 4.35, "spec budget (12,500)", fontsize=8, va="top")
    ax.axvspan(PARENT_BUDGET_STEP, df["step"].max(), color="grey", alpha=0.06)
    ax.text(25000, 4.35, "extension (out of issue scope)",
            ha="center", va="top", fontsize=8, style="italic", color="dimgrey")
    ax.set_xlabel("Backbone step (B=1024)")
    ax.set_ylabel("training loss")
    ax.set_title("Backbone training loss — base run + continuations to 37,500 (200-step MA)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "backbone_loss.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    gm_vs_step()
    backbone_loss()
    print("plots written to", HERE)
