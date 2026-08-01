"""Plots for the B=1024 retrain of the τ=0.90 winner (#369).

Reads:
  ../../experiments/2026-07-03_b1024_traj_ckpts/results/gm_table.csv
  ../../experiments/2026-07-03_b1024_traj_ckpts/results/b512_seed2/  (gm.csv, steps_loss.csv)
  ../../experiments/2026-07-03_b1024_traj_ckpts/runs/bb_..._losses.csv (+ _r2.._r4)

Emits into this directory:
  gm_vs_step.png       — GM-Rel MASE per head vs backbone step, B=1024 vs B=512
  backbone_loss.png    — contrastive loss (200-step MA), B=1024 vs B=512
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
EXP = HERE.parent.parent.parent / "experiments" / "2026-07-03_b1024_traj_ckpts"
GM = pd.read_csv(EXP / "results" / "gm_table.csv")
SEED2 = EXP / "results" / "b512_seed2"
RUNS = EXP / "runs"
TAG = "bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b1024_cpc_l_emb10_enc10_tau090_b1024"

PARENT_BUDGET_STEP = 12500
PARENT_BEST_LOSS_STEP_RAW = 533  # parent's own best-loss step (unsnapped)
COLOR_2L = "#1f77b4"
COLOR_6L = "#d62728"
XMAX = 51500


def _b1024(head: str) -> tuple[list[int], list[float]]:
    rows = GM[(GM["source"] == "retrain") & (GM["head"] == head)].copy()
    rows["step"] = rows["ckpt"].str.replace("step", "").astype(int)
    rows = rows.sort_values("step")
    return rows["step"].tolist(), rows["gm"].tolist()


def _b512(head: str) -> tuple[list[int], list[float]]:
    """One curve: the parent's best point (seed 1), then every seed-2 re-run point."""
    best = GM[(GM["source"] == "parent_366") & (GM["head"] == head) & (GM["ckpt"] == "best")]
    s2 = pd.read_csv(SEED2 / "gm.csv")
    s2 = s2[s2["head"] == head].sort_values("step")
    return ([PARENT_BEST_LOSS_STEP_RAW] + s2["step"].tolist(),
            [float(best["gm"].iloc[0])] + s2["gm"].tolist())


def gm_vs_step() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for head, colour in [("2L", COLOR_2L), ("6L", COLOR_6L)]:
        steps, gms = _b1024(head)
        ax.plot(steps, gms, "o-", color=colour, linewidth=1.7, markersize=5,
                label=f"B=1024, {head} head")
        for s, g in zip(steps, gms):
            ax.text(s, g + 0.0022, f"{g:.4f}", ha="center", va="bottom",
                    fontsize=7, color=colour, rotation=90)
        steps, gms = _b512(head)
        ax.plot(steps, gms, "s--", color=colour, linewidth=1.2, markersize=4,
                alpha=0.65, label=f"B=512, {head} head")
    ax.text(0.99, 0.02,
            "B=512 curve: seed-1 best point (step 533) + seed-2 re-run points",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="dimgrey")

    ax.axvline(PARENT_BUDGET_STEP, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(PARENT_BUDGET_STEP, 1.198, "  spec budget (12,500 steps)", fontsize=8, va="top")

    ax.set_xlabel("Backbone step")
    ax.set_ylabel("GM-Relative MASE  (lower is better)")
    ax.set_xlim(0, XMAX)
    # Ceiling must clear the highest data point (B=512 2L 1.1943 at 45k).
    ax.set_ylim(1.112, 1.205)
    ax.set_xticks([500, 12500, 20000, 25000, 30000, 35000, 40000, 45000, 50000])
    ax.set_xticklabels(["500", "12500", "20k", "25k", "30k", "35k", "40k", "45k", "50k"], fontsize=8)
    ax.set_title("GM-Rel MASE vs backbone step — arm C (τ=0.90), B=1024 vs B=512")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # Legend outside the axes so it can never occlude data.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.045), ncol=4,
              fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "gm_vs_step.png", dpi=140)
    plt.close(fig)


def _b1024_losses() -> pd.DataFrame:
    frames = []
    for suffix in ["", "_r2", "_r3", "_r4"]:
        path = RUNS / f"{TAG}{suffix}_losses.csv"
        if path.exists():
            frames.append(pd.read_csv(path)[["step", "loss"]])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("step", keep="last")
    return df.sort_values("step")


def backbone_loss() -> None:
    # Batch-coded colours, deliberately distinct from the GM plot's
    # head-coded blue/red so the two figures cannot be cross-read.
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for label, df, colour in [
        ("B=1024 (seed 1)", _b1024_losses(), "#6a3d9a"),
        ("B=512 (seed 2)", pd.read_csv(SEED2 / "steps_loss.csv"), "#ff8c00"),
    ]:
        df = df.sort_values("step")
        df["ma"] = df["loss"].rolling(200, min_periods=1).mean()
        df = df[df["step"] >= 200]
        ax.plot(df["step"], df["ma"], color=colour, linewidth=1.3, label=label)
        # Mark the global MA minimum of each curve (the caption's anchor).
        i = df["ma"].idxmin()
        ax.plot(df.loc[i, "step"], df.loc[i, "ma"], "o", color=colour,
                markersize=6, fillstyle="none", markeredgewidth=1.4)
        ax.annotate(f"min {df.loc[i, 'ma']:.2f}",
                    (df.loc[i, "step"], df.loc[i, "ma"]),
                    textcoords="offset points", xytext=(8, -11),
                    fontsize=7.5, color=colour)
    # The B=1024 extension minimum, second anchor of the caption.
    b1024 = _b1024_losses().sort_values("step")
    b1024["ma"] = b1024["loss"].rolling(200, min_periods=1).mean()
    ext = b1024[b1024["step"] >= PARENT_BUDGET_STEP]
    j = ext["ma"].idxmin()
    ax.plot(ext.loc[j, "step"], ext.loc[j, "ma"], "o", color="#6a3d9a",
            markersize=6, fillstyle="none", markeredgewidth=1.4)
    ax.annotate(f"extension min {ext.loc[j, 'ma']:.2f}",
                (ext.loc[j, "step"], ext.loc[j, "ma"]),
                textcoords="offset points", xytext=(8, -11),
                fontsize=7.5, color="#6a3d9a")
    ax.axvline(PARENT_BUDGET_STEP, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(PARENT_BUDGET_STEP + 400, 4.45, "spec budget (12,500)", fontsize=8, va="top")
    ax.set_xlim(0, XMAX)
    # Bounds must include every post-warmup feature: the early dips
    # (global MA minima 3.23 / 3.38) and the B=512 rebound peak (4.43 at
    # step ~5,650). Only the step<~300 warmup transient may run off-scale
    # — clipping a real extremum once misled the report's caption.
    ax.set_ylim(3.15, 4.5)
    ax.set_xlabel("Backbone step")
    ax.set_ylabel("training loss")
    ax.set_title("Backbone training loss (200-step MA) — B=1024 vs B=512")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "backbone_loss.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    gm_vs_step()
    backbone_loss()
    print("plots written to", HERE)
