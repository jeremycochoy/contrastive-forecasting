#!/usr/bin/env python3
"""Trajectory plot — transformer-encoder run vs τ=0.10 GRU baseline.

6 panels in a 2×3 grid, log/log axes:
  row 0:  loss        · 1-AUC          · 1-Top1
  row 1:  gap_ratio   · U_temporal     · U_batch

AUC and Top-1 are plotted as `1 − x` on a log y-axis so improvements
move the curve down and the near-1 region isn't squashed.

`gap = ff - fp` is intentionally excluded (it's not informative when ff
and fp can drift up/down together with the same gap). `gap_ratio =
(1-ff)/(1-fp)` is plotted instead — added as a CSV column on this run,
so the baseline line is missing from that panel.

Baseline trajectory is concatenated from the three segments that cover
0..50k (original 15k from-scratch + 50k continuation that got preempted
mid-way + r2 re-provision after preempt). Duplicates by step kept as the
later segment's value. Truncated to the current run's max step so both
curves share the same x range.

x-axis starts at step 1000 (early-step transients dominate y-range).
y-axis per panel is hand-picked to bracket the smoothed (window=200)
data observed after step 1000 with ~20% headroom.

Output: experiments/2026-05-10_exp_transformer_encoder/plots/
        transformer_vs_baseline_loglog.png
"""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt

# Plot script lives in the worktree (`cf-transformer-encoder`) but reads
# checkpoints / sync data from the MAIN checkout (per CLAUDE.md rule 4 —
# sync dirs never live under a worktree). Resolve paths explicitly.
WORKTREE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MAIN_CHECKOUT = os.environ.get(
    "CFCAST_REPO_ROOT", "/home/jupyter/contrastive-forecasting")

EXP_DIR = os.path.join(WORKTREE_ROOT, "experiments",
                       "2026-05-10_exp_transformer_encoder")
OUT_PATH = os.path.join(EXP_DIR, "plots", "transformer_vs_baseline_loglog.png")

TRANSFORMER_CSV = os.path.join(
    MAIN_CHECKOUT, "sync_transformer_encoder", "checkpoints",
    "transformer_encoder_tau_0_10_50k_losses.csv")

BASELINE_SEGMENTS = [
    os.path.join(MAIN_CHECKOUT, "sync_tau_sweep", "checkpoints",
                 "tau_sweep_0_10_losses.csv"),                  # 1..15000
    os.path.join(MAIN_CHECKOUT, "sync_tau_sweep_0_10_50k", "checkpoints",
                 "tau_sweep_0_10_50k_losses.csv"),              # 14901..24500
    os.path.join(MAIN_CHECKOUT, "sync_tau_sweep_0_10_50k", "checkpoints",
                 "tau_sweep_0_10_50k_r2_losses.csv"),           # 23901..50000
]

SMOOTH_WINDOW = 200   # rolling-mean window in steps; CSV is logged every step
X_MIN = 5000          # log-x left edge — cuts the early-step rapid descent

# (key, panel title, transform, ylim) — ylim hand-picked to bracket the
# smoothed range observed in [X_MIN, max_step] for both series with
# headroom on a log y-axis. Re-tighten as the run progresses if needed.
PANELS = [
    ("loss",       "loss",                        lambda s: s,        (6.5,  8.0)),
    ("auc",        "1 − AUC  (lower is better)",  lambda s: 1.0 - s,  (0.09, 0.13)),
    ("top1",       "1 − Top1  (lower is better)", lambda s: 1.0 - s,  (0.22, 0.32)),
    ("gap_ratio",  "gap_ratio = (1−ff)/(1−fp)",   lambda s: s,        (0.38, 0.45)),
    ("u_temporal", "U_temporal",                  lambda s: s,        (3e-2, 7e-2)),
    ("u_batch",    "U_batch",                     lambda s: s,        (6e-2, 1.3e-1)),
]


def smooth(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=max(1, w // 4)).mean()


def load_baseline(max_step: int) -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in BASELINE_SEGMENTS if os.path.exists(p)]
    if not parts:
        raise FileNotFoundError("no baseline segments found")
    df = pd.concat(parts, ignore_index=True)
    # Resume overlap: keep the LATER segment's value at duplicated steps.
    df = df.drop_duplicates(subset="step", keep="last").sort_values("step")
    df = df[df["step"] <= max_step].reset_index(drop=True)
    return df


def main() -> None:
    if not os.path.exists(TRANSFORMER_CSV):
        raise FileNotFoundError(TRANSFORMER_CSV)
    te = pd.read_csv(TRANSFORMER_CSV).sort_values("step").reset_index(drop=True)
    if te.empty:
        raise RuntimeError("transformer CSV is empty")

    max_step = int(te["step"].max())
    baseline = load_baseline(max_step)
    print(f"transformer rows: {len(te)} (steps 1..{max_step})")
    print(f"baseline rows  : {len(baseline)} (steps "
          f"{int(baseline.step.min())}..{int(baseline.step.max())}, "
          f"truncated to max_step={max_step})")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, (key, title, transform, ylim) in zip(axes, PANELS):
        if key in baseline.columns:
            ax.plot(baseline["step"],
                    smooth(transform(baseline[key]), SMOOTH_WINDOW),
                    label="τ=0.10 GRU baseline", color="#1f77b4",
                    linewidth=1.5, alpha=0.85)
        if key in te.columns:
            ax.plot(te["step"],
                    smooth(transform(te[key]), SMOOTH_WINDOW),
                    label="τ=0.10 transformer encoder", color="#ff7f0e",
                    linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=X_MIN, right=max_step)
        ax.set_ylim(*ylim)
        ax.set_xlabel("step (log)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle=":", alpha=0.3)
        if key == "loss":
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"transformer-encoder τ=0.10 vs GRU τ=0.10 baseline · "
        f"smoothed (window={SMOOTH_WINDOW} steps) · log/log · "
        f"x≥{X_MIN}",
        fontsize=11)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=140)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
