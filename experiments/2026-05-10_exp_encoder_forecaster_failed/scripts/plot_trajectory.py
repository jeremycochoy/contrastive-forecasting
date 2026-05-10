#!/usr/bin/env python3
"""Trajectory comparison: encoder+forecaster (τ=0.10) vs τ=0.10 baseline.

Reads both losses CSVs, smooths with a 1000-step moving average, and
renders a 2x2 panel: loss / gap / per-batch AUC / per-batch top1.

Output: experiments/2026-05-10_exp_encoder_forecaster/plots/trajectory.png

Run from the worktree (so the source CSVs in main checkout are
accessible via REPO_ROOT):
    PYTHONPATH=. python experiments/2026-05-10_exp_encoder_forecaster/scripts/plot_trajectory.py
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
# Read training losses from the MAIN checkout (where checkpoints are
# saved). The worktree's "checkpoints/" is empty by design.
MAIN = Path("/home/jupyter/contrastive-forecasting")
EXP_DIR = REPO / "experiments/2026-05-10_exp_encoder_forecaster"
OUT = EXP_DIR / "plots/trajectory.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# The canonical τ=0.10 baseline (used in tau-sweep RESULTS.md) was preempted
# at step 15001 and resumed in separate chunks; the canonical _FINAL.pth is
# best_loss within the first 15k. The enc+fcst arm trains 50k from scratch,
# so the baseline curve naturally ends earlier.
ARMS = [
    ("τ=0.10 baseline (6L fcst, 15k trained)", "tau_sweep_0_10",            "#888888",
     MAIN / "sync_tau_sweep/checkpoints/tau_sweep_0_10_losses.csv"),
    ("encoder+forecaster (6L+6L, bf16, 50k)",  "enc_fcst_tau_0_10_50k",     "#d62728",
     MAIN / "checkpoints/enc_fcst_tau_0_10_50k_losses.csv"),
]


def smooth(x, w=1000):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_traj(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    cols = {}
    cols["step"] = np.array([int(r["step"]) for r in rows])
    for k in ("loss", "gap", "auc", "top1"):
        cols[k] = np.array([float(r[k]) for r in rows])
    return cols


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    ax_loss, ax_gap = axes[0]
    ax_auc, ax_top1 = axes[1]

    for label, _name, color, path in ARMS:
        d = load_traj(path)
        if d is None:
            print(f"[plot] SKIP {label}: {path} not found")
            continue
        s = d["step"]
        w = min(1000, max(1, len(s) // 50))
        s_sm = s[w - 1:]
        ax_loss.plot(s_sm, smooth(d["loss"], w), color=color, label=label, lw=1.4)
        ax_gap.plot(s_sm, smooth(d["gap"], w), color=color, label=label, lw=1.4)
        ax_auc.plot(s_sm, smooth(d["auc"], w), color=color, label=label, lw=1.4)
        ax_top1.plot(s_sm, smooth(d["top1"], w), color=color, label=label, lw=1.4)
        n = len(s)
        print(f"[plot] {label}: {n} steps, "
              f"final ema_loss≈{np.mean(d['loss'][-200:]):.4f}, "
              f"final auc(per-batch)≈{np.mean(d['auc'][-200:]):.4f}")

    ax_loss.set_ylabel("loss (1k-step MA)")
    ax_gap.set_ylabel("gap = positive − cross-batch (1k-step MA)")
    ax_auc.set_ylabel("per-batch AUC (training, 1k-step MA)")
    ax_top1.set_ylabel("per-batch top-1 (training, 1k-step MA)")
    for ax in axes.ravel():
        ax.grid(alpha=0.3, ls="--")
        ax.set_xlabel("step")
    ax_loss.legend(loc="upper right", fontsize=9)

    fig.suptitle("Encoder+Forecaster τ=0.10 vs τ=0.10 baseline — training trajectory",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {OUT}")


if __name__ == "__main__":
    main()
