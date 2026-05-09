#!/usr/bin/env python3
"""Exp 2 — encoder comparison plot: gru vs residual_silu @ τ=0.10.

Both backbones trained 15k steps under identical recipe (T_RAW=4096,
d_model=384, num_layers=6, n_heads=6, freq+seas emb dim 3, rev_norm
ewma span 128, mixup 0.3, AdamW lr 1e-3 wd 0.1). Only --encoder-type
differs.

Inputs:
  experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_v2.csv
    — final 6-metric eval values for both backbones.
  sync_tau_sweep/checkpoints/tau_sweep_0_10_losses.csv
    — gru training trajectory (15k full).
  sync_exp2_residual_silu/checkpoints/exp2_residual_silu_tau_0_10_losses.csv
    — residual_silu training trajectory (15k full).

Output:
  experiments/2026-05-09_exp_tau_encoder/plots/encoder_comparison.png
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_CSV = REPO / "experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_v2.csv"
TRAJ_GRU = REPO / "sync_tau_sweep/checkpoints/tau_sweep_0_10_losses.csv"
TRAJ_RES = REPO / "sync_exp2_residual_silu/checkpoints/exp2_residual_silu_tau_0_10_losses.csv"
OUT = REPO / "experiments/2026-05-09_exp_tau_encoder/plots/encoder_comparison.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

BETA = dict(r2_random=0.6839, r2_naive=0.6080, u_temporal=0.0375,
            u_batch=0.0762, auc=0.8966, top1=0.7531)

ARMS = [
    ("gru",            "tau_sweep_0_10",            "#d62728", TRAJ_GRU),
    ("residual_silu",  "exp2_residual_silu_tau_0_10","#17becf", TRAJ_RES),
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
    return dict(
        step=np.array([int(r["step"]) for r in rows]),
        loss=np.array([float(r["loss"]) for r in rows]),
        r2_random=np.array([float(r["r2_random"]) for r in rows]),
        r2_naive=np.array([float(r["r2_naive"]) for r in rows]),
        u_temporal=np.array([float(r["u_temporal"]) for r in rows]),
        u_batch=np.array([float(r["u_batch"]) for r in rows]),
        auc=np.array([float(r["auc"]) for r in rows]),
        top1=np.array([float(r["top1"]) for r in rows]),
    )


def main():
    eval_rows: dict[str, dict] = {}
    with open(EVAL_CSV) as f:
        for row in csv.DictReader(f):
            eval_rows[row["name"]] = {
                "r2_random": float(row["r2_random"]),
                "r2_naive": float(row["r2_naive"]),
                "u_temporal": float(row["u_temporal"]),
                "u_batch": float(row["u_batch"]),
                "auc": float(row["auc"]),
                "top1": float(row["top1"]),
            }

    traj: dict[str, dict] = {}
    for label, name, _color, path in ARMS:
        t = load_traj(path)
        if t is not None:
            traj[name] = t

    metrics = [
        ("r2_random", "R²_random",  None,        BETA["r2_random"]),
        ("r2_naive",  "R²_naive",   None,        BETA["r2_naive"]),
        ("u_temporal","U_temporal", None,        BETA["u_temporal"]),
        ("u_batch",   "U_batch",    None,        BETA["u_batch"]),
        ("auc",       "AUC",        (0.88, 0.93), BETA["auc"]),
        ("top1",      "Top-1",      (0.70, 0.78), BETA["top1"]),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()
    for ax, (key, title, ylim, beta_ref) in zip(axs, metrics):
        for label, name, color, _ in ARMS:
            t = traj.get(name)
            if t is None:
                continue
            arr = t[key]
            sm = smooth(arr)
            x = t["step"][len(t["step"]) - len(sm):] if len(sm) > 0 else t["step"]
            ev = eval_rows.get(name, {}).get(key)
            label_full = f"{label}"
            if ev is not None:
                label_full += f"  (final eval={ev:.4f})"
            ax.plot(x, sm if len(sm) > 0 else arr, color=color, label=label_full,
                    linewidth=1.8)
            # mark held-out eval value at the last step
            if ev is not None:
                last_step = int(t["step"][-1])
                ax.scatter([last_step], [ev], color=color, marker="*",
                           s=130, edgecolor="black", linewidth=0.6, zorder=5)
        if beta_ref is not None:
            ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=0.8,
                       label=f"backbone-β 167k = {beta_ref:.4f}")
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "Exp 2 — encoder comparison @ τ=0.10 (gru vs residual_silu, 15k steps each, "
        "1000-step MA in-training; ★ = held-out FINAL.pth eval)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
