#!/usr/bin/env python3
"""Extended τ-sweep comparison plot (Exp 1A v2 + Exp 1B learnable_τ).

Adapted from `plot_tau_sweep_comparison.py`. Adds:
  - τ=0.20 v2 trajectory (from sync_tau_sweep_arm5_v2; 7.8k partial)
  - τ=learnable init=0.10 trajectory (from sync_tau_sweep_learnable; 15k full)

Drops the original τ=0.20 trajectory (CSV never recovered) and the
"resync partial" trajectory (superseded by v2).

Outputs:
  experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_comparison.png
  experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_trajectories.png
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_CSV = REPO / "experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_v2.csv"
SYNC = REPO / "sync_tau_sweep/checkpoints"
SYNC_V2 = REPO / "sync_tau_sweep_arm5_v2/checkpoints"
SYNC_LEARN = REPO / "sync_tau_sweep_learnable/checkpoints"
OUT_COMP = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_comparison.png"
OUT_TRAJ = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_trajectories.png"
OUT_COMP.parent.mkdir(parents=True, exist_ok=True)

# (display_name, eval_csv_name, color, csv_path)
ARMS = [
    ("τ=0.03",          "tau_sweep_0_03",            "#1f77b4",
     SYNC / "tau_sweep_0_03_losses.csv"),
    ("τ=0.05",          "tau_sweep_0_05",            "#2ca02c",
     SYNC / "tau_sweep_0_05_losses.csv"),
    ("τ=0.07",          "tau_sweep_0_07",            "#9467bd",
     SYNC / "tau_sweep_0_07_losses.csv"),
    ("τ=0.10",          "tau_sweep_0_10",            "#d62728",
     SYNC / "tau_sweep_0_10_losses.csv"),
    ("τ=0.20 v2 (7.8k)","tau_sweep_0_20_v2",         "#ff7f0e",
     SYNC_V2 / "tau_sweep_0_20_v2_losses.csv"),
    ("τ=learnable init0.10", "tau_sweep_learnable_0_10", "#17becf",
     SYNC_LEARN / "tau_sweep_learnable_0_10_losses.csv"),
]

# Reference points (backbone-beta_167k held-out batch from the original RESULTS).
BETA = dict(r2_random=0.6839, r2_naive=0.6080, u_temporal=0.0375,
            u_batch=0.0762, auc=0.8966, top1=0.7531)


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
    # --- Eval CSV (final values per backbone) ---
    eval_rows: dict[str, dict] = {}
    with open(EVAL_CSV) as f:
        for row in csv.DictReader(f):
            eval_rows[row["name"]] = {
                "tau": float(row["tau"]),
                "r2_random": float(row["r2_random"]),
                "r2_naive": float(row["r2_naive"]),
                "u_temporal": float(row["u_temporal"]),
                "u_batch": float(row["u_batch"]),
                "auc": float(row["auc"]),
                "top1": float(row["top1"]),
            }

    # --- Trajectory CSVs ---
    traj: dict[str, dict] = {}
    for label, name, _color, csv_path in ARMS:
        t = load_traj(csv_path)
        if t is not None:
            traj[name] = t

    metrics = [
        ("r2_random", "R²_random", None, BETA["r2_random"]),
        ("r2_naive",  "R²_naive",  None, BETA["r2_naive"]),
        ("u_temporal", "U_temporal", None, BETA["u_temporal"]),
        ("u_batch",   "U_batch",     None, BETA["u_batch"]),
        ("auc",       "AUC",       (0.882, 0.900),  BETA["auc"]),
        ("top1",      "Top-1",     (0.72, 0.76),  BETA["top1"]),
    ]

    # --- Comparison plot: 1 row of bars/markers per arm ---
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    arm_labels = [a[0] for a in ARMS]
    arm_names = [a[1] for a in ARMS]
    arm_colors = [a[2] for a in ARMS]
    xs = np.arange(len(arm_labels))
    for ax, (key, title, ylim, beta_ref) in zip(axs, metrics):
        ys = [eval_rows[name][key] if name in eval_rows else None
              for name in arm_names]
        ys_clean = [y for y in ys if y is not None]
        xs_clean = [x for x, y in zip(xs, ys) if y is not None]
        cs_clean = [c for c, y in zip(arm_colors, ys) if y is not None]
        ax.bar(xs_clean, ys_clean, color=cs_clean, edgecolor="black",
               linewidth=0.5)
        if beta_ref is not None:
            ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=0.8,
                       label=f"backbone-β 167k = {beta_ref:.4f}")
            ax.legend(loc="best", fontsize=7)
        ax.set_title(f"{title} — held-out eval-batch (FINAL.pth)")
        ax.set_xticks(xs)
        ax.set_xticklabels(arm_labels, rotation=30, ha="right", fontsize=8)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3, axis="y")
        for x, y in zip(xs_clean, ys_clean):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7)

    fig.suptitle(
        "τ-sweep extended — held-out eval (B=256, skip=50M wrap, seed=0). "
        "v2 = τ=0.20 redo @ step 7800 best_loss. learnable: init τ=0.10, "
        "converged to τ≈0.069 by step 15k.",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_COMP, dpi=110, bbox_inches="tight")
    print(f"saved {OUT_COMP}")

    # --- Trajectory plot: 6 metric panels ---
    fig2, axs2 = plt.subplots(2, 3, figsize=(16, 8))
    axs2 = axs2.flatten()
    for ax, (key, title, ylim, beta_ref) in zip(axs2, metrics):
        for label, name, color, _ in ARMS:
            t = traj.get(name)
            if t is None:
                continue
            arr = t[key]
            sm = smooth(arr)
            x = t["step"][len(t["step"]) - len(sm):] if len(sm) > 0 else t["step"]
            ax.plot(x, sm if len(sm) > 0 else arr, color=color, label=label,
                    linewidth=1.6)
        if beta_ref is not None:
            ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=0.8,
                       label=f"backbone-β 167k = {beta_ref:.4f}")
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig2.suptitle(
        "τ-sweep extended — training trajectories (1000-step MA). "
        "6 arms: 0.03/0.05/0.07/0.10/0.20-v2(7.8k partial)/learnable_τ_init0.10.",
        fontsize=11)
    fig2.tight_layout()
    fig2.savefig(OUT_TRAJ, dpi=110, bbox_inches="tight")
    print(f"saved {OUT_TRAJ}")


if __name__ == "__main__":
    main()
