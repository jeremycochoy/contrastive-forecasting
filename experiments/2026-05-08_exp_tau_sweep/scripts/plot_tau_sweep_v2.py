#!/usr/bin/env python3
"""τ-sweep training-trajectory plot.

Renders 6-panel training trajectories (1000-step MA) for the 6 trained
arms: {τ=0.03, 0.05, 0.07, 0.10, learnable_τ_init0.10, 0.20}. Single
τ=0.20 trace is the v2 retrain (the only run with a full trajectory CSV).

The held-out per-backbone comparison is the multisample plot
(`tau_sweep_eval_multisample.png`), produced by
`plot_tau_sweep_multisample.py`. This script renders only the trajectory.

Output: experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_trajectories.png
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
SYNC = REPO / "sync_tau_sweep/checkpoints"
SYNC_V2 = REPO / "sync_tau_sweep_arm5_v2/checkpoints"
SYNC_LEARN = REPO / "sync_tau_sweep_learnable/checkpoints"
OUT_TRAJ = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_trajectories.png"
OUT_TRAJ.parent.mkdir(parents=True, exist_ok=True)

# Arms in numerical-τ order with τ=0.10 and learnable_τ_init0.10 adjacent.
# (display_label, run_name, color, csv_path)
ARMS = [
    ("τ=0.03",            "tau_sweep_0_03",           "#1f77b4",
     SYNC / "tau_sweep_0_03_losses.csv"),
    ("τ=0.05",            "tau_sweep_0_05",           "#2ca02c",
     SYNC / "tau_sweep_0_05_losses.csv"),
    ("τ=0.07",            "tau_sweep_0_07",           "#9467bd",
     SYNC / "tau_sweep_0_07_losses.csv"),
    ("τ=0.10",            "tau_sweep_0_10",           "#d62728",
     SYNC / "tau_sweep_0_10_losses.csv"),
    ("τ=0.10 → 0.069 (learnable)", "tau_sweep_learnable_0_10", "#17becf",
     SYNC_LEARN / "tau_sweep_learnable_0_10_losses.csv"),
    ("τ=0.20",            "tau_sweep_0_20_v2",        "#ff7f0e",
     SYNC_V2 / "tau_sweep_0_20_v2_losses.csv"),
]

# backbone-beta_167k held-out reference, same batch as the eval CSV.
BETA = dict(r2_random=0.6839, r2_naive=0.6080, u_temporal=0.0375,
            u_batch=0.0762, auc=0.8966, top1=0.7531)


def smooth(x, w):
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


def main() -> None:
    traj: dict[str, dict] = {}
    for label, name, _color, csv_path in ARMS:
        t = load_traj(csv_path)
        if t is not None:
            traj[name] = t

    # Per-metric ylim zooms; per-metric smoothing window. R² panels zoomed
    # to the data-relevant range (negative R² is meaningless). AUC / Top-1
    # use a smaller window (100-step MA) so finer in-training detail is
    # visible; the heavier-curve metrics (R², U, loss) stay at 400.
    # (key, title, ylim, beta_ref, smooth_w)
    metrics = [
        ("r2_random", "R²_random", (0.60, 0.85), BETA["r2_random"], 400),
        ("r2_naive",  "R²_naive",  (0.45, 0.80), BETA["r2_naive"],  400),
        ("u_temporal", "U_temporal", None,       BETA["u_temporal"], 400),
        ("u_batch",   "U_batch",     None,       BETA["u_batch"],    400),
        ("auc",       "AUC",        (0.882, 0.910), BETA["auc"],    100),
        ("top1",      "Top-1",      (0.72, 0.77),   BETA["top1"],   100),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs = axs.flatten()
    for ax, (key, title, ylim, beta_ref, smooth_w) in zip(axs, metrics):
        for label, name, color, _ in ARMS:
            t = traj.get(name)
            if t is None:
                continue
            arr = t[key]
            sm = smooth(arr, smooth_w)
            x = t["step"][len(t["step"]) - len(sm):] if len(sm) > 0 else t["step"]
            ax.plot(x, sm if len(sm) > 0 else arr, color=color,
                    label=label, linewidth=1.6)
        if beta_ref is not None:
            ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=0.8,
                       label=f"backbone-β 167k = {beta_ref:.4f}")
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim is not None:
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(
        "τ-sweep — training trajectories (R² / U: 400-step MA, "
        "AUC / Top-1: 100-step MA; 6 arms, 15k steps).",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_TRAJ, dpi=110, bbox_inches="tight")
    print(f"saved {OUT_TRAJ}")


if __name__ == "__main__":
    main()
