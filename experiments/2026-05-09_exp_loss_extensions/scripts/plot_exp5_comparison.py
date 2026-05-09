#!/usr/bin/env python3
"""Exp 5 comparison plot — 4 trajectories: baseline τ=0.20, Exp 3, Exp 4-only,
Exp 5. Mirrors `plot_loss_extensions.py` but extends to 4 arms with the new
Exp 5 (skip-step f-negatives) trajectory + held-out eval bar.

Outputs:
  experiments/2026-05-09_exp_loss_extensions/plots/exp5_comparison.png

Tight zoom on AUC (0.50-0.95) and Top-1 (0.20-0.80) to keep both the chance-
level Exp 3 trajectory and the high-AUC arms readable.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_CSV = REPO / "experiments/2026-05-09_exp_loss_extensions/results/loss_extensions_metrics.csv"
OUT_PATH = REPO / "experiments/2026-05-09_exp_loss_extensions/plots/exp5_comparison.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# (display_label, eval_csv_name, color, csv_path)
ARMS = [
    ("baseline τ=0.20",
     "tau_sweep_0_20",
     "#1f77b4",
     REPO / "sync_tau_sweep_arm5_resync/checkpoints/tau_sweep_0_20_losses.csv"),
    ("Exp 3 +(h_t,f_t) pos",
     "exp3_pos_htft_tau_0_20",
     "#d62728",
     REPO / "sync_exp3_pos_htft/checkpoints/exp3_pos_htft_tau_0_20_losses.csv"),
    ("Exp 4-only +f-cross-bc negs",
     "exp4_only_fnegs_tau_0_20",
     "#2ca02c",
     REPO / "sync_exp4_only_fnegs/checkpoints/exp4_only_fnegs_tau_0_20_losses.csv"),
    ("Exp 5 +skip-f negs (t↔t+2)",
     "exp5_skip_fnegs_tau_0_20",
     "#9467bd",
     REPO / "sync_exp5_skip_fnegs/checkpoints/exp5_skip_fnegs_tau_0_20_losses.csv"),
]


def smooth(x: np.ndarray, w: int = 1000) -> np.ndarray:
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
        r2_random=np.array([float(r["r2_random"]) for r in rows]),
        r2_naive=np.array([float(r["r2_naive"]) for r in rows]),
        u_temporal=np.array([float(r["u_temporal"]) for r in rows]),
        u_batch=np.array([float(r["u_batch"]) for r in rows]),
        auc=np.array([float(r["auc"]) for r in rows]),
        top1=np.array([float(r["top1"]) for r in rows]),
    )


def main() -> None:
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
    for label, name, _color, csv_path in ARMS:
        t = load_traj(csv_path)
        if t is not None:
            traj[name] = t
            print(f"loaded trajectory {name}: {len(t['step'])} rows")
        else:
            print(f"missing trajectory CSV for {name}: {csv_path}")

    # 6-panel layout.
    metrics = [
        ("r2_random",  "R² random",   None),
        ("r2_naive",   "R² naive",    None),
        ("u_temporal", "U temporal",  None),
        ("u_batch",    "U batch",     None),
        ("auc",        "AUC",         (0.50, 0.95)),
        ("top1",       "Top-1",       (0.20, 0.80)),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    axs = axs.flatten()
    for ax, (key, title, ylim) in zip(axs, metrics):
        for label, name, color, _ in ARMS:
            t = traj.get(name)
            if t is None:
                continue
            arr = t[key]
            sm = smooth(arr)
            x = t["step"][len(t["step"]) - len(sm):] if len(sm) > 0 else t["step"]
            y = sm if len(sm) > 0 else arr
            ax.plot(x, y, color=color, label=label, linewidth=1.6)
            # Held-out eval point at the rightmost x position.
            ev = eval_rows.get(name)
            if ev is not None and key in ev:
                x_eval = float(t["step"][-1])
                ax.scatter([x_eval], [ev[key]], color=color,
                           edgecolor="black", s=42, zorder=5)
        # Guide line for AUC chance.
        if key == "auc":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                       label="chance (0.50)")
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "Loss-extension comparison — 1000-step MA + held-out eval points "
        "(open circles). 4 arms @ τ=0.20.",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=110, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
