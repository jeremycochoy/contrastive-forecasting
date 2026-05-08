#!/usr/bin/env python3
"""5-arm τ-sweep comparison plot.

Two-row figure:
  Row 1 — final-state values vs τ (one panel per metric, dots-and-line).
  Row 2 — training-time trajectories (4 arms with full CSV; τ=0.20 final
          marker only because that arm's trajectory CSV was lost).

Inputs:
  experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics.csv
    — final 6-metric eval values from eval_tau_sweep_metrics.py
  sync_tau_sweep/checkpoints/tau_sweep_<safe>_losses.csv
    — per-step trajectory (4 arms only)

Output:
  experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_comparison.png
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_CSV = REPO / "experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics.csv"
SYNC = REPO / "sync_tau_sweep/checkpoints"
OUT = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_comparison.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

ARMS = [("0_03", 0.03), ("0_05", 0.05), ("0_07", 0.07), ("0_10", 0.10), ("0_20", 0.20)]
COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e"]  # blue/green/purple/red/orange

# Reference points (from REPORT.md, backbone-beta_167k held-out batch).
BETA = dict(r2_random=0.6839, r2_naive=0.6080, u_temporal=0.0375,
            u_batch=0.0762, auc=0.8966, top1=0.7531)


def smooth(x, w=200):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    # --- Eval CSV (final values per arm) ---
    eval_rows = {}
    with open(EVAL_CSV) as f:
        for row in csv.DictReader(f):
            tau = float(row["tau"])
            eval_rows[tau] = {k: float(v) for k, v in row.items() if k not in ("name", "tau")}
    taus = sorted(eval_rows.keys())

    # --- Trajectory CSVs (4 arms have them; τ=0.20 doesn't) ---
    traj = {}
    for safe, tau in ARMS:
        path = SYNC / f"tau_sweep_{safe}_losses.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        traj[tau] = dict(
            step=np.array([int(r["step"]) for r in rows]),
            loss=np.array([float(r["loss"]) for r in rows]),
            r2_random=np.array([float(r["r2_random"]) for r in rows]),
            r2_naive=np.array([float(r["r2_naive"]) for r in rows]),
            u_temporal=np.array([float(r["u_temporal"]) for r in rows]),
            u_batch=np.array([float(r["u_batch"]) for r in rows]),
            auc=np.array([float(r["auc"]) for r in rows]),
            top1=np.array([float(r["top1"]) for r in rows]),
        )

    # --- Figure: 2 rows, 6 metric columns + loss column = layout 3 cols ---
    metrics = [
        ("r2_random", "R²_random", (0.0, 1.0), BETA["r2_random"]),
        ("r2_naive",  "R²_naive",  (0.0, 1.0), BETA["r2_naive"]),
        ("u_temporal", "U_temporal (per-slice)", None, BETA["u_temporal"]),
        ("u_batch",   "U_batch (per-slice)",     None, BETA["u_batch"]),
        ("auc",       "AUC",       (0.87, 0.92),  BETA["auc"]),
        ("top1",      "Top-1",     (0.70, 0.80),  BETA["top1"]),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()

    # Row 1: final-state vs τ (one panel per metric, all 6 in 2x3)
    for ax, (key, title, ylim, beta_ref) in zip(axs, metrics):
        ys = [eval_rows[t][key] for t in taus]
        ax.plot(taus, ys, marker="o", color="black", linewidth=1.5)
        for t, y, c in zip(taus, ys, COLORS):
            ax.scatter([t], [y], color=c, s=80, zorder=3, edgecolor="black", linewidth=0.5)
        if beta_ref is not None:
            ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=0.8,
                       label=f"backbone-β 167k = {beta_ref:.4f}")
        ax.set_title(f"{title} — final eval-batch")
        ax.set_xlabel("τ")
        ax.set_xscale("log")
        ax.set_xticks(taus)
        ax.set_xticklabels([f"{t}" for t in taus])
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        if beta_ref is not None:
            ax.legend(loc="best", fontsize=8)
        for t, y in zip(taus, ys):
            ax.annotate(f"{y:.3f}", (t, y), textcoords="offset points",
                        xytext=(8, 4), fontsize=7)

    fig.suptitle(
        "τ-sweep — final eval-batch values (5 arms × 15k steps, B=256, "
        "held-out HF batch skip=50M wrap, eval seed=0)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")

    # --- Trajectory plot (separate file) ---
    OUT2 = OUT.parent / "tau_sweep_trajectories.png"
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))
    axs2 = axs2.flatten()
    for ax, (key, title, ylim, beta_ref) in zip(axs2, metrics):
        for tau, t_data, c in [(t, traj[t], COLORS[i]) for i, (_, t) in enumerate(ARMS) if t in traj]:
            sm = smooth(t_data[key])
            x = t_data["step"][len(t_data["step"]) - len(sm):] if len(sm) > 0 else t_data["step"]
            ax.plot(x, sm if len(sm) > 0 else t_data[key],
                    color=c, label=f"τ={tau}", linewidth=1.5)
        # τ=0.20 final-state marker (no trajectory available)
        ax.scatter([15000], [eval_rows[0.20][key]], color=COLORS[4], marker="*",
                   s=120, edgecolor="black", linewidth=0.7,
                   label=f"τ=0.2 (eval only, no traj)", zorder=5)
        if beta_ref is not None:
            ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    fig2.suptitle(
        "τ-sweep — training trajectories (4 arms, smoothed 200; "
        "τ=0.20 trajectory CSV lost — final eval value only as ★)",
        fontsize=11)
    fig2.tight_layout()
    fig2.savefig(OUT2, dpi=110, bbox_inches="tight")
    print(f"saved {OUT2}")


if __name__ == "__main__":
    main()
