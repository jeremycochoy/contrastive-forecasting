#!/usr/bin/env python3
"""Loss-extension comparison plots: baseline τ=0.20 vs Exp 3 vs Exp 4-only.

Outputs:
  experiments/2026-05-09_exp_loss_extensions/plots/loss_extensions_trajectories.png
  experiments/2026-05-09_exp_loss_extensions/plots/loss_extensions_eval.png

Uses 1000-step MA smoothing, 6-panel layout (R²_rand, R²_naive, U_t, U_b,
AUC, Top-1). The baseline τ=0.20 trajectory CSV from the original Exp-1
run only retained ~4300 steps locally (its sync was incomplete) — its
final FINAL.pth eval point is taken from the per-arm eval CSV.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
EVAL_CSV = REPO / "experiments/2026-05-09_exp_loss_extensions/results/loss_extensions_metrics.csv"
OUT_TRAJ = REPO / "experiments/2026-05-09_exp_loss_extensions/plots/loss_extensions_trajectories.png"
OUT_EVAL = REPO / "experiments/2026-05-09_exp_loss_extensions/plots/loss_extensions_eval.png"
OUT_TRAJ.parent.mkdir(parents=True, exist_ok=True)

# (display_label, eval_csv_name, color, csv_path)
ARMS = [
    ("baseline τ=0.20",
     "tau_sweep_0_20",
     "#1f77b4",
     REPO / "sync_tau_sweep_arm5_resync/checkpoints/tau_sweep_0_20_losses.csv"),
    ("Exp 3 +(h_t,f_t) pos (cumulative)",
     "exp3_pos_htft_tau_0_20",
     "#d62728",
     REPO / "sync_exp3_pos_htft/checkpoints/exp3_pos_htft_tau_0_20_losses.csv"),
    ("Exp 4-only +f-cross-bc negs",
     "exp4_only_fnegs_tau_0_20",
     "#2ca02c",
     REPO / "sync_exp4_only_fnegs/checkpoints/exp4_only_fnegs_tau_0_20_losses.csv"),
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

    # 6-panel layout. Tight zoom on AUC + Top-1 chosen so Exp-3 chance band
    # at ~0.50/0.20 stays visible while baseline + Exp-4 trace the 0.88-0.93
    # range.
    metrics = [
        ("r2_random",  "R² random",   None),
        ("r2_naive",   "R² naive",    None),
        ("u_temporal", "U temporal",  None),
        ("u_batch",    "U batch",     None),
        ("auc",        "AUC",         (0.45, 0.95)),
        ("top1",       "Top-1",       (0.10, 0.85)),
    ]

    # --- Trajectory plot -----------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
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
        # Guide line for AUC chance / Top-1 random level (1/T).
        if key == "auc":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8,
                       label="chance (0.50)")
        elif key == "top1":
            # 1/T = 1/256 ≈ 0.0039; the chance Top-1 over T-1 candidates is
            # tiny. With B*C cross-batch negatives in scope it is harder to
            # call cleanly — leave the dotted reference off for Top-1 to
            # avoid implying the chance level shown is absolute.
            pass
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        "Loss-extension trajectories — 1000-step MA. "
        "Baseline τ=0.20 (~4.3k steps locally), Exp 3 (15k), Exp 4-only (15k).",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_TRAJ, dpi=110, bbox_inches="tight")
    print(f"saved {OUT_TRAJ}")

    # --- Held-out eval bar chart --------------------------------------------
    fig2, axs2 = plt.subplots(2, 3, figsize=(16, 8))
    axs2 = axs2.flatten()
    arm_labels = [a[0] for a in ARMS]
    arm_names = [a[1] for a in ARMS]
    arm_colors = [a[2] for a in ARMS]
    xs = np.arange(len(arm_labels))

    eval_metrics = [
        ("r2_random",  "R² random",  None),
        ("r2_naive",   "R² naive",   None),
        ("u_temporal", "U temporal", None),
        ("u_batch",    "U batch",    None),
        ("auc",        "AUC",        (0.45, 0.95)),
        ("top1",       "Top-1",      (0.10, 0.85)),
    ]

    for ax, (key, title, ylim) in zip(axs2, eval_metrics):
        ys = [eval_rows.get(name, {}).get(key) for name in arm_names]
        ys_clean = [y for y in ys if y is not None]
        xs_clean = [x for x, y in zip(xs, ys) if y is not None]
        cs_clean = [c for c, y in zip(arm_colors, ys) if y is not None]
        ax.bar(xs_clean, ys_clean, color=cs_clean, edgecolor="black",
               linewidth=0.5)
        ax.set_title(f"{title} — held-out eval (FINAL.pth)")
        ax.set_xticks(xs)
        ax.set_xticklabels(arm_labels, rotation=20, ha="right", fontsize=8)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3, axis="y")
        for x, y in zip(xs_clean, ys_clean):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7)

    fig2.suptitle(
        "Held-out eval (B=256, skip=50M wrap, seed=0) — 3 loss shapes @ τ=0.20.",
        fontsize=11)
    fig2.tight_layout()
    fig2.savefig(OUT_EVAL, dpi=110, bbox_inches="tight")
    print(f"saved {OUT_EVAL}")


if __name__ == "__main__":
    main()
