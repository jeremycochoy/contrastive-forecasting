#!/usr/bin/env python3
"""τ-sweep short-trajectory zoom — AUC and Top-1 only, log/log, x∈[5k,15k].

A 1×2 companion to `plot_tau_sweep_v2.py`. Same ARMS, same y-windows
(AUC 0.882–0.910, Top-1 0.72–0.77 — matching the short trajectory plot
exactly), but rendered in log/log with x cropped to the converged
region [5k, 15k] and AUC/Top-1 smoothed harder (260-step MA — about
50% above the short-trajectory plot's 175-step window).

Output: experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_auc_top1_loglog.png
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
SYNC_030 = REPO / "sync_tau_sweep_0_30/checkpoints"
SYNC_080 = REPO / "sync_tau_sweep_0_80/checkpoints"
OUT = REPO / "experiments/2026-05-08_exp_tau_sweep/plots/tau_sweep_v2_auc_top1_loglog.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Same ARMS list as plot_tau_sweep_v2.py.
ARMS = [
    ("τ=0.03",                     "tau_sweep_0_03",           "#1f77b4",
     SYNC / "tau_sweep_0_03_losses.csv"),
    ("τ=0.05",                     "tau_sweep_0_05",           "#2ca02c",
     SYNC / "tau_sweep_0_05_losses.csv"),
    ("τ=0.07",                     "tau_sweep_0_07",           "#9467bd",
     SYNC / "tau_sweep_0_07_losses.csv"),
    ("τ=0.10 → 0.069 (learnable)", "tau_sweep_learnable_0_10", "#17becf",
     SYNC_LEARN / "tau_sweep_learnable_0_10_losses.csv"),
    ("τ=0.20 → 0.07 (learnable)",  "tau_sweep_learnable_0_20", "#bcbd22",
     SYNC_LEARN / "tau_sweep_learnable_0_20_losses.csv"),
    ("τ=0.10",                     "tau_sweep_0_10",           "#d62728",
     SYNC / "tau_sweep_0_10_losses.csv"),
    ("τ=0.20",                     "tau_sweep_0_20_v2",        "#ff7f0e",
     SYNC_V2 / "tau_sweep_0_20_v2_losses.csv"),
    ("τ=0.30",                     "tau_sweep_0_30",           "#e377c2",
     SYNC_030 / "tau_sweep_0_30_losses.csv"),
    ("τ=0.50",                     "tau_sweep_0_50",           "#8c564b",
     SYNC / "tau_sweep_0_50_losses.csv"),
    ("τ=0.80",                     "tau_sweep_0_80",           "#7f7f7f",
     SYNC_080 / "tau_sweep_0_80_losses.csv"),
]

BETA = dict(auc=0.8966, top1=0.7531)


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
        auc=np.array([float(r["auc"]) for r in rows]),
        top1=np.array([float(r["top1"]) for r in rows]),
    )


def main() -> None:
    traj: dict[str, dict] = {}
    for label, name, _color, csv_path in ARMS:
        t = load_traj(csv_path)
        if t is None:
            continue
        # Clip to ≤ 15k like the short trajectory plot does.
        keep = t["step"] <= 15000
        t = {k: v[keep] for k, v in t.items()}
        traj[name] = t

    SMOOTH_W = 260  # ~50% more than the short trajectory plot's 175.
    metrics = [
        ("auc",  "AUC",   (0.882, 0.910), BETA["auc"]),
        ("top1", "Top-1", (0.72, 0.77),   BETA["top1"]),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (key, title, ylim, beta_ref) in zip(axs, metrics):
        for label, name, color, _ in ARMS:
            t = traj.get(name)
            if t is None:
                continue
            arr = t[key]
            sm = smooth(arr, SMOOTH_W)
            x = t["step"][len(t["step"]) - len(sm):] if len(sm) > 0 else t["step"]
            ax.plot(x, sm if len(sm) > 0 else arr, color=color,
                    label=label, linewidth=1.6)
        ax.axhline(beta_ref, color="gray", linestyle="--", linewidth=0.9,
                   label=f"backbone-β 167k = {beta_ref:.4f}")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5000, 15000)
        ax.set_ylim(ylim)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle(
        f"τ-sweep — AUC / Top-1 zoom (log/log; {SMOOTH_W}-step MA; "
        "x∈[5k, 15k]; y windows match short-trajectory plot).",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
