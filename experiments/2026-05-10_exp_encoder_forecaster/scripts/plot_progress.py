#!/usr/bin/env python3
"""Progress plot — encoder+forecaster (6L+6L, bf16) vs τ=0.10 baseline.

Reads per-step training-batch metrics from the losses CSVs and renders a
2x3 panel of trajectories smoothed with a moving average:

    loss (log-y)         |  U_temporal              |  U_batch
    1 - per-batch AUC    |  1 - per-batch top-1     |  1 - error-gap-closure
    (all log-y)            (log-y)                    (log-y)

x-axis is `log(step)` on every panel. The `1 - metric` transform on a
log y-axis (used in the τ-sweep plots) magnifies residual differences
near the saturation point (auc/top1 → 1) while compressing the noisy
plateau.

Error-gap-closure is a derived metric:
    egc = 1 - (1 - ff) / (1 - fp)
where `ff` is the cosine similarity between forecast and future
embeddings (positive pair) and `fp` is the cosine similarity between
forecast and past embeddings (cross-batch reference). When fp < 1,
egc=1 ⇒ perfect (ff=1), egc=0 ⇒ no improvement over fp, egc<0 ⇒ fp > ff.
Guards: rows with fp ≥ 1 (denominator ≤ 0) are masked out of the plot.

Outputs:
    experiments/2026-05-10_exp_encoder_forecaster/plots/progress.png

Re-runnable. Worktree path is the script root; source CSVs are read
from the MAIN checkout where the training writes them. No GPU usage.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# This script lives in <worktree>/experiments/2026-05-10_exp_encoder_forecaster/scripts.
# Its output plot goes back into the same worktree experiment dir.
WORKTREE = Path(__file__).resolve().parents[3]
EXP_DIR = WORKTREE / "experiments/2026-05-10_exp_encoder_forecaster"
OUT = EXP_DIR / "plots/progress.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Training CSVs are written by the live run into the MAIN checkout. We
# read from there read-only — never write into that tree.
MAIN = Path("/home/jupyter/contrastive-forecasting")

# (display_label, color, linestyle, lw, csv_path)
ARMS = [
    ("τ=0.10 baseline (6L fcst, 15k trained)",
     "#888888", "-", 1.6,
     MAIN / "sync_tau_sweep/checkpoints/tau_sweep_0_10_losses.csv"),
    ("τ=0.10 long-run reference (48k–150k)",
     "#bbbbbb", "--", 1.2,
     MAIN / "sync_tau_sweep_0_10_150k/checkpoints/tau_sweep_0_10_150k_losses.csv"),
    ("encoder+forecaster (6L+6L, bf16)",
     "#d62728", "-", 1.7,
     MAIN / "checkpoints/enc_fcst_tau_0_10_50k_losses.csv"),
]


COLS = ("loss", "ff", "fp", "u_temporal", "u_batch", "auc", "top1")


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_traj(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    out = {"step": np.array([int(r["step"]) for r in rows])}
    for k in COLS:
        out[k] = np.array([float(r[k]) for r in rows], dtype=np.float64)
    # derived: error-gap-closure = 1 - (1-ff)/(1-fp)
    denom = 1.0 - out["fp"]
    # mask fp >= 1 (denom <= 0); rare in practice
    with np.errstate(divide="ignore", invalid="ignore"):
        egc = np.where(denom > 0, 1.0 - (1.0 - out["ff"]) / denom, np.nan)
    out["egc"] = egc
    return out


def _aligned_smooth(steps, values, w):
    """Smooth values and return (x, y) aligned to the right edge of the window."""
    sm = smooth(values, w)
    if len(sm) == len(values):
        return steps, sm
    x = steps[len(steps) - len(sm):]
    return x, sm


def last_window_mean(arr, n=200):
    if len(arr) == 0:
        return float("nan")
    a = arr[-n:]
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def plot_panel(ax, traj_by_label, *, key, ylabel, transform, logy):
    """Render one metric panel for all arms.

    transform(y) optionally maps y → 1-y so log-y shows residual to 1.
    """
    for (label, color, ls, lw, _path), arm in traj_by_label:
        if arm is None:
            continue
        n = len(arm["step"])
        # Window proportional to trajectory length, capped at 1000.
        w = max(20, min(1000, n // 40))
        y = arm[key]
        y = transform(y) if transform is not None else y
        # If transformed, mask non-positive entries before log so the plot
        # doesn't error out (no clipping silently changes the curve).
        if logy and transform is not None:
            mask = np.isfinite(y) & (y > 0)
            if not mask.any():
                continue
            x_steps = arm["step"][mask]
            y = y[mask]
        else:
            mask = np.isfinite(y)
            x_steps = arm["step"][mask]
            y = y[mask]
        x_sm, y_sm = _aligned_smooth(x_steps, y, w)
        ax.plot(x_sm, y_sm, color=color, linestyle=ls,
                linewidth=lw, label=label)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both", ls=":")


def main() -> None:
    # Load all CSVs once.
    traj = []
    for arm in ARMS:
        d = load_traj(arm[-1])
        traj.append((arm, d))
        label = arm[0]
        if d is None:
            print(f"[plot] SKIP {label}: {arm[-1]} not found")
        else:
            n = len(d["step"])
            print(
                f"[plot] {label}: {n} steps, last≈{int(d['step'][-1])} "
                f"loss={last_window_mean(d['loss']):.3f} "
                f"u_t={last_window_mean(d['u_temporal']):.3f} "
                f"u_b={last_window_mean(d['u_batch']):.3f} "
                f"auc={last_window_mean(d['auc']):.4f} "
                f"top1={last_window_mean(d['top1']):.4f} "
                f"egc={last_window_mean(d['egc']):.3f}"
            )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_loss, ax_ut, ax_ub = axes[0]
    ax_auc, ax_top1, ax_egc = axes[1]

    plot_panel(ax_loss, traj, key="loss", ylabel="loss  (log-y, MA)",
               transform=None, logy=True)
    plot_panel(ax_ut, traj, key="u_temporal",
               ylabel="U_temporal  (MA)", transform=None, logy=False)
    plot_panel(ax_ub, traj, key="u_batch",
               ylabel="U_batch  (MA)", transform=None, logy=False)
    plot_panel(ax_auc, traj, key="auc",
               ylabel="1 - per-batch AUC  (log-y, MA; lower = better)",
               transform=lambda y: 1.0 - y, logy=True)
    plot_panel(ax_top1, traj, key="top1",
               ylabel="1 - per-batch top-1  (log-y, MA; lower = better)",
               transform=lambda y: 1.0 - y, logy=True)
    plot_panel(ax_egc, traj, key="egc",
               ylabel="1 - error-gap-closure  (log-y, MA; lower = better)",
               transform=lambda y: 1.0 - y, logy=True)

    # One shared legend at the top, drawn from the first non-empty panel.
    handles, labels = ax_loss.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(handles), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.985))

    fig.suptitle(
        "Encoder+forecaster (6L+6L, bf16, τ=0.10) progress — per-batch "
        "training metrics; MA window = min(1000, len/40); log-x on all panels; "
        "log-y on loss / 1-AUC / 1-top1 / 1-egc",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {OUT}")


if __name__ == "__main__":
    main()
