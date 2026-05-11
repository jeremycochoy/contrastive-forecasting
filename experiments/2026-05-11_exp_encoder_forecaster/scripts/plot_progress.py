#!/usr/bin/env python3
"""Progress plot — encoder+forecaster v2 (6L+6L, dropkey=0.7, bf16) vs τ=0.10.

Reads per-step training-batch metrics from the losses CSVs and renders a
2x3 panel of trajectories smoothed with a moving average:

    loss                 |  U_temporal              |  U_batch
    1 - AUC              |  1 - top-1               |  1 - top-3

`auc` / `top1` / `top3` are the canonical retrieval metrics: positive at
lag 0 ranked against 12 negatives (4 same-sample temporal lags
{1,2,4,8} + 8 cross-batch at the positive time). The baseline τ=0.10
CSVs predate top-3 (only `auc` + `top1` were logged at the time), so
the 1-top3 panel shows only the new arm.

Two PNGs are produced in one call:

    plots/progress.png        — log-x on all panels; log-y on
                                loss / 1-AUC / 1-top1 / 1-top3
    plots/progress_linear.png — same panels, linear x; linear y for
                                loss / U_temporal / U_batch; the AUC /
                                top-1 / top-3 panels still show `1-metric`
                                (linear y) so small residuals near 1 are
                                visible.

Error-gap-closure is a derived metric:
    egc = 1 - (1 - ff) / (1 - fp)
where `ff` is the cosine similarity between forecast and future
embeddings (positive pair) and `fp` is the cosine similarity between
forecast and past embeddings (cross-batch reference). When fp < 1,
egc=1 ⇒ perfect (ff=1), egc=0 ⇒ no improvement over fp, egc<0 ⇒ fp > ff.
Guards: rows with fp ≥ 1 (denominator ≤ 0) are masked out of the plot.

The τ=0.10 baseline is concatenated from multiple resume chunks
(0–15k, 15k–24.5k, 24.5k–50k); on overlapping ranges the earliest
chunk's row wins.

The x-axis upper bound is clipped to the SHORTER of the two
trajectories — i.e. `min(baseline_max_step, new_arm_max_step)` plus a
small margin.

Re-runnable. No GPU usage.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# This script lives in <worktree>/experiments/2026-05-11_exp_encoder_forecaster/scripts.
WORKTREE = Path(__file__).resolve().parents[3]
EXP_DIR = WORKTREE / "experiments/2026-05-11_exp_encoder_forecaster"
OUT_LOG = EXP_DIR / "plots/progress.png"
OUT_LIN = EXP_DIR / "plots/progress_linear.png"
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

# Training CSVs are written by the live run into the MAIN checkout. We
# read from there read-only — never write into that tree.
MAIN = Path("/home/jupyter/contrastive-forecasting")

# Color palette: saturated, high-contrast.
#   - τ=0.10 baseline (long-trained, concatenated chunks) → tab:blue
#   - attempt-1 (shared (T,T) dropkey mask, NaN'd at step 11700) → tab:orange
#   - per-(B,head) headline arm (dropkey=0.7, fresh per row+head) → tab:red
C_BASELINE = "#1f77b4"  # tab:blue
C_ATTEMPT1 = "#ff7f0e"  # tab:orange
C_HEADLINE = "#d62728"  # tab:red

# τ=0.10 baseline is split across resume chunks. Concatenate in step
# order; on overlap, the earliest chunk's row wins.
BASELINE_CHUNKS = [
    MAIN / "sync_tau_sweep/checkpoints/tau_sweep_0_10_losses.csv",                  # 1–15000
    MAIN / "sync_tau_sweep_0_10_50k/checkpoints/tau_sweep_0_10_50k_losses.csv",     # 14901–24500
    MAIN / "sync_tau_sweep_0_10_50k/checkpoints/tau_sweep_0_10_50k_r2_losses.csv",  # 23901–50000
    MAIN / "sync_tau_sweep_0_10_150k/checkpoints/tau_sweep_0_10_150k_losses.csv",   # 48001–150000
]

NEW_ARM_CSV = MAIN / "checkpoints/enc_fcst_dropkey07_pb_50k_losses.csv"
ATTEMPT1_CSV = MAIN / "checkpoints/enc_fcst_dropkey07_50k_attempt1_losses.csv"

# (display_label, color, linestyle, lw, csv_path_or_chunks, is_concat)
ARMS = [
    ("τ=0.10 baseline (long-trained, 0–150k available)",
     C_BASELINE, "-", 1.6,
     BASELINE_CHUNKS,
     True),
    ("attempt-1 (shared (T,T) dropkey mask, NaN'd @ 11700)",
     C_ATTEMPT1, "-", 1.4,
     ATTEMPT1_CSV,
     False),
    ("encoder+forecaster v2 (6L+6L, dropkey=0.7 per-(B,head), bf16)",
     C_HEADLINE, "-", 1.8,
     NEW_ARM_CSV,
     False),
]

COLS = ("loss", "ff", "fp", "u_temporal", "u_batch", "auc", "top1", "top3",
        "r2_random", "r2_naive")


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
        # tolerate missing columns (older CSVs may not have all metric names)
        if k in rows[0]:
            out[k] = np.array([float(r[k]) for r in rows], dtype=np.float64)
        else:
            out[k] = np.full(len(rows), np.nan, dtype=np.float64)
    denom = 1.0 - out["fp"]
    with np.errstate(divide="ignore", invalid="ignore"):
        egc = np.where(denom > 0, 1.0 - (1.0 - out["ff"]) / denom, np.nan)
    out["egc"] = egc
    return out


def load_concat_traj(paths: list[Path]) -> dict | None:
    seen: set[int] = set()
    merged: list[dict] = []
    found_any = False
    for p in paths:
        if not p.exists():
            print(f"[plot] WARN concat chunk missing: {p}")
            continue
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        found_any = True
        for r in rows:
            s = int(r["step"])
            if s in seen:
                continue
            seen.add(s)
            merged.append(r)
    if not found_any or not merged:
        return None
    merged.sort(key=lambda r: int(r["step"]))
    out = {"step": np.array([int(r["step"]) for r in merged])}
    for k in COLS:
        if k in merged[0]:
            out[k] = np.array([float(r[k]) for r in merged], dtype=np.float64)
        else:
            out[k] = np.full(len(merged), np.nan, dtype=np.float64)
    denom = 1.0 - out["fp"]
    with np.errstate(divide="ignore", invalid="ignore"):
        egc = np.where(denom > 0, 1.0 - (1.0 - out["ff"]) / denom, np.nan)
    out["egc"] = egc
    return out


def _aligned_smooth(steps, values, w):
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


def plot_panel(ax, traj_by_label, *, key, ylabel, panel_title,
               transform, logy, logx, xlim):
    xmin, xmax = xlim
    for (label, color, ls, lw, _path, _ref), arm in traj_by_label:
        if arm is None:
            continue
        # Truncate the trajectory to the visible x-window FIRST, then
        # smooth. Otherwise the long baseline (150k rows) gets a huge
        # smoothing window (1000) and its first plottable point is step
        # 1000 — invisible when the new arm is still under 1000 steps.
        # Clip-then-smooth gives both arms comparable effective windows
        # within the visible range.
        in_arm = (arm["step"] >= xmin) & (arm["step"] <= xmax)
        if not in_arm.any():
            continue
        steps_clip = arm["step"][in_arm]
        y_clip = arm[key][in_arm]
        n = len(steps_clip)
        # Smoothing window proportional to the *visible* trajectory.
        w = max(20, min(1000, n // 40))
        y = transform(y_clip) if transform is not None else y_clip
        if logy and transform is not None:
            mask = np.isfinite(y) & (y > 0)
        else:
            mask = np.isfinite(y)
        if not mask.any():
            continue
        x_steps = steps_clip[mask]
        y = y[mask]
        x_sm, y_sm = _aligned_smooth(x_steps, y, w)
        ax.plot(x_sm, y_sm, color=color, linestyle=ls, linewidth=lw,
                label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_title(panel_title, fontsize=10)
    ax.set_xlabel("step" + (" (log)" if logx else ""))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both", ls=":")


def render_figure(traj, *, logx, logy_metrics, out_path, scale_tag, xmax,
                  xmin_log=1):
    xmin = xmin_log if logx else 0
    xlim = (xmin, xmax)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    ax_loss, ax_ut, ax_ub, ax_r2r = axes[0]
    ax_r2n, ax_auc, ax_top1, ax_top3 = axes[1]

    plot_panel(ax_loss, traj, key="loss",
               ylabel="loss" + ("  (log-y)" if "loss" in logy_metrics else ""),
               panel_title="Loss (MA)",
               transform=None, logy="loss" in logy_metrics, logx=logx, xlim=xlim)
    plot_panel(ax_ut, traj, key="u_temporal",
               ylabel="U_temporal",
               panel_title="U_temporal — dimension usage along time",
               transform=None, logy=False, logx=logx, xlim=xlim)
    plot_panel(ax_ub, traj, key="u_batch",
               ylabel="U_batch",
               panel_title="U_batch — dimension usage across batch",
               transform=None, logy=False, logx=logx, xlim=xlim)
    plot_panel(ax_r2r, traj, key="r2_random",
               ylabel="R²_random",
               panel_title="R²_random",
               transform=None, logy=False, logx=logx, xlim=xlim)
    plot_panel(ax_r2n, traj, key="r2_naive",
               ylabel="R²_naive",
               panel_title="R²_naive",
               transform=None, logy=False, logx=logx, xlim=xlim)
    plot_panel(ax_auc, traj, key="auc",
               ylabel="1 − AUC",
               panel_title="1 − AUC",
               transform=lambda y: 1.0 - y, logy="auc" in logy_metrics,
               logx=logx, xlim=xlim)
    plot_panel(ax_top1, traj, key="top1",
               ylabel="1 − top-1",
               panel_title="1 − top-1",
               transform=lambda y: 1.0 - y, logy="top1" in logy_metrics,
               logx=logx, xlim=xlim)
    plot_panel(ax_top3, traj, key="top3",
               ylabel="1 − top-3",
               panel_title="1 − top-3",
               transform=lambda y: 1.0 - y, logy="top3" in logy_metrics,
               logx=logx, xlim=xlim)

    handles, labels = ax_loss.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(handles), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, 0.985))

    fig.suptitle(
        f"encoder+forecaster v2 (dropkey=0.7) — per-(B,head) vs shared-mask "
        f"attempt-1 vs τ=0.10 baseline, all clipped to {xmax:,} ({scale_tag})",
        fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    new_arm = load_traj(NEW_ARM_CSV)
    if new_arm is None or len(new_arm["step"]) == 0:
        raise SystemExit(f"new-arm CSV missing or empty: {NEW_ARM_CSV}")
    baseline = load_concat_traj(BASELINE_CHUNKS)
    if baseline is None or len(baseline["step"]) == 0:
        raise SystemExit("baseline chunks all missing or empty")
    new_arm_max = int(new_arm["step"].max())
    baseline_max = int(baseline["step"].max())
    # Right edge = the new arm's most recent step (or attempt-1's, while
    # the new arm hasn't passed it). All three arms get clipped to the
    # same window before smoothing so the curves end at the same x.
    attempt1 = load_traj(ATTEMPT1_CSV)
    attempt1_max = int(attempt1["step"].max()) if attempt1 is not None else 0
    xmax = max(new_arm_max, attempt1_max)
    xmax = min(xmax, baseline_max)

    # Log-x lower bound: snap to the smaller of the two arms' first
    # smoothed point inside the visible window, so the [1..first_step]
    # decade isn't wasted whitespace. Smoothing window is computed on
    # the truncated trajectory (matching plot_panel).
    def _first_smoothed_step_clipped(steps_arr: np.ndarray, x_hi: int) -> int:
        clip = steps_arr[(steps_arr >= 1) & (steps_arr <= x_hi)]
        n = len(clip)
        if n == 0:
            return 1
        w = max(20, min(1000, n // 40))
        idx = min(n - 1, max(0, w - 1))
        return int(clip[idx])
    first_step = max(
        1,
        min(_first_smoothed_step_clipped(new_arm["step"], xmax),
            _first_smoothed_step_clipped(baseline["step"], xmax)),
    )

    traj = []
    for arm in ARMS:
        _label, _color, _ls, _lw, path_or_chunks, is_concat = arm
        d = load_concat_traj(path_or_chunks) if is_concat else load_traj(path_or_chunks)
        traj.append((arm, d))
        label = arm[0]
        if d is None:
            print(f"[plot] SKIP {label}: source(s) not found")
        else:
            n = len(d["step"])
            print(f"[plot] {label}: {n} steps, last≈{int(d['step'][-1])} "
                  f"loss={last_window_mean(d['loss']):.3f} "
                  f"u_t={last_window_mean(d['u_temporal']):.3f} "
                  f"u_b={last_window_mean(d['u_batch']):.3f} "
                  f"r2_rand={last_window_mean(d['r2_random']):.4f} "
                  f"r2_naive={last_window_mean(d['r2_naive']):.4f} "
                  f"auc={last_window_mean(d['auc']):.4f} "
                  f"top1={last_window_mean(d['top1']):.4f} "
                  f"top3={last_window_mean(d['top3']):.4f}")

    print(f"[plot] new-arm max = {new_arm_max}, baseline max = {baseline_max}; "
          f"xmax = min = {xmax}")

    render_figure(traj, logx=True,
                  logy_metrics={"loss", "auc", "top1", "top3"},
                  out_path=OUT_LOG, scale_tag="log–log",
                  xmax=xmax, xmin_log=first_step)

    render_figure(traj, logx=False, logy_metrics=set(),
                  out_path=OUT_LIN, scale_tag="linear", xmax=xmax)


if __name__ == "__main__":
    main()
