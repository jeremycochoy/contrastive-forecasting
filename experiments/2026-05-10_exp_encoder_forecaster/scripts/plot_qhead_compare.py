#!/usr/bin/env python3
"""Compare in-progress q-head training vs R9_E13 reference baseline q-head.

Both runs use the R9_E13 recipe (xfmr 12L causal quantile head, e_then_f
input, cosine schedule + 2k warmup), but on different backbones:

  * R9_E13 reference   — backbone-beta_167k (blue, tab:blue / #1f77b4)
  * our in-progress    — enc_fcst_tau_0_10_50k_FINAL.pth (red, tab:red / #d62728)

Reads per-step q-head training-loss CSVs from the MAIN checkout, never
writes there. Produces two PNGs under the worktree experiment dir:

  * qhead_compare.png         — linear x; log y on loss panels (primary)
  * qhead_compare_loglog.png  — log x; log y on loss panels (companion)

Columns compared are the INTERSECTION of both CSV headers (so the
script works whether the head trainer logs raw `loss` only or also
emits per-quantile `loss_q*` rows). Smoothing: moving average with
`min(1000, len // 40)` window, lower-bounded by 1 for very short
trajectories (our run is at ~200 steps when this is first re-rendered).

X-axis upper bound is clipped to `min(max_step_ours, max_step_ref)`
plus a small margin, so the comparison is at the SAME step count, not
stretched. The reference's full 60k trajectory is therefore drawn only
up to our current step.

Re-runnable. `Agg` backend, no GPU.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Script lives in <worktree>/experiments/2026-05-10_exp_encoder_forecaster/scripts.
WORKTREE = Path(__file__).resolve().parents[3]
EXP_DIR = WORKTREE / "experiments/2026-05-10_exp_encoder_forecaster"
OUT_LIN = EXP_DIR / "plots/qhead_compare.png"
OUT_LOG = EXP_DIR / "plots/qhead_compare_loglog.png"
OUT_LIN.parent.mkdir(parents=True, exist_ok=True)

# CSVs are written by the live runs in the MAIN checkout; read-only.
MAIN = Path("/home/jupyter/contrastive-forecasting")

# Reference: R9_E13 baseline q-head on backbone-beta_167k (60k recipe).
REF_CSV = MAIN / "sync_qhead_beta_rd9/checkpoints/R9_E13_xfmr12L_quant_moirai_cosine_e_then_f_60k_losses.csv"
REF_LABEL = "R9_E13 ref (backbone-beta_167k, 60k recipe)"

# Our in-progress arm: same head recipe, on enc_fcst_tau_0_10_50k_FINAL backbone.
OURS_CSV = MAIN / "checkpoints/enc_fcst_qhead_xfmr12L_quant_30k_losses.csv"
OURS_LABEL = "ours (enc+fcst backbone, 30k recipe, in progress)"

# Same convention as plot_progress.py / plot_square_diagram.py.
C_REF = "#1f77b4"   # tab:blue
C_OURS = "#d62728"  # tab:red


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing-mean moving average. Returns input if w<=1 or shorter than w."""
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _aligned_smooth(steps: np.ndarray, values: np.ndarray, w: int):
    sm = smooth(values, w)
    if len(sm) == len(values):
        return steps, sm
    x = steps[len(steps) - len(sm):]
    return x, sm


def window(n: int) -> int:
    """Smoothing window: min(1000, n // 40), floor 1."""
    return max(1, min(1000, n // 40))


def last_window_mean(arr: np.ndarray, n: int = 200) -> float:
    if len(arr) == 0:
        return float("nan")
    a = arr[-n:]
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else float("nan")


def load_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]] | None:
    """Read CSV; return (column_list, {col: float-array}). None if missing/empty.

    `step` is converted to int via float (handles "1.0"). All other columns
    are coerced to float; non-numeric rows are dropped.
    """
    if not path.exists():
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "step" not in cols:
        return None
    out: dict[str, list[float]] = {c: [] for c in cols}
    for r in rows:
        try:
            for c in cols:
                out[c].append(float(r[c]))
        except (ValueError, TypeError):
            # skip header dupes / partial rows
            continue
    arrs = {c: np.array(out[c], dtype=np.float64) for c in cols}
    # int-ify step
    arrs["step"] = arrs["step"].astype(np.int64)
    return cols, arrs


def is_loss_col(name: str) -> bool:
    """Loss panels get log-y. Treat any column with 'loss' in its name as loss-like."""
    return "loss" in name.lower()


def plot_panel(ax, key, ours, ref, *, logx, logy, xlim):
    xmin, xmax = xlim
    for arr, color, label in (
        (ref, C_REF, REF_LABEL),
        (ours, C_OURS, OURS_LABEL),
    ):
        if arr is None or key not in arr or len(arr[key]) == 0:
            continue
        steps = arr["step"]
        y = arr[key]
        mask = np.isfinite(y)
        # Loss panels with log-y require positive values.
        if logy:
            mask = mask & (y > 0)
        if not mask.any():
            continue
        # Clip to the visible step window BEFORE choosing the smoothing
        # window — otherwise a long reference trajectory (e.g. 60k rows)
        # picks w=1000 and its first smoothed point lands past xmax,
        # silently dropping the ref curve. Sizing w on the in-window
        # slice keeps both arms comparable at the same visible scale.
        win_mask = mask & (steps >= xmin) & (steps <= xmax)
        if not win_mask.any():
            continue
        x_steps = steps[win_mask]
        y = y[win_mask]
        n = len(x_steps)
        w = window(n)
        x_sm, y_sm = _aligned_smooth(x_steps, y, w)
        if len(x_sm) == 0:
            continue
        ax.plot(x_sm, y_sm, color=color, linewidth=1.6, label=label)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_title(key, fontsize=10)
    ax.set_xlabel("step" + (" (log)" if logx else ""))
    ax.set_ylabel(key + ("  (log-y)" if logy else ""))
    ax.grid(alpha=0.3, which="both", ls=":")


def render_figure(ours, ref, shared_metric_cols, *, logx, out_path, scale_tag, xmax, xmin_log=1):
    xmin = xmin_log if logx else 0
    xlim = (xmin, xmax)
    n_panels = len(shared_metric_cols)
    if n_panels == 0:
        print(f"[plot] no shared numeric metric columns to plot; skipping {out_path}")
        return
    # Grid: up to 3 columns; rows as needed.
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False)
    for i, key in enumerate(shared_metric_cols):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        plot_panel(
            ax, key, ours, ref,
            logx=logx, logy=is_loss_col(key), xlim=xlim,
        )
    # Hide unused panels.
    for j in range(n_panels, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    # Shared legend at top.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            ncol=len(handles), fontsize=9, frameon=False,
            bbox_to_anchor=(0.5, 0.985),
        )
    ours_max = int(ours["step"].max()) if ours is not None and len(ours["step"]) else 0
    fig.suptitle(
        f"q-head training: ours (step ~{ours_max:,}) vs R9_E13 reference  "
        f"({scale_tag}, x clipped to ≈{xmax:,})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ref_load = load_csv(REF_CSV)
    ours_load = load_csv(OURS_CSV)
    if ref_load is None:
        raise SystemExit(f"reference CSV missing or empty: {REF_CSV}")
    if ours_load is None:
        raise SystemExit(f"ours CSV missing or empty: {OURS_CSV}")
    ref_cols, ref = ref_load
    ours_cols, ours = ours_load

    # Intersect columns; metrics = shared columns minus housekeeping.
    shared = [c for c in ours_cols if c in ref_cols]
    # Drop housekeeping columns from the panel set; we still plot vs `step`.
    drop = {"step", "hf_rows_consumed", "epoch", "wallclock", "lr"}
    metric_cols = [c for c in shared if c not in drop]
    print(f"[plot] ref cols    : {ref_cols}")
    print(f"[plot] ours cols   : {ours_cols}")
    print(f"[plot] shared      : {shared}")
    print(f"[plot] metric cols : {metric_cols}")

    ours_max = int(ours["step"].max()) if len(ours["step"]) else 0
    ref_max = int(ref["step"].max()) if len(ref["step"]) else 0
    print(f"[plot] ours max step = {ours_max}, ref max step = {ref_max}")
    if ours_max == 0:
        raise SystemExit("ours CSV has no rows yet")
    max_step = min(ours_max, ref_max)
    # Tiny margin so curves don't run flush against the right edge.
    # Use 2% of max_step, floor 5.
    margin = max(5, int(0.02 * max_step))
    xmax = max_step + margin

    # Log-x lower bound: snap to the smoothed first step (mirrors plot_progress).
    def _first_smoothed_step(steps_arr: np.ndarray) -> int:
        n = len(steps_arr)
        w = window(n)
        idx = min(n - 1, max(0, w - 1))
        return int(steps_arr[idx]) if n else 1
    first_step = max(1, min(_first_smoothed_step(ours["step"]),
                            _first_smoothed_step(ref["step"])))

    # Summary log line — same shape as plot_progress.
    for label, arr in ((REF_LABEL, ref), (OURS_LABEL, ours)):
        n = len(arr["step"])
        if n == 0:
            print(f"[plot] {label}: empty")
            continue
        last = int(arr["step"][-1])
        pieces = [f"{label}: {n} steps, last={last}"]
        for c in metric_cols:
            pieces.append(f"{c}={last_window_mean(arr[c]):.4f}")
        print("[plot] " + ", ".join(pieces))

    # Skip the first 500 steps in BOTH plots: that window is dominated by
    # the warmup ramp where both arms drop fast on the head's still-random
    # init, and it visually compresses the post-warmup regime where the
    # actual gap lives. The log-x lower bound is also snapped here so the
    # two arms share an identical visible window.
    XMIN_SKIP = 500
    render_figure(
        ours, ref, metric_cols,
        logx=True, out_path=OUT_LIN, scale_tag="log–log, step ≥ 500",
        xmax=xmax, xmin_log=XMIN_SKIP,
    )
    render_figure(
        ours, ref, metric_cols,
        logx=True, out_path=OUT_LOG, scale_tag="log–log, step ≥ 500",
        xmax=xmax, xmin_log=XMIN_SKIP,
    )


if __name__ == "__main__":
    main()
