"""4-arm comparison plot — 2x2 grid, log-log, high resolution.

Arms (consistent color + thickness across all four panels):
  default HP 30k       — Blue   (was #6)
  MOIRAI HP 30k        — Green  (was #9)
  MOIRAI HP resumed    — Red    (was v1, #10 attempt 1)
  MOIRAI HP fresh      — Orange (was v3, #10 attempt 2, bs256 fresh init)

Caveat: the fresh arm uses bs=256 (vs bs=96 for the other three) to fit
~70% of VRAM; the contrastive loss has cross-batch negatives, so larger
batch raises the absolute loss floor. Do NOT directly compare loss/gap
values between the bs=256 (orange) and bs=96 curves — focus on shape,
plateau point relative to its own start, and τ trajectory.

Panels (log-log throughout):
  TL: Backbone contrastive loss vs step
  TR: Gap (feat-feat − feat-pred) vs step
  BL: Learnable τ trajectory vs step
  BR: τ=0.07 fixed-reference loss vs main loss (PR #107 diagnostic)

The MOIRAI HP resumed curve is stitched onto the MOIRAI HP 30k prefix
on its left side (same data) so its rolling-mean has continuous left
context across the resume boundary. MOIRAI HP 30k is plotted last in
each panel with a dashed style so you can verify the stitch overlap.

Output: plots/full4096_v3_4arm.png at DPI 300.
"""
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
OUT = os.path.join(ROOT, "plots", "full4096_v3_4arm.png")

# Consistent color + lw across all panels.
COL_DEFAULT  = "tab:blue"
COL_MOIRAI   = "tab:green"
COL_RESUMED  = "tab:red"
COL_FRESH    = "tab:orange"
LW           = 1.8
LW_DASHED    = 1.4
ROLL         = 500


ARM_DEFAULT = {
    "label":   "default HP B96 30k",
    "color":   COL_DEFAULT,
    "loss_csv": f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv",
    "tau_csv":  f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv",
}
ARM_MOIRAI = {
    "label":   "MOIRAI HP B96 30k",
    "color":   COL_MOIRAI,
    "loss_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
    "tau_csv":  f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv",
}
ARM_RESUMED = {
    "label":     "MOIRAI HP B96 resumed",
    "color":     COL_RESUMED,
    "loss_csv":  f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL_run1/moirai_hp_FINAL/checkpoints/tiny_full4096_moirai_hp_FINAL_losses.csv",
    "log_path":  f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL_run1/moirai_hp_FINAL/run.log",
    "log_marker": "=== run_full4096_moirai_hp_FINAL: starting ===",
}
ARM_FRESH = {
    "label":     "MOIRAI HP B256",
    "color":     COL_FRESH,
    "loss_csv":  f"{ROOT}/sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/checkpoints/tiny_full4096_moirai_hp_FRESH_losses.csv",
    "log_path":  f"{ROOT}/sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/run.log",
    "log_marker": "=== run_full4096_moirai_hp_FRESH: starting ===",
}


def load_loss(path):
    parts = []
    prev = path + ".prev"
    if os.path.exists(prev):
        try:
            parts.append(pd.read_csv(prev))
        except Exception as e:
            print(f"  warn: skipping unreadable {prev}: {e}", file=sys.stderr)
    if os.path.exists(path):
        parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return (
        df.drop_duplicates(subset="step", keep="last")
          .sort_values("step")
          .reset_index(drop=True)
    )


def load_tau_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["step", "tau"])
    df = pd.read_csv(path).sort_values("step").reset_index(drop=True)
    df["step"] = df["step"].clip(lower=1)
    return df


def extract_tau_from_log(log_path, marker):
    """Extract τ trajectory from the LAST run-section in the log.

    The remote run.log is append-only, so multiple restart attempts (e.g.,
    OOM retries during the fresh-arm bring-up) leave their step counters
    interleaved in the same file. Reading from the FIRST marker mixes
    early failed-run τ values with the current run, producing misleading
    duplicates and gaps. We instead seek to the LAST occurrence of the
    starting marker and only parse from there.
    """
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["step", "tau"])
    with open(log_path) as f:
        lines = f.readlines()
    last_marker_idx = None
    for i, line in enumerate(lines):
        if marker in line:
            last_marker_idx = i
    if last_marker_idx is None:
        return pd.DataFrame(columns=["step", "tau"])
    pat = re.compile(r"^\[\s*(\d+)\].+τ=([0-9.]+)")
    rows = []
    for line in lines[last_marker_idx + 1:]:
        m = pat.match(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))
    if not rows:
        return pd.DataFrame(columns=["step", "tau"])
    return (
        pd.DataFrame(rows, columns=["step", "tau"])
          .drop_duplicates("step", keep="last")
          .sort_values("step")
          .reset_index(drop=True)
    )


def stitched(df_left, df_right, col_left, col_right):
    if len(df_right) == 0:
        if len(df_left) == 0:
            return pd.DataFrame(columns=["step", "value"])
        return df_left[["step", col_left]].rename(columns={col_left: "value"})
    cut = int(df_right["step"].min())
    L = df_left[df_left["step"] < cut][["step", col_left]].rename(columns={col_left: "value"})
    R = df_right[["step", col_right]].rename(columns={col_right: "value"})
    return pd.concat([L, R], ignore_index=True).reset_index(drop=True)


def plot_smoothed(ax, df, col, color, label, lw=LW, dashed=False):
    if len(df) == 0 or col not in df.columns:
        return
    series = df[col]
    if (series <= 0).any():
        # Loss/gap can dip to 0 transiently; clip for log axis without
        # masking the real dynamics.
        series = series.clip(lower=1e-6)
    smooth = series.rolling(ROLL, min_periods=1).mean()
    ax.plot(df["step"], series, color=color, alpha=0.05, linewidth=0.4)
    style = (0, (4, 3)) if dashed else "-"
    ax.plot(df["step"], smooth, color=color, linewidth=lw, linestyle=style,
            label=f"{label} (n={len(df):,})")


def plot_loss_panel(ax, df_default, df_moirai, df_resumed, df_fresh):
    plot_smoothed(ax, df_default, "loss", COL_DEFAULT, ARM_DEFAULT["label"])
    g_full = stitched(df_moirai, df_resumed, "loss", "loss")
    if len(g_full):
        smooth = g_full["value"].rolling(ROLL, min_periods=1).mean()
        ax.plot(g_full["step"], g_full["value"], color=COL_RESUMED, alpha=0.05, linewidth=0.4)
        ax.plot(g_full["step"], smooth, color=COL_RESUMED, linewidth=LW,
                label=f"{ARM_RESUMED['label']} (n={len(df_resumed):,} new + {ARM_MOIRAI['label']} prefix)")
    plot_smoothed(ax, df_moirai, "loss", COL_MOIRAI, ARM_MOIRAI["label"], lw=LW_DASHED, dashed=True)
    plot_smoothed(ax, df_fresh, "loss", COL_FRESH, ARM_FRESH["label"], lw=LW)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("contrastive loss (log)")
    ax.set_title("Backbone contrastive loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_gap_panel(ax, df_default, df_moirai, df_resumed, df_fresh):
    plot_smoothed(ax, df_default, "gap", COL_DEFAULT, ARM_DEFAULT["label"])
    g_full = stitched(df_moirai, df_resumed, "gap", "gap")
    if len(g_full):
        smooth = g_full["value"].rolling(ROLL, min_periods=1).mean()
        ax.plot(g_full["step"], g_full["value"], color=COL_RESUMED, alpha=0.05, linewidth=0.4)
        ax.plot(g_full["step"], smooth, color=COL_RESUMED, linewidth=LW,
                label=f"{ARM_RESUMED['label']} (n={len(df_resumed):,} new + {ARM_MOIRAI['label']} prefix)")
    plot_smoothed(ax, df_moirai, "gap", COL_MOIRAI, ARM_MOIRAI["label"], lw=LW_DASHED, dashed=True)
    plot_smoothed(ax, df_fresh, "gap", COL_FRESH, ARM_FRESH["label"], lw=LW)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("gap (log) — feat-feat − feat-pred")
    ax.set_title("Contrastive gap (higher = better discrimination)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def _anchor_tau_init(df, init_step=1, init_tau=0.07):
    """Prepend (step=init_step, tau=init_tau) so each τ curve is anchored
    at the same starting point on the log-x axis. Skipped if the series
    already has a point at step ≤ init_step (e.g., default/MOIRAI HP
    which include step 0 clipped to 1)."""
    if len(df) == 0:
        return df
    if int(df["step"].iloc[0]) <= init_step:
        return df
    head = pd.DataFrame({"step": [init_step], "tau": [init_tau]})
    return pd.concat([head, df[["step", "tau"]]], ignore_index=True)


def _stitch_tau(df_left, df_right):
    """Same idea as `stitched()` but for τ trajectories: take df_left's
    points for steps strictly before df_right's first step, then append
    df_right. Both must have columns ['step', 'tau']."""
    if len(df_right) == 0:
        return df_left
    if len(df_left) == 0:
        return df_right
    cut = int(df_right["step"].min())
    L = df_left[df_left["step"] < cut][["step", "tau"]]
    R = df_right[["step", "tau"]]
    return pd.concat([L, R], ignore_index=True).reset_index(drop=True)


def plot_tau_panel(ax, df_default_tau, df_moirai_tau, df_resumed_tau, df_fresh_tau):
    # Resumed run continued from MOIRAI HP 30k's step-30000 state, so its
    # τ history for [0, 30k] IS the MOIRAI HP 30k τ trajectory. Stitch them
    # into one continuous red curve so the [1, 30k] range isn't drawn as
    # a flat line from the (1, 0.07) anchor to the first observed point at
    # step 30100.
    df_resumed_tau = _stitch_tau(df_moirai_tau, df_resumed_tau)
    df_fresh_tau   = _anchor_tau_init(df_fresh_tau)
    # Plot order matters here: red is stitched with the same MOIRAI HP B96
    # 30k data on its [1, 30k] prefix, so plotting red AFTER green would
    # hide green entirely. Draw red first (back), then green dashed (on
    # top) so the dashes show through over the [1, 30k] overlap segment.
    if len(df_default_tau):
        ax.plot(df_default_tau["step"], df_default_tau["tau"], color=COL_DEFAULT, linewidth=LW,
                marker="o", markersize=4,
                label=f"{ARM_DEFAULT['label']} (n={len(df_default_tau)} pts; reconstructed)")
    if len(df_resumed_tau):
        ax.plot(df_resumed_tau["step"], df_resumed_tau["tau"], color=COL_RESUMED, linewidth=LW,
                label=f"{ARM_RESUMED['label']} (n={len(df_resumed_tau):,} pts; MOIRAI HP 30k prefix)")
    if len(df_moirai_tau):
        # Bigger dash pattern (8 on, 4 off) at thicker lw so the green is
        # visibly distinct from the solid red underneath in the [1, 30k]
        # overlap region.
        ax.plot(df_moirai_tau["step"], df_moirai_tau["tau"], color=COL_MOIRAI, linewidth=2.2,
                linestyle=(0, (8, 4)),
                label=f"{ARM_MOIRAI['label']} (n={len(df_moirai_tau):,} pts)")
    if len(df_fresh_tau):
        ax.plot(df_fresh_tau["step"], df_fresh_tau["tau"], color=COL_FRESH, linewidth=LW,
                label=f"{ARM_FRESH['label']} (n={len(df_fresh_tau):,} pts; init-anchored)")
    ax.scatter([1], [0.07], color="black", marker="^", s=70, zorder=5,
               label="init τ=0.07 at step 0 (anchored at step 1 for log-x)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("τ (log)")
    ax.set_title("Learnable τ trajectory")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))


def plot_tau_ref_panel(ax, df_default, df_moirai, df_resumed, df_fresh):
    """Main learnable-τ loss vs τ=0.07 fixed-reference loss.

    Main losses use the arm color; τ=0.07 refs use distinct visible colors
    (dark purple for resumed-ref, brown for fresh-ref) so the dashed
    overlay is clearly visible even when it sits close to its main curve.
    """
    REF_RESUMED_COLOR = "tab:purple"
    REF_FRESH_COLOR   = "tab:brown"
    main = stitched(df_moirai, df_resumed, "loss", "loss")
    if len(main):
        smooth = main["value"].rolling(ROLL, min_periods=1).mean()
        ax.plot(main["step"], main["value"], color=COL_RESUMED, alpha=0.05, linewidth=0.4)
        ax.plot(main["step"], smooth, color=COL_RESUMED, linewidth=LW,
                label=f"{ARM_RESUMED['label']} main loss")
    if len(df_resumed) and "loss_tau_ref" in df_resumed.columns:
        ref = stitched(df_default, df_resumed, "loss", "loss_tau_ref")
        if len(ref):
            smooth = ref["value"].rolling(ROLL, min_periods=1).mean()
            ax.plot(ref["step"], ref["value"], color=REF_RESUMED_COLOR, alpha=0.04, linewidth=0.3)
            ax.plot(ref["step"], smooth, color=REF_RESUMED_COLOR, linewidth=LW,
                    linestyle=(0, (4, 3)),
                    label=f"{ARM_RESUMED['label']} τ=0.07 ref (default-HP prefix ∪ loss_tau_ref)")
    if len(df_fresh):
        smooth = df_fresh["loss"].rolling(ROLL, min_periods=1).mean()
        ax.plot(df_fresh["step"], df_fresh["loss"], color=COL_FRESH, alpha=0.05, linewidth=0.4)
        ax.plot(df_fresh["step"], smooth, color=COL_FRESH, linewidth=LW,
                label=f"{ARM_FRESH['label']} main loss")
        if "loss_tau_ref" in df_fresh.columns:
            smooth_ref = df_fresh["loss_tau_ref"].rolling(ROLL, min_periods=1).mean()
            ax.plot(df_fresh["step"], df_fresh["loss_tau_ref"],
                    color=REF_FRESH_COLOR, alpha=0.04, linewidth=0.3)
            ax.plot(df_fresh["step"], smooth_ref, color=REF_FRESH_COLOR, linewidth=LW,
                    linestyle=(0, (4, 3)),
                    label=f"{ARM_FRESH['label']} τ=0.07 ref (loss_tau_ref column)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel("contrastive loss (log)")
    ax.set_title("τ=0.07 fixed reference vs main learnable-τ loss (PR #107)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def main():
    df_default = load_loss(ARM_DEFAULT["loss_csv"])
    df_moirai  = load_loss(ARM_MOIRAI["loss_csv"])
    df_resumed = load_loss(ARM_RESUMED["loss_csv"])
    df_fresh   = load_loss(ARM_FRESH["loss_csv"])
    df_default_tau = load_tau_csv(ARM_DEFAULT["tau_csv"])
    df_moirai_tau  = load_tau_csv(ARM_MOIRAI["tau_csv"])
    df_resumed_tau = extract_tau_from_log(ARM_RESUMED["log_path"], ARM_RESUMED["log_marker"])
    df_fresh_tau   = extract_tau_from_log(ARM_FRESH["log_path"], ARM_FRESH["log_marker"])

    last_step = max(
        int(df_resumed["step"].max()) if len(df_resumed) else 0,
        int(df_fresh["step"].max())   if len(df_fresh) else 0,
        30000,
    )
    # x-axis lower bound at 1 so the τ-init scatter (step=1) and the
    # init-anchored τ curves are visible. An earlier setting of 10
    # cut off every curve's starting point under the log-x.
    XLIM = (1, last_step + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    plot_loss_panel(   axes[0, 0], df_default, df_moirai, df_resumed, df_fresh)
    plot_gap_panel(    axes[0, 1], df_default, df_moirai, df_resumed, df_fresh)
    plot_tau_panel(    axes[1, 0], df_default_tau, df_moirai_tau, df_resumed_tau, df_fresh_tau)
    plot_tau_ref_panel(axes[1, 1], df_default, df_moirai, df_resumed, df_fresh)
    for ax in axes.flat:
        ax.set_xlim(*XLIM)
    axes[1, 0].set_xlabel("step (log)")
    axes[1, 1].set_xlabel("step (log)")

    fig.suptitle(
        f"Full-4096 4-arm comparison — log-log "
        f"[resumed step={int(df_resumed['step'].max()) if len(df_resumed) else 0:,}, "
        f"fresh step={int(df_fresh['step'].max()) if len(df_fresh) else 0:,}]",
        fontsize=14,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"  default HP 30k    : n={len(df_default):,}")
    print(f"  MOIRAI HP 30k     : n={len(df_moirai):,}")
    print(f"  MOIRAI HP resumed : n={len(df_resumed):,}, last step "
          f"{int(df_resumed['step'].max()) if len(df_resumed) else 0:,}")
    print(f"  MOIRAI HP fresh   : n={len(df_fresh):,}, last step "
          f"{int(df_fresh['step'].max()) if len(df_fresh) else 0:,}, "
          f"has loss_tau_ref={'loss_tau_ref' in df_fresh.columns if len(df_fresh) else False}")


if __name__ == "__main__":
    main()
