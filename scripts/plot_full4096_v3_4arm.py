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
    "label":   "default HP 30k",
    "color":   COL_DEFAULT,
    "loss_csv": f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv",
    "tau_csv":  f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv",
}
ARM_MOIRAI = {
    "label":   "MOIRAI HP 30k",
    "color":   COL_MOIRAI,
    "loss_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
    "tau_csv":  f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv",
}
ARM_RESUMED = {
    "label":     "MOIRAI HP resumed",
    "color":     COL_RESUMED,
    "loss_csv":  f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL_run1/moirai_hp_FINAL/checkpoints/tiny_full4096_moirai_hp_FINAL_losses.csv",
    "log_path":  f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL_run1/moirai_hp_FINAL/run.log",
    "log_marker": "=== run_full4096_moirai_hp_FINAL: starting ===",
}
ARM_FRESH = {
    "label":     "MOIRAI HP fresh",
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
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["step", "tau"])
    rows, found, pat = [], False, re.compile(r"^\[\s*(\d+)\].+τ=([0-9.]+)")
    with open(log_path) as f:
        for line in f:
            if not found:
                if marker in line:
                    found = True
                continue
            m = pat.match(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    if not rows:
        return pd.DataFrame(columns=["step", "tau"])
    return (
        pd.DataFrame(rows, columns=["step", "tau"])
          .drop_duplicates("step")
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
                label=f"{ARM_RESUMED['label']} (n={len(df_resumed):,} new + MOIRAI HP 30k prefix)")
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
                label=f"{ARM_RESUMED['label']} (n={len(df_resumed):,} new + MOIRAI HP 30k prefix)")
    plot_smoothed(ax, df_moirai, "gap", COL_MOIRAI, ARM_MOIRAI["label"], lw=LW_DASHED, dashed=True)
    plot_smoothed(ax, df_fresh, "gap", COL_FRESH, ARM_FRESH["label"], lw=LW)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("gap (log) — feat-feat − feat-pred")
    ax.set_title("Contrastive gap (higher = better discrimination)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_tau_panel(ax, df_default_tau, df_moirai_tau, df_resumed_tau, df_fresh_tau):
    if len(df_default_tau):
        ax.plot(df_default_tau["step"], df_default_tau["tau"], color=COL_DEFAULT, linewidth=LW,
                marker="o", markersize=4,
                label=f"{ARM_DEFAULT['label']} (n={len(df_default_tau)} pts; reconstructed)")
    if len(df_moirai_tau):
        ax.plot(df_moirai_tau["step"], df_moirai_tau["tau"], color=COL_MOIRAI, linewidth=LW_DASHED,
                linestyle=(0, (4, 3)),
                label=f"{ARM_MOIRAI['label']} (n={len(df_moirai_tau):,} pts)")
    if len(df_resumed_tau):
        ax.plot(df_resumed_tau["step"], df_resumed_tau["tau"], color=COL_RESUMED, linewidth=LW,
                label=f"{ARM_RESUMED['label']} (n={len(df_resumed_tau):,} pts)")
    if len(df_fresh_tau):
        ax.plot(df_fresh_tau["step"], df_fresh_tau["tau"], color=COL_FRESH, linewidth=LW,
                label=f"{ARM_FRESH['label']} (n={len(df_fresh_tau):,} pts)")
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
    """Main learnable-τ loss vs τ=0.07 fixed-reference loss."""
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
            ax.plot(ref["step"], ref["value"], color=COL_RESUMED, alpha=0.04, linewidth=0.3)
            ax.plot(ref["step"], smooth, color=COL_RESUMED, linewidth=LW_DASHED,
                    linestyle=(0, (3, 2)),
                    label=f"{ARM_RESUMED['label']} τ=0.07 ref (default-HP prefix ∪ loss_tau_ref)")
    if len(df_fresh):
        smooth = df_fresh["loss"].rolling(ROLL, min_periods=1).mean()
        ax.plot(df_fresh["step"], df_fresh["loss"], color=COL_FRESH, alpha=0.05, linewidth=0.4)
        ax.plot(df_fresh["step"], smooth, color=COL_FRESH, linewidth=LW,
                label=f"{ARM_FRESH['label']} main loss (bs256)")
        if "loss_tau_ref" in df_fresh.columns:
            smooth_ref = df_fresh["loss_tau_ref"].rolling(ROLL, min_periods=1).mean()
            ax.plot(df_fresh["step"], smooth_ref, color=COL_FRESH, linewidth=LW_DASHED,
                    linestyle=(0, (3, 2)),
                    label=f"{ARM_FRESH['label']} τ=0.07 ref")
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
