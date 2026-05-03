"""Various-gaps plot — 8-panel 4x2 grid showing the contrastive similarity
metrics and their meaningful pairwise gaps for the 4-arm comparison.

Raw similarity metrics (computed by `src.models.compute_metrics`):
  ff           — forecast vs future        (positive pair, want HIGH)
  fp           — forecast vs past          (should be lower than ff)
  tp           — past   vs future          (data's intrinsic temporal coherence)
  cross_batch  — forecast vs other-series' future (cross-batch negative, want LOW)

Derived gaps (positive = better discrimination):
  gap_ff_fp = ff − fp           (directional forecast: future-aware vs past-aware)
  gap_ff_cb = ff − cross_batch  (model contrastive separation, the training signal)
  gap_ff_tp = ff − tp           (value model adds beyond raw temporal coherence)
  gap_tp_cb = tp − cross_batch  (intrinsic temporal coherence in the data alone)

All eight panels rendered in LINEAR x/y (not log-log), as requested. Same
arm color/style scheme as the main 4-arm plot.

Output: plots/full4096_gaps.png at DPI 300.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
OUT = os.path.join(ROOT, "plots", "full4096_gaps.png")

COL_DEFAULT  = "tab:blue"
COL_MOIRAI   = "tab:green"
COL_RESUMED  = "tab:red"
COL_FRESH    = "tab:orange"
LW           = 1.8
LW_DASHED    = 2.0
ROLL         = 500


ARMS = [
    {
        "label":   "default HP B96 30k",
        "color":   COL_DEFAULT,
        "style":   "solid",
        "lw":      LW,
        "csv":     f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv",
    },
    {
        "label":   "MOIRAI HP B96 30k",
        "color":   COL_MOIRAI,
        "style":   (0, (8, 4)),
        "lw":      LW_DASHED,
        "csv":     f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
    },
    {
        "label":   "MOIRAI HP B96 resumed",
        "color":   COL_RESUMED,
        "style":   "solid",
        "lw":      LW,
        "csv":     f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL_run1/moirai_hp_FINAL/checkpoints/tiny_full4096_moirai_hp_FINAL_losses.csv",
        # The resumed CSV starts at step 30001 because the trainer appends
        # to a CSV file whose first 30k rows we did NOT push (we only
        # pushed the 30k checkpoint). For the gap plot we stitch on the
        # MOIRAI HP 30k prefix so the curve has continuous left context.
        "prefix_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
    },
    {
        "label":   "MOIRAI HP B256",
        "color":   COL_FRESH,
        "style":   "solid",
        "lw":      LW,
        "csv":     f"{ROOT}/sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/checkpoints/tiny_full4096_moirai_hp_FRESH_losses.csv",
    },
]


def load_csv(path):
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


def load_arm(arm):
    df = load_csv(arm["csv"])
    prefix = arm.get("prefix_csv")
    if prefix and len(df):
        df_prefix = load_csv(prefix)
        if len(df_prefix):
            cut = int(df["step"].min())
            df_prefix = df_prefix[df_prefix["step"] < cut]
            df = pd.concat([df_prefix, df], ignore_index=True)
            df = df.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)
    # Derived gaps
    if "ff" in df.columns:
        if "fp" in df.columns:
            df["gap_ff_fp"] = df["ff"] - df["fp"]
        if "cross_batch" in df.columns:
            df["gap_ff_cb"] = df["ff"] - df["cross_batch"]
        if "tp" in df.columns:
            df["gap_ff_tp"] = df["ff"] - df["tp"]
            df["gap_tp_cb"] = df["tp"] - df["cross_batch"] if "cross_batch" in df.columns else None
    return df


def plot_metric(ax, dfs, col, title, ylabel):
    for arm, df in dfs:
        if len(df) == 0 or col not in df.columns:
            continue
        series = df[col]
        smooth = series.rolling(ROLL, min_periods=1).mean()
        ax.plot(df["step"], series, color=arm["color"], alpha=0.05, linewidth=0.4)
        ax.plot(df["step"], smooth, color=arm["color"], linewidth=arm["lw"],
                linestyle=arm["style"],
                label=f"{arm['label']} (n={len(df):,})")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)


def main():
    dfs = [(arm, load_arm(arm)) for arm in ARMS]

    fig, axes = plt.subplots(4, 2, figsize=(15, 18), sharex=True)

    # Left column: raw similarities. Right column: derived gaps.
    plot_metric(axes[0, 0], dfs, "ff",          "ff — forecast vs future (positive)",       "similarity")
    plot_metric(axes[0, 1], dfs, "gap_ff_fp",   "gap_ff_fp = ff − fp (directional)",        "gap")
    plot_metric(axes[1, 0], dfs, "fp",          "fp — forecast vs past",                    "similarity")
    plot_metric(axes[1, 1], dfs, "gap_ff_cb",   "gap_ff_cb = ff − cross_batch (training signal)", "gap")
    plot_metric(axes[2, 0], dfs, "tp",          "tp — past vs future (data temporal coherence)", "similarity")
    plot_metric(axes[2, 1], dfs, "gap_ff_tp",   "gap_ff_tp = ff − tp (value beyond raw temporal)", "gap")
    plot_metric(axes[3, 0], dfs, "cross_batch", "cross_batch — forecast vs other-series future (negative)", "similarity")
    plot_metric(axes[3, 1], dfs, "gap_tp_cb",   "gap_tp_cb = tp − cross_batch (data alone, no model)", "gap")

    axes[3, 0].set_xlabel("step")
    axes[3, 1].set_xlabel("step")

    fig.suptitle(
        "Various contrastive gaps — raw similarities (left) + derived gaps (right). "
        "Linear x/y. Smoothed with rolling mean of 500.",
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT}")
    for arm, df in dfs:
        cols = ", ".join(c for c in ("ff", "fp", "tp", "cross_batch") if c in df.columns)
        print(f"  {arm['label']:<26}: n={len(df):,}  [{cols}]")


if __name__ == "__main__":
    main()
