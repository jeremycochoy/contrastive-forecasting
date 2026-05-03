"""4-panel gap-only plot — produces TWO files: one in linear x/y, one in
log-log. Same arms and color scheme as the main 4-arm plot.

Gaps (all four are meaningful in our contrastive setup):
  TL: gap_ff_fp = ff − fp           (directional: future-aware forecast)
  TR: gap_ff_cb = ff − cross_batch  (training signal — the contrastive separation we optimize)
  BL: gap_ff_tp = ff − tp           (model value above raw temporal coherence)
  BR: gap_tp_cb = tp − cross_batch  (data alone — intrinsic temporal coherence, no model)

Outputs:
  plots/full4096_gaps_only.png         (linear x/y)
  plots/full4096_gaps_only_loglog.png  (log x/y; gap clipped at 1e-3 for negative/zero values)
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
OUT_LINEAR = os.path.join(ROOT, "plots", "full4096_gaps_only.png")
OUT_LOGLOG = os.path.join(ROOT, "plots", "full4096_gaps_only_loglog.png")

COL_DEFAULT  = "tab:blue"
COL_MOIRAI   = "tab:green"
COL_RESUMED  = "tab:red"
COL_FRESH    = "tab:orange"
LW           = 1.8
LW_DASHED    = 2.2
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

GAPS = [
    # Order = importance: training signal, then temporal-arrow detection,
    # then model uplift over a data-only baseline, then data-only sanity
    # check (model-independent). Arrow in title:
    # ↑ higher-is-better, ↓ lower-is-better, → should-be-same-across-runs.
    ("gap_ff_cb", "↑ gap_ff_cb = ff − cross_batch   (cross-series discrimination)"),
    ("gap_ff_fp", "↑ gap_ff_fp = ff − fp   (temporal discrimination)"),
    ("gap_ff_tp", "↑ gap_ff_tp = ff − tp   (naive discrimination)"),
    ("gap_tp_cb", "→ gap_tp_cb = tp − cross_batch   (self-temporal vs cross series)"),
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
    if "ff" in df.columns:
        if "fp" in df.columns:
            df["gap_ff_fp"] = df["ff"] - df["fp"]
        if "cross_batch" in df.columns:
            df["gap_ff_cb"] = df["ff"] - df["cross_batch"]
        if "tp" in df.columns:
            df["gap_ff_tp"] = df["ff"] - df["tp"]
            if "cross_batch" in df.columns:
                df["gap_tp_cb"] = df["tp"] - df["cross_batch"]
    return df


def plot_gap(ax, dfs, col, title, log=False):
    # Plot order: default (blue) first, then resumed (red, solid),
    # then MOIRAI 30k (green dashed) LAST so its dashes are visible
    # over the red curve in the [1, 30k] overlap region (the resumed
    # arm stitches the MOIRAI HP 30k prefix on its left side, so the
    # two curves carry identical data there). Fresh (orange) plotted
    # in the original position so the prefix stack is unaffected.
    arm_order_idx = {
        "default HP B96 30k":      0,
        "MOIRAI HP B96 resumed":   1,
        "MOIRAI HP B256":          2,
        "MOIRAI HP B96 30k":       3,  # last → on top
    }
    ordered = sorted(dfs, key=lambda pair: arm_order_idx.get(pair[0]["label"], 99))
    for arm, df in ordered:
        if len(df) == 0 or col not in df.columns:
            continue
        series = df[col]
        if log:
            # On log scale, clip non-positive values so the smoothing/render
            # doesn't crash. Values < 1e-3 are essentially noise floor.
            series = series.clip(lower=1e-3)
        smooth = series.rolling(ROLL, min_periods=1).mean()
        ax.plot(df["step"], series, color=arm["color"], alpha=0.05, linewidth=0.4)
        ax.plot(df["step"], smooth, color=arm["color"], linewidth=arm["lw"],
                linestyle=arm["style"],
                label=f"{arm['label']} (n={len(df):,})")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("gap")
    ax.grid(True, which="both" if log else "major", alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def render(dfs, log, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True)
    flat = axes.flat
    for ax, (col, title) in zip(flat, GAPS):
        plot_gap(ax, dfs, col, title, log=log)
    axes[1, 0].set_xlabel("step" + (" (log)" if log else ""))
    axes[1, 1].set_xlabel("step" + (" (log)" if log else ""))
    fig.suptitle(
        "Various contrastive gaps — "
        + ("log-log" if log else "linear x/y")
        + ", smoothed with rolling mean of 500.",
        fontsize=13,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    dfs = [(arm, load_arm(arm)) for arm in ARMS]
    render(dfs, log=False, out_path=OUT_LINEAR)
    render(dfs, log=True,  out_path=OUT_LOGLOG)
    for arm, df in dfs:
        print(f"  {arm['label']:<26}: n={len(df):,}")


if __name__ == "__main__":
    main()
