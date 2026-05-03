"""3-panel plot — #6 + #9 (30k baselines) + #10 in-flight FINAL retrain.

The previous version (v1) plotted #10 only on [30k, 165k]. That produces
a smoothing discontinuity at step 30k (rolling mean has no left
context) and makes the curves harder to compare visually.

This version (v2):

- Panel 1 (backbone loss). #10's curve is *stitched* with #9's [0,30k]
  data on its left side, so its rolling mean has continuous left
  context across the resume boundary. Green is drawn FIRST, red on
  top of the [0,30k] segment, so red stays visible. Smoothing window
  bumped 100 → 500 to tame the late-#10 batch-noise (the model is in
  a flatter loss region where per-batch variance is a larger
  fraction of the mean — real, not a plot artifact).

- Panel 2 (τ trajectory). Unchanged.

- Panel 3 (τ=0.07 reference diagnostic, PR #107). Two continuous
  curves spanning [1, 165k]:
    * "main learnable-τ loss"  — red(#9) [0,30k] + green(#10) [30k,]
    * "τ≈0.07 reference loss"  — blue(#6) [0,30k] + purple(#10
       loss_tau_ref column) [30k,]
  We use #6's main loss as the τ≈0.07 reference for [0,30k] because
  #6's τ stays within ~30% of 0.07 throughout that window (descends
  0.07 → 0.047, mostly between 0.05–0.07), the closest available
  proxy. The purple right-half is the actual fixed-τ=0.07 loss
  computed under torch.no_grad() per PR #107. The two halves stitch
  cleanly because at step 30k both arms are seeing similar τ in
  practice.

Output: plots/full4096_with_final_3panel.png at DPI 300.
"""
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
OUT = os.path.join(ROOT, "plots", "full4096_with_final_3panel.png")

ARM6 = {
    "label": "#6 default HP (30k)",
    "color": "tab:blue",
    "loss_csv": f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv",
    "tau_csv":  f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv",
}
ARM9 = {
    "label": "#9 MOIRAI HP (30k)",
    "color": "tab:red",
    "loss_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
    "tau_csv":  f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv",
}
ARM10 = {
    "label": "#10 FINAL (resumed from #9 30k)",
    "color": "tab:green",
    "loss_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/checkpoints/tiny_full4096_moirai_hp_FINAL_losses.csv",
    "log_path": f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/run.log",
    "log_marker": "=== run_full4096_moirai_hp_FINAL: starting ===",
}

ROLL = 500   # uniform smoothing for ALL arms — by using the same window we
             # can verify visually that red(#9) and the [0, 30k] portion of
             # the green-stitched (#9 + #10) curve coincide exactly. If they
             # don't, the stitch / smoothing has a bug. If they do, any
             # divergence at >30k is real model behavior.


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


def extract_final_tau_from_log(log_path, marker):
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


def stitched(df_left, df_right, col_left="loss", col_right="loss"):
    """Stitch two losses CSVs end-to-end. Use df_left[col_left] for steps
    < df_right[step].min(), and df_right[col_right] for steps >= that.
    Returns a DataFrame with columns step, value (no overlap).
    """
    if len(df_right) == 0:
        if len(df_left) == 0:
            return pd.DataFrame(columns=["step", "value"])
        return df_left[["step", col_left]].rename(columns={col_left: "value"})
    cut = int(df_right["step"].min())
    L = df_left[df_left["step"] < cut][["step", col_left]].rename(columns={col_left: "value"})
    R = df_right[["step", col_right]].rename(columns={col_right: "value"})
    return pd.concat([L, R], ignore_index=True).reset_index(drop=True)


def plot_loss_panel(ax, df6, df9, df10):
    # All three curves use the SAME rolling window (ROLL) so visual overlap
    # in the [0, 30k] region is a true equality check on the stitched data.
    # Plot order: blue (#6), green (#10 stitched, solid, full opacity),
    # red (#9, DASHED so green underneath stays visible where they coincide).
    if len(df6):
        sm6 = df6["loss"].rolling(ROLL, min_periods=1).mean()
        ax.plot(df6["step"], df6["loss"], color=ARM6["color"], alpha=0.05, linewidth=0.4)
        ax.plot(
            df6["step"], sm6, color=ARM6["color"], linewidth=1.5,
            label=f"{ARM6['label']}  (n={len(df6):,}, last={df6['loss'].iloc[-1]:.3f})",
        )

    g_full = stitched(df9, df10)
    if len(g_full):
        sm10 = g_full["value"].rolling(ROLL, min_periods=1).mean()
        ax.plot(g_full["step"], g_full["value"], color=ARM10["color"], alpha=0.05, linewidth=0.4)
        ax.plot(
            g_full["step"], sm10, color=ARM10["color"], linewidth=2.2,
            label=f"{ARM10['label']}  (n={len(df10):,} new + #9 prefix; roll={ROLL})",
        )

    if len(df9):
        sm9 = df9["loss"].rolling(ROLL, min_periods=1).mean()
        ax.plot(df9["step"], df9["loss"], color=ARM9["color"], alpha=0.05, linewidth=0.4)
        ax.plot(
            df9["step"], sm9, color=ARM9["color"], linewidth=1.4,
            linestyle=(0, (4, 3)),  # dashed
            label=f"{ARM9['label']}  (n={len(df9):,}, last={df9['loss'].iloc[-1]:.3f}, DASHED so #10 stitch is visible underneath)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("contrastive loss (log)")
    ax.set_title(f"Backbone contrastive loss — log-log, uniform smoothing roll={ROLL}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)


def plot_tau_panel(ax, df10_tau):
    for arm in (ARM6, ARM9):
        df = load_tau_csv(arm["tau_csv"])
        if len(df) == 0:
            continue
        if len(df) < 50:
            ax.plot(
                df["step"], df["tau"], color=arm["color"], linewidth=1.7,
                marker="o", markersize=4,
                label=f"{arm['label']}  (n={len(df)} pts; reconstructed from snapshots)",
            )
        else:
            ax.plot(
                df["step"], df["tau"], color=arm["color"], linewidth=1.4,
                label=f"{arm['label']}  (n={len(df):,} pts; per-step grep)",
            )
    if len(df10_tau):
        ax.plot(
            df10_tau["step"], df10_tau["tau"], color=ARM10["color"], linewidth=1.4,
            label=f"{ARM10['label']}  (n={len(df10_tau):,} pts; per-100-step grep, #10 section only)",
        )
    ax.scatter(
        [1], [0.07], color="black", marker="^", s=70, zorder=5,
        label="init τ=0.07 at step 0 (anchored at step 1 for log-x)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("τ (log)")
    ax.set_title("Learnable τ trajectory — same arch + data, optimizer HP differs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))


def plot_tau_ref_panel(ax, df6, df9, df10):
    """Two continuous curves, both spanning [1, 165k]:

    * "main learnable-τ loss"  : red(#9) ∪ green(#10 main loss)
    * "τ≈0.07 reference loss"  : blue(#6 main loss; τ stays close to 0.07
                                  through [0,30k]) ∪ purple(#10 loss_tau_ref
                                  column from PR #107)

    The gap between the two curves shows whether learnable τ is buying
    anything vs the canonical 0.07.
    """
    # Main loss: #9 + #10's loss column.
    main = stitched(df9, df10, "loss", "loss")
    if len(main):
        smooth = main["value"].rolling(ROLL, min_periods=1).mean()
        ax.plot(main["step"], main["value"], color=ARM10["color"], alpha=0.05, linewidth=0.4)
        ax.plot(
            main["step"], smooth, color=ARM10["color"], linewidth=1.7,
            label=f"main learnable-τ loss  ({ARM9['label'].split()[0]} + {ARM10['label'].split()[0]} stitched)",
        )

    # τ≈0.07 reference: #6's loss (its τ stays close to 0.07) + #10's
    # loss_tau_ref column (actual fixed-τ=0.07 reference, PR #107).
    if len(df10) and "loss_tau_ref" in df10.columns:
        ref = stitched(df6, df10, "loss", "loss_tau_ref")
        if len(ref):
            smooth = ref["value"].rolling(ROLL, min_periods=1).mean()
            ax.plot(ref["step"], ref["value"], color="tab:purple", alpha=0.05, linewidth=0.4)
            ax.plot(
                ref["step"], smooth, color="tab:purple", linewidth=1.7,
                label=f"τ≈0.07 reference  ({ARM6['label'].split()[0]} loss [τ→0.047] ∪ #10 loss_tau_ref [τ=0.07 fixed])",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel("contrastive loss (log)")
    ax.set_title("τ=0.07 reference loss vs main learnable-τ loss (PR #107 diagnostic, extended)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)


def main():
    df6 = load_loss(ARM6["loss_csv"])
    df9 = load_loss(ARM9["loss_csv"])
    df10 = load_loss(ARM10["loss_csv"])
    df10_tau = extract_final_tau_from_log(ARM10["log_path"], ARM10["log_marker"])

    last_step = int(df10["step"].max()) if len(df10) else 30000
    XLIM = (1, max(last_step + 1, 30000))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 14), sharex=True)
    plot_loss_panel(ax1, df6, df9, df10)
    plot_tau_panel(ax2, df10_tau)
    plot_tau_ref_panel(ax3, df6, df9, df10)
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(*XLIM)

    fig.suptitle(
        f"Full-4096 #6 vs #9 (30k) + #10 in-flight FINAL — log-log "
        f"(#10 step={last_step:,}/498,000 = {last_step/498000*100:.1f}%)",
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"  #6  bb_loss n={len(df6):,}")
    print(f"  #9  bb_loss n={len(df9):,}")
    print(f"  #10 bb_loss n={len(df10):,}, last step: {last_step:,}, "
          f"has loss_tau_ref={'loss_tau_ref' in df10.columns}")


if __name__ == "__main__":
    main()
