"""3-panel plot covering the full-4096 30k baselines (#6, #9) AND the
in-flight FINAL retrain (#10) which resumes from #9's 30k.

Panels (all log-log):
  1. Backbone contrastive loss vs step — three curves (#6, #9, #10).
     #10 picks up at step 30001 and continues to wherever the live
     training has reached. Plus the τ=0.07 fixed-reference loss for
     #10 only (PR #107 added this column to the trainer's CSV; #6 and
     #9 do not have it).
  2. Learnable τ vs step — three curves. #6 sparse (snapshot
     reconstruction), #9 dense from per-step grep, #10 dense from the
     #10 section of the run.log only (the log was pre-seeded with
     #9's, we filter to lines after the FINAL "starting" marker).
  3. Recovery / quantile-head loss is intentionally omitted here — the
     prior 3-panel report (PR #102) covered it for #6 vs #9 at 30k,
     and #10's qhead is trained AFTER backbone completes, so we have
     no qhead data for #10 yet. Instead, panel 3 is the τ=0.07 fixed
     reference loss alone — comparable across runs in principle but
     only #10 has it logged so we plot just one curve and call it
     out as a #10-only diagnostic.

Output: plots/full4096_with_final_3panel.png at DPI 300.

Reads:
  sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv
  sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv
  sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv
  sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv
  sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/checkpoints/tiny_full4096_moirai_hp_FINAL_losses.csv
  sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/run.log   (filtered to #10 section)
"""
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
OUT = os.path.join(ROOT, "plots", "full4096_with_final_3panel.png")

ARMS_30K = [
    {
        "label": "#6 default HP (30k)",
        "color": "tab:blue",
        "loss_csv": f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv",
        "tau_csv": f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv",
    },
    {
        "label": "#9 MOIRAI HP (30k)",
        "color": "tab:red",
        "loss_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
        "tau_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv",
    },
]

ARM_FINAL = {
    "label": "#10 FINAL (resumed from #9 30k)",
    "color": "tab:green",
    "loss_csv": f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/checkpoints/tiny_full4096_moirai_hp_FINAL_losses.csv",
    "log_path": f"{ROOT}/sync_realonly_full4096_moirai_hp_FINAL/moirai_hp_FINAL/run.log",
    "log_marker": "=== run_full4096_moirai_hp_FINAL: starting ===",
}

ROLL = 100


def load_loss(path):
    """Per-step losses CSV; merge .prev rotation if present, dedupe."""
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
    """Parse #10's τ trajectory from the post-marker section of the log.

    The log file is pre-seeded with #9's content; we skip every line up
    to (and including) the FINAL `starting` marker, then grep
    `[<step>] ... τ=<value>` lines.
    """
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=["step", "tau"])
    rows = []
    pat = re.compile(r"^\[\s*(\d+)\].+τ=([0-9.]+)")
    found = False
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
    df = pd.DataFrame(rows, columns=["step", "tau"]).drop_duplicates("step")
    return df.sort_values("step").reset_index(drop=True)


def plot_loss_panel(ax):
    # 30k baselines
    for arm in ARMS_30K:
        df = load_loss(arm["loss_csv"])
        if len(df) == 0:
            continue
        smooth = df["loss"].rolling(ROLL, min_periods=1).mean()
        last = df["loss"].iloc[-1]
        ax.plot(df["step"], df["loss"], color=arm["color"], alpha=0.10, linewidth=0.4)
        ax.plot(
            df["step"], smooth, color=arm["color"], linewidth=1.7,
            label=f"{arm['label']}  (n={len(df):,}, last={last:.3f})",
        )
    # FINAL retrain (in flight): plot main loss + the new fixed-τ ref column
    df10 = load_loss(ARM_FINAL["loss_csv"])
    if len(df10) > 0:
        smooth10 = df10["loss"].rolling(ROLL, min_periods=1).mean()
        last10 = df10["loss"].iloc[-1]
        ax.plot(df10["step"], df10["loss"], color=ARM_FINAL["color"], alpha=0.08, linewidth=0.4)
        ax.plot(
            df10["step"], smooth10, color=ARM_FINAL["color"], linewidth=1.7,
            label=f"{ARM_FINAL['label']}  (n={len(df10):,}, last={last10:.3f})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("contrastive loss (log)")
    ax.set_title("Backbone contrastive loss — full-4096, log-log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)


def plot_tau_panel(ax):
    # 30k arms from canonical tau_trajectory.csv
    for arm in ARMS_30K:
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
    # #10 from log
    df10 = extract_final_tau_from_log(ARM_FINAL["log_path"], ARM_FINAL["log_marker"])
    if len(df10) > 0:
        ax.plot(
            df10["step"], df10["tau"], color=ARM_FINAL["color"], linewidth=1.4,
            label=f"{ARM_FINAL['label']}  (n={len(df10):,} pts; per-100-step grep from #10 section)",
        )
    # init point shared by all arms
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


def plot_tau_ref_panel(ax):
    """τ=0.07 fixed-reference loss — only #10 has this column (PR #107).

    Plotted alongside #10's main loss so the divergence is visible on
    a single axis: when learnable τ is below 0.07, the fixed-τ=0.07
    reference is harder (cosine sims spread out more), so loss_tau_ref
    > loss; the gap is a measure of "what τ is buying us".
    """
    df10 = load_loss(ARM_FINAL["loss_csv"])
    if len(df10) == 0 or "loss_tau_ref" not in df10.columns:
        ax.text(0.5, 0.5, "no loss_tau_ref column found in #10 CSV",
                ha="center", va="center", transform=ax.transAxes)
        return
    smooth_main = df10["loss"].rolling(ROLL, min_periods=1).mean()
    smooth_ref = df10["loss_tau_ref"].rolling(ROLL, min_periods=1).mean()
    ax.plot(df10["step"], df10["loss"], color="tab:green", alpha=0.10, linewidth=0.4)
    ax.plot(
        df10["step"], smooth_main, color="tab:green", linewidth=1.7,
        label=f"#10 main loss (learnable τ; last={df10['loss'].iloc[-1]:.3f})",
    )
    ax.plot(df10["step"], df10["loss_tau_ref"], color="tab:purple",
            alpha=0.10, linewidth=0.4)
    ax.plot(
        df10["step"], smooth_ref, color="tab:purple", linewidth=1.7,
        label=f"#10 τ=0.07 fixed reference (no grad; last={df10['loss_tau_ref'].iloc[-1]:.3f})",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("contrastive loss (log)")
    ax.set_xlabel("step (log)")
    ax.set_title("#10 only: τ=0.07 fixed reference loss vs main loss (PR #107 diagnostic)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=9)


def main():
    df10 = load_loss(ARM_FINAL["loss_csv"])
    last_step = int(df10["step"].max()) if len(df10) else 30000
    XLIM = (1, max(last_step + 1, 30000))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 14), sharex=True)
    plot_loss_panel(ax1)
    plot_tau_panel(ax2)
    plot_tau_ref_panel(ax3)
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(*XLIM)

    fig.suptitle(
        f"Full-4096 #6 vs #9 (30k) and #10 in-flight FINAL retrain — log-log "
        f"(#10 step={last_step:,}/498,000 = {last_step/498000*100:.1f}%)",
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Wrote {OUT}")
    print(f"  #10 backbone CSV rows: {len(df10):,}, last step: {last_step:,}")


if __name__ == "__main__":
    main()
