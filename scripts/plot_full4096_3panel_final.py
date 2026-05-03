"""3-panel final comparison plot for the full-4096 30k-step learnable-τ runs.

Compares:
- #6 (default HP):  lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999)
- #9 (MOIRAI HP):   lr=1e-3, weight_decay=0.10, betas=(0.9, 0.98)

Both runs have completed STAGE H (backbone 30k + qhead 30k). All three series
are full per-step on #9 (live run.log preserved). #6's backbone & qhead loss
CSVs are full (sync_loop preserved them) but the τ trace was reconstructed
from periodic checkpoint snapshots after the May 2 run.log overwrite incident.

Three vertically-stacked panels, all log-log, high-resolution:
  Panel 1: Backbone contrastive loss vs step
  Panel 2: Learnable τ vs step
  Panel 3: Recovery / quantile head loss vs step

The τ y-axis on log scale spans ≈0.04 → 0.08 (half a decade), enough to make
the divergence between the two HP regimes legible on its own scale.

Output: plots/full4096_3panel_final.png at DPI 300.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
OUT_PNG = os.path.join(ROOT, "plots", "full4096_3panel_final.png")
OUT_PNG_EXP = os.path.join(
    ROOT, "experiments", "2026-05-02_exp_realonly_full4096_moirai_hp", "plots",
    "full4096_3panel_final.png",
)

ARMS = [
    {
        "label": "#6 default HP",
        "color": "tab:blue",
        "bb_loss":  f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv",
        "tau":      f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv",
        "head_loss": f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/R1q_realonly_full4096_learnable_tau_losses.csv",
        "tau_note": "reconstructed from .pth snapshots (run.log overwritten May 2)",
    },
    {
        "label": "#9 MOIRAI HP",
        "color": "tab:red",
        "bb_loss":  f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
        "tau":      f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv",
        "head_loss": f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/R1q_realonly_full4096_moirai_hp_losses.csv",
        "tau_note": "live run.log per-step",
    },
]

ROLL = 100  # rolling-mean window in steps for raw loss smoothing
XLIM = (1, 30000)


def load_loss(path):
    """Load a per-step losses CSV (columns: step, loss, hf_rows_consumed).

    Merge any rotated `.prev` companion (the sync_loop's one-deep rotation)
    so we don't drop history across a rotation boundary; dedupe by step.
    """
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
        return pd.DataFrame(columns=["step", "loss"])
    df = pd.concat(parts, ignore_index=True)
    return (
        df.drop_duplicates(subset="step", keep="last")
          .sort_values("step")
          .reset_index(drop=True)
    )


def load_tau(path):
    if not os.path.exists(path):
        return [], []
    df = pd.read_csv(path)
    df = df.sort_values("step").reset_index(drop=True)
    # Anchor step=0 init at step=1 so log-x can render it.
    df["step"] = df["step"].clip(lower=1)
    return df["step"].tolist(), df["tau"].tolist()


def plot_loss_panel(ax, arms, key, title, ylabel, last_fmt=":.3f"):
    """Generic loss panel: faded raw + 100-step rolling mean per arm."""
    for arm in arms:
        df = load_loss(arm[key])
        if len(df) == 0:
            print(f"  skip empty: {arm[key]}", file=sys.stderr)
            continue
        smooth = df["loss"].rolling(ROLL, min_periods=1).mean()
        last = df["loss"].iloc[-1]
        ax.plot(df["step"], df["loss"], color=arm["color"], alpha=0.10, linewidth=0.5)
        ax.plot(
            df["step"], smooth,
            color=arm["color"], linewidth=1.8,
            label=f"{arm['label']} (n={len(df):,}, last={format(last, last_fmt[1:])})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(*XLIM)


def plot_tau_panel(ax, arms):
    for arm in arms:
        xs, ys = load_tau(arm["tau"])
        if not xs:
            print(f"  skip empty tau: {arm['tau']}", file=sys.stderr)
            continue
        # Sparse #6 series (~14 pts from snapshots) gets markers; dense #9
        # (~272 pts from grep) gets a thin line.
        if len(xs) < 50:
            ax.plot(
                xs, ys, color=arm["color"], linewidth=1.8, marker="o", markersize=5,
                label=f"{arm['label']} (n={len(xs)} pts; {arm['tau_note']}; last τ={ys[-1]:.4f})",
            )
        else:
            ax.plot(
                xs, ys, color=arm["color"], linewidth=1.4,
                label=f"{arm['label']} (n={len(xs):,} pts; {arm['tau_note']}; last τ={ys[-1]:.4f})",
            )
    ax.scatter(
        [1], [0.07], color="black", marker="^", s=70, zorder=5,
        label="init τ=0.07 at step 0 (anchored at step 1 for log-x)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("τ (log)")
    ax.set_title("Learnable τ trajectory — same data + arch, only optimizer HP differs")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim(*XLIM)
    # Make the narrow τ range readable on log-y: explicit ticks at the values
    # both arms actually visit (≈0.04 → 0.08).
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))


def main():
    fig, (ax_bb, ax_tau, ax_head) = plt.subplots(3, 1, figsize=(11, 14), sharex=True)

    plot_loss_panel(
        ax_bb, ARMS, "bb_loss",
        title="Backbone contrastive loss — full-4096, 30k steps",
        ylabel="contrastive loss (log)",
    )

    plot_tau_panel(ax_tau, ARMS)

    plot_loss_panel(
        ax_head, ARMS, "head_loss",
        title="Recovery head (R1q quantile) loss — frozen-backbone, 30k steps",
        ylabel="quantile head loss (log)",
    )
    ax_head.set_xlabel("step (log)")

    fig.suptitle(
        "Full-4096 30k-step comparison — default HP vs MOIRAI HP (learnable τ, identical data + arch)",
        fontsize=13,
    )
    fig.tight_layout()

    for out in (OUT_PNG, OUT_PNG_EXP):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Wrote {out}")

    # Status print to stdout for quick verification.
    for arm in ARMS:
        bb = load_loss(arm["bb_loss"])
        head = load_loss(arm["head_loss"])
        xs, _ = load_tau(arm["tau"])
        print(
            f"  {arm['label']}: bb_loss n={len(bb):,}, head_loss n={len(head):,}, τ pts={len(xs):,}"
        )


if __name__ == "__main__":
    main()
