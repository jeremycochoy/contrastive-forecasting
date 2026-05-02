"""Compare the two full-4096 30k-step learnable-τ runs:
- #6 (default HP):  lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999)  — DONE 30k steps
- #9 (MOIRAI HP):   lr=1e-3, weight_decay=0.10, betas=(0.9, 0.98)    — partial (in flight)

Top panel: backbone contrastive loss vs step (log-log). 100-step rolling mean.
Bottom panel: learnable τ vs step (log-x, linear-y for narrow τ range).

τ data is loaded from `<sync_arm_dir>/tau_trajectory.csv` (canonical CSVs
maintained alongside each run's other artifacts; columns step, tau,
log_inv_tau, source). For #6 those rows came from periodic .pth model
state extraction (the run.log was wiped during a fresh-instance relaunch
incident); for #9 they came from the live run.log per-step grep.

Both x-axes are forced to span the same range (1 → 30000) so the data
lines up vertically across panels.
"""
import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting"
OUT_PNG = os.path.join(ROOT, "plots", "full4096_default_vs_moirai_hp.png")

ARMS = [
    ("#6 default HP",
     "tab:blue",
     f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/checkpoints/tiny_realonly_full4096_learnable_tau_losses.csv",
     f"{ROOT}/sync_realonly_full4096_learnable_tau/learnable/tau_trajectory.csv"),
    ("#9 MOIRAI HP",
     "tab:red",
     f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/checkpoints/tiny_realonly_full4096_moirai_hp_losses.csv",
     f"{ROOT}/sync_realonly_full4096_moirai_hp/moirai_hp/tau_trajectory.csv"),
]

ROLL = 100


def load_loss(path):
    """Load losses CSV. Merge .prev rotation if present, dedupe by step."""
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
    return df.drop_duplicates(subset="step", keep="last").sort_values("step").reset_index(drop=True)


def load_tau_csv(path):
    """Load (step, τ) series from a tau_trajectory.csv (step, tau, log_inv_tau, source)."""
    if not os.path.exists(path):
        return [], []
    df = pd.read_csv(path)
    df = df.sort_values("step").reset_index(drop=True)
    # Anchor the init step=0 at step=1 so the log-x axis can render it.
    df["step"] = df["step"].clip(lower=1)
    return df["step"].tolist(), df["tau"].tolist()


fig, (ax_loss, ax_tau) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# Both panels span the same x range so the curves line up vertically.
XLIM = (1, 30000)

# --- Backbone loss panel (log-log) ---
for label, color, csv_path, log_path in ARMS:
    df = load_loss(csv_path)
    if len(df) == 0:
        print(f"  skip empty: {csv_path}", file=sys.stderr)
        continue
    smooth = df["loss"].rolling(ROLL, min_periods=1).mean()
    ax_loss.plot(df["step"], df["loss"], color=color, alpha=0.10, linewidth=0.5)
    last = df["loss"].iloc[-1]
    ax_loss.plot(df["step"], smooth, color=color, linewidth=1.7,
                 label=f"{label} (n={len(df):,}, last={last:.3f})")
ax_loss.set_xscale("log")
ax_loss.set_yscale("log")
ax_loss.set_ylabel("contrastive loss (log)")
ax_loss.set_title("Backbone contrastive loss — full-4096, 30k step budget — default HP vs MOIRAI HP")
ax_loss.grid(True, which="both", alpha=0.25)
ax_loss.legend(loc="best", fontsize=10)
ax_loss.set_xlim(*XLIM)

# --- τ trajectory panel (log-x, linear-y; τ range too narrow for log-y) ---
xs6, ys6 = load_tau_csv(ARMS[0][3])
ax_tau.plot(xs6, ys6, color="tab:blue", linewidth=1.7, marker="o", markersize=5,
            label=f"#6 default HP   (n={len(xs6)} pts from tau_trajectory.csv, last τ={ys6[-1]:.4f})")

xs9, ys9 = load_tau_csv(ARMS[1][3])
ax_tau.plot(xs9, ys9, color="tab:red", linewidth=1.4,
            label=f"#9 MOIRAI HP    (n={len(xs9):,} pts from tau_trajectory.csv, last τ={ys9[-1]:.4f}, partial)")

# Mark init point shared by both arms.
ax_tau.scatter([1], [0.07], color="black", marker="^", s=70, zorder=5,
               label="init τ=0.07 at step 0 (anchored at step 1 for log-x)")

ax_tau.set_xscale("log")
ax_tau.set_xlabel("step (log)")
ax_tau.set_ylabel("τ")
ax_tau.set_title("Learnable τ trajectory — same data + arch, only optimizer HP differs")
ax_tau.grid(True, which="both", alpha=0.25)
ax_tau.set_xlim(*XLIM)
ax_tau.legend(loc="best", fontsize=10)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
print(f"Wrote {OUT_PNG}")
print(f"  #6 loss n={len(load_loss(ARMS[0][2])):,} (DONE), τ pts={len(xs6)}")
print(f"  #9 loss n={len(load_loss(ARMS[1][2])):,} (partial), τ pts={len(xs9):,}")
