"""Training loss curves for v10, v11c, v12 vs the v7-base (pre-JEPA) baseline.

Two panels:
  left  — linear y, full 0–50k window
  right — log y / log x, zoomed to the early divergence window (500–50k)

v11c is the concatenation of two CSVs (the pre-reboot 0–5400 segment and the
post-reboot 5000–50000 continuation). v10/v12/baseline are single CSVs.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT = Path("/home/jupyter/contrastive-forecasting/checkpoints")

# (label, color, lw, csv_paths_in_order)
ARMS = [
    ("baseline (v7 base) — 6L enc + 6L fcst, legacy conv, fp32",
     "tab:gray",   1.6, ["enc_fcst_dk09_hsl_b256_fp32_50k_losses.csv"]),
    ("v10 — JEPA (6L enc + 1L fcst), GRU enc, LEGACY conv, fp32",
     "tab:blue",   2.2, ["enc_fcst_v10_jepa_enc6_fcst1_50k_losses.csv"]),
    ("v11c — JEPA, GRU enc, NEW conv, fp32",
     "tab:purple", 2.2, ["enc_fcst_v11c_jepa_newconv_fp32_50k_losses.csv",
                         "enc_fcst_v11c_cont_from5k_50k_losses.csv"]),
    ("v12 — JEPA, residual_silu enc, NEW conv, fp32",
     "tab:green",  2.2, ["enc_fcst_v12_jepa_newconv_residualsilu_fp32_50k_losses.csv"]),
]


def load_concat(paths):
    steps_all, loss_all = [], []
    seen_max_step = 0
    for name in paths:
        p = CKPT / name
        if not p.exists():
            print(f"  SKIP missing {name}")
            continue
        steps, losses = [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    s = int(row["step"]); l = float(row["loss"])
                except (KeyError, ValueError):
                    continue
                # only include steps strictly past what we already have, so the
                # post-reboot segment doesn't overwrite the pre-reboot tail
                if s > seen_max_step:
                    steps.append(s); losses.append(l)
        if steps:
            steps_all.extend(steps); loss_all.extend(losses)
            seen_max_step = max(steps_all)
    return np.array(steps_all), np.array(loss_all)


def ema(y, alpha=0.02):
    out = np.empty_like(y, dtype=float)
    if len(y) == 0:
        return out
    m = float(y[0])
    for i, v in enumerate(y):
        m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

for label, color, lw, paths in ARMS:
    s, l = load_concat(paths)
    if len(s) == 0:
        continue
    ax1.plot(s, l,      color=color, lw=lw * 0.3, alpha=0.18)
    ax1.plot(s, ema(l), color=color, lw=lw,      label=label)
    ax2.plot(s, ema(l), color=color, lw=lw,      label=label)
    print(f"{label[:60]:<60} steps {int(s.min())}-{int(s.max())}  "
          f"final loss={l[-1]:.3f}")

ax1.set_xlabel("step"); ax1.set_ylabel("loss")
ax1.set_title("Linear, full 0–50k")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 50000)
ax1.set_ylim(1.0, 5.5)
ax1.legend(loc="upper right", fontsize=10)

ax2.set_xlabel("step (log)"); ax2.set_ylabel("loss (log)")
ax2.set_title("log / log, 500–50k")
ax2.grid(True, which="both", alpha=0.3)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlim(500, 50000)
ax2.set_ylim(1.0, 5.0)

plt.suptitle(
    "Training contrastive loss — v10 / v11c / v12 vs pre-JEPA baseline",
    fontsize=13, y=1.00,
)
plt.tight_layout()

out = Path(__file__).resolve().parent.parent / "plots" / "v10_v11c_v12_vs_baseline_loss.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
