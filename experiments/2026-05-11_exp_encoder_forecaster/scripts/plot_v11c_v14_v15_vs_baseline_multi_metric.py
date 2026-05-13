"""Multi-metric log/log comparison for the five arms below:

  - baseline (pre-JEPA, 6L+6L, legacy conv, fp32, dk=0.9, τ=0.10)
  - τ=0.10 baseline (tau-sweep, pre-transformer-encoder)
  - v11c (JEPA 6L+1L, NEW conv, dk=0.9 — best so far, MASE 1.388)
  - v14   (JEPA 6L+6L, NEW conv, dk=0.9 — MASE 1.650)
  - v15   (JEPA 6L+4L, NEW conv, dk=0.9 — MASE 1.671)

Panels (2x3): loss | loss_tau_ref | (1-AUC) | (1-top1) | (1-ff)/(1-fp) |
u_temporal & u_batch combined.

x-axis log, starting at step 500. Curves smoothed with EMA(α=0.02), faint
raw underlay.

The "gap" panel plots (1 − ff) / (1 − fp) — the asymmetry between forecaster
matching its own forecast vs the past — not the raw `gap` column. Computed
per-row from ff and fp; div-by-zero is clipped to a small positive floor.

v11c is the concatenation of the pre-reboot 0–5800 segment and the
post-reboot 5001–50000 continuation; segments are joined with `step >
seen_max` so the post-reboot CSV picks up cleanly.
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT = Path("/home/jupyter/contrastive-forecasting/checkpoints")

ARMS = [
    ("baseline (v7 base) — pre-JEPA 6L+6L, legacy conv, fp32, dk=0.9, τ=0.10",
     "tab:gray",   1.6, ["enc_fcst_dk09_hsl_b256_fp32_50k_losses.csv"]),
    ("τ=0.10 baseline (tau-sweep, pre-transformer-encoder)",
     "tab:cyan",   1.6, ["enc_fcst_tau_0_10_50k_losses.csv"]),
    ("v11c — JEPA 6L+1L, NEW conv, dk=0.9 (MASE 1.388)",
     "tab:purple", 2.2, ["enc_fcst_v11c_jepa_newconv_fp32_50k_losses.csv",
                         "enc_fcst_v11c_cont_from5k_50k_losses.csv"]),
    ("v14 — JEPA 6L+6L, NEW conv, dk=0.9 (MASE 1.650)",
     "tab:red",    2.2, ["enc_fcst_v14_jepa_enc6_fcst6_dk09_newconv_fp32_50k_losses.csv"]),
    ("v15 — JEPA 6L+4L, NEW conv, dk=0.9 (MASE 1.671)",
     "tab:orange", 2.2, ["enc_fcst_v15_jepa_enc6_fcst4_dk09_newconv_fp32_50k_losses.csv"]),
]

# (csv_col, title, transform, ylim).  transform "1minus_log" plots 1-y on log y.
# For the combined u_temporal+u_batch panel we list u_temporal here and
# overlay u_batch in the plotting loop.
# "gap_metric" is a synthetic column computed as (1 - ff) / (1 - fp) — the
# value-of-interest "gap" per Jeremy. ff and fp are loaded from CSV and the
# ratio is computed in the plotting loop (with clipping to avoid divide-by-
# zero / negative denominators).
METRICS = [
    ("loss",          "Contrastive loss",                     "log",        (1.2, 8.0)),
    # loss_tau_ref goes negative (down to ~-2.7) → linear scale, not log.
    ("loss_tau_ref",  "loss_tau_ref (τ=0.07 reference)",      "linear",     (-3.0, 12.0)),
    ("auc",           "1 − Retrieval AUC (4×128)",            "1minus_log", (1e-8, 1.0)),
    ("top1",          "1 − Retrieval top-1",                  "1minus_log", (1e-8, 1.0)),
    # (1-ff)/(1-fp) spans ~1e-3 (final v11c → final baseline) to ~0.2 (early)
    # → log y for readable separation. Floor at 1e-4 to keep the axis tame.
    ("gap_metric",    "(1 − ff) / (1 − fp)",                  "log",        (1e-3, 3e-1)),
    ("u_temporal",    "Dim usage — u_temporal (solid) / u_batch (dashed)",
                                                              "log",        (1e-2, 0.8)),
]

# Columns we actually need to load from each CSV. ff/fp drive the synthetic
# gap_metric panel; the raw "gap" column is no longer plotted.
COLS_NEEDED = ["loss", "loss_tau_ref", "ff", "fp", "auc", "top1",
               "u_temporal", "u_batch"]

XMIN, XMAX = 500, 50000


def load_concat(paths):
    seen_max_step = 0
    steps_all = []
    cols_all = {c: [] for c in COLS_NEEDED}
    for name in paths:
        p = CKPT / name
        if not p.exists():
            print(f"  SKIP missing {name}")
            continue
        new_steps = []
        with open(p) as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    s = int(row["step"])
                except (KeyError, ValueError):
                    continue
                if s <= seen_max_step:
                    continue
                new_steps.append(s)
                steps_all.append(s)
                for c in COLS_NEEDED:
                    try:
                        cols_all[c].append(float(row[c]))
                    except (KeyError, ValueError):
                        cols_all[c].append(float("nan"))
        if new_steps:
            seen_max_step = max(seen_max_step, max(new_steps))
    return np.array(steps_all), {k: np.array(v) for k, v in cols_all.items()}


def ema(y, alpha=0.02):
    out = np.empty_like(y, dtype=float)
    if len(y) == 0:
        return out
    # initialise at first finite value
    init = next((float(v) for v in y if not math.isnan(v)), 0.0)
    m = init
    for i, v in enumerate(y):
        if not math.isnan(v):
            m = alpha * v + (1 - alpha) * m
        out[i] = m
    return out


fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.flatten()

for label, color, lw, paths in ARMS:
    s, cols = load_concat(paths)
    if len(s) == 0:
        continue
    keep = s >= XMIN
    s_k = s[keep]
    if len(s_k) == 0:
        continue

    # Synthesize gap_metric = (1 - ff) / (1 - fp). Mask the denominator when
    # |1-fp| is tiny (avoid blow-up); clip the resulting ratio at 1e-4 so
    # log y renders cleanly without -inf.
    ff_arr = cols["ff"][keep]
    fp_arr = cols["fp"][keep]
    denom = np.where(np.abs(1.0 - fp_arr) > 1e-4, 1.0 - fp_arr, np.nan)
    gap_metric = np.clip((1.0 - ff_arr) / denom, 1e-4, None)

    for ax, (m, _t, transform, _yl) in zip(axes, METRICS):
        if m == "gap_metric":
            y = gap_metric
        else:
            y = cols[m][keep]
        if np.all(np.isnan(y)):
            continue
        if transform == "1minus_log":
            y_plot = np.clip(1.0 - y, 1e-8, None)
        else:
            y_plot = y
        # faint raw underlay
        ax.plot(s_k, y_plot, color=color, lw=lw * 0.3, alpha=0.18)
        # EMA solid line — only this one carries the legend handle
        ax.plot(s_k, ema(y_plot), color=color, lw=lw, label=label)

    # Overlay u_batch on the last (u_temporal) panel as dashed
    ax_dim = axes[5]
    yb = cols["u_batch"][keep]
    if not np.all(np.isnan(yb)):
        ax_dim.plot(s_k, yb, color=color, lw=lw * 0.3, alpha=0.18, linestyle="--")
        ax_dim.plot(s_k, ema(yb), color=color, lw=lw, linestyle="--")


for ax, (m, title, transform, ylim) in zip(axes, METRICS):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("step (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    if transform in ("log", "1minus_log"):
        ax.set_yscale("log")
    elif transform == "linear":
        ax.set_yscale("linear")
    ax.set_ylim(*ylim)

# Shared legend at bottom (only solid EMA handles)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=9, frameon=True)

plt.suptitle(
    "v11c / v14 / v15 vs pre-JEPA baseline + tau-sweep τ=0.10 — multi-metric "
    f"log/log (x∈[{XMIN}, {XMAX}], EMA α=0.02; raw faint)",
    fontsize=13, y=0.995,
)
plt.tight_layout(rect=(0, 0.06, 1, 0.99))

out = Path(__file__).resolve().parent.parent / "plots" / "v11c_v14_v15_vs_baseline_multi_metric.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)


# Last-step table for the PR body / report
print("\n=== last-step values per arm ===")
print(f"{'arm':<70} {'step':>6} {'loss':>7} {'loss_tau_ref':>13} "
      f"{'(1-ff)/(1-fp)':>14} "
      f"{'1-AUC':>10} {'1-top1':>10} {'u_t':>6} {'u_b':>6}")
for label, color, lw, paths in ARMS:
    s, cols = load_concat(paths)
    if len(s) == 0:
        continue
    i = -1
    auc_minus = max(1.0 - cols['auc'][i], 1e-8)
    top1_minus = max(1.0 - cols['top1'][i], 1e-8)
    ff_i = cols['ff'][i]
    fp_i = cols['fp'][i]
    if abs(1.0 - fp_i) > 1e-4:
        gap_i = (1.0 - ff_i) / (1.0 - fp_i)
    else:
        gap_i = float('nan')
    print(f"{label[:70]:<70} {int(s[i]):>6d} "
          f"{cols['loss'][i]:>7.3f} {cols['loss_tau_ref'][i]:>13.3f} "
          f"{gap_i:>14.4f} "
          f"{auc_minus:>10.2e} {top1_minus:>10.2e} "
          f"{cols['u_temporal'][i]:>6.3f} {cols['u_batch'][i]:>6.3f}")
