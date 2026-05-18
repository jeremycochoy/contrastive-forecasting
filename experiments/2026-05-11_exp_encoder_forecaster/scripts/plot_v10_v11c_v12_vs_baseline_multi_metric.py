"""Multi-metric log/log comparison — same 2x3 panel layout as before, focused
on the four arms: pre-JEPA baseline, v10, v11c, v12.

Panels: loss | loss_tau_ref | (1-AUC) | (1-top1) | u_temporal | u_batch.
All log y; log x ∈ [500, 50000]. (1-AUC) and (1-top1) make the saturated
retrieval metrics readable.

v11c is the concatenation of the pre-reboot 0–5400 segment and the post-reboot
5000–50000 continuation.
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# In-repo training-loss CSVs (pulled from elisa
# /home/jupyter/contrastive-forecasting/checkpoints, byte-verified).
CKPT = Path(__file__).resolve().parent.parent / "results" / "train_losses"

ARMS = [
    ("baseline (v7 base) — 6L+6L, legacy conv, fp32",
     "tab:gray",   1.6, ["enc_fcst_dk09_hsl_b256_fp32_50k_losses.csv"]),
    ("v10 — JEPA (6L+1L), GRU enc, LEGACY conv, fp32",
     "tab:blue",   2.2, ["enc_fcst_v10_jepa_enc6_fcst1_50k_losses.csv"]),
    ("v11c — JEPA, GRU enc, NEW conv, fp32",
     "tab:purple", 2.2, ["enc_fcst_v11c_jepa_newconv_fp32_50k_losses.csv",
                         "enc_fcst_v11c_cont_from5k_50k_losses.csv"]),
    ("v12 — JEPA, residual_silu enc, NEW conv, fp32",
     "tab:green",  2.2, ["enc_fcst_v12_jepa_newconv_residualsilu_fp32_50k_losses.csv"]),
]

METRICS = [
    # (csv_col, title, transform, ylim)
    ("loss",          "Contrastive loss",                     "log",        (1.0, 30.0)),
    ("loss_tau_ref",  "loss_tau_ref (τ=0.07 ref)",            "log",        (1.0, 12.0)),
    ("auc",           "1 − Retrieval AUC (4×128)",            "1minus_log", (1e-7, 1.0)),
    ("top1",          "1 − Retrieval top-1",                  "1minus_log", (1e-7, 1.0)),
    ("u_temporal",    "Dim usage — temporal",                 "log",        (3e-3, 0.7)),
    ("u_batch",       "Dim usage — batch",                    "log",        (3e-3, 0.7)),
]

XMIN, XMAX = 500, 50000


def load_concat(paths):
    seen_max_step = 0
    steps_all, cols_all = [], {row[0]: [] for row in METRICS}
    for name in paths:
        p = CKPT / name
        if not p.exists():
            print(f"  SKIP missing {name}")
            continue
        with open(p) as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    s = int(row["step"])
                except (KeyError, ValueError):
                    continue
                if s <= seen_max_step:
                    continue
                steps_all.append(s)
                for m, *_ in METRICS:
                    try:
                        cols_all[m].append(float(row[m]))
                    except (KeyError, ValueError):
                        cols_all[m].append(float("nan"))
        if steps_all:
            seen_max_step = max(steps_all)
    return np.array(steps_all), {k: np.array(v) for k, v in cols_all.items()}


def ema(y, alpha=0.02):
    out = np.empty_like(y, dtype=float)
    if len(y) == 0:
        return out
    m = float(y[0]) if not math.isnan(y[0]) else 0.0
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
    for ax, (m, _t, transform, _yl) in zip(axes, METRICS):
        y = cols[m][keep]
        if np.all(np.isnan(y)):
            continue
        if transform == "1minus_log":
            y_plot = np.clip(1.0 - y, 1e-8, None)
        else:
            y_plot = y
        ax.plot(s_k, y_plot,      color=color, lw=lw * 0.3, alpha=0.18)
        ax.plot(s_k, ema(y_plot), color=color, lw=lw,        label=label)

for ax, (m, title, transform, ylim) in zip(axes, METRICS):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("step (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale("log")
    ax.set_xlim(XMIN, XMAX)
    if transform in ("log", "1minus_log"):
        ax.set_yscale("log")
    ax.set_ylim(*ylim)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=10, frameon=True)

plt.suptitle(
    "v10 / v11c / v12 vs pre-JEPA baseline — multi-metric log/log "
    f"(x∈[{XMIN}, {XMAX}], y log on loss / loss_tau_ref / (1-AUC) / (1-top1) / dim-usage)",
    fontsize=13, y=0.995,
)
plt.tight_layout(rect=(0, 0.06, 1, 0.99))

out = Path(__file__).resolve().parent.parent / "plots" / "v10_v11c_v12_vs_baseline_multi_metric.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)

# Last-step table for the PR body / report
print("\n=== last-step values per arm ===")
print(f"{'arm':<55} {'step':>6} {'loss':>7} {'1-AUC':>10} {'1-top1':>10} {'u_t':>6} {'u_b':>6}")
for label, color, lw, paths in ARMS:
    s, cols = load_concat(paths)
    if len(s) == 0:
        continue
    i = -1
    auc_minus = max(1.0 - cols['auc'][i], 1e-8)
    top1_minus = max(1.0 - cols['top1'][i], 1e-8)
    print(f"{label[:55]:<55} {int(s[i]):>6d} "
          f"{cols['loss'][i]:>7.3f} {auc_minus:>10.2e} {top1_minus:>10.2e} "
          f"{cols['u_temporal'][i]:>6.3f} {cols['u_batch'][i]:>6.3f}")
