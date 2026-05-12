"""Multi-metric comparison of all v10/v11 runs from the 2026-05-12 session.

Plots six metrics in a 2x3 grid:
  loss (log y)             | loss_tau_ref (linear, τ=0.07 ref)  | AUC
  top1                     | u_temporal (dim usage time)        | u_batch (dim usage batch)

Runs included
-------------
Fresh (start from step 0):
  v10   = baseline JEPA 6L+1L, LEGACY conv, pure fp32        (BLUE solid, thick)
  v11   = same, NEW conv + all-fp16 (FAILED)                 (RED solid)
  v11b  = same, NEW conv + pemb-fp32 + rest-fp16 (FAILED)    (ORANGE solid)
  v11c  = same, NEW conv + PURE fp32 (in-progress)           (PURPLE solid, thick)

Resume from v6's best_loss (precision-saga arms):
  v10d  = fp32 control                                       (light-blue dashed)
  v10c  = bf16+fp32loss (FAILED, bf16 cos-sim wall)          (gray dashed)
  v10e  = hybrid granular-fp32 (FAILED)                      (slategray dashed)
  v10g  = attn-bf16+ffn-bf16+deprecated-conv-fp32 (FAILED)   (brown dashed)
  v10f  = ffn-bf16 only                                      (teal dashed)
  v10h  = attn-fp16+ffn-fp16                                 (green dashed)
  v10i  = all-fp16 (residual+attn+ffn)                       (darkgreen dashed)
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT = Path("/home/jupyter/contrastive-forecasting/checkpoints")

# (filename, label, color, linestyle, linewidth)
RUNS = [
    # --- Fresh runs (start from step 0) ---
    ("enc_fcst_v10_jepa_enc6_fcst1_50k_losses.csv",
     "v10 fresh — LEGACY conv, fp32  (baseline)",
     "tab:blue",      "-",  2.5),
    ("enc_fcst_v11_jepa_newconv_allfp16_FAILED_5k_losses.csv",
     "v11 fresh — NEW conv, all-fp16  (FAILED)",
     "tab:red",       "-",  1.5),
    ("enc_fcst_v11b_jepa_newconv_pembfp32_FAILED_7600_losses.csv",
     "v11b fresh — NEW conv, pemb-fp32 + rest fp16  (FAILED)",
     "tab:orange",    "-",  1.5),
    ("enc_fcst_v11c_jepa_newconv_fp32_50k_losses.csv",
     "v11c fresh — NEW conv, pure fp32  (in-progress)",
     "tab:purple",    "-",  2.5),
    # --- Resume v6 best_loss (precision-saga arms) ---
    ("enc_fcst_v10d_fp32_resume_v6_15k_losses.csv",
     "v10d resume-v6 — pure fp32 control",
     "lightskyblue",  "--", 1.2),
    ("enc_fcst_v10c_bf16_fp32loss_resume_v6_FAILED_4000_losses.csv",
     "v10c resume-v6 — bf16+fp32-loss  (FAILED)",
     "gray",          "--", 1.2),
    ("enc_fcst_v10e_hybrid_resume_v6_FAILED_4500_losses.csv",
     "v10e resume-v6 — hybrid granular-fp32  (FAILED)",
     "slategray",     "--", 1.2),
    ("enc_fcst_v10g_attnffnbf16_deprconv_resume_v6_FAILED_4000_losses.csv",
     "v10g resume-v6 — attn-bf16 + ffn-bf16  (FAILED)",
     "saddlebrown",   "--", 1.2),
    ("enc_fcst_v10f_ffn_bf16_resume_v6_15k_losses.csv",
     "v10f resume-v6 — ffn-bf16 only",
     "tab:cyan",      "--", 1.2),
    ("enc_fcst_v10h_attnffnfp16_deprconv_resume_v6_15k_losses.csv",
     "v10h resume-v6 — attn-fp16 + ffn-fp16",
     "tab:green",     "--", 1.2),
    ("enc_fcst_v10i_all_fp16_resume_v6_15k_losses.csv",
     "v10i resume-v6 — all-fp16 body",
     "darkgreen",     "--", 1.2),
]

METRICS = [
    # (csv_col, title, log_y, ylim)
    ("loss",          "Contrastive loss (log y)",         True,  (0.5, 30.0)),
    ("loss_tau_ref",  "loss_tau_ref (τ=0.07 ref)",        False, (0.0, 4.5)),
    ("auc",           "Retrieval AUC (4×128)",            False, (0.5, 1.005)),
    ("top1",          "Retrieval top-1",                  False, (0.0, 1.05)),
    ("u_temporal",    "Dim usage — temporal",             False, (0.0, 0.65)),
    ("u_batch",       "Dim usage — batch",                False, (0.0, 0.65)),
]


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


def load(path):
    steps, cols = [], {row[0]: [] for row in METRICS}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                steps.append(int(row["step"]))
            except (KeyError, ValueError):
                continue
            for m, *_unused in METRICS:
                try:
                    cols[m].append(float(row[m]))
                except (KeyError, ValueError):
                    cols[m].append(float("nan"))
    return np.array(steps), {k: np.array(v) for k, v in cols.items()}


fig, axes = plt.subplots(2, 3, figsize=(22, 12))
axes = axes.flatten()

xmax_global = 0
for fname, label, color, ls, lw in RUNS:
    path = CKPT / fname
    if not path.exists():
        print(f"  SKIP missing {path.name}")
        continue
    s, cols = load(path)
    if len(s) == 0:
        continue
    xmax_global = max(xmax_global, int(s.max()))
    for ax, (m, _title, _log, _ylim) in zip(axes, METRICS):
        y = cols[m]
        if np.all(np.isnan(y)):
            continue
        ax.plot(s, y,      color=color, ls=ls, lw=lw * 0.3, alpha=0.15)
        ax.plot(s, ema(y), color=color, ls=ls, lw=lw,        label=label)

for ax, (m, title, log, ylim) in zip(axes, METRICS):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(0, xmax_global)
    ax.set_ylim(*ylim)

# Single legend below the grid, 3 columns
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", bbox_to_anchor=(0.5, -0.01),
           ncol=3, fontsize=9, frameon=True)

plt.suptitle(
    "v10 / v11* — full multi-metric comparison  (2026-05-12 session: "
    "conv-placement (legacy vs new sa_input) + precision saga (fp32/fp16/bf16))",
    fontsize=13, y=0.995,
)
plt.tight_layout(rect=(0, 0.09, 1, 0.99))

out = Path(__file__).resolve().parent.parent / "plots" / "v10_v11_all_runs_multi_metric.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)

# Also print the table of last-step values per run for the PR body
print("\n=== last-step values per run ===")
print(f"{'run':<58} {'step':>6} {'loss':>7} {'auc':>6} {'top1':>6} {'u_t':>5} {'u_b':>5}")
for fname, label, *_ in RUNS:
    path = CKPT / fname
    if not path.exists():
        continue
    s, cols = load(path)
    if len(s) == 0:
        continue
    i = -1
    print(f"{label[:58]:<58} {int(s[i]):>6d} "
          f"{cols['loss'][i]:>7.3f} {cols['auc'][i]:>6.3f} {cols['top1'][i]:>6.3f} "
          f"{cols['u_temporal'][i]:>5.3f} {cols['u_batch'][i]:>5.3f}")
