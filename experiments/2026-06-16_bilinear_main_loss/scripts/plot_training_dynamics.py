#!/usr/bin/env python3
"""#350 — training-dynamics log-log panels, the bilinear-W arm (solid) against
the #348 τ-scaled-dot-product + CPC baseline (dashed). `loss_tau_ref` is the
CPC-free, τ=0.07 contrastive reference computed identically for both arms, so it
is directly comparable regardless of each arm's training objective. Reads the
two backbones' losses CSVs. Writes plots/training_dynamics.png.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
E348 = f"{W}/2026-06-15_no_encoder_redo/runs"
E350 = f"{W}/2026-06-16_bilinear_main_loss/runs"
BLUE, RED = "#1f77b4", "#d62728"
RUNS = {
    "bilinear W + CPC (this work)": (f"{E350}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear_losses.csv", BLUE, "-"),
    "τ-dot product + CPC (#348)":   (f"{E348}/bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_cpc_losses.csv", RED, "--"),
}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "training_dynamics.png")
SMOOTH = 25
START_STEP = 100
PANELS = [
    ("loss_tau_ref", "contrastive reference loss (norm-InfoNCE τ=0.07)  (↓)", lambda v: v),
    ("cpc_aux",      "CPC InfoNCE term value", lambda v: v),
    ("r2_naive",     "1 − R²_naive  (↓)", lambda v: 1 - v),
    ("auc",          "1 − retrieval AUC  (↓)", lambda v: 1 - v),
]


def load(path):
    if not os.path.exists(path):
        return None
    d = {}
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                d.setdefault(k, []).append(float(v) if v not in ("", None) else float("nan"))
            except (ValueError, TypeError):
                d.setdefault(k, []).append(float("nan"))
    return d


def smooth(y, w):
    out, run = [], []
    for v in y:
        run.append(v)
        if len(run) > w:
            run.pop(0)
        win = [x for x in run if x == x]
        out.append(sum(win) / len(win) if win else float("nan"))
    return out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, title, tf) in zip(axes.flat, PANELS):
        for lab, (path, c, ls) in RUNS.items():
            d = load(path)
            if not d or col not in d or "step" not in d:
                continue
            y = [tf(v) for v in d[col]]
            sm = smooth(y, SMOOTH)
            step = d["step"]
            xs = [s for s, v in zip(step, sm) if v == v and v > 0 and s >= START_STEP]
            ys = [v for s, v in zip(step, sm) if v == v and v > 0 and s >= START_STEP]
            if xs:
                ax.plot(xs, ys, color=c, ls=ls, lw=1.6, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Training dynamics (log-log; solid = bilinear W, dashed = τ baseline)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
