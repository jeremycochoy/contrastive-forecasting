#!/usr/bin/env python3
"""#344 — training dynamics: do the CPC arms' contrastive trajectory and
late-training stability differ from their baselines, and how does the CPC
term itself evolve? Reads the four backbone losses.csv (two baselines, two
CPC arms) and writes plots/training_dynamics.png.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = "/home/jupyter/workspaces/contrastive-forecasting/experiments"
RUNS = {
    "enc3 baseline": (f"{W}/2026-06-10_stopgrad_positive/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_losses.csv", "C0", "--"),
    "enc3 + CPC":    (f"{W}/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv", "C0", "-"),
    "enc6 baseline": (f"{W}/2026-06-11_stopgrad_capacity/runs/bb_allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024_losses.csv", "C3", "--"),
    "enc6 + CPC":    (f"{W}/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc6_sgpos_qk_aon_b1024_cpc_losses.csv", "C3", "-"),
}
OUT = os.path.join(os.path.dirname(__file__), "..", "plots", "training_dynamics.png")


def load(path):
    if not os.path.exists(path):
        return None
    cols = {}
    for r in csv.DictReader(open(path)):
        for k, v in r.items():
            try:
                cols.setdefault(k, []).append(float(v) if v not in ("", None) else float("nan"))
            except (ValueError, TypeError):
                cols.setdefault(k, []).append(float("nan"))
    return cols


def smooth(xs, ys, k=25):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - k)
        win = [y for y in ys[lo:i + 1] if y == y]
        out.append(sum(win) / len(win) if win else float("nan"))
    return xs, out


def panel(ax, col, title, ylabel, logy=False, only_cpc=False):
    for label, (path, color, ls) in RUNS.items():
        if only_cpc and "CPC" not in label:
            continue
        d = load(path)
        if not d or col not in d:
            continue
        step, y = smooth(d["step"], d[col])
        ax.plot(step, y, color=color, ls=ls, lw=1.6, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    panel(axs[0, 0], "loss_tau_ref",
          "Contrastive reference loss (normalized InfoNCE, τ=0.07)\n— comparable across runs, CPC-term-free",
          "loss_tau_ref")
    panel(axs[0, 1], "cpc_aux",
          "CPC InfoNCE auxiliary term (the added loss)",
          "cpc_aux", only_cpc=True)
    panel(axs[1, 0], "u_batch",
          "U_batch — batch-wise embedding dimensions in use",
          "U_batch")
    panel(axs[1, 1], "gap_ratio",
          "gap_ratio = (1−ff)/(1−fp) — forecast-vs-future gap (lower better)",
          "gap_ratio")
    fig.suptitle("#344 CPC InfoNCE auxiliary — training dynamics (CPC solid, baseline dashed)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
