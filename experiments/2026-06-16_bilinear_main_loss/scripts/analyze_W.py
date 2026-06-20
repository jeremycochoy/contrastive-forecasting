#!/usr/bin/env python3
"""#350 — what did the learnable main-loss W become? Scans the bilinear
backbone checkpoints (periodic _Nk, _best_loss, _final) and, for each, reports
how the learned W departs from its (1/τ₀)·I = 10·I init:

  diag_mean    mean of diag(W)  → effective inverse-temperature 1/τ_eff
  tau_eff      1 / diag_mean    → effective temperature (init 0.10)
  offdiag_rms  RMS of the off-diagonal entries (init 0)
  offdiag_frac ||offdiag||_F / ||W||_F   (share of W's energy off the diagonal)
  asymmetry    ||W − Wᵀ||_F / ||W||_F
  dev_from_init ||W − 10·I||_F / ||10·I||_F

A W that stays ≈ 10·I (offdiag_frac ≈ 0, tau_eff ≈ 0.10) means τ was already
near-optimal; a large offdiag_frac / asymmetry means the bilinear is using
structure a scalar τ cannot express. Writes results/W_evolution.csv and
plots/W_evolution.png.
"""
import csv
import glob
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RUNS = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss/runs"
RES = "/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss/results"
PLOT = os.path.join(os.path.dirname(__file__), "..", "plots", "W_evolution.png")
TAG = "bb_allt08_xftrip_nobn_noenc_sgpos_qk_aon_b1024_bilinear"
TAU0 = 0.10


def step_of(path):
    b = os.path.basename(path)
    if "_best_loss" in b:
        return ("best_loss", -1)
    if b.endswith("_final.pth"):
        return ("final", 10 ** 9)
    m = re.search(r"_(\d+)k\.pth$", b)
    return (b, int(m.group(1)) * 1000) if m else (b, 0)


def stats(W):
    H = W.shape[0]
    diag = torch.diagonal(W)
    off = W - torch.diag(diag)
    fro = W.norm().item()
    dm = diag.mean().item()
    return {
        "diag_mean": dm,
        "tau_eff": (1.0 / dm) if dm else float("nan"),
        "offdiag_rms": (off.pow(2).sum() / (H * H - H)).sqrt().item(),
        "offdiag_frac": off.norm().item() / fro if fro else float("nan"),
        "asymmetry": (W - W.t()).norm().item() / fro if fro else float("nan"),
        "dev_from_init": (W - torch.eye(H) / TAU0).norm().item()
        / (torch.eye(H) / TAU0).norm().item(),
    }


def main():
    paths = [p for p in glob.glob(f"{RUNS}/{TAG}_*.pth") if "optimizer" not in p]
    rows = []
    for p in sorted(paths, key=lambda q: step_of(q)[1]):
        name, order = step_of(p)
        sd = torch.load(p, map_location="cpu")
        sd = sd.get("model_state_dict", sd)
        if "main_w.weight" not in sd:
            continue
        s = stats(sd["main_w.weight"].float())
        s["ckpt"] = name
        s["order"] = order
        rows.append(s)
    if not rows:
        print("no bilinear checkpoints with main_w yet")
        return
    os.makedirs(RES, exist_ok=True)
    keys = ["ckpt", "diag_mean", "tau_eff", "offdiag_rms", "offdiag_frac",
            "asymmetry", "dev_from_init"]
    with open(f"{RES}/W_evolution.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    for r in rows:
        print(f"{r['ckpt']:<14} τ_eff={r['tau_eff']:.4f}  "
              f"offdiag_frac={r['offdiag_frac']:.4f}  "
              f"asym={r['asymmetry']:.4f}  dev={r['dev_from_init']:.4f}")

    periodic = [r for r in rows if r["order"] not in (-1, 10 ** 9)]
    if len(periodic) >= 2:
        xs = [r["order"] // 1000 for r in periodic]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(xs, [r["tau_eff"] for r in periodic], "o-", color="C0",
                 label="τ_eff = 1/mean(diag W)")
        ax1.axhline(TAU0, color="C0", ls=":", lw=1, label="τ₀ = 0.10 (init)")
        ax1.set_xlabel("training step (k)")
        ax1.set_ylabel("effective temperature τ_eff", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax2 = ax1.twinx()
        ax2.plot(xs, [r["offdiag_frac"] for r in periodic], "s-", color="C3",
                 label="off-diagonal energy fraction")
        ax2.set_ylabel("||offdiag||_F / ||W||_F", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="best")
        ax1.set_title("Learned main-loss W: effective temperature and off-diagonal structure")
        fig.tight_layout()
        os.makedirs(os.path.dirname(PLOT), exist_ok=True)
        fig.savefig(PLOT, dpi=110, bbox_inches="tight")
        print("wrote", os.path.abspath(PLOT))


if __name__ == "__main__":
    main()
