#!/usr/bin/env python3
"""Generate figures for the joint ARMA × correlation experiment."""

import argparse
import json
import os
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import generate_arma_correlated_batch  # noqa: E402
from train_arma_head import extract_per_channel_h, evaluate as evaluate_arma  # noqa: E402
from train_correlation_head import (  # noqa: E402
    extract_h_hat, JointCorrelationHead, evaluate as evaluate_corr,
    empirical_diff_corr_baseline, empirical_position_corr,
)

from src.models import ConfigurableModel
from src.recovery import GRURecoveryHead
from src.correlation import correlation_to_pairs

PAIR_LABELS = ["(1,2)", "(1,3)", "(1,4)", "(2,3)", "(2,4)", "(3,4)"]


def load_backbone(args, device):
    model = ConfigurableModel(
        C=args.C, H=args.H, W=args.W,
        encoder_type=args.encoder_type,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers, nhead=args.nhead, ffn_mult=args.ffn_mult,
        dropout=args.dropout, activation=args.activation,
        depthwise_conv=args.depthwise_conv,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    return model.to(device).eval()


def plot_data_samples(args, device, out_path, n=3):
    y, C, ar, ma = generate_arma_correlated_batch(
        batch_size=n, T_raw=args.T_raw, K=args.C, dimension=args.dimension,
        seed=42, device=device,
    )
    y = y.cpu().numpy(); C = C.cpu().numpy(); ar = ar.cpu().numpy(); ma = ma.cpu().numpy()
    fig, axes = plt.subplots(n, 3, figsize=(14, 2.6 * n),
                              gridspec_kw={"width_ratios": [3, 1, 1.2]})
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        ax = axes[i, 0]
        for k in range(args.C):
            ax.plot(y[i, :, k], lw=0.7, label=f"ch {k+1}")
        ax.set_title(f"Sample {i+1}: 4 ARMA channels with correlated innovations")
        ax.set_xlabel("t"); ax.set_ylabel("y")
        if i == 0: ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        ax2 = axes[i, 1]
        im = ax2.imshow(C[i], vmin=0, vmax=1, cmap="viridis")
        ax2.set_title("True C")
        ax2.set_xticks(range(args.C)); ax2.set_yticks(range(args.C))
        for r in range(args.C):
            for c in range(args.C):
                ax2.text(c, r, f"{C[i, r, c]:.2f}", ha="center", va="center",
                         color="white" if C[i, r, c] < 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = axes[i, 2]
        idx = np.arange(args.dimension); w = 0.18
        for k in range(args.C):
            offset = (k - 1.5) * w
            ax3.bar(idx + offset - 0.4, ar[i, k], w * 0.4, color=f"C{k}",
                    label=f"AR ch{k+1}" if i == 0 else None)
            ax3.bar(idx + offset + 0.4, ma[i, k], w * 0.4, color=f"C{k}", alpha=0.5,
                    hatch="//")
        ax3.set_title("AR (solid) / MA (hatched) per ch")
        ax3.set_xticks(idx); ax3.set_xlabel("coef idx")
        if i == 0: ax3.legend(fontsize=6, ncol=2)
        ax3.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_training_curves(backbone_results, arma_head_results, corr_head_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ax = axes[0]
    if backbone_results and os.path.exists(backbone_results):
        with open(backbone_results) as f:
            br = json.load(f)
        steps = [m["step"] for m in br["metrics_log"]]
        gaps = [m["val_ff_fp_gap"] for m in br["metrics_log"]]
        loss = [m["loss"] for m in br["metrics_log"]]
        ax.plot(steps, gaps, color="steelblue", label="FF-FP gap")
        ax.set_xlabel("step"); ax.set_ylabel("gap", color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax.set_title("corrV6 backbone (ARMA × correlation, channel-mixing live)")
        ax2 = ax.twinx()
        ax2.plot(steps, loss, color="indianred", alpha=0.6, lw=0.9)
        ax2.set_ylabel("loss", color="indianred")
        ax2.tick_params(axis="y", labelcolor="indianred")
        ax.grid(alpha=0.3)

    for j, (path, title) in enumerate([(arma_head_results, "ARMA head MSE"),
                                       (corr_head_results, "Correlation head MSE")]):
        ax = axes[j + 1]
        if path and os.path.exists(path):
            with open(path) as f:
                hr = json.load(f)
            history = hr.get("history_excerpt", [])
            if history:
                ep = [h["epoch"] for h in history]
                vl = [h["val_loss"] for h in history]
                tl = [h["train_loss"] for h in history]
                ax.plot(ep, tl, color="steelblue", lw=0.9, alpha=0.7, label="train")
                ax.plot(ep, vl, color="indianred", lw=1.3, label="val")
                ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
                ax.set_title(title); ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_correlation_recovery(data, out_path):
    pred, true = data["pred"], data["true"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        t, p = true[:, j], pred[:, j]
        ax.scatter(t, p, alpha=0.35, s=14, color="steelblue", edgecolors="none")
        ax.plot([0, 1], [0, 1], "r--", lw=1, label="y = x")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel(f"True {PAIR_LABELS[j]}"); ax.set_ylabel(f"Predicted {PAIR_LABELS[j]}")
        if np.std(p) > 1e-8:
            r = np.corrcoef(t, p)[0, 1]; mae = np.mean(np.abs(t - p))
            ax.set_title(f"{PAIR_LABELS[j]}  r={r:.3f}  MAE={mae:.3f}")
        else:
            ax.set_title(PAIR_LABELS[j])
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.suptitle("Correlation recovery — per-pair scatter (corrV6)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_arma_recovery(arma_arrays, out_path, dim=4):
    ar_pred, ar_true = arma_arrays["ar_pred"].reshape(-1, dim), arma_arrays["ar_true"].reshape(-1, dim)
    ma_pred, ma_true = arma_arrays["ma_pred"].reshape(-1, dim), arma_arrays["ma_true"].reshape(-1, dim)
    fig, axes = plt.subplots(2, dim, figsize=(4 * dim, 8))
    threshold = 0.05
    for row, (kind, p_arr, t_arr) in enumerate([("AR", ar_pred, ar_true), ("MA", ma_pred, ma_true)]):
        for j in range(dim):
            ax = axes[row, j]
            t = t_arr[:, j]; p = p_arr[:, j]
            ax.scatter(t, p, alpha=0.2, s=8, color="steelblue", edgecolors="none")
            ax.plot([-1, 1], [-1, 1], "r--", lw=1)
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
            ax.set_xlabel(f"True {kind}[{j}]"); ax.set_ylabel(f"Predicted {kind}[{j}]")
            mask = np.abs(t) > threshold
            if mask.sum() > 1 and np.std(p) > 1e-8:
                r = np.corrcoef(t[mask], p[mask])[0, 1]
                ax.set_title(f"{kind}[{j}]  r={r:.3f}")
            else:
                ax.set_title(f"{kind}[{j}]")
            ax.grid(alpha=0.3)
    plt.suptitle("ARMA coefficient recovery (corrV6)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison(data, out_path):
    pred, true, diffbase, posbase = data["pred"], data["true"], data["diffbase"], data["posbase"]
    rows = []
    for j in range(6):
        r_head = np.corrcoef(true[:, j], pred[:, j])[0, 1] if np.std(pred[:, j]) > 1e-8 else 0
        r_diff = np.corrcoef(true[:, j], diffbase[:, j])[0, 1] if np.std(diffbase[:, j]) > 1e-8 else 0
        r_pos = np.corrcoef(true[:, j], posbase[:, j])[0, 1] if np.std(posbase[:, j]) > 1e-8 else 0
        rows.append((PAIR_LABELS[j], r_head, r_diff, r_pos))
    labels, r_head, r_diff, r_pos = zip(*rows)
    idx = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(idx - w, r_head, w, label="contrastive head", color="indianred")
    ax.bar(idx,     r_diff, w, label="diff(y) corrcoef", color="steelblue")
    ax.bar(idx + w, r_pos,  w, label="y corrcoef",       color="goldenrod")
    ax.set_xticks(idx); ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r vs ground truth")
    ax.set_title("Correlation recovery: head vs analytic baselines (corrV6)")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--arma-head-path", type=str, required=True)
    p.add_argument("--corr-head-path", type=str, required=True)
    p.add_argument("--encoder-type", type=str, default="gru")
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--ffn-mult", type=float, default=4.0)
    p.add_argument("--activation", type=str, default="gelu")
    p.add_argument("--depthwise-conv", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--intermediate-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-gru-layers", type=int, default=2)
    p.add_argument("--dimension", type=int, default=4)
    p.add_argument("--T-raw", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=400)
    p.add_argument("--out-dir", type=str, default="experiments/contrastive-arma-correlation/figures")
    p.add_argument("--backbone-results", type=str, default=None)
    p.add_argument("--arma-results", type=str, default=None)
    p.add_argument("--corr-results", type=str, default=None)
    args = p.parse_args()

    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_backbone(args, device)

    # Heads
    arma_head = GRURecoveryHead(
        H=args.H, hidden_dim=args.hidden_dim, num_arma_params=args.dimension,
        num_gru_layers=args.num_gru_layers, bidirectional=True,
    )
    arma_head.load_state_dict(torch.load(args.arma_head_path, map_location=device))
    arma_head = arma_head.to(device).eval()

    corr_head = JointCorrelationHead(
        H=args.H, K=args.C, hidden_dim=args.hidden_dim, num_gru_layers=args.num_gru_layers,
    )
    corr_head.load_state_dict(torch.load(args.corr_head_path, map_location=device))
    corr_head = corr_head.to(device).eval()

    print("Generating data sample plot…")
    plot_data_samples(args, device, out / "data_samples_v6.png", n=3)
    print("Generating training curves…")
    plot_training_curves(args.backbone_results, args.arma_results, args.corr_results,
                          out / "training_curves_v6.png")

    print("Evaluating ARMA head…")
    arma_summary, arma_arrays = evaluate_arma(arma_head, model, args, device,
                                               num_samples=args.num_samples)
    print(f"  ARMA: improvement={arma_summary['improvement_vs_zero']:.2f}x, "
          f"sign={arma_summary['sign_agreement']:.3f}")

    print("Evaluating Correlation head…")
    corr_summary, corr_arrays = evaluate_corr(corr_head, model, args, device,
                                                num_samples=args.num_samples)
    print(f"  Corr: improvement_vs_zero={corr_summary['improvement_vs_zero']:.2f}x, "
          f"mean_r={np.mean([p['pearson_r'] for p in corr_summary['per_pair']]):.3f}")

    print("Plotting correlation recovery…")
    plot_correlation_recovery(corr_arrays, out / "correlation_recovery_v6.png")
    print("Plotting ARMA recovery…")
    plot_arma_recovery(arma_arrays, out / "arma_recovery_v6.png", dim=args.dimension)
    print("Plotting baseline comparison…")
    plot_baseline_comparison(corr_arrays, out / "baseline_comparison_v6.png")

    np.savez_compressed(out / "eval_arrays_v6.npz",
                         **arma_arrays, **{f"corr_{k}": v for k, v in corr_arrays.items()})
    print(f"\nDone. Figures written to {out}/")


if __name__ == "__main__":
    main()
