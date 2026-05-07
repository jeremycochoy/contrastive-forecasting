#!/usr/bin/env python3
"""Generate the corrV5 figures: data samples, training curves, true-vs-predicted
bar charts, scatter per pair, error distributions, correlation-matrix heatmaps,
head-vs-baseline comparison. Mirrors `evaluate_and_plot.py` but uses the
joint-channel backbone + GRU head."""

import argparse
import json
import os
import sys
import pathlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.correlation import generate_correlated_batch, correlation_to_pairs, pairs_to_correlation
from joint_channel_model import JointChannelModel, GRUCorrelationHead  # noqa: E402
from train_joint_channel_head import (  # noqa: E402
    extract_joint_latent, make_batch,
)
from correlation_recovery import (  # noqa: E402
    empirical_corr_baseline, empirical_position_corr,
)

PAIR_LABELS = ["(1,2)", "(1,3)", "(1,4)", "(2,3)", "(2,4)", "(3,4)"]


def load_backbone(args, device):
    model = JointChannelModel(
        C=args.C, H=args.H, W=args.W,
        encoder_type=args.encoder_type,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        depthwise_conv=args.depthwise_conv,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_head(args, device):
    head = GRUCorrelationHead(
        H=args.H, K=args.C,
        hidden_dim=args.hidden_dim, num_gru_layers=args.num_gru_layers,
    )
    head.load_state_dict(torch.load(args.head_path, map_location=device))
    return head.to(device).eval()


def collect(head, model, args, device, num_samples=400, seed_base=10000):
    K = args.C
    bs = args.batch_size
    n_batches = (num_samples + bs - 1) // bs
    out_pred, out_true, out_diff, out_pos, out_x, out_C = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(n_batches):
            this_bs = min(bs, num_samples - i * bs)
            x, C, target = make_batch(this_bs, K, args.T_raw, device, seed=seed_base + i,
                                       sampler=args.sampler, n_factors=args.n_factors)
            h = extract_joint_latent(model, x)
            pred = head(h)
            out_pred.append(pred.cpu().numpy())
            out_true.append(target.cpu().numpy())
            out_diff.append(empirical_corr_baseline(x).cpu().numpy())
            out_pos.append(empirical_position_corr(x).cpu().numpy())
            out_x.append(x.cpu().numpy())
            out_C.append(C.cpu().numpy())
    return {
        "pred": np.concatenate(out_pred, axis=0),
        "true": np.concatenate(out_true, axis=0),
        "diffbase": np.concatenate(out_diff, axis=0),
        "posbase": np.concatenate(out_pos, axis=0),
        "x": np.concatenate(out_x, axis=0),
        "C": np.concatenate(out_C, axis=0),
    }


def plot_data_samples(args, device, out_path, n=4):
    x, C = generate_correlated_batch(
        batch_size=n, T_raw=args.T_raw, K=args.C, seed=42,
        device=device, sampler=args.sampler, n_factors=args.n_factors,
    )
    x = x.cpu().numpy(); C = C.cpu().numpy()
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n),
                             gridspec_kw={"width_ratios": [3, 1]})
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        ax = axes[i, 0]
        for k in range(args.C):
            ax.plot(x[i, :, k], lw=0.7, label=f"ch {k+1}")
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("t"); ax.set_ylabel("x (z-scored walk)")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
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
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_training_curves(backbone_results_path, head_results_path, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    if backbone_results_path and os.path.exists(backbone_results_path):
        with open(backbone_results_path) as f:
            br = json.load(f)
        steps = [m["step"] for m in br["metrics_log"]]
        gaps = [m["val_ff_fp_gap"] for m in br["metrics_log"]]
        loss = [m["loss"] for m in br["metrics_log"]]
        ax.plot(steps, gaps, color="steelblue", label="FF–FP gap")
        ax.set_xlabel("step"); ax.set_ylabel("FF–FP gap", color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax.set_title("Joint-channel backbone training (corrV5)")
        ax2 = ax.twinx()
        ax2.plot(steps, loss, color="indianred", alpha=0.6, lw=0.9, label="loss")
        ax2.set_ylabel("loss", color="indianred")
        ax2.tick_params(axis="y", labelcolor="indianred")
        ax.grid(alpha=0.3)

    ax = axes[1]
    if head_results_path and os.path.exists(head_results_path):
        with open(head_results_path) as f:
            hr = json.load(f)
        history = hr.get("history_excerpt", [])
        if history:
            ep = [h["epoch"] for h in history]
            vl = [h["val_loss"] for h in history]
            tl = [h["train_loss"] for h in history]
            ax.plot(ep, tl, color="steelblue", lw=0.9, alpha=0.7, label="train")
            ax.plot(ep, vl, color="indianred", lw=1.3, label="val")
            ax.set_yscale("log")
            ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
            ax.set_title("GRU recovery head")
            ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_true_vs_predicted_bars(data, out_path, n=6):
    pred, true = data["pred"][:n], data["true"][:n]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes = axes.flatten()
    idx = np.arange(len(PAIR_LABELS)); w = 0.38
    for i in range(n):
        ax = axes[i]
        ax.bar(idx - w/2, true[i], w, label="True", color="steelblue")
        ax.bar(idx + w/2, pred[i], w, label="Predicted", color="indianred")
        ax.set_title(f"Sample {i+1}")
        ax.set_xticks(idx); ax.set_xticklabels(PAIR_LABELS)
        ax.set_ylim(0, 1); ax.set_ylabel("correlation")
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("Pairwise correlations: true vs predicted (6 samples)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_scatter_per_pair(data, out_path):
    pred, true = data["pred"], data["true"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        t, p = true[:, j], pred[:, j]
        ax.scatter(t, p, alpha=0.35, s=14, color="steelblue", edgecolors="none")
        lims = [0, 1]
        ax.plot(lims, lims, "r--", lw=1, label="y = x")
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
        ax.set_xlabel(f"True {PAIR_LABELS[j]}"); ax.set_ylabel(f"Predicted {PAIR_LABELS[j]}")
        if np.std(p) > 1e-8:
            r = np.corrcoef(t, p)[0, 1]
            mae = np.mean(np.abs(t - p))
            ax.set_title(f"{PAIR_LABELS[j]}  r={r:.3f}  MAE={mae:.3f}")
        else:
            ax.set_title(PAIR_LABELS[j])
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    plt.suptitle("Per-pair true vs predicted scatter", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_error_distributions(data, out_path):
    pred, true = data["pred"], data["true"]
    abs_err = np.abs(pred - true)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()
    for j, ax in enumerate(axes):
        ax.hist(abs_err[:, j], bins=40, alpha=0.7, color="seagreen", edgecolor="black", lw=0.4)
        m = abs_err[:, j].mean()
        ax.axvline(m, color="red", ls="--", lw=1.4, label=f"mean={m:.3f}")
        ax.set_xlabel("|true - predicted|"); ax.set_ylabel("count")
        ax.set_title(f"Error: {PAIR_LABELS[j]}")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.suptitle("Absolute error per pair", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_corr_heatmaps(data, out_path, K=4, n=6):
    pred, true = data["pred"][:n], data["true"][:n]
    pred_C = pairs_to_correlation(torch.from_numpy(pred).float(), K=K).numpy()
    true_C = pairs_to_correlation(torch.from_numpy(true).float(), K=K).numpy()
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4))
    for i in range(n):
        for row, (mat, title) in enumerate([(true_C[i], "True"), (pred_C[i], "Predicted")]):
            ax = axes[row, i]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.set_title(f"{title}\nSample {i+1}", fontsize=10)
            for r in range(K):
                for c in range(K):
                    ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center",
                            color="white" if mat[r, c] < 0.5 else "black", fontsize=7)
    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="correlation")
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
    ax.bar(idx,     r_diff, w, label="diff(x) corrcoef", color="steelblue")
    ax.bar(idx + w, r_pos,  w, label="x corrcoef",       color="goldenrod")
    ax.set_xticks(idx); ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r vs ground truth")
    ax.set_title("Recovery quality per pair: head vs analytic baselines (corrV5)")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--head-path", type=str, required=True)
    p.add_argument("--encoder-type", type=str, default="gru")
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--ffn-mult", type=float, default=4.0)
    p.add_argument("--depthwise-conv", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--intermediate-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-gru-layers", type=int, default=2)
    p.add_argument("--T-raw", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sampler", type=str, default="uniform", choices=["factor", "uniform"])
    p.add_argument("--n-factors", type=int, default=2)
    p.add_argument("--num-samples", type=int, default=800)
    p.add_argument("--out-dir", type=str, default="figures_v5")
    p.add_argument("--backbone-results", type=str, default=None)
    p.add_argument("--head-results", type=str, default=None)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_backbone(args, device)
    head = load_head(args, device)

    print("Generating data sample plot…")
    plot_data_samples(args, device, out / "data_samples_v5.png")
    print("Generating training curves…")
    plot_training_curves(args.backbone_results, args.head_results, out / "training_curves_v5.png")

    print("Collecting predictions on eval set…")
    data = collect(head, model, args, device, num_samples=args.num_samples)
    print(f"  N={data['pred'].shape[0]}")

    print("Plotting figures…")
    plot_true_vs_predicted_bars(data, out / "true_vs_predicted_bars_v5.png")
    plot_scatter_per_pair(data, out / "scatter_per_pair_v5.png")
    plot_error_distributions(data, out / "error_distributions_v5.png")
    plot_corr_heatmaps(data, out / "corr_matrix_heatmaps_v5.png", K=args.C, n=6)
    plot_baseline_comparison(data, out / "baseline_comparison_v5.png")

    np.savez_compressed(
        out / "eval_arrays_v5.npz",
        pred=data["pred"], true=data["true"],
        diffbase=data["diffbase"], posbase=data["posbase"],
    )
    print(f"\nDone. Figures written to {out}/")


if __name__ == "__main__":
    main()
