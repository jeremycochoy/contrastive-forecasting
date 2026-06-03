#!/usr/bin/env python3
"""
Recovery head training for the joint-channel backbone.

Mirrors `correlation_recovery.py` but the backbone is `JointChannelModel`
and the head is `GRUCorrelationHead` operating on [B, T, H] latents.
"""

import argparse
import json
import os
import sys
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.correlation import generate_correlated_batch, correlation_to_pairs

# Add this directory to path so we can import joint_channel_model.
HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from joint_channel_model import JointChannelModel, GRUCorrelationHead  # noqa: E402
from correlation_recovery import (  # noqa: E402
    empirical_corr_baseline, empirical_position_corr, print_summary,
)


def extract_joint_latent(model: JointChannelModel, x: torch.Tensor) -> torch.Tensor:
    """Run the backbone, return the encoder/transformer latent [B, T, H]."""
    with torch.no_grad():
        f, o = model(x)  # both [B, T, 1, H]
    # Use the original latent (the "h" tap point — same convention as the
    # ARMA recovery and the per-channel correlation recovery).
    return o.squeeze(2)  # [B, T, H]


def make_batch(batch_size, K, T_raw, device, seed, sampler, n_factors):
    x, C = generate_correlated_batch(
        batch_size=batch_size, T_raw=T_raw, K=K, seed=seed,
        device=device, sampler=sampler, n_factors=n_factors,
    )
    return x, C, correlation_to_pairs(C, K=K)


def evaluate(head, model, K, T_raw, device, num_samples=400, batch_size=16,
             seed_base=10000, sampler="uniform", n_factors=2):
    head.eval()
    all_pred, all_true, all_diffbase, all_posbase = [], [], [], []
    n_batches = (num_samples + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(n_batches):
            bs = min(batch_size, num_samples - i * batch_size)
            x, _C, target = make_batch(
                bs, K, T_raw, device, seed=seed_base + i,
                sampler=sampler, n_factors=n_factors,
            )
            h = extract_joint_latent(model, x)  # [B, T, H]
            pred = head(h)
            all_pred.append(pred.cpu().numpy())
            all_true.append(target.cpu().numpy())
            all_diffbase.append(empirical_corr_baseline(x).cpu().numpy())
            all_posbase.append(empirical_position_corr(x).cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)
    diffbase = np.concatenate(all_diffbase, axis=0)
    posbase = np.concatenate(all_posbase, axis=0)
    num_pairs = pred.shape[1]
    per_pair = []
    for j in range(num_pairs):
        t, p, db, pb = true[:, j], pred[:, j], diffbase[:, j], posbase[:, j]
        per_pair.append({
            "pair_index": j,
            "mse": float(np.mean((t - p) ** 2)),
            "mae": float(np.mean(np.abs(t - p))),
            "pearson_r": float(np.corrcoef(t, p)[0, 1]) if np.std(p) > 1e-8 else 0.0,
            "diff_baseline_mse": float(np.mean((t - db) ** 2)),
            "diff_baseline_pearson_r": float(np.corrcoef(t, db)[0, 1]) if np.std(db) > 1e-8 else 0.0,
            "pos_baseline_mse": float(np.mean((t - pb) ** 2)),
            "pos_baseline_pearson_r": float(np.corrcoef(t, pb)[0, 1]) if np.std(pb) > 1e-8 else 0.0,
            "true_mean": float(np.mean(t)),
            "true_std": float(np.std(t)),
        })
    overall_mse = float(np.mean((true - pred) ** 2))
    summary = {
        "num_samples": int(true.shape[0]),
        "overall_mse": overall_mse,
        "overall_mae": float(np.mean(np.abs(true - pred))),
        "zero_baseline_mse": float(np.mean(true ** 2)),
        "mean_baseline_mse": float(np.mean((true - true.mean()) ** 2)),
        "diff_baseline_mse": float(np.mean((true - diffbase) ** 2)),
        "pos_baseline_mse": float(np.mean((true - posbase) ** 2)),
        "improvement_vs_zero": float(np.mean(true ** 2)) / overall_mse if overall_mse > 0 else 0.0,
        "improvement_vs_mean": float(np.mean((true - true.mean()) ** 2)) / overall_mse if overall_mse > 0 else 0.0,
        "improvement_vs_pos_baseline": float(np.mean((true - posbase) ** 2)) / overall_mse if overall_mse > 0 else 0.0,
        "improvement_vs_diff_baseline": float(np.mean((true - diffbase) ** 2)) / overall_mse if overall_mse > 0 else 0.0,
        "per_pair": per_pair,
    }
    arrays = {"pred": pred, "true": true, "diffbase": diffbase, "posbase": posbase}
    return summary, arrays


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model-path", type=str, required=True)
    # Backbone arch
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
    # Head
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-gru-layers", type=int, default=2)
    # Training
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T-raw", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sampler", type=str, default="uniform", choices=["factor", "uniform"])
    p.add_argument("--n-factors", type=int, default=2)
    p.add_argument("--head-path", type=str, default="corr_jc_head.pth")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--eval-samples", type=int, default=400)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=2000)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load backbone (frozen)
    print(f"Loading joint-channel backbone from {args.model_path}")
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
    for q in model.parameters():
        q.requires_grad = False

    K = args.C
    head = GRUCorrelationHead(
        H=args.H, K=K, hidden_dim=args.hidden_dim,
        num_gru_layers=args.num_gru_layers,
    )
    head = head.to(device)
    n_params = sum(q.numel() for q in head.parameters() if q.requires_grad)
    print(f"GRU head params: {n_params:,}")

    if args.evaluate:
        best_path = args.head_path.replace(".pth", "_best.pth")
        load_path = best_path if os.path.exists(best_path) else args.head_path
        print(f"Loading head from {load_path}")
        head.load_state_dict(torch.load(load_path, map_location=device))
        summary, _ = evaluate(
            head, model, K, args.T_raw, device,
            num_samples=args.eval_samples, batch_size=args.batch_size,
            sampler=args.sampler, n_factors=args.n_factors,
        )
        print_summary(summary)
        return

    optimizer = optim.AdamW(head.parameters(), lr=args.lr)

    # Fixed val set, built in chunks to keep the GRU encoder workspace small.
    val_size = max(args.batch_size * 4, 64)
    val_chunks_h, val_chunks_t = [], []
    for i in range(0, val_size, args.batch_size):
        bs_i = min(args.batch_size, val_size - i)
        xv, _Cv, tv = make_batch(
            bs_i, K, args.T_raw, device, seed=10**6 + i,
            sampler=args.sampler, n_factors=args.n_factors,
        )
        val_chunks_h.append(extract_joint_latent(model, xv))
        val_chunks_t.append(tv)
    h_val = torch.cat(val_chunks_h, dim=0)
    target_val = torch.cat(val_chunks_t, dim=0)

    print(f"\nTraining GRU head for {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}, sampler={args.sampler}")

    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        x, _C, target = make_batch(
            args.batch_size, K, args.T_raw, device, seed=epoch,
            sampler=args.sampler, n_factors=args.n_factors,
        )
        h = extract_joint_latent(model, x)
        pred = head(h)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            pred_val = head(h_val)
            val_loss = F.mse_loss(pred_val, target_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(head.state_dict(), args.head_path.replace(".pth", "_best.pth"))

        if epoch % args.log_every == 0:
            print(
                f"[Epoch {epoch:6d}] train={loss.item():.6f} | val={val_loss:.6f} | "
                f"best={best_val:.6f}@{best_epoch}",
                flush=True,
            )
            history.append({
                "epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                "best_val": best_val, "best_epoch": best_epoch,
            })
        if epoch % args.save_every == 0:
            torch.save(head.state_dict(), args.head_path)

    torch.save(head.state_dict(), args.head_path)
    print(f"\nTraining complete. Best val_loss={best_val:.6f} at epoch {best_epoch}")

    head.load_state_dict(torch.load(args.head_path.replace(".pth", "_best.pth"), map_location=device))
    summary, _ = evaluate(
        head, model, K, args.T_raw, device,
        num_samples=args.eval_samples, batch_size=args.batch_size,
        sampler=args.sampler, n_factors=args.n_factors,
    )
    print_summary(summary)

    summary["head_type"] = "gru_correlation"
    summary["head_n_params"] = n_params
    summary["best_epoch"] = best_epoch
    summary["best_val_loss"] = best_val
    summary["history_excerpt"] = history[-50:]
    results_path = args.head_path.replace(".pth", "_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
