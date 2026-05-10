#!/usr/bin/env python3
"""
Correlation recovery head on top of the frozen corrV6 backbone.

Reuses the V5 GRU recipe (h=128, 2 layers, bidirectional, mean-pool over time,
single linear projection to 6 outputs). Input is the channel-mixed latent
`h_hat` from `model.forward(x)`. We collapse the C dim into the feature dim
and project back to H before the GRU, mirroring the V5 design (the projection
moves from inside the backbone to inside the head).
"""

import argparse
import json
import os
import sys
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import generate_arma_correlated_batch  # noqa: E402

V5_DIR = HERE.parent / "contrastive-correlation"
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))
from joint_channel_model import GRUCorrelationHead  # noqa: E402

from src.models import ConfigurableModel
from src.correlation import correlation_to_pairs


class JointCorrelationHead(nn.Module):
    """Wraps a Linear(C·H → H) projection in front of the V5 GRUCorrelationHead.

    NOTE: this collapses the channel dim with a sample-independent linear map
    BEFORE the GRU sees the sequence. A linear projection cannot compute
    second-order cross-channel statistics (h^c1 · h^c2), which is exactly the
    information per-sample correlation lives in. Kept for the V7 baseline;
    use `JointCorrelationHeadDirect` instead.

    Input  : h_hat from `ConfigurableModel.forward` shaped [B, T, C, H].
    Output : [B, K(K-1)/2] pairwise correlations.
    """

    def __init__(self, H: int = 1024, K: int = 4, hidden_dim: int = 128,
                 num_gru_layers: int = 2):
        super().__init__()
        self.K = K
        self.H = H
        self.input_proj = nn.Linear(K * H, H)
        self.head = GRUCorrelationHead(
            H=H, K=K, hidden_dim=hidden_dim, num_gru_layers=num_gru_layers,
        )

    def forward(self, h_hat: torch.Tensor) -> torch.Tensor:
        """h_hat: [B, T, C, H] → [B, num_pairs] in [0, 1]."""
        flat = h_hat.flatten(2)             # [B, T, C*H]
        x = self.input_proj(flat)           # [B, T, H]
        return self.head(x)


class JointCorrelationHeadDirect(nn.Module):
    """GRU sees [B, T, C·H] directly (input_size = C·H), no projection ahead.

    The bidirectional GRU's internal gates are nonlinear in the input
    sequence, so over time the hidden state can accumulate quadratic
    cross-channel statistics — which is what per-sample correlation lives
    in. The Linear-projection head (V5 / `JointCorrelationHead`) cannot
    because a sample-independent linear map across channels destroys
    second-order cross-channel structure.
    """

    def __init__(self, H: int = 1024, K: int = 4, hidden_dim: int = 128,
                 num_gru_layers: int = 2, dropout: float = 0.1,
                 init_bias: float = 0.45):
        super().__init__()
        self.K = K
        self.num_pairs = K * (K - 1) // 2
        self.gru = nn.GRU(
            input_size=K * H,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        gru_out = hidden_dim * 2
        self.output_layers = nn.Sequential(
            nn.Linear(gru_out, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, self.num_pairs)
        with torch.no_grad():
            self.out.bias.fill_(float(init_bias))
            self.out.weight.mul_(0.1)

    def forward(self, h_hat: torch.Tensor) -> torch.Tensor:
        """h_hat: [B, T, C, H] → [B, num_pairs] in [0, 1]."""
        flat = h_hat.flatten(2)              # [B, T, C*H]
        gru_out, _ = self.gru(flat)          # [B, T, hidden*2]
        pooled = gru_out.mean(dim=1)         # [B, hidden*2]
        feat = self.output_layers(pooled)    # [B, hidden]
        return self.out(feat).clamp(0.0, 1.0)


def extract_h_hat(model, x):
    """Return the channel-mixed latent h_hat shaped [B, T, C, H]."""
    with torch.no_grad():
        h_hat, _h = model(x)
    return h_hat


def make_batch(batch_size, K, T_raw, dim, device, seed):
    y, C, ar, ma = generate_arma_correlated_batch(
        batch_size=batch_size, T_raw=T_raw, K=K, dimension=dim, seed=seed, device=device,
    )
    return y, C, correlation_to_pairs(C, K=K)


def empirical_diff_corr_baseline(y: torch.Tensor, K: int):
    """Per-batch corrcoef of diff(y). Approximation only (the ARMA filter
    means diff(y) ≠ ε), but useful as a sanity baseline."""
    diffs = y[:, 1:] - y[:, :-1]
    diffs = diffs - diffs.mean(dim=1, keepdim=True)
    std = diffs.std(dim=1, keepdim=True).clamp(min=1e-6)
    z = diffs / std
    Tm1 = z.shape[1]
    corr = torch.einsum("bti,btj->bij", z, z) / Tm1
    return correlation_to_pairs(corr, K=K)


def empirical_position_corr(y: torch.Tensor, K: int):
    z = y - y.mean(dim=1, keepdim=True)
    std = z.std(dim=1, keepdim=True).clamp(min=1e-6)
    z = z / std
    T = z.shape[1]
    corr = torch.einsum("bti,btj->bij", z, z) / T
    return correlation_to_pairs(corr, K=K)


def evaluate(head, model, args, device, num_samples=400):
    head.eval()
    K = args.C
    bs = args.batch_size
    n = (num_samples + bs - 1) // bs
    out_pred, out_true, out_diff, out_pos = [], [], [], []
    with torch.no_grad():
        for i in range(n):
            this_bs = min(bs, num_samples - i * bs)
            y, _C, target = make_batch(
                this_bs, K, args.T_raw, args.dimension, device, seed=10000 + i
            )
            h_hat = extract_h_hat(model, y)
            pred = head(h_hat)
            out_pred.append(pred.cpu().numpy())
            out_true.append(target.cpu().numpy())
            out_diff.append(empirical_diff_corr_baseline(y, K).cpu().numpy())
            out_pos.append(empirical_position_corr(y, K).cpu().numpy())
    pred = np.concatenate(out_pred, axis=0)
    true = np.concatenate(out_true, axis=0)
    diffbase = np.concatenate(out_diff, axis=0)
    posbase = np.concatenate(out_pos, axis=0)
    num_pairs = pred.shape[1]
    per_pair = []
    for j in range(num_pairs):
        t = true[:, j]; p = pred[:, j]
        db = diffbase[:, j]; pb = posbase[:, j]
        per_pair.append({
            "pair_index": j,
            "mse": float(np.mean((t - p) ** 2)),
            "mae": float(np.mean(np.abs(t - p))),
            "pearson_r": float(np.corrcoef(t, p)[0, 1]) if np.std(p) > 1e-8 else 0.0,
            "diff_baseline_pearson_r": float(np.corrcoef(t, db)[0, 1]) if np.std(db) > 1e-8 else 0.0,
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
        "improvement_vs_diff_baseline": float(np.mean((true - diffbase) ** 2)) / overall_mse if overall_mse > 0 else 0.0,
        "per_pair": per_pair,
    }
    arrays = {"pred": pred, "true": true, "diffbase": diffbase, "posbase": posbase}
    return summary, arrays


def print_summary(s):
    print("\n=== Correlation Recovery ===")
    print(f"Samples:                {s['num_samples']}")
    print(f"Overall MSE:            {s['overall_mse']:.6f}")
    print(f"Overall MAE:            {s['overall_mae']:.6f}")
    print(f"Zero baseline:          {s['zero_baseline_mse']:.6f}")
    print(f"Mean baseline:          {s['mean_baseline_mse']:.6f}")
    print(f"diff(y) baseline MSE:   {s['diff_baseline_mse']:.6f}  (corrcoef of diff(y))")
    print(f"y corrcoef baseline:    {s['pos_baseline_mse']:.6f}")
    print(f"Improvement vs zero:    {s['improvement_vs_zero']:.2f}x")
    print(f"Improvement vs mean:    {s['improvement_vs_mean']:.2f}x")
    print(f"Improvement vs diffbase:{s['improvement_vs_diff_baseline']:.2f}x")
    for p in s["per_pair"]:
        print(f"  pair[{p['pair_index']}]  r_head={p['pearson_r']:.3f}  "
              f"r_diffbase={p['diff_baseline_pearson_r']:.3f}  "
              f"r_posbase={p['pos_baseline_pearson_r']:.3f}  mae={p['mae']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model-path", type=str, required=True)
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
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T-raw", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head-path", type=str, default="armacorr_head_corr.pth")
    p.add_argument("--head-kind", type=str, default="direct",
                   choices=["projected", "direct"],
                   help="projected: V5 Linear(C*H→H)+GRU. direct: GRU sees [B,T,C*H].")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--eval-samples", type=int, default=400)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=2000)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Loading backbone from {args.model_path}")
    model = ConfigurableModel(
        C=args.C, H=args.H, W=args.W,
        encoder_type=args.encoder_type,
        intermediate_dim=args.intermediate_dim,
        num_layers=args.num_layers, nhead=args.nhead, ffn_mult=args.ffn_mult,
        dropout=args.dropout, activation=args.activation,
        depthwise_conv=args.depthwise_conv,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()
    for q in model.parameters():
        q.requires_grad = False

    K = args.C
    if args.head_kind == "projected":
        head = JointCorrelationHead(
            H=args.H, K=K, hidden_dim=args.hidden_dim, num_gru_layers=args.num_gru_layers,
        ).to(device)
    else:
        head = JointCorrelationHeadDirect(
            H=args.H, K=K, hidden_dim=args.hidden_dim, num_gru_layers=args.num_gru_layers,
        ).to(device)
    n_params = sum(q.numel() for q in head.parameters() if q.requires_grad)
    print(f"Correlation head ({args.head_kind}): GRU h={args.hidden_dim}, params={n_params:,}")

    if args.evaluate:
        best = args.head_path.replace(".pth", "_best.pth")
        load = best if os.path.exists(best) else args.head_path
        head.load_state_dict(torch.load(load, map_location=device))
        s, _ = evaluate(head, model, args, device, num_samples=args.eval_samples)
        print_summary(s)
        return

    optimizer = optim.AdamW(head.parameters(), lr=args.lr)

    # Fixed val set
    val_size = max(args.batch_size * 4, 64)
    val_hh, val_t = [], []
    for i in range(0, val_size, args.batch_size):
        bs = min(args.batch_size, val_size - i)
        y, _C, target = make_batch(bs, K, args.T_raw, args.dimension, device, seed=10**6 + i)
        val_hh.append(extract_h_hat(model, y))
        val_t.append(target)
    val_hh = torch.cat(val_hh, dim=0)
    val_t = torch.cat(val_t, dim=0)
    print(f"val set: {val_hh.shape[0]} samples")

    print(f"\nTraining {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}")

    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        y, _C, target = make_batch(args.batch_size, K, args.T_raw, args.dimension, device, seed=epoch)
        h_hat = extract_h_hat(model, y)
        pred = head(h_hat)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            pred_val = head(val_hh)
            val_loss = F.mse_loss(pred_val, val_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(head.state_dict(), args.head_path.replace(".pth", "_best.pth"))
        if epoch % args.log_every == 0:
            print(f"[Epoch {epoch:6d}] train={loss.item():.6f} | val={val_loss:.6f} | "
                  f"best={best_val:.6f}@{best_epoch}", flush=True)
            history.append({
                "epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss,
                "best_val": best_val, "best_epoch": best_epoch,
            })
        if epoch % args.save_every == 0:
            torch.save(head.state_dict(), args.head_path)

    torch.save(head.state_dict(), args.head_path)
    print(f"\nDone. Best val_loss={best_val:.6f} @ {best_epoch}")
    head.load_state_dict(torch.load(args.head_path.replace(".pth", "_best.pth"), map_location=device))
    s, _ = evaluate(head, model, args, device, num_samples=args.eval_samples)
    print_summary(s)
    s["head_kind"] = "JointCorrelationHead (V5 GRU + Linear(C*H, H))"
    s["head_n_params"] = n_params
    s["best_epoch"] = best_epoch
    s["best_val_loss"] = best_val
    s["history_excerpt"] = history
    out = args.head_path.replace(".pth", "_results.json")
    with open(out, "w") as f:
        json.dump(s, f, indent=2)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
