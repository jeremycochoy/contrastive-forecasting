#!/usr/bin/env python3
"""
ARMA recovery head on top of the frozen corrV6 backbone.

Re-uses `src/recovery.py:GRURecoveryHead` (the ARMA-V2-winning recipe with
hidden_dim=128, 2 layers, bidirectional). Input is the per-channel encoder
latent `h` from `model.forward(x)`; output is 4 AR + 4 MA coefficients per
channel. MSE loss against the padded ground-truth coefficients from the data
generator.
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

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import generate_arma_correlated_batch  # noqa: E402

from src.models import ConfigurableModel
from src.recovery import GRURecoveryHead


def extract_per_channel_h(model, x):
    """Return h reshaped to [B*C, T, H] for the ARMA head."""
    with torch.no_grad():
        _h_hat, h = model(x)  # h: [B, T, C, H]
    B, T, C, H = h.shape
    return h.permute(0, 2, 1, 3).reshape(B * C, T, H), B, C


def make_batch(batch_size, K, T_raw, dim, device, seed):
    y, C, ar, ma = generate_arma_correlated_batch(
        batch_size=batch_size, T_raw=T_raw, K=K, dimension=dim, seed=seed, device=device,
    )
    return y, C, ar, ma


def evaluate(head, model, args, device, num_samples=200):
    K = args.C
    head.eval()
    ar_pred_all, ma_pred_all = [], []
    ar_true_all, ma_true_all = [], []
    bs = args.batch_size
    n = (num_samples + bs - 1) // bs
    with torch.no_grad():
        for i in range(n):
            this_bs = min(bs, num_samples - i * bs)
            y, _C, ar_true, ma_true = make_batch(
                this_bs, K, args.T_raw, args.dimension, device, seed=10000 + i,
            )
            h_bc, B_, C_ = extract_per_channel_h(model, y)
            ar_pred, ma_pred = head(h_bc)  # [B*C, T, dim]
            # Average over time
            ar_pred = ar_pred.mean(dim=1).reshape(B_, C_, args.dimension)
            ma_pred = ma_pred.mean(dim=1).reshape(B_, C_, args.dimension)
            ar_pred_all.append(ar_pred.cpu().numpy())
            ma_pred_all.append(ma_pred.cpu().numpy())
            ar_true_all.append(ar_true.cpu().numpy())
            ma_true_all.append(ma_true.cpu().numpy())
    ar_pred = np.concatenate(ar_pred_all, axis=0)  # [N, K, dim]
    ma_pred = np.concatenate(ma_pred_all, axis=0)
    ar_true = np.concatenate(ar_true_all, axis=0)
    ma_true = np.concatenate(ma_true_all, axis=0)
    flat_pred = np.concatenate([ar_pred.reshape(-1), ma_pred.reshape(-1)])
    flat_true = np.concatenate([ar_true.reshape(-1), ma_true.reshape(-1)])
    overall_mse = float(np.mean((flat_pred - flat_true) ** 2))
    zero_baseline = float(np.mean(flat_true ** 2))
    threshold = 0.05
    nonzero_mask = np.abs(flat_true) > threshold
    if nonzero_mask.sum() > 1:
        sign_agree = float(np.mean(
            np.sign(flat_pred[nonzero_mask]) == np.sign(flat_true[nonzero_mask])
        ))
    else:
        sign_agree = 0.0
    # Per-coefficient correlations
    per_coef = []
    for kind, p_arr, t_arr in [("AR", ar_pred, ar_true), ("MA", ma_pred, ma_true)]:
        for j in range(args.dimension):
            t = t_arr.reshape(-1, args.dimension)[:, j]
            p = p_arr.reshape(-1, args.dimension)[:, j]
            mask = np.abs(t) > threshold
            if mask.sum() > 1 and np.std(p) > 1e-8:
                r = float(np.corrcoef(t[mask], p[mask])[0, 1])
                mae = float(np.mean(np.abs(t[mask] - p[mask])))
            else:
                r = 0.0; mae = float(np.mean(np.abs(t - p)))
            per_coef.append({"coef": f"{kind}[{j}]", "pearson_r": r, "mae": mae,
                              "n_nonzero": int(mask.sum())})
    summary = {
        "num_samples": int(ar_true.shape[0]),
        "overall_mse": overall_mse,
        "zero_baseline_mse": zero_baseline,
        "improvement_vs_zero": zero_baseline / overall_mse if overall_mse > 0 else 0.0,
        "sign_agreement": sign_agree,
        "per_coef": per_coef,
    }
    arrays = {"ar_pred": ar_pred, "ma_pred": ma_pred,
              "ar_true": ar_true, "ma_true": ma_true}
    return summary, arrays


def print_summary(s):
    print("\n=== ARMA Recovery ===")
    print(f"Samples:               {s['num_samples']}")
    print(f"Overall MSE:           {s['overall_mse']:.6f}")
    print(f"Zero baseline MSE:     {s['zero_baseline_mse']:.6f}")
    print(f"Improvement vs zero:   {s['improvement_vs_zero']:.2f}x")
    print(f"Sign agreement (>{0.05}): {s['sign_agreement']:.4f}")
    print("Per coefficient:")
    for c in s["per_coef"]:
        print(f"  {c['coef']:>5}  r={c['pearson_r']:.3f}  MAE={c['mae']:.4f}  "
              f"n_nonzero={c['n_nonzero']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model-path", type=str, required=True)
    # Backbone
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
    # Head
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-gru-layers", type=int, default=2)
    p.add_argument("--dimension", type=int, default=4)  # max p, q
    # Training
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--T-raw", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head-path", type=str, default="armacorr_head_arma.pth")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--eval-samples", type=int, default=200)
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
        num_layers=args.num_layers,
        nhead=args.nhead,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        activation=args.activation,
        depthwise_conv=args.depthwise_conv,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()
    for q in model.parameters():
        q.requires_grad = False

    head = GRURecoveryHead(
        H=args.H, hidden_dim=args.hidden_dim,
        num_arma_params=args.dimension, num_gru_layers=args.num_gru_layers,
        bidirectional=True,
    ).to(device)
    n_params = sum(q.numel() for q in head.parameters() if q.requires_grad)
    print(f"ARMA head: GRU h={args.hidden_dim}, layers={args.num_gru_layers}, params={n_params:,}")

    if args.evaluate:
        best = args.head_path.replace(".pth", "_best.pth")
        load = best if os.path.exists(best) else args.head_path
        head.load_state_dict(torch.load(load, map_location=device))
        s, _ = evaluate(head, model, args, device, num_samples=args.eval_samples)
        print_summary(s)
        return

    optimizer = optim.AdamW(head.parameters(), lr=args.lr)

    # Fixed val set
    K = args.C
    val_size = max(args.batch_size * 4, 64)
    val_h = []; val_ar = []; val_ma = []
    for i in range(0, val_size, args.batch_size):
        bs = min(args.batch_size, val_size - i)
        y, _C, ar_t, ma_t = make_batch(bs, K, args.T_raw, args.dimension, device, seed=10**6 + i)
        h_bc, B_, C_ = extract_per_channel_h(model, y)
        val_h.append(h_bc)
        val_ar.append(ar_t.reshape(B_ * C_, args.dimension))
        val_ma.append(ma_t.reshape(B_ * C_, args.dimension))
    val_h = torch.cat(val_h, dim=0)
    val_ar = torch.cat(val_ar, dim=0)
    val_ma = torch.cat(val_ma, dim=0)
    print(f"val set: {val_h.shape[0]} (B*C) sequences")

    print(f"\nTraining {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}")

    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        y, _C, ar_t, ma_t = make_batch(
            args.batch_size, K, args.T_raw, args.dimension, device, seed=epoch,
        )
        h_bc, B_, C_ = extract_per_channel_h(model, y)
        ar_pred, ma_pred = head(h_bc)  # [B*C, T, dim]
        ar_pred = ar_pred.mean(dim=1)  # [B*C, dim]
        ma_pred = ma_pred.mean(dim=1)
        ar_t_flat = ar_t.reshape(B_ * C_, args.dimension)
        ma_t_flat = ma_t.reshape(B_ * C_, args.dimension)
        loss = F.mse_loss(ar_pred, ar_t_flat) + F.mse_loss(ma_pred, ma_t_flat)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            ar_pv, ma_pv = head(val_h)
            ar_pv = ar_pv.mean(dim=1); ma_pv = ma_pv.mean(dim=1)
            val_loss = (F.mse_loss(ar_pv, val_ar) + F.mse_loss(ma_pv, val_ma)).item()

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
    s["head_kind"] = "GRURecoveryHead (h=128, 2 layers)"
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
