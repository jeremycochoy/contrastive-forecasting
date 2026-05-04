#!/usr/bin/env python3
"""
ARMA parameter recovery script for Tiny backbone.

Trains a lightweight GRU head on top of a frozen contrastive backbone to
recover the AR/MA coefficients. Tests whether the backbone's latent
representations encode the structure of the generating process.

Usage:
    python scripts/recover_v3.py --device cuda --model-path checkpoints/tiny_best_gap.pth
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.models import ConfigurableModel
from src.recovery import create_recovery_head, parameter_loss
from src.synthetic import generate_synthetic_batch

# -- Backbone architecture (must match checkpoint) -----------------------------
BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)

NUM_ARMA_PARAMS = 4
RECOVERY_CONFIG = dict(
    model_type="gru", H=512, hidden_dim=128, num_gru_layers=2,
    num_arma_params=NUM_ARMA_PARAMS,
)
T_RAW = 1024


def parse_args():
    p = argparse.ArgumentParser(description="Train ARMA recovery head (Tiny)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-path", required=True,
                   help="Path to trained backbone checkpoint")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-path", default="recovery_head.pth")
    p.add_argument("--eval-samples", type=int, default=200)
    return p.parse_args()


def extract_features(backbone, x):
    """Extract per-channel latent features from the frozen backbone."""
    W = backbone.W
    H = backbone.H
    with torch.no_grad():
        if backbone.rev_norm is not None:
            x = backbone.rev_norm(x, mode='norm')
        B, T_raw, C = x.shape
        T = T_raw // W
        xr = x.view(B, T, W, C).permute(0, 1, 3, 2)
        _, h = backbone.transformer(xr)  # h = original latents
        h = h.reshape(B, C, T, H)
        return h.permute(0, 2, 1, 3).reshape(B * C, T, H)


def prepare_batch(batch_size, device, seed=None):
    """Generate synthetic data and extract true ARMA parameters as targets."""
    from src.arma import generate_arma_batch
    C = BACKBONE_CONFIG["C"]

    # Use ARMA (not ARIMA) for recovery targets — we recover the underlying
    # ARMA coefficients, not the integrated process parameters.
    x, parameters = generate_arma_batch(
        batch_size=batch_size, T_raw=T_RAW, C=C, seed=seed, dimension=8)
    x = x.to(device)

    true_ar, true_ma = [], []
    for ar_poly, ma_poly in parameters:
        ar_coeffs = -ar_poly[1:]
        ma_coeffs = ma_poly[1:]
        ar_padded = np.pad(ar_coeffs,
                           (0, max(0, NUM_ARMA_PARAMS - len(ar_coeffs)))
                           )[:NUM_ARMA_PARAMS]
        ma_padded = np.pad(ma_coeffs,
                           (0, max(0, NUM_ARMA_PARAMS - len(ma_coeffs)))
                           )[:NUM_ARMA_PARAMS]
        true_ar.append(ar_padded)
        true_ma.append(ma_padded)

    true_ar = torch.tensor(np.array(true_ar), dtype=torch.float32).to(device)
    true_ma = torch.tensor(np.array(true_ma), dtype=torch.float32).to(device)
    return x, true_ar, true_ma


def main():
    args = parse_args()
    device = torch.device(args.device)
    C = BACKBONE_CONFIG["C"]
    H = BACKBONE_CONFIG["H"]

    # -- Load frozen backbone --------------------------------------------------
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    backbone.load_state_dict(torch.load(args.model_path, map_location=device))
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print(f"Backbone loaded from {args.model_path}")

    # -- Recovery head ---------------------------------------------------------
    recovery_head = create_recovery_head(**RECOVERY_CONFIG).to(device)
    n_params = sum(p.numel() for p in recovery_head.parameters() if p.requires_grad)
    print(f"Recovery head parameters: {n_params:,}")

    optimizer = optim.Adam(recovery_head.parameters(), lr=args.lr)

    # -- Fixed validation set --------------------------------------------------
    x_val, ar_val, ma_val = prepare_batch(args.batch_size, device, seed=0)
    h_val = extract_features(backbone, x_val)

    # -- Train -----------------------------------------------------------------
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        recovery_head.train()
        optimizer.zero_grad()

        x_train, ar_train, ma_train = prepare_batch(
            args.batch_size, device, seed=epoch)
        h_train = extract_features(backbone, x_train)

        pred_ar, pred_ma = recovery_head(h_train)
        loss, ar_loss, ma_loss = parameter_loss(pred_ar, pred_ma,
                                                ar_train, ma_train)
        loss.backward()
        optimizer.step()

        # Validation
        recovery_head.eval()
        with torch.no_grad():
            pred_ar_v, pred_ma_v = recovery_head(h_val)
            val_loss, _, _ = parameter_loss(pred_ar_v, pred_ma_v,
                                            ar_val, ma_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_epoch = epoch
            torch.save(recovery_head.state_dict(), args.save_path)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | train={loss.item():.6f} "
                  f"| val={val_loss.item():.6f} "
                  f"| best={best_val_loss:.6f}@{best_epoch}")

    # -- Evaluate --------------------------------------------------------------
    recovery_head.load_state_dict(torch.load(args.save_path, map_location=device))
    recovery_head.eval()

    all_errors, all_baselines = [], []
    pred_ar_all, pred_ma_all = [], []
    true_ar_all, true_ma_all = [], []

    with torch.no_grad():
        for i in range(args.eval_samples):
            x_test, ar_true, ma_true = prepare_batch(1, device, seed=i + 10000)
            h_test = extract_features(backbone, x_test)
            pred_ar, pred_ma = recovery_head(h_test)
            _, ar_err, ma_err = parameter_loss(pred_ar, pred_ma, ar_true, ma_true)

            baseline = (F.mse_loss(torch.zeros_like(ar_true), ar_true) +
                        F.mse_loss(torch.zeros_like(ma_true), ma_true))

            all_errors.append(ar_err.item() + ma_err.item())
            all_baselines.append(baseline.item())

            pred_ar_all.append(pred_ar.mean(dim=1).cpu())
            pred_ma_all.append(pred_ma.mean(dim=1).cpu())
            true_ar_all.append(ar_true.cpu())
            true_ma_all.append(ma_true.cpu())

    mean_error = np.mean(all_errors)
    mean_baseline = np.mean(all_baselines)
    improvement = mean_baseline / mean_error

    pred_ar_np = torch.cat(pred_ar_all).numpy()
    pred_ma_np = torch.cat(pred_ma_all).numpy()
    true_ar_np = torch.cat(true_ar_all).numpy()
    true_ma_np = torch.cat(true_ma_all).numpy()

    ar_nonzero = true_ar_np != 0
    ma_nonzero = true_ma_np != 0
    sign_ar = np.mean(np.sign(pred_ar_np[ar_nonzero]) == np.sign(true_ar_np[ar_nonzero]))
    sign_ma = np.mean(np.sign(pred_ma_np[ma_nonzero]) == np.sign(true_ma_np[ma_nonzero]))

    print(f"\n{'='*50}")
    print(f"Recovery Results ({args.eval_samples} samples)")
    print(f"{'='*50}")
    print(f"Mean MSE:          {mean_error:.6f}")
    print(f"Baseline MSE:      {mean_baseline:.6f}")
    print(f"Improvement ratio: {improvement:.2f}x")
    print(f"Sign agreement AR: {sign_ar:.2%}")
    print(f"Sign agreement MA: {sign_ma:.2%}")


if __name__ == "__main__":
    main()
