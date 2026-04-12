#!/usr/bin/env python3
"""
ARMA parameter recovery training pipeline for ConfigurableModel (v2 architecture).
Adapted from train_parameter_recovery.py to support the new configurable backbone.

Usage:
    python train_parameter_recovery_v2.py --device cuda \
        --model-path arch_search_phase4_best.pth \
        --encoder-type gru --H 1024 --num-layers 12 \
        --nhead 8 --ffn-mult 4 --activation gelu --depthwise-conv 3 \
        --model-type deepgru --epochs 20000
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.arma import generate_arma_batch
from train_contrastive_v2 import ConfigurableModel

# Import all recovery head classes from the original script
from train_parameter_recovery import (
    create_recovery_head,
    parameter_loss,
)


def create_parameter_loss(loss_type):
    """Create a configurable loss function for parameter recovery.

    Returns a function with signature (pred_ar, pred_ma, true_ar, true_ma) -> (total_loss, ar_loss, ma_loss).
    """
    def _loss_fn(pred_ar, pred_ma, true_ar, true_ma):
        pred_ar_avg = pred_ar.mean(dim=1)
        pred_ma_avg = pred_ma.mean(dim=1)

        if loss_type == "mse":
            ar_loss = F.mse_loss(pred_ar_avg, true_ar)
            ma_loss = F.mse_loss(pred_ma_avg, true_ma)
        elif loss_type == "huber":
            ar_loss = F.huber_loss(pred_ar_avg, true_ar, delta=0.1)
            ma_loss = F.huber_loss(pred_ma_avg, true_ma, delta=0.1)
        elif loss_type == "l1":
            ar_loss = F.l1_loss(pred_ar_avg, true_ar)
            ma_loss = F.l1_loss(pred_ma_avg, true_ma)
        elif loss_type == "weighted_mse":
            # 2x weight on first coefficient (index 0), 1x on the rest
            weights = torch.ones(true_ar.shape[-1], device=true_ar.device)
            weights[0] = 2.0
            ar_diff = (pred_ar_avg - true_ar) ** 2
            ma_diff = (pred_ma_avg - true_ma) ** 2
            ar_loss = (ar_diff * weights).mean()
            ma_loss = (ma_diff * weights).mean()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return ar_loss + ma_loss, ar_loss, ma_loss

    return _loss_fn


def extract_latent_features(model, x):
    """Extract latent features from the pre-trained ConfigurableModel, per-channel."""
    with torch.no_grad():
        h_hat, h = model(x)
        B, T, C, H = h.shape
        h_reshaped = h.permute(0, 2, 1, 3).reshape(B * C, T, H)
        return h_reshaped


def prepare_training_data(batch_size, num_arma_params, device, T_raw=4096, C=4, seed=None, dimension=4):
    """Generate training data with known ARMA parameters."""
    x, parameters = generate_arma_batch(batch_size=batch_size, T_raw=T_raw, C=C, seed=seed, dimension=dimension)
    x = x.to(device)

    true_ar_params = []
    true_ma_params = []

    for ar_poly, ma_poly in parameters:
        ar_coeffs = -ar_poly[1:]
        ma_coeffs = ma_poly[1:]
        ar_padded = np.pad(ar_coeffs, (0, max(0, num_arma_params - len(ar_coeffs))), mode='constant')[:num_arma_params]
        ma_padded = np.pad(ma_coeffs, (0, max(0, num_arma_params - len(ma_coeffs))), mode='constant')[:num_arma_params]
        true_ar_params.append(ar_padded)
        true_ma_params.append(ma_padded)

    true_ar = torch.tensor(np.array(true_ar_params), dtype=torch.float32).to(device)
    true_ma = torch.tensor(np.array(true_ma_params), dtype=torch.float32).to(device)

    return x, true_ar, true_ma


def evaluate(param_head, model, num_arma_params, device, num_samples=200, dimension=4):
    """Evaluate parameter recovery performance."""
    param_head.eval()

    all_ar_errors = []
    all_ma_errors = []
    all_total_errors = []
    all_baseline_errors = []
    # For sign agreement and correlation
    all_pred_ar = []
    all_pred_ma = []
    all_true_ar = []
    all_true_ma = []

    with torch.no_grad():
        for i in range(num_samples):
            x_test, ar_true, ma_true = prepare_training_data(
                batch_size=1, num_arma_params=num_arma_params, device=device,
                seed=i + 1000, dimension=dimension
            )
            h_test = extract_latent_features(model, x_test)

            pred_ar, pred_ma = param_head(h_test)
            _, ar_error, ma_error = parameter_loss(pred_ar, pred_ma, ar_true, ma_true)
            ar_error = ar_error.item()
            ma_error = ma_error.item()
            total_error = ar_error + ma_error

            baseline_ar = F.mse_loss(torch.zeros_like(ar_true), ar_true).item()
            baseline_ma = F.mse_loss(torch.zeros_like(ma_true), ma_true).item()
            baseline_total = baseline_ar + baseline_ma

            all_ar_errors.append(ar_error)
            all_ma_errors.append(ma_error)
            all_total_errors.append(total_error)
            all_baseline_errors.append(baseline_total)

            # Store predictions and targets for sign/correlation metrics
            pred_ar_avg = pred_ar.mean(dim=1)  # [B*C, num_arma_params]
            pred_ma_avg = pred_ma.mean(dim=1)
            all_pred_ar.append(pred_ar_avg.cpu())
            all_pred_ma.append(pred_ma_avg.cpu())
            all_true_ar.append(ar_true.cpu())
            all_true_ma.append(ma_true.cpu())

    # Concatenate all predictions and targets
    all_pred_ar_t = torch.cat(all_pred_ar, dim=0).numpy()  # [N, num_arma_params]
    all_pred_ma_t = torch.cat(all_pred_ma, dim=0).numpy()
    all_true_ar_t = torch.cat(all_true_ar, dim=0).numpy()
    all_true_ma_t = torch.cat(all_true_ma, dim=0).numpy()

    # Sign agreement: fraction where sign(pred) == sign(true), ignoring zero-padded entries
    ar_nonzero = all_true_ar_t != 0
    ma_nonzero = all_true_ma_t != 0
    sign_agree_ar = float(np.mean(
        np.sign(all_pred_ar_t[ar_nonzero]) == np.sign(all_true_ar_t[ar_nonzero])
    )) if ar_nonzero.any() else 0.0
    sign_agree_ma = float(np.mean(
        np.sign(all_pred_ma_t[ma_nonzero]) == np.sign(all_true_ma_t[ma_nonzero])
    )) if ma_nonzero.any() else 0.0

    # Mean Pearson correlation across coefficient indices
    num_params = all_true_ar_t.shape[1]
    ar_corrs = []
    ma_corrs = []
    for j in range(num_params):
        # Only compute correlation if there is variance in both pred and true
        if np.std(all_true_ar_t[:, j]) > 1e-8 and np.std(all_pred_ar_t[:, j]) > 1e-8:
            ar_corrs.append(float(np.corrcoef(all_pred_ar_t[:, j], all_true_ar_t[:, j])[0, 1]))
        if np.std(all_true_ma_t[:, j]) > 1e-8 and np.std(all_pred_ma_t[:, j]) > 1e-8:
            ma_corrs.append(float(np.corrcoef(all_pred_ma_t[:, j], all_true_ma_t[:, j])[0, 1]))
    correlation_ar = float(np.mean(ar_corrs)) if ar_corrs else 0.0
    correlation_ma = float(np.mean(ma_corrs)) if ma_corrs else 0.0

    results = {
        'mean_ar_error': float(np.mean(all_ar_errors)),
        'mean_ma_error': float(np.mean(all_ma_errors)),
        'mean_total_error': float(np.mean(all_total_errors)),
        'mean_baseline_error': float(np.mean(all_baseline_errors)),
        'std_ar_error': float(np.std(all_ar_errors)),
        'std_ma_error': float(np.std(all_ma_errors)),
        'std_total_error': float(np.std(all_total_errors)),
        'improvement_ratio': float(np.mean(all_baseline_errors) / np.mean(all_total_errors)),
        'sign_agreement_ar': sign_agree_ar,
        'sign_agreement_ma': sign_agree_ma,
        'correlation_ar': correlation_ar,
        'correlation_ma': correlation_ma,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ARMA parameter recovery head (v2 backbone)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-path", type=str, default="arch_search_phase4_best.pth",
                        help="Path to pre-trained ConfigurableModel")

    # Backbone architecture (must match the saved model)
    parser.add_argument("--encoder-type", type=str, default="gru",
                        choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv"])
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=32)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn-mult", type=float, default=4.0)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    parser.add_argument("--depthwise-conv", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--intermediate-dim", type=int, default=None)

    # Recovery head config
    parser.add_argument("--model-type", type=str, default="deepgru",
                        choices=["mlp", "gru", "resmlp", "attention", "grupool", "deepgru", "deepgrupool"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-gru-layers", type=int, default=2)
    parser.add_argument("--num-arma-params", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--loss-type", type=str, default="mse",
                        choices=["mse", "huber", "l1", "weighted_mse"])

    # Training
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head-path", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)

    args = parser.parse_args()

    if args.head_path is None:
        args.head_path = f"parameter_recovery_v2_{args.model_type}.pth"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load pre-trained ConfigurableModel
    print(f"Loading ConfigurableModel from {args.model_path}")
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
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Backbone loaded: encoder={args.encoder_type}, H={args.H}, layers={args.num_layers}")

    # Create recovery head
    param_head = create_recovery_head(
        args.model_type, H=args.H, hidden_dim=args.hidden_dim,
        num_arma_params=args.num_arma_params, num_gru_layers=args.num_gru_layers
    )
    num_params = sum(p.numel() for p in param_head.parameters() if p.requires_grad)
    print(f"Recovery head: {args.model_type}, params={num_params:,}")

    if args.evaluate:
        print(f"Loading recovery head from {args.head_path}")
        best_path = args.head_path.replace('.pth', '_best.pth')
        if os.path.exists(best_path):
            param_head.load_state_dict(torch.load(best_path, map_location=device))
            print(f"  (loaded best checkpoint)")
        else:
            param_head.load_state_dict(torch.load(args.head_path, map_location=device))
        param_head = param_head.to(device)

        results = evaluate(param_head, model, args.num_arma_params, device,
                           num_samples=args.eval_samples, dimension=args.dimension)

        print("\n=== Parameter Recovery Performance ===")
        print(f"Mean AR Error:    {results['mean_ar_error']:.6f} +/- {results['std_ar_error']:.6f}")
        print(f"Mean MA Error:    {results['mean_ma_error']:.6f} +/- {results['std_ma_error']:.6f}")
        print(f"Mean Total Error: {results['mean_total_error']:.6f} +/- {results['std_total_error']:.6f}")
        print(f"Baseline Error:   {results['mean_baseline_error']:.6f}")
        print(f"Improvement:      {results['improvement_ratio']:.2f}x better than zero-baseline")
        print(f"Sign Agreement AR: {results['sign_agreement_ar']:.4f}")
        print(f"Sign Agreement MA: {results['sign_agreement_ma']:.4f}")
        print(f"Correlation AR:    {results['correlation_ar']:.4f}")
        print(f"Correlation MA:    {results['correlation_ma']:.4f}")
        return

    # Create configurable loss function
    loss_fn = create_parameter_loss(args.loss_type)
    print(f"Loss function: {args.loss_type}")

    param_head = param_head.to(device)
    optimizer = optim.Adam(param_head.parameters(), lr=args.lr)

    # Fixed validation set
    x_val, ar_val, ma_val = prepare_training_data(
        batch_size=32, num_arma_params=args.num_arma_params, device=device,
        seed=0, dimension=args.dimension
    )
    h_val = extract_latent_features(model, x_val)

    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        param_head.train()
        optimizer.zero_grad()

        x_train, ar_train, ma_train = prepare_training_data(
            batch_size=args.batch_size, num_arma_params=args.num_arma_params,
            device=device, seed=epoch, dimension=args.dimension
        )
        h_train = extract_latent_features(model, x_train)

        pred_ar, pred_ma = param_head(h_train)
        loss, ar_loss, ma_loss = loss_fn(pred_ar, pred_ma, ar_train, ma_train)

        loss.backward()
        optimizer.step()

        # Validation
        param_head.eval()
        with torch.no_grad():
            pred_ar_val, pred_ma_val = param_head(h_val)
            val_loss, val_ar_loss, val_ma_loss = loss_fn(pred_ar_val, pred_ma_val, ar_val, ma_val)

        val_loss_item = val_loss.item()

        if val_loss_item < best_val_loss:
            best_val_loss = val_loss_item
            best_epoch = epoch
            torch.save(param_head.state_dict(), args.head_path.replace('.pth', '_best.pth'))

        if epoch % args.log_every == 0:
            print(f"[Epoch {epoch:6d}] train={loss.item():.6f} (AR={ar_loss.item():.6f} MA={ma_loss.item():.6f}) | "
                  f"val={val_loss_item:.6f} (AR={val_ar_loss.item():.6f} MA={val_ma_loss.item():.6f}) | "
                  f"best={best_val_loss:.6f}@{best_epoch}")

        if epoch % args.save_every == 0:
            torch.save(param_head.state_dict(), args.head_path)

    # Final save
    torch.save(param_head.state_dict(), args.head_path)
    print(f"\nTraining complete. Best val_loss={best_val_loss:.6f} at epoch {best_epoch}")

    # Final evaluation with best model
    param_head.load_state_dict(torch.load(args.head_path.replace('.pth', '_best.pth'), map_location=device))
    results = evaluate(param_head, model, args.num_arma_params, device,
                       num_samples=args.eval_samples, dimension=args.dimension)

    print(f"\n=== Final Evaluation ===")
    print(f"Mean AR Error:    {results['mean_ar_error']:.6f} +/- {results['std_ar_error']:.6f}")
    print(f"Mean MA Error:    {results['mean_ma_error']:.6f} +/- {results['std_ma_error']:.6f}")
    print(f"Mean Total Error: {results['mean_total_error']:.6f} +/- {results['std_total_error']:.6f}")
    print(f"Baseline Error:   {results['mean_baseline_error']:.6f}")
    print(f"Improvement:      {results['improvement_ratio']:.2f}x better than zero-baseline")
    print(f"Sign Agreement AR: {results['sign_agreement_ar']:.4f}")
    print(f"Sign Agreement MA: {results['sign_agreement_ma']:.4f}")
    print(f"Correlation AR:    {results['correlation_ar']:.4f}")
    print(f"Correlation MA:    {results['correlation_ma']:.4f}")

    # Save results
    results_path = args.head_path.replace('.pth', '_results.json')
    results['model_type'] = args.model_type
    results['backbone'] = {
        'encoder_type': args.encoder_type, 'H': args.H,
        'num_layers': args.num_layers, 'nhead': args.nhead,
        'ffn_mult': args.ffn_mult, 'activation': args.activation,
    }
    results['best_epoch'] = best_epoch
    results['best_val_loss'] = best_val_loss
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
