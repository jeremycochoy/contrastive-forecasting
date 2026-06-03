#!/usr/bin/env python3
"""
ARMA parameter recovery training pipeline.
Converted from parameter_recovery_experiment.ipynb with multiple model architectures.

Usage:
    # Baseline MLP (original notebook model)
    python train_parameter_recovery.py --model-type mlp --device cuda:1

    # GRU-based model (processes temporal structure)
    python train_parameter_recovery.py --model-type gru --device cuda:1

    # Deep residual MLP
    python train_parameter_recovery.py --model-type resmlp --device cuda:1

    # Attention-based temporal aggregation
    python train_parameter_recovery.py --model-type attention --device cuda:1

    # Evaluate a trained model
    python train_parameter_recovery.py --evaluate --head-path parameter_recovery_head.pth --model-type mlp --device cuda:1
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.arma import generate_arma_batch
from src.network import SimpleModel
from src.recovery import (
    ParameterRecoveryHead,
    GRURecoveryHead,
    ResidualMLPRecoveryHead,
    AttentionRecoveryHead,
    GRUPoolRecoveryHead,
    DeepGRURecoveryHead,
    DeepGRUPoolRecoveryHead,
    create_recovery_head,
    parameter_loss,
)


# =============================================================================
# Data & Feature Extraction
# =============================================================================

def extract_latent_features(model, x):
    """Extract latent features from the pre-trained model, per-channel."""
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
        ar_coeffs = -ar_poly[1:]  # Remove constant term and negate
        ma_coeffs = ma_poly[1:]   # Remove constant term

        ar_padded = np.pad(ar_coeffs, (0, max(0, num_arma_params - len(ar_coeffs))), mode='constant')[:num_arma_params]
        ma_padded = np.pad(ma_coeffs, (0, max(0, num_arma_params - len(ma_coeffs))), mode='constant')[:num_arma_params]

        true_ar_params.append(ar_padded)
        true_ma_params.append(ma_padded)

    true_ar = torch.tensor(np.array(true_ar_params), dtype=torch.float32).to(device)
    true_ma = torch.tensor(np.array(true_ma_params), dtype=torch.float32).to(device)

    return x, true_ar, true_ma


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(param_head, model, num_arma_params, device, num_samples=200, dimension=4):
    """Evaluate parameter recovery performance."""
    param_head.eval()

    all_ar_errors = []
    all_ma_errors = []
    all_total_errors = []
    all_baseline_errors = []

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

    results = {
        'mean_ar_error': float(np.mean(all_ar_errors)),
        'mean_ma_error': float(np.mean(all_ma_errors)),
        'mean_total_error': float(np.mean(all_total_errors)),
        'mean_baseline_error': float(np.mean(all_baseline_errors)),
        'std_ar_error': float(np.std(all_ar_errors)),
        'std_ma_error': float(np.std(all_ma_errors)),
        'std_total_error': float(np.std(all_total_errors)),
        'improvement_ratio': float(np.mean(all_baseline_errors) / np.mean(all_total_errors)),
    }

    return results


# =============================================================================
# Main training loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ARMA parameter recovery head")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--model-path", type=str, default="trained_simple_model_H1024.pth",
                        help="Path to pre-trained contrastive model")
    parser.add_argument("--model-type", type=str, default="mlp",
                        choices=["mlp", "gru", "resmlp", "attention", "grupool", "deepgru", "deepgrupool"],
                        help="Recovery head architecture")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for recovery head")
    parser.add_argument("--num-arma-params", type=int, default=4, help="Number of AR/MA params to recover")
    parser.add_argument("--dimension", type=int, default=4, help="ARMA dimension for data generation")
    parser.add_argument("--epochs", type=int, default=30000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"],
                        help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine", "plateau"],
                        help="LR scheduler type")
    parser.add_argument("--head-path", type=str, default=None,
                        help="Path to save/load recovery head (default: parameter_recovery_{model_type}.pth)")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--eval-samples", type=int, default=200, help="Number of evaluation samples")
    parser.add_argument("--log-every", type=int, default=100, help="Log every N epochs")
    parser.add_argument("--save-every", type=int, default=5000, help="Save checkpoint every N epochs")

    # Model-specific args
    parser.add_argument("--H", type=int, default=1024, help="Latent dim of pre-trained model")
    parser.add_argument("--C", type=int, default=4, help="Number of channels")
    parser.add_argument("--W", type=int, default=32, help="Patch size")
    parser.add_argument("--num-layers", type=int, default=12, help="Transformer layers in pre-trained model")

    args = parser.parse_args()

    if args.head_path is None:
        args.head_path = f"parameter_recovery_{args.model_type}.pth"

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    # Load pre-trained model
    print(f"Loading pre-trained model from {args.model_path}")
    model = SimpleModel(C=args.C, H=args.H, W=args.W, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Pre-trained model loaded and frozen")

    # Create recovery head
    param_head = create_recovery_head(
        args.model_type, H=args.H, hidden_dim=args.hidden_dim,
        num_arma_params=args.num_arma_params
    )

    # Count parameters
    num_params = sum(p.numel() for p in param_head.parameters() if p.requires_grad)
    print(f"Recovery head parameters: {num_params:,}")

    if args.evaluate:
        # Load and evaluate
        print(f"Loading recovery head from {args.head_path}")
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
        return

    param_head = param_head.to(device)

    # Setup optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(param_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(param_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-6)

    # Fixed validation set
    x_val, ar_val, ma_val = prepare_training_data(
        batch_size=32, num_arma_params=args.num_arma_params, device=device,
        seed=0, dimension=args.dimension
    )
    h_val = extract_latent_features(model, x_val)

    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Save path: {args.head_path}")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # Training step
        param_head.train()
        optimizer.zero_grad()

        x_train, ar_train, ma_train = prepare_training_data(
            batch_size=args.batch_size, num_arma_params=args.num_arma_params,
            device=device, seed=epoch, dimension=args.dimension
        )
        h_train = extract_latent_features(model, x_train)

        pred_ar, pred_ma = param_head(h_train)
        loss, ar_loss, ma_loss = parameter_loss(pred_ar, pred_ma, ar_train, ma_train)

        loss.backward()
        optimizer.step()

        # Validation
        param_head.eval()
        with torch.no_grad():
            pred_ar_val, pred_ma_val = param_head(h_val)
            val_loss, val_ar_loss, val_ma_loss = parameter_loss(pred_ar_val, pred_ma_val, ar_val, ma_val)

        val_loss_item = val_loss.item()

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss_item)
            else:
                scheduler.step()

        if val_loss_item < best_val_loss:
            best_val_loss = val_loss_item
            best_epoch = epoch
            torch.save(param_head.state_dict(), args.head_path.replace('.pth', '_best.pth'))

        if epoch % args.log_every == 0:
            lr_current = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch:6d}] train={loss.item():.6f} (AR={ar_loss.item():.6f} MA={ma_loss.item():.6f}) | "
                  f"val={val_loss_item:.6f} (AR={val_ar_loss.item():.6f} MA={val_ma_loss.item():.6f}) | "
                  f"best={best_val_loss:.6f}@{best_epoch} | lr={lr_current:.2e}")

        if epoch % args.save_every == 0:
            torch.save(param_head.state_dict(), args.head_path)

    # Final save
    torch.save(param_head.state_dict(), args.head_path)
    print(f"\nTraining complete. Final model saved to {args.head_path}")
    print(f"Best model (val_loss={best_val_loss:.6f} at epoch {best_epoch}) saved to {args.head_path.replace('.pth', '_best.pth')}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    # Load best model for evaluation
    param_head.load_state_dict(torch.load(args.head_path.replace('.pth', '_best.pth'), map_location=device))
    results = evaluate(param_head, model, args.num_arma_params, device,
                       num_samples=args.eval_samples, dimension=args.dimension)

    print(f"Mean AR Error:    {results['mean_ar_error']:.6f} +/- {results['std_ar_error']:.6f}")
    print(f"Mean MA Error:    {results['mean_ma_error']:.6f} +/- {results['std_ma_error']:.6f}")
    print(f"Mean Total Error: {results['mean_total_error']:.6f} +/- {results['std_total_error']:.6f}")
    print(f"Baseline Error:   {results['mean_baseline_error']:.6f}")
    print(f"Improvement:      {results['improvement_ratio']:.2f}x better than zero-baseline")

    # Save results
    results_path = args.head_path.replace('.pth', '_results.json')
    results['model_type'] = args.model_type
    results['hidden_dim'] = args.hidden_dim
    results['epochs'] = args.epochs
    results['lr'] = args.lr
    results['best_epoch'] = best_epoch
    results['best_val_loss'] = best_val_loss
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
