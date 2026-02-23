#!/usr/bin/env python3
"""
Contrastive learning training pipeline for ARMA time series forecasting.
Converted from forecast_arma.ipynb.

Usage:
    python train_contrastive.py --device cuda --total-steps 2000000 --batch-size 16
    python train_contrastive.py --device cuda:1 --total-steps 100000 --lr 1e-4 --save-path my_model.pth
"""

import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from types import SimpleNamespace

from arma import generate_arma_batch
from loss import contrastive_latent_loss
from network import SimpleModel


def generate_random_batch(batch_size=16, T_raw=4096, C=4, mean=0.0, std=1.0, seed=None, dimension=4):
    """Generate multivariate random walks using ARMA processes."""
    X, _ = generate_arma_batch(batch_size=batch_size, T_raw=T_raw, C=C, mean=mean, std=std, seed=seed, dimension=dimension)
    return X


def compute_metrics(f_lat, o_lat, cld):
    """Compute forecast-future, forecast-past, future-past, and cross-batch cosine similarities."""
    fn = F.normalize(f_lat, p=2, dim=-1)
    on = F.normalize(o_lat, p=2, dim=-1)
    hyh = fn[:, :-cld, :, :]
    hyn = on[:, cld:, :, :]
    hxn = on[:, :-cld, :, :]

    ff = (hyh * hyn).sum(-1).mean().item()
    fp = (hyh * hxn).sum(-1).mean().item()
    tp = (hyn * hxn).sum(-1).mean().item()

    B, T, C, H = hyh.shape
    hyh_exp = hyh.unsqueeze(0)
    hyn_exp = hyn.unsqueeze(1)
    sims_cross_batch = (hyh_exp * hyn_exp).sum(-1)
    mask_batch = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch.device)
    mask_batch = mask_batch.view(B, B, 1, 1)
    sims_masked = sims_cross_batch.masked_fill(~mask_batch, 0)
    cross_batch = sims_masked.mean().item()

    return ff, fp, tp, cross_batch


def train_step(model, optimizer, C, H, W, batch_size, device, spec, dimension=4):
    """Execute a single training step."""
    model.train()
    optimizer.zero_grad()

    x_train = generate_random_batch(batch_size, T_raw=4096, C=C, dimension=dimension).to(device)
    Bt = x_train.shape[0]
    T = x_train.shape[1] // W

    x_train = x_train.view(Bt, T, W, C).permute(0, 1, 3, 2)

    f_flat, o_flat = model.transformer(x_train)
    f_lat = f_flat.reshape(Bt, C, T, H).permute(0, 2, 1, 3)
    o_lat = o_flat.reshape(Bt, C, T, H).permute(0, 2, 1, 3)

    loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=spec)
    loss.backward()
    optimizer.step()

    return loss.item(), f_lat.detach(), o_lat.detach()


def validation_step(model, x_val, spec, H):
    """Execute validation step."""
    model.eval()
    with torch.no_grad():
        fv_flat, ov_flat = model.transformer(x_val)
        Bv, T, C, W = x_val.shape
        fv = fv_flat.reshape(Bv, C, T, H).permute(0, 2, 1, 3)
        ov = ov_flat.reshape(Bv, C, T, H).permute(0, 2, 1, 3)
        return fv, ov


def main():
    parser = argparse.ArgumentParser(description="Train contrastive forecasting model on ARMA data")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--C", type=int, default=4, help="Number of channels")
    parser.add_argument("--H", type=int, default=1024, help="Latent embedding dimension")
    parser.add_argument("--W", type=int, default=32, help="Patch/window size")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--dimension", type=int, default=4, help="ARMA dimension (max p,q)")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive loss temperature")
    parser.add_argument("--loss-shape", type=str, default="cosine_similarity_batch_no_time_neg",
                        help="Loss function variant")
    parser.add_argument("--val-every", type=int, default=500, help="Validate every N steps")
    parser.add_argument("--save-every", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--save-path", type=str, default="trained_simple_model.pth", help="Model save path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--T-raw", type=int, default=4096, help="Raw time series length")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize model
    model = SimpleModel(C=args.C, H=args.H, W=args.W, num_layers=args.num_layers)

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Fixed validation set
    x_val = generate_random_batch(args.batch_size, T_raw=args.T_raw, C=args.C, seed=0, dimension=args.dimension).to(device)
    Bv, Tr, _ = x_val.shape
    T = Tr // args.W
    x_val = x_val.view(Bv, T, args.W, args.C).permute(0, 1, 3, 2)

    # Training configuration
    spec = SimpleNamespace(train_configuration={
        'contrastive_divergence_temperature': args.temperature,
        'contrastive_latent_noise': None,
        'loss_shape': args.loss_shape,
        'contrastive_latent_delay': 0
    })
    cld = spec.train_configuration['contrastive_latent_delay'] + 1

    print(f"Model: SimpleModel(C={args.C}, H={args.H}, W={args.W}, num_layers={args.num_layers})")
    print(f"Training for {args.total_steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Loss: {args.loss_shape}, temperature={args.temperature}")

    # Training loop
    for step in range(start_step + 1, args.total_steps + 1):
        loss_val, f_lat, o_lat = train_step(
            model, optimizer, args.C, args.H, args.W,
            args.batch_size, device, spec, args.dimension
        )

        if step % args.val_every == 0 or step == args.total_steps:
            train_ff, train_fp, train_tp, train_cb = compute_metrics(f_lat, o_lat, cld)

            fv, ov = validation_step(model, x_val, spec, args.H)
            val_ff, val_fp, val_tp, val_cb = compute_metrics(fv, ov, cld)

            print(f"[Step {step}] loss={loss_val:.4f} | "
                  f"train FF={train_ff:.4f} FP={train_fp:.4f} TP={train_tp:.4f} CB={train_cb:.4f} | "
                  f"val FF={val_ff:.4f} FP={val_fp:.4f} TP={val_tp:.4f} CB={val_cb:.4f}")

        if step % args.save_every == 0:
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Checkpoint saved to {args.save_path}")

    # Final save
    torch.save(model.state_dict(), args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
