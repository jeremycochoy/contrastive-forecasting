#!/usr/bin/env python3
"""
Enhanced contrastive learning training pipeline with configurable architecture.
Supports multiple encoder types and transformer configurations.

Usage:
    # Baseline (matches original)
    python train_contrastive_v2.py --device cuda --encoder-type mlp

    # GRU encoder with more heads
    python train_contrastive_v2.py --device cuda --encoder-type gru --nhead 16

    # TimeFM-like config
    python train_contrastive_v2.py --device cuda --encoder-type residual_silu --H 1280 --num-layers 20 --nhead 16 --ffn-mult 1
"""

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from types import SimpleNamespace

from arma import generate_arma_batch
from loss import contrastive_latent_loss
from encoders import create_encoder
from blocks import TransformerBlock, Simple_channel_mixing_module
from checkpoint import save_training_state, load_training_state


class ConfigurableModel(torch.nn.Module):
    """SimpleModel with configurable encoder and transformer."""
    def __init__(self, C, H, W, encoder_type='mlp', intermediate_dim=None,
                 num_layers=12, nhead=4, ffn_mult=2, dropout=0.1,
                 activation='gelu', depthwise_conv=3):
        super().__init__()
        self.C = C
        self.H = H
        self.W = W

        self.encoder = create_encoder(encoder_type, W, H, intermediate_dim)

        self.transformer = TransformerBlock(
            dimension_e=H,
            nhead=nhead,
            num_layers=num_layers,
            feedforward_mult=ffn_mult,
            dropout=dropout,
            input_to_latent=self.encoder,
            depthwise_conv=depthwise_conv,
        )
        # Override activation if requested
        if activation != 'gelu':
            act_fn = torch.nn.functional.silu if activation == 'silu' else torch.nn.functional.gelu
            for layer in self.transformer.layers:
                layer.activation = act_fn

        self.channel_mixing_module = Simple_channel_mixing_module(H=H, C=C)

    def forward(self, x):
        B, T_raw, C = x.shape
        W = self.W
        H = self.H
        assert T_raw % W == 0
        T = T_raw // W
        x = x.view(B, T, W, C).permute(0, 1, 3, 2)  # [B, T, C, W]
        x, x_original = self.transformer(x)
        x = x.reshape(B, C, T, H).permute(0, 2, 1, 3).reshape(B, T, C * H)
        x_original = x_original.reshape(B, C, T, H).permute(0, 2, 1, 3)
        x = self.channel_mixing_module(x)
        x = x.reshape(B, T, C, H)
        return x, x_original


def generate_random_batch(batch_size=16, T_raw=4096, C=4, seed=None, dimension=4):
    X, _ = generate_arma_batch(batch_size=batch_size, T_raw=T_raw, C=C, seed=seed, dimension=dimension)
    return X


def compute_metrics(f_lat, o_lat, cld):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Architecture search for contrastive forecasting")
    # Architecture
    parser.add_argument("--encoder-type", type=str, default="mlp",
                        choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv"])
    parser.add_argument("--intermediate-dim", type=int, default=None,
                        help="Encoder intermediate dimension (default varies by type)")
    parser.add_argument("--H", type=int, default=1024, help="Latent embedding dimension")
    parser.add_argument("--W", type=int, default=32, help="Patch/window size")
    parser.add_argument("--C", type=int, default=4, help="Number of channels")
    parser.add_argument("--num-layers", type=int, default=12, help="Transformer layers")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads")
    parser.add_argument("--ffn-mult", type=float, default=2.0, help="FFN dimension multiplier")
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    parser.add_argument("--depthwise-conv", type=int, default=3, help="Causal conv kernel (0=disable)")
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dimension", type=int, default=4, help="ARMA dimension")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss-shape", type=str, default="cosine_similarity_batch_no_time_neg")
    parser.add_argument("--T-raw", type=int, default=4096)
    # Logging
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--save-path", type=str, default="arch_search_model.pth")
    parser.add_argument("--experiment-id", type=str, default="default",
                        help="Experiment ID for logging")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize model
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

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    model = model.to(device)
    n_params = count_parameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Restore optimizer state, step counter, and best-tracking if resuming
    start_step = 0
    best_val_ff_restored = -float('inf')
    best_step_restored = 0
    if args.resume:
        restored = load_training_state(optimizer, args.resume, device=device)
        start_step = restored['step']
        best_val_ff_restored = restored['best_val_ff']
        best_step_restored = restored['best_step']

    # Fixed validation set
    x_val = generate_random_batch(args.batch_size, T_raw=args.T_raw, C=args.C, seed=0, dimension=args.dimension).to(device)

    spec = SimpleNamespace(train_configuration={
        'contrastive_divergence_temperature': args.temperature,
        'contrastive_latent_noise': None,
        'loss_shape': args.loss_shape,
        'contrastive_latent_delay': 0
    })
    cld = spec.train_configuration['contrastive_latent_delay'] + 1

    # Print config
    config = {
        'experiment_id': args.experiment_id,
        'encoder_type': args.encoder_type,
        'intermediate_dim': args.intermediate_dim,
        'H': args.H, 'W': args.W, 'C': args.C,
        'num_layers': args.num_layers, 'nhead': args.nhead,
        'ffn_mult': args.ffn_mult, 'activation': args.activation,
        'depthwise_conv': args.depthwise_conv, 'dropout': args.dropout,
        'total_steps': args.total_steps, 'batch_size': args.batch_size,
        'lr': args.lr, 'dimension': args.dimension,
        'temperature': args.temperature, 'loss_shape': args.loss_shape,
        'n_params': n_params,
    }
    print(f"Experiment: {args.experiment_id}")
    print(f"Model: encoder={args.encoder_type}, H={args.H}, layers={args.num_layers}, "
          f"nhead={args.nhead}, ffn_mult={args.ffn_mult}, act={args.activation}, "
          f"conv_k={args.depthwise_conv}")
    print(f"Parameters: {n_params:,}")
    print(f"Training: {args.total_steps} steps, bs={args.batch_size}, lr={args.lr}")

    # Metrics tracking (restored from checkpoint if resuming)
    best_val_ff = best_val_ff_restored
    best_step = best_step_restored
    metrics_log = []
    start_time = time.time()

    # Training loop
    for step in range(start_step + 1, args.total_steps + 1):
        model.train()
        optimizer.zero_grad()

        x_train = generate_random_batch(args.batch_size, T_raw=args.T_raw, C=args.C, dimension=args.dimension).to(device)
        B, T_raw_actual, C_actual = x_train.shape
        T = T_raw_actual // args.W
        x_reshaped = x_train.view(B, T, args.W, C_actual).permute(0, 1, 3, 2)

        f_flat, o_flat = model.transformer(x_reshaped)
        f_lat = f_flat.reshape(B, C_actual, T, args.H).permute(0, 2, 1, 3)
        o_lat = o_flat.reshape(B, C_actual, T, args.H).permute(0, 2, 1, 3)

        loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=spec)
        loss.backward()
        optimizer.step()

        if step % args.val_every == 0 or step == args.total_steps:
            train_ff, train_fp, train_tp, train_cb = compute_metrics(f_lat.detach(), o_lat.detach(), cld)

            model.eval()
            with torch.no_grad():
                Bv, Tr, Cv = x_val.shape
                Tv = Tr // args.W
                xv = x_val.view(Bv, Tv, args.W, Cv).permute(0, 1, 3, 2)
                fv, ov = model.transformer(xv)
                fv = fv.reshape(Bv, Cv, Tv, args.H).permute(0, 2, 1, 3)
                ov = ov.reshape(Bv, Cv, Tv, args.H).permute(0, 2, 1, 3)
                val_ff, val_fp, val_tp, val_cb = compute_metrics(fv, ov, cld)

            elapsed = time.time() - start_time
            steps_done = step - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            eta_min = (args.total_steps - step) / steps_per_sec / 60

            entry = {
                'step': step, 'loss': loss.item(),
                'train_ff': train_ff, 'train_fp': train_fp, 'train_tp': train_tp, 'train_cb': train_cb,
                'val_ff': val_ff, 'val_fp': val_fp, 'val_tp': val_tp, 'val_cb': val_cb,
                'val_ff_fp_gap': val_ff - val_fp,
                'elapsed_sec': elapsed, 'steps_per_sec': steps_per_sec,
            }
            metrics_log.append(entry)

            print(f"[Step {step}] loss={loss.item():.4f} | "
                  f"train FF={train_ff:.4f} FP={train_fp:.4f} TP={train_tp:.4f} CB={train_cb:.4f} | "
                  f"val FF={val_ff:.4f} FP={val_fp:.4f} TP={val_tp:.4f} CB={val_cb:.4f} | "
                  f"gap={val_ff-val_fp:.4f} | {steps_per_sec:.1f} step/s | ETA {eta_min:.0f}min")

            # Track best
            if val_ff > best_val_ff:
                best_val_ff = val_ff
                best_step = step
                best_path = args.save_path.replace('.pth', '_best.pth')
                torch.save(model.state_dict(), best_path)
                save_training_state(optimizer, best_path,
                                    step=step, best_val_ff=best_val_ff,
                                    best_step=best_step)

        if step % args.save_every == 0:
            torch.save(model.state_dict(), args.save_path)
            save_training_state(optimizer, args.save_path,
                                step=step, best_val_ff=best_val_ff,
                                best_step=best_step)
            print(f"  -> Checkpoint saved to {args.save_path}")

    # Final save
    torch.save(model.state_dict(), args.save_path)
    save_training_state(optimizer, args.save_path,
                        step=step, best_val_ff=best_val_ff,
                        best_step=best_step)
    total_time = time.time() - start_time

    # Save results
    results = {
        **config,
        'best_val_ff': best_val_ff,
        'best_step': best_step,
        'total_time_sec': total_time,
        'final_metrics': metrics_log[-1] if metrics_log else None,
        'metrics_log': metrics_log,
    }
    results_path = f"arch_search_{args.experiment_id}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val FF: {best_val_ff:.4f} at step {best_step}")
    print(f"Results saved to {results_path}")
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
