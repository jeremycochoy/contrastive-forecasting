#!/usr/bin/env python3
"""
Contrastive forecasting training script.

Trains the best architecture (GRU encoder, 12-layer transformer, H=1024)
on synthetic ARMA data using a contrastive latent loss. The model learns
to predict future latent states from past observations.

Usage:
    # Train from scratch on GPU
    python scripts/train.py --device cuda

    # Resume from a checkpoint
    python scripts/train.py --device cuda --resume checkpoint.pth

    # Custom training length
    python scripts/train.py --device cuda --total-steps 500000 --save-every 50000
"""

import argparse
import time
import torch
import torch.optim as optim
from types import SimpleNamespace

from src.models import ConfigurableModel, generate_random_batch, compute_metrics, count_parameters
from src.loss import contrastive_latent_loss
from src.checkpoint import save_training_state, load_training_state, safe_save_path

# -- Best architecture (from architecture search experiments) ----------------
# GRU encoder, 12 transformer layers, 8 heads, FFN 4x, GELU, causal conv k=3
MODEL_CONFIG = dict(
    C=4, H=1024, W=32,
    encoder_type="gru", num_layers=12, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
)

# -- Loss configuration -----------------------------------------------------
LOSS_SPEC = SimpleNamespace(train_configuration={
    "contrastive_divergence_temperature": 0.07,
    "contrastive_latent_noise": None,
    "loss_shape": "cosine_similarity_batch_no_time_neg",
    "contrastive_latent_delay": 0,
})
CLD = LOSS_SPEC.train_configuration["contrastive_latent_delay"] + 1


def parse_args():
    p = argparse.ArgumentParser(description="Train contrastive forecasting model")
    p.add_argument("--device", default="cuda", help="Device (cuda, cpu, cuda:1, ...)")
    p.add_argument("--total-steps", type=int, default=500000, help="Total training steps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-path", default="model.pth", help="Checkpoint save path")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--val-every", type=int, default=500, help="Validate every N steps")
    p.add_argument("--save-every", type=int, default=10000, help="Save checkpoint every N steps")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # -- Model ---------------------------------------------------------------
    model = ConfigurableModel(**MODEL_CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # -- Resume --------------------------------------------------------------
    start_step, best_gap, best_step = 0, -float("inf"), 0
    if args.resume:
        args.save_path = safe_save_path(args.save_path, args.resume)
        model.load_state_dict(torch.load(args.resume, map_location=device))
        restored = load_training_state(optimizer, args.resume, device=device)
        start_step = restored["step"]
        best_gap = restored["best_val_ff"]
        best_step = restored["best_step"]
        print(f"Resumed from {args.resume} at step {start_step}")

    print(f"Device: {device} | Params: {count_parameters(model):,}")
    print(f"Training for {args.total_steps} steps, bs={args.batch_size}, lr={args.lr}")

    # -- Fixed validation batch ----------------------------------------------
    x_val = generate_random_batch(args.batch_size, seed=0).to(device)

    # -- Training loop -------------------------------------------------------
    t0 = time.time()
    for step in range(start_step + 1, args.total_steps + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass through transformer only (skips channel mixing)
        x = generate_random_batch(args.batch_size).to(device)
        B, T_raw, C = x.shape
        W, H = MODEL_CONFIG["W"], MODEL_CONFIG["H"]
        T = T_raw // W
        xr = x.view(B, T, W, C).permute(0, 1, 3, 2)
        f_flat, o_flat = model.transformer(xr)
        f_lat = f_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
        o_lat = o_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)

        loss = contrastive_latent_loss((f_lat, o_lat), validation=False, spec=LOSS_SPEC)
        loss.backward()
        optimizer.step()

        # -- Validation & logging --------------------------------------------
        if step % args.val_every == 0 or step == args.total_steps:
            model.eval()
            with torch.no_grad():
                Bv, Tr, Cv = x_val.shape
                Tv = Tr // W
                xv = x_val.view(Bv, Tv, W, Cv).permute(0, 1, 3, 2)
                fv, ov = model.transformer(xv)
                fv = fv.reshape(Bv, Cv, Tv, H).permute(0, 2, 1, 3)
                ov = ov.reshape(Bv, Cv, Tv, H).permute(0, 2, 1, 3)
                val_ff, val_fp, _, _ = compute_metrics(fv, ov, CLD)

            gap = val_ff - val_fp
            elapsed = time.time() - t0
            sps = (step - start_step) / elapsed
            eta = (args.total_steps - step) / sps / 60

            print(f"[{step:>7d}] loss={loss.item():.4f}  "
                  f"FF={val_ff:.4f}  FP={val_fp:.4f}  gap={gap:.4f}  "
                  f"{sps:.1f} step/s  ETA {eta:.0f}min")

            # Save best by contrastive gap
            if gap > best_gap:
                best_gap, best_step = gap, step
                best_path = args.save_path.replace(".pth", "_best.pth")
                torch.save(model.state_dict(), best_path)
                save_training_state(optimizer, best_path,
                                    step=step, best_val_ff=best_gap, best_step=best_step)

        # -- Periodic checkpoint ---------------------------------------------
        if step % args.save_every == 0:
            torch.save(model.state_dict(), args.save_path)
            save_training_state(optimizer, args.save_path,
                                step=step, best_val_ff=best_gap, best_step=best_step)
            print(f"  -> Saved checkpoint to {args.save_path}")

    # -- Final save ----------------------------------------------------------
    torch.save(model.state_dict(), args.save_path)
    save_training_state(optimizer, args.save_path,
                        step=args.total_steps, best_val_ff=best_gap, best_step=best_step)
    total = time.time() - t0
    print(f"\nDone in {total/60:.1f} min. Best gap={best_gap:.4f} at step {best_step}")


if __name__ == "__main__":
    main()
