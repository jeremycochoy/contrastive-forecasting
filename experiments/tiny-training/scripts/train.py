#!/usr/bin/env python3
"""
Contrastive forecasting training script (v3).

Trains the Tiny backbone on synthetic composite ARIMA data (TimesFM recipe)
with RevEWMNorm. Uses rolling EMA of train loss/gap for best checkpoint
selection. Saves periodic snapshots that are never overwritten.

Usage:
    # Train from scratch
    python scripts/train_v3.py --device cuda

    # Resume from checkpoint
    python scripts/train_v3.py --device cuda --resume checkpoints/model_200k.pth

    # Custom LR
    python scripts/train_v3.py --device cuda --lr 3e-4
"""

import argparse
import os
import time
import torch
import torch.optim as optim
from types import SimpleNamespace

from src.models import ConfigurableModel, compute_metrics, count_parameters
from src.synthetic import generate_synthetic_batch
from src.dataloader import create_dataloader
from src.loss import contrastive_latent_loss
from src.checkpoint import save_training_state, load_training_state, safe_save_path

# -- Tiny architecture ---------------------------------------------------------
MODEL_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)

# -- Loss configuration --------------------------------------------------------
LOSS_SPEC = SimpleNamespace(train_configuration={
    "contrastive_divergence_temperature": 0.07,
    "contrastive_latent_noise": None,
    "loss_shape": "cosine_similarity_batch_no_time_neg",
    "contrastive_latent_delay": 0,
})
CLD = LOSS_SPEC.train_configuration["contrastive_latent_delay"] + 1

# -- Data configuration --------------------------------------------------------
T_RAW = 1024
DIMENSION = 8  # ARMA(p,q) with p,q in [1,8]


def parse_args():
    p = argparse.ArgumentParser(description="Train contrastive forecasting model (v3)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-steps", type=int, default=500000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", default="checkpoints",
                   help="Directory for all checkpoints")
    p.add_argument("--run-name", default="tiny",
                   help="Run name prefix for checkpoint files")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--log-every", type=int, default=100,
                   help="Log metrics every N steps")
    p.add_argument("--save-every", type=int, default=100000,
                   help="Save snapshot every N steps (never overwritten)")
    p.add_argument("--ema-decay", type=float, default=0.99,
                   help="EMA decay for rolling loss/gap tracking")
    p.add_argument("--data-dir", default=None,
                   help="Directory of parquet shards (HF data). "
                        "If omitted, uses synthetic data only.")
    return p.parse_args()


def random_sign_flip(x: torch.Tensor) -> torch.Tensor:
    """Randomly flip the sign of each channel independently.

    For each (batch, channel) pair, flip the sign with 50% probability.
    This is a free augmentation: if x(t) is a valid series, -x(t) is too.
    """
    B, T, C = x.shape
    signs = torch.where(torch.rand(B, 1, C, device=x.device) < 0.5,
                        torch.ones(1, device=x.device),
                        -torch.ones(1, device=x.device))
    return x * signs


def forward_step(model, x):
    """Apply RevEWMNorm + transformer (skip channel mixing)."""
    W = model.W
    H = model.H
    if model.rev_norm is not None:
        x = model.rev_norm(x, mode='norm')
    B, T_raw, C = x.shape
    T = T_raw // W
    xr = x.view(B, T, W, C).permute(0, 1, 3, 2)
    f_flat, o_flat = model.transformer(xr)
    f_lat = f_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    o_lat = o_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    return f_lat, o_lat


def save_snapshot(model, optimizer, path, step, best_gap, best_gap_step,
                  best_loss, best_loss_step):
    """Save model + optimizer state to a unique path."""
    torch.save(model.state_dict(), path)
    save_training_state(optimizer, path, step=step,
                        best_val_ff=best_gap, best_step=best_gap_step)
    print(f"  -> Saved {path}")


def _has_checkpoints(save_dir, run_name):
    """Check if any checkpoint files exist for this run name."""
    import glob
    return len(glob.glob(os.path.join(save_dir, f"{run_name}_*.pth"))) > 0


def safe_run_name(save_dir, run_name):
    """If any checkpoints already exist for this run name, auto-increment.

    Prevents overwriting best/periodic checkpoints when restarting a run
    (e.g., after a spike or with a new LR).
    """
    if not _has_checkpoints(save_dir, run_name):
        return run_name

    n = 2
    while True:
        candidate = f"{run_name}_r{n}"
        if not _has_checkpoints(save_dir, candidate):
            print(f"  [checkpoint] Run name '{run_name}' has existing "
                  f"checkpoints. Branching to '{candidate}'.")
            return candidate
        n += 1


def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.save_dir, exist_ok=True)

    # Protect existing checkpoints when restarting
    if args.resume:
        args.run_name = safe_run_name(args.save_dir, args.run_name)

    # -- Model -----------------------------------------------------------------
    model = ConfigurableModel(**MODEL_CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # -- Resume ----------------------------------------------------------------
    start_step = 0
    best_gap, best_gap_step = -float("inf"), 0
    best_loss, best_loss_step = float("inf"), 0
    ema_loss, ema_gap = None, None

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        restored = load_training_state(optimizer, args.resume, device=device)
        start_step = restored["step"]
        best_gap = restored["best_val_ff"]
        best_gap_step = restored["best_step"]
        print(f"Resumed from {args.resume} at step {start_step}")

    print(f"Device: {device} | Params: {count_parameters(model):,}")
    print(f"Training for {args.total_steps} steps, bs={args.batch_size}, "
          f"lr={args.lr}, T={T_RAW}")
    print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")

    # -- Data source -----------------------------------------------------------
    data_iter = None
    if args.data_dir:
        data_loader = create_dataloader(
            args.data_dir, batch_size=args.batch_size,
            C=MODEL_CONFIG["C"], shuffle=True)
        data_iter = iter(data_loader)
        print(f"Data: parquet shards from {args.data_dir} "
              f"({len(data_loader)} batches/epoch)")
    else:
        print("Data: synthetic only (no --data-dir)")

    # -- Training loop ---------------------------------------------------------
    t0 = time.time()
    for step in range(start_step + 1, args.total_steps + 1):
        model.train()
        optimizer.zero_grad()

        # Get batch from parquet shards or synthetic fallback
        if data_iter is not None:
            try:
                x = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x = next(data_iter)
        else:
            x, = generate_synthetic_batch(
                batch_size=args.batch_size, T_raw=T_RAW,
                C=MODEL_CONFIG["C"], dimension=DIMENSION)

        x = x.to(device)
        x = random_sign_flip(x)

        f_lat, o_lat = forward_step(model, x)
        loss = contrastive_latent_loss((f_lat, o_lat), validation=False,
                                       spec=LOSS_SPEC)
        loss.backward()
        optimizer.step()

        # -- Rolling metrics (EMA) ---------------------------------------------
        loss_val = loss.item()
        with torch.no_grad():
            val_ff, val_fp, _, _ = compute_metrics(f_lat, o_lat, CLD)
        gap_val = val_ff - val_fp

        if ema_loss is None:
            ema_loss = loss_val
            ema_gap = gap_val
        else:
            d = args.ema_decay
            ema_loss = d * ema_loss + (1 - d) * loss_val
            ema_gap = d * ema_gap + (1 - d) * gap_val

        # -- Logging -----------------------------------------------------------
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step - start_step) / elapsed
            eta = (args.total_steps - step) / sps / 3600

            print(f"[{step:>7d}] loss={loss_val:.4f}  ema_loss={ema_loss:.4f}  "
                  f"gap={gap_val:.4f}  ema_gap={ema_gap:.4f}  "
                  f"{sps:.1f} sps  ETA {eta:.1f}h")

        # -- Best gap checkpoint -----------------------------------------------
        if ema_gap > best_gap:
            best_gap, best_gap_step = ema_gap, step
            path = os.path.join(args.save_dir,
                                f"{args.run_name}_best_gap.pth")
            save_snapshot(model, optimizer, path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step)

        # -- Best loss checkpoint ----------------------------------------------
        if ema_loss < best_loss:
            best_loss, best_loss_step = ema_loss, step
            path = os.path.join(args.save_dir,
                                f"{args.run_name}_best_loss.pth")
            save_snapshot(model, optimizer, path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step)

        # -- Periodic snapshot (never overwritten) -----------------------------
        if step % args.save_every == 0:
            path = os.path.join(args.save_dir,
                                f"{args.run_name}_{step // 1000}k.pth")
            save_snapshot(model, optimizer, path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step)

    # -- Final save ------------------------------------------------------------
    path = os.path.join(args.save_dir, f"{args.run_name}_final.pth")
    save_snapshot(model, optimizer, path, args.total_steps,
                  best_gap, best_gap_step, best_loss, best_loss_step)

    total = time.time() - t0
    print(f"\nDone in {total/3600:.1f}h. "
          f"Best gap={best_gap:.4f} at step {best_gap_step}, "
          f"Best loss={best_loss:.4f} at step {best_loss_step}")


if __name__ == "__main__":
    main()
