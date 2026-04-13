#!/usr/bin/env python3
"""
Training script for the forecasting head on top of a frozen contrastive backbone.

Trains a GRU-based ForecastingHead to decode backbone forecaster latents
into future normalized values. The backbone is kept frozen; only the head
is trained.

Usage:
    # Train from scratch
    python scripts/train_forecasting_head.py \
        --backbone-path checkpoints/tiny_best_gap.pth \
        --device cuda:1

    # Resume from checkpoint
    python scripts/train_forecasting_head.py \
        --backbone-path checkpoints/tiny_best_gap.pth \
        --device cuda:1 \
        --resume checkpoints/forecast_head_best.pth
"""

import argparse
import csv
import math
import os
import sys
import time

import torch
import torch.optim as optim

from src.models import ConfigurableModel, count_parameters
from src.dataloader import create_hf_dataloader
from src.forecasting_head import (
    ForecastingHead,
    W,
    FORECAST_LEN,
    extract_forecaster_latents,
    compute_valid_targets,
)

# -- Backbone architecture (must match checkpoint) ---------------------------
BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)

# -- Head architecture -------------------------------------------------------
HEAD_CONFIG = dict(
    H=512, hidden_dim=128, num_gru_layers=2, forecast_len=FORECAST_LEN, dropout=0.1,
)

T_RAW = 1024


def parse_args():
    p = argparse.ArgumentParser(
        description="Train forecasting head on frozen contrastive backbone")
    p.add_argument("--backbone-path", required=True,
                   help="Path to trained backbone checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-dir", default="checkpoints",
                   help="Directory for all checkpoints")
    p.add_argument("--run-name", default="forecast_head",
                   help="Run name prefix for checkpoint files")
    p.add_argument("--resume", default=None,
                   help="Path to forecasting head checkpoint to resume from")
    p.add_argument("--log-every", type=int, default=500,
                   help="Log metrics every N steps")
    p.add_argument("--save-every", type=int, default=5000,
                   help="Save snapshot every N steps (never overwritten)")
    p.add_argument("--hf-repo",
                   default="jeremycochoy/contrastive-training-tiny-bundles",
                   help="HuggingFace dataset repo ID for streaming")
    p.add_argument("--hf-path", default="tiny_mixed_v1",
                   help="Subdirectory within the HF repo")
    p.add_argument("--skip-rows", type=int, default=0,
                   help="HF rows to skip (for data position resume)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


class CSVLogger:
    """Buffered per-step loss CSV logger."""

    def __init__(self, path: str, flush_every: int = 100):
        self.path = path
        self.flush_every = flush_every
        self._buffer = []
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if os.path.getsize(path) == 0:
            self._writer.writerow(["step", "loss", "hf_rows_consumed"])
            self._file.flush()

    def log(self, step: int, loss: float, hf_rows_consumed: int):
        self._buffer.append([step, loss, hf_rows_consumed])
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if self._buffer:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer = []

    def close(self):
        self.flush()
        self._file.close()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # -- Load frozen backbone --------------------------------------------------
    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    backbone.load_state_dict(
        torch.load(args.backbone_path, map_location=device, weights_only=True))
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print(f"Backbone loaded from {args.backbone_path} "
          f"({count_parameters(backbone):,} params, frozen)")

    # -- Forecasting head ------------------------------------------------------
    head = ForecastingHead(**HEAD_CONFIG).to(device)
    n_head_params = count_parameters(head)
    print(f"Forecasting head: {n_head_params:,} trainable params")

    optimizer = optim.AdamW(head.parameters(), lr=args.lr)

    # -- Resume ----------------------------------------------------------------
    start_step = 0
    best_loss = float("inf")
    best_loss_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        head.load_state_dict(ckpt)
        # Try loading optimizer + metadata from companion file
        optim_path = args.resume.replace(".pth", "_optimizer.pth")
        if os.path.exists(optim_path):
            meta = torch.load(optim_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(meta["optimizer_state_dict"])
            start_step = meta.get("step", 0)
            best_loss = meta.get("best_loss", float("inf"))
            best_loss_step = meta.get("best_loss_step", 0)
            print(f"Resumed from {args.resume} at step {start_step} "
                  f"(best_loss={best_loss:.6f})")
        else:
            print(f"Loaded head weights from {args.resume} (no optimizer state)")

    # -- CSV logger ------------------------------------------------------------
    csv_path = os.path.join(args.save_dir, f"{args.run_name}_losses.csv")
    csv_logger = CSVLogger(csv_path)
    print(f"Loss CSV: {csv_path}")

    # -- Data ------------------------------------------------------------------
    C = BACKBONE_CONFIG["C"]
    rows_per_step = args.batch_size * C
    hf_rows_consumed = start_step * rows_per_step + args.skip_rows

    data_loader = create_hf_dataloader(
        args.hf_repo, batch_size=args.batch_size, C=C,
        path_in_repo=args.hf_path, skip_rows=hf_rows_consumed)
    data_iter = iter(data_loader)
    print(f"Data: HF streaming from {args.hf_repo}/{args.hf_path} "
          f"(skip={hf_rows_consumed} rows)")

    print(f"\nTraining for {args.total_steps} steps, bs={args.batch_size}, "
          f"lr={args.lr}")
    print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")
    sys.stdout.flush()

    # -- Training loop ---------------------------------------------------------
    t0 = time.time()
    ema_loss = None
    ema_decay = 0.99

    for step in range(start_step + 1, args.total_steps + 1):
        head.train()
        optimizer.zero_grad()

        # Data loading
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            x = next(data_iter)
        hf_rows_consumed += rows_per_step
        x = x.to(device)

        # Extract forecaster latents from frozen backbone
        f_bc, x_norm = extract_forecaster_latents(backbone, x)

        # Compute valid targets
        targets, T_valid = compute_valid_targets(x_norm, W=W, forecast_len=FORECAST_LEN)
        targets = targets.to(device)

        # Forward through head
        preds = head(f_bc)[:, :T_valid, :]

        # MSE loss
        loss = torch.nn.functional.mse_loss(preds, targets)

        # NaN detection
        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n*** NaN/Inf DETECTED at step {step} ***")
            emerg_path = os.path.join(
                args.save_dir, f"{args.run_name}_EMERGENCY_{step}.pth")
            torch.save(head.state_dict(), emerg_path)
            print(f"  Emergency checkpoint: {emerg_path}")
            csv_logger.close()
            sys.exit(1)

        # Backward + step
        loss.backward()
        optimizer.step()

        # EMA tracking
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss_val

        # Per-step CSV logging
        csv_logger.log(step, loss_val, hf_rows_consumed)

        # Console logging
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step - start_step) / elapsed
            eta = (args.total_steps - step) / sps / 3600

            print(f"[{step:>7d}] loss={loss_val:.6f}  ema_loss={ema_loss:.6f}  "
                  f"{sps:.1f} sps  ETA {eta:.1f}h")
            sys.stdout.flush()

        # Best checkpoint
        if step % args.log_every == 0 and ema_loss < best_loss:
            best_loss = ema_loss
            best_loss_step = step
            path = os.path.join(args.save_dir, f"{args.run_name}_best.pth")
            torch.save(head.state_dict(), path)
            _save_optim_meta(optimizer, path, step, best_loss, best_loss_step)
            print(f"  -> New best: {path} (ema_loss={ema_loss:.6f})")

        # Periodic snapshot
        if step % args.save_every == 0:
            path = os.path.join(
                args.save_dir, f"{args.run_name}_{step // 1000}k.pth")
            torch.save(head.state_dict(), path)
            _save_optim_meta(optimizer, path, step, best_loss, best_loss_step)
            print(f"  -> Saved {path}")

    # -- Final save ------------------------------------------------------------
    path = os.path.join(args.save_dir, f"{args.run_name}_final.pth")
    torch.save(head.state_dict(), path)
    _save_optim_meta(optimizer, path, args.total_steps, best_loss, best_loss_step)
    csv_logger.close()

    total = time.time() - t0
    print(f"\nDone in {total / 3600:.1f}h. "
          f"Best loss={best_loss:.6f} at step {best_loss_step}")


def _save_optim_meta(optimizer, model_path, step, best_loss, best_loss_step):
    """Save optimizer state and metadata to companion file."""
    optim_path = model_path.replace(".pth", "_optimizer.pth")
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_loss": best_loss,
        "best_loss_step": best_loss_step,
    }, optim_path)


if __name__ == "__main__":
    main()
