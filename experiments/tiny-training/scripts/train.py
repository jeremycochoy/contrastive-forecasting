#!/usr/bin/env python3
"""
Contrastive forecasting training script (v3).

Trains the Tiny backbone on synthetic composite ARIMA data (TimesFM recipe)
with RevEWMNorm. Uses rolling EMA of train loss/gap for best checkpoint
selection. Saves periodic snapshots that are never overwritten.

Features:
  - Per-step loss CSV logger (buffered writes every 100 steps)
  - NaN detection with emergency checkpoint + immediate stop
  - HF row counter for data traceability

Usage:
    # Train from scratch
    python scripts/train_v3.py --device cuda

    # Resume from checkpoint
    python scripts/train_v3.py --device cuda --resume checkpoints/model_200k.pth

    # Custom LR
    python scripts/train_v3.py --device cuda --lr 3e-4
"""

import argparse
import csv
import math
import os
import sys
import time
import torch
import torch.optim as optim
from types import SimpleNamespace

from src.models import ConfigurableModel, compute_metrics, count_parameters
from src.synthetic import generate_synthetic_batch
from src.dataloader import create_dataloader, create_hf_dataloader
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
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", default="checkpoints",
                   help="Directory for all checkpoints")
    p.add_argument("--run-name", default="tiny",
                   help="Run name prefix for checkpoint files")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--log-every", type=int, default=100,
                   help="Log metrics every N steps")
    p.add_argument("--save-every", type=int, default=10000,
                   help="Save snapshot every N steps (never overwritten)")
    p.add_argument("--ema-decay", type=float, default=0.99,
                   help="EMA decay for rolling loss/gap tracking")
    p.add_argument("--grad-clip", type=float, default=None,
                   help="Max gradient norm for clipping (None=disabled)")
    p.add_argument("--data-dir", default=None,
                   help="Directory of local parquet shards. "
                        "If omitted, uses --hf-repo or synthetic fallback.")
    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace dataset repo ID for streaming "
                        "(e.g. 'user/contrastive-training-tiny-bundles').")
    p.add_argument("--hf-path", default=None,
                   help="Subdirectory within the HF repo "
                        "(e.g. 'tiny_mixed_v1').")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
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
                  best_loss, best_loss_step, ema_loss=None, ema_gap=None,
                  hf_rows_consumed=0):
    """Save model + optimizer + complete training state to a unique path."""
    import numpy as _np
    torch.save(model.state_dict(), path)
    save_training_state(
        optimizer, path, step=step,
        best_val_ff=best_gap, best_step=best_gap_step,
        best_loss=best_loss, best_loss_step=best_loss_step,
        ema_loss=ema_loss, ema_gap=ema_gap,
        hf_rows_consumed=hf_rows_consumed,
        rng_state_torch=torch.get_rng_state(),
        rng_state_numpy=_np.random.get_state(),
    )
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


class CSVLogger:
    """Buffered per-step loss CSV logger.

    Writes step,loss,gap,ff,fp,hf_rows_consumed to a CSV file.
    Flushes to disk every `flush_every` steps to reduce I/O overhead.
    """

    def __init__(self, path: str, flush_every: int = 100):
        self.path = path
        self.flush_every = flush_every
        self._buffer = []
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        # Write header only if file is empty/new
        if os.path.getsize(path) == 0:
            self._writer.writerow([
                "step", "loss", "gap", "ff", "fp", "hf_rows_consumed"
            ])
            self._file.flush()

    def log(self, step: int, loss: float, gap: float, ff: float, fp: float,
            hf_rows_consumed: int):
        self._buffer.append([step, loss, gap, ff, fp, hf_rows_consumed])
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
    import numpy as _np
    _np.random.seed(args.seed)

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
        best_loss = restored.get("best_loss", float("inf"))
        best_loss_step = restored.get("best_loss_step", 0)
        ema_loss = restored.get("ema_loss", None)
        ema_gap = restored.get("ema_gap", None)
        # Restore RNG state for reproducibility
        if restored.get("rng_state_torch") is not None:
            torch.set_rng_state(restored["rng_state_torch"])
        if restored.get("rng_state_numpy") is not None:
            import numpy as _np2
            _np2.random.set_state(restored["rng_state_numpy"])
        print(f"Resumed from {args.resume} at step {start_step}")

    print(f"Device: {device} | Params: {count_parameters(model):,}")
    print(f"Training for {args.total_steps} steps, bs={args.batch_size}, "
          f"lr={args.lr}, T={T_RAW}")
    print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")

    # -- CSV logger ------------------------------------------------------------
    csv_path = os.path.join(args.save_dir, f"{args.run_name}_losses.csv")
    csv_logger = CSVLogger(csv_path, flush_every=100)
    print(f"Loss CSV: {csv_path}")

    # -- Data source -----------------------------------------------------------
    rows_per_step = args.batch_size * MODEL_CONFIG["C"]  # bs * C rows per step
    # Use restored count if available (accounts for skipped all-NaN rows);
    # fall back to step-based estimate for old checkpoints.
    if args.resume and restored.get("hf_rows_consumed", 0) > 0:
        hf_rows_consumed = restored["hf_rows_consumed"]
    else:
        hf_rows_consumed = start_step * rows_per_step

    data_iter = None
    if args.data_dir:
        data_loader = create_dataloader(
            args.data_dir, batch_size=args.batch_size,
            C=MODEL_CONFIG["C"], shuffle=True)
        data_iter = iter(data_loader)
        print(f"Data: local shards from {args.data_dir} "
              f"({len(data_loader)} batches/epoch)")
    elif args.hf_repo:
        data_loader = create_hf_dataloader(
            args.hf_repo, batch_size=args.batch_size,
            C=MODEL_CONFIG["C"], path_in_repo=args.hf_path,
            skip_rows=hf_rows_consumed)
        data_iter = iter(data_loader)
        print(f"Data: HF streaming from {args.hf_repo} "
              f"({rows_per_step} rows/step, skip={hf_rows_consumed} rows)")
    else:
        print("Data: synthetic only (no --data-dir or --hf-repo)")
    sys.stdout.flush()

    # -- Training loop ---------------------------------------------------------
    t0 = time.time()
    # Timing accumulators (reset every log_every steps)
    t_data_sum, t_fwd_sum, t_bwd_sum, t_step_sum = 0.0, 0.0, 0.0, 0.0
    timing_count = 0

    for step in range(start_step + 1, args.total_steps + 1):
        t_step_start = time.perf_counter()

        model.train()
        optimizer.zero_grad()

        # -- Data loading (timing) -------------------------------------------
        t_data_start = time.perf_counter()
        if data_iter is not None:
            try:
                x = next(data_iter)
            except StopIteration:
                print(f"\n=== Data exhausted at step {step} "
                      f"(hf_rows_consumed={hf_rows_consumed}) ===")
                # Save final checkpoint and exit — never re-see data
                path = os.path.join(args.save_dir,
                                    f"{args.run_name}_final.pth")
                save_snapshot(model, optimizer, path, step - 1,
                              best_gap, best_gap_step,
                              best_loss, best_loss_step,
                              ema_loss=ema_loss, ema_gap=ema_gap,
                              hf_rows_consumed=hf_rows_consumed)
                csv_logger.flush()
                print(f"Done in {time.time() - t0:.0f}s "
                      f"({step - 1 - start_step} steps)")
                break
        else:
            x, = generate_synthetic_batch(
                batch_size=args.batch_size, T_raw=T_RAW,
                C=MODEL_CONFIG["C"], dimension=DIMENSION)

        hf_rows_consumed += rows_per_step

        x = x.to(device)
        x = random_sign_flip(x)
        t_data_end = time.perf_counter()

        # -- Forward pass (timing) -------------------------------------------
        t_fwd_start = time.perf_counter()
        f_lat, o_lat = forward_step(model, x)
        loss = contrastive_latent_loss((f_lat, o_lat), validation=False,
                                       spec=LOSS_SPEC)
        t_fwd_end = time.perf_counter()

        # -- NaN detection ---------------------------------------------------
        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n*** NaN/Inf DETECTED at step {step} ***")
            print(f"  loss={loss_val}, hf_rows_consumed={hf_rows_consumed}")
            print(f"  Batch shape: {x.shape}")
            print(f"  Batch stats: min={x.min().item():.4f}, "
                  f"max={x.max().item():.4f}, "
                  f"mean={x.mean().item():.4f}, "
                  f"std={x.std().item():.4f}")
            print(f"  Any NaN in input: {torch.isnan(x).any().item()}")

            # Save emergency checkpoint
            emerg_path = os.path.join(
                args.save_dir, f"{args.run_name}_EMERGENCY_{step}.pth")
            save_snapshot(model, optimizer, emerg_path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step,
                          ema_loss=ema_loss, ema_gap=ema_gap,
                          hf_rows_consumed=hf_rows_consumed)
            print(f"  Emergency checkpoint: {emerg_path}")

            # Flush CSV and close
            csv_logger.close()
            sys.stdout.flush()
            sys.exit(1)

        # -- Backward pass (timing) ------------------------------------------
        t_bwd_start = time.perf_counter()
        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        t_bwd_end = time.perf_counter()

        t_step_end = time.perf_counter()

        # Accumulate timing
        t_data_sum += (t_data_end - t_data_start)
        t_fwd_sum += (t_fwd_end - t_fwd_start)
        t_bwd_sum += (t_bwd_end - t_bwd_start)
        t_step_sum += (t_step_end - t_step_start)
        timing_count += 1

        # -- Rolling metrics (EMA) ---------------------------------------------
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

        # -- Per-step CSV logging ----------------------------------------------
        csv_logger.log(step, loss_val, gap_val, val_ff, val_fp,
                       hf_rows_consumed)

        # -- Logging -----------------------------------------------------------
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step - start_step) / elapsed
            eta = (args.total_steps - step) / sps / 3600

            print(f"[{step:>7d}] loss={loss_val:.4f}  ema_loss={ema_loss:.4f}  "
                  f"gap={gap_val:.4f}  ema_gap={ema_gap:.4f}  "
                  f"{sps:.1f} sps  ETA {eta:.1f}h")

            # Timing summary
            n = timing_count
            print(f"  timing: data={t_data_sum/n*1000:.1f}ms  "
                  f"fwd={t_fwd_sum/n*1000:.1f}ms  "
                  f"bwd={t_bwd_sum/n*1000:.1f}ms  "
                  f"total={t_step_sum/n*1000:.1f}ms")
            sys.stdout.flush()
            t_data_sum, t_fwd_sum, t_bwd_sum, t_step_sum = 0.0, 0.0, 0.0, 0.0
            timing_count = 0

        # -- Best checkpoints (only check on log steps to reduce I/O) ---------
        if step % args.log_every == 0:
            if ema_gap > best_gap:
                best_gap, best_gap_step = ema_gap, step
                path = os.path.join(args.save_dir,
                                    f"{args.run_name}_best_gap.pth")
                save_snapshot(model, optimizer, path, step,
                              best_gap, best_gap_step, best_loss, best_loss_step,
                              ema_loss=ema_loss, ema_gap=ema_gap,
                              hf_rows_consumed=hf_rows_consumed)

            if ema_loss < best_loss:
                best_loss, best_loss_step = ema_loss, step
                path = os.path.join(args.save_dir,
                                    f"{args.run_name}_best_loss.pth")
                save_snapshot(model, optimizer, path, step,
                              best_gap, best_gap_step, best_loss, best_loss_step,
                              ema_loss=ema_loss, ema_gap=ema_gap,
                              hf_rows_consumed=hf_rows_consumed)

        # -- Periodic snapshot (never overwritten) -----------------------------
        if step % args.save_every == 0:
            path = os.path.join(args.save_dir,
                                f"{args.run_name}_{step // 1000}k.pth")
            save_snapshot(model, optimizer, path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step,
                          ema_loss=ema_loss, ema_gap=ema_gap,
                          hf_rows_consumed=hf_rows_consumed)

    # -- Final save ------------------------------------------------------------
    path = os.path.join(args.save_dir, f"{args.run_name}_final.pth")
    save_snapshot(model, optimizer, path, args.total_steps,
                  best_gap, best_gap_step, best_loss, best_loss_step,
                  ema_loss=ema_loss, ema_gap=ema_gap,
                  hf_rows_consumed=hf_rows_consumed)

    # Flush and close CSV
    csv_logger.close()

    total = time.time() - t0
    print(f"\nDone in {total/3600:.1f}h. "
          f"Best gap={best_gap:.4f} at step {best_gap_step}, "
          f"Best loss={best_loss:.4f} at step {best_loss_step}")


if __name__ == "__main__":
    main()
