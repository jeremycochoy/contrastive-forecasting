#!/usr/bin/env python3
"""
Contrastive training with optional periodic-synth mix.

Extends experiments/tiny-training/scripts/train.py with a ``--mix-ratio``
flag: each batch is ``(1 - mix_ratio) * bs`` rows drawn from the HF stream
plus ``mix_ratio * bs`` rows drawn on-the-fly from the periodic synthesizer
in ``src.synthetic_periodic``. See experiments/periodic-synth-mix/DESIGN.md.

Both arms (CONTROL: ``--mix-ratio 0.0``, MIX: ``--mix-ratio 0.5``) share
every other hyperparameter so the only independent variable at matched
compute is the presence of the periodic synth half.

Usage
-----
    # CONTROL arm (pure HF stream, same as v3b at 30k)
    python experiments/periodic-synth-mix/scripts/train.py \
        --device cuda --run-name tiny_v3c_ctrl \
        --total-steps 30000 --batch-size 24 --lr 1e-4 \
        --hf-repo jeremycochoy/contrastive-training-base-bundles \
        --hf-path base_mixed_v1 --mix-ratio 0.0

    # MIX arm (50/50 HF + periodic synth)
    python experiments/periodic-synth-mix/scripts/train.py \
        --device cuda --run-name tiny_v3c_mix \
        --total-steps 30000 --batch-size 24 --lr 1e-4 \
        --hf-repo jeremycochoy/contrastive-training-base-bundles \
        --hf-path base_mixed_v1 --mix-ratio 0.5
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
from src.dataloader import (
    create_dataloader,
    create_hf_dataloader,
    create_mixed_periodic_dataloader,
)
from src.loss import contrastive_latent_loss
from src.checkpoint import save_training_state, load_training_state, safe_save_path

# -- Tiny architecture (identical to v3b) -----------------------------------
MODEL_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
    rev_norm_span=32,
)

LOSS_SPEC = SimpleNamespace(train_configuration={
    "contrastive_divergence_temperature": 0.07,
    "contrastive_latent_noise": None,
    "loss_shape": "cosine_similarity_batch_no_time_neg",
    "contrastive_latent_delay": 0,
})
CLD = LOSS_SPEC.train_configuration["contrastive_latent_delay"] + 1

T_RAW = 1024
DIMENSION = 8


def parse_args():
    p = argparse.ArgumentParser(description="Train contrastive backbone with periodic-synth mix")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=24,
                   help="Effective batch size (HF + synth combined)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--run-name", default="tiny_v3c")
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--hf-repo", default=None,
                   help="HuggingFace dataset repo ID (required unless --mix-ratio=1.0).")
    p.add_argument("--hf-path", default=None,
                   help="Subdirectory within the HF repo (e.g. 'base_mixed_v1').")
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mix-ratio", type=float, default=0.0,
                   help="Fraction of each batch drawn from the on-the-fly "
                        "periodic synthesizer. 0.0 = pure HF (CONTROL arm). "
                        "0.5 = 50/50 (MIX arm). 1.0 = pure synth.")
    p.add_argument("--synth-seed", type=int, default=None,
                   help="Seed for the periodic synth generator. "
                        "Defaults to --seed + 10_000 so HF skip_rows and "
                        "synth draws don't share an RNG stream.")
    return p.parse_args()


def random_sign_flip(x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    signs = torch.where(torch.rand(B, 1, C, device=x.device) < 0.5,
                        torch.ones(1, device=x.device),
                        -torch.ones(1, device=x.device))
    return x * signs


def forward_step(model, x):
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
                  hf_rows_consumed=0, synth_rows_consumed=0):
    import numpy as _np
    torch.save(model.state_dict(), path)
    save_training_state(
        optimizer, path, step=step,
        best_val_ff=best_gap, best_step=best_gap_step,
        best_loss=best_loss, best_loss_step=best_loss_step,
        ema_loss=ema_loss, ema_gap=ema_gap,
        hf_rows_consumed=hf_rows_consumed,
        synth_rows_consumed=synth_rows_consumed,
        rng_state_torch=torch.get_rng_state(),
        rng_state_numpy=_np.random.get_state(),
    )
    print(f"  -> Saved {path}")


def _has_checkpoints(save_dir, run_name):
    import glob
    return len(glob.glob(os.path.join(save_dir, f"{run_name}_*.pth"))) > 0


def safe_run_name(save_dir, run_name):
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
    def __init__(self, path: str, flush_every: int = 100):
        self.path = path
        self.flush_every = flush_every
        self._buffer = []
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if os.path.getsize(path) == 0:
            self._writer.writerow([
                "step", "loss", "gap", "ff", "fp", "tp", "cross_batch",
                "hf_rows_consumed", "synth_rows_consumed",
            ])
            self._file.flush()

    def log(self, step: int, loss: float, gap: float, ff: float, fp: float,
            tp: float, cross_batch: float,
            hf_rows_consumed: int, synth_rows_consumed: int):
        self._buffer.append([step, loss, gap, ff, fp, tp, cross_batch,
                             hf_rows_consumed, synth_rows_consumed])
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

    if not 0.0 <= args.mix_ratio <= 1.0:
        raise ValueError(f"--mix-ratio must be in [0, 1], got {args.mix_ratio}")
    if args.mix_ratio < 1.0 and args.hf_repo is None:
        raise ValueError("--hf-repo is required unless --mix-ratio=1.0")

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

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
    restored = {}

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
        try:
            if restored.get("rng_state_torch") is not None:
                rng = restored["rng_state_torch"]
                if not isinstance(rng, torch.ByteTensor):
                    rng = rng.byte()
                torch.set_rng_state(rng)
            if restored.get("rng_state_numpy") is not None:
                _np.random.set_state(restored["rng_state_numpy"])
        except Exception as e:
            print(f"  [checkpoint] WARNING: Could not restore RNG state: {e}")
        print(f"Resumed from {args.resume} at step {start_step}")

    print(f"Device: {device} | Params: {count_parameters(model):,}")
    print(f"Training for {args.total_steps} steps, bs={args.batch_size}, "
          f"lr={args.lr}, T={T_RAW}, mix_ratio={args.mix_ratio}")
    print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")

    # -- CSV logger ------------------------------------------------------------
    csv_path = os.path.join(args.save_dir, f"{args.run_name}_losses.csv")
    csv_logger = CSVLogger(csv_path, flush_every=100)
    print(f"Loss CSV: {csv_path}")

    # -- Data source -----------------------------------------------------------
    C = MODEL_CONFIG["C"]
    synth_bs = int(round(args.batch_size * args.mix_ratio))
    hf_bs = args.batch_size - synth_bs
    hf_rows_per_step = hf_bs * C
    synth_rows_per_step = synth_bs * C

    if args.resume and restored.get("hf_rows_consumed", 0) > 0:
        hf_rows_consumed = restored["hf_rows_consumed"]
        synth_rows_consumed = restored.get("synth_rows_consumed", 0)
    else:
        hf_rows_consumed = start_step * hf_rows_per_step
        synth_rows_consumed = start_step * synth_rows_per_step

    synth_seed = args.synth_seed if args.synth_seed is not None else args.seed + 10_000

    if args.mix_ratio == 0.0:
        data_loader = create_hf_dataloader(
            args.hf_repo, batch_size=args.batch_size,
            C=C, path_in_repo=args.hf_path,
            split=args.split, skip_rows=hf_rows_consumed)
        print(f"Data: HF streaming from {args.hf_repo} "
              f"({hf_rows_per_step} rows/step, skip={hf_rows_consumed} rows)")
    else:
        data_loader = create_mixed_periodic_dataloader(
            repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
            mix_ratio=args.mix_ratio,
            path_in_repo=args.hf_path, split=args.split,
            skip_rows=hf_rows_consumed, T_raw=T_RAW, seed=synth_seed,
        )
        print(f"Data: MIX {(1-args.mix_ratio)*100:.0f}% HF + "
              f"{args.mix_ratio*100:.0f}% periodic synth")
        print(f"  HF sub-batch: bs={hf_bs}, {hf_rows_per_step} rows/step, "
              f"skip={hf_rows_consumed} rows")
        print(f"  Synth sub-batch: bs={synth_bs}, {synth_rows_per_step} rows/step, "
              f"seed={synth_seed}")

    data_iter = iter(data_loader)
    sys.stdout.flush()

    # -- Training loop ---------------------------------------------------------
    t0 = time.time()
    t_data_sum, t_fwd_sum, t_bwd_sum, t_step_sum = 0.0, 0.0, 0.0, 0.0
    timing_count = 0

    for step in range(start_step + 1, args.total_steps + 1):
        t_step_start = time.perf_counter()

        model.train()
        optimizer.zero_grad()

        # -- Data loading -----------------------------------------------------
        t_data_start = time.perf_counter()
        try:
            x = next(data_iter)
        except StopIteration:
            epoch_num = hf_rows_consumed // (121500 * max(hf_rows_per_step, 1)) + 1
            print(f"\n=== Epoch boundary at step {step} "
                  f"(hf_rows={hf_rows_consumed}, epoch {epoch_num}) ===")
            sys.stdout.flush()
            data_iter = iter(data_loader)
            x = next(data_iter)

        hf_rows_consumed += hf_rows_per_step
        synth_rows_consumed += synth_rows_per_step

        x = x.to(device)
        x = random_sign_flip(x)
        t_data_end = time.perf_counter()

        # -- Forward ----------------------------------------------------------
        t_fwd_start = time.perf_counter()
        f_lat, o_lat = forward_step(model, x)
        loss = contrastive_latent_loss((f_lat, o_lat), validation=False,
                                       spec=LOSS_SPEC)
        t_fwd_end = time.perf_counter()

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n*** NaN/Inf DETECTED at step {step} ***")
            print(f"  loss={loss_val}, hf_rows={hf_rows_consumed}, "
                  f"synth_rows={synth_rows_consumed}")
            print(f"  Batch shape: {x.shape}")
            print(f"  Batch stats: min={x.min().item():.4f}, "
                  f"max={x.max().item():.4f}, "
                  f"mean={x.mean().item():.4f}, "
                  f"std={x.std().item():.4f}")
            print(f"  Any NaN in input: {torch.isnan(x).any().item()}")
            emerg_path = os.path.join(
                args.save_dir, f"{args.run_name}_EMERGENCY_{step}.pth")
            save_snapshot(model, optimizer, emerg_path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step,
                          ema_loss=ema_loss, ema_gap=ema_gap,
                          hf_rows_consumed=hf_rows_consumed,
                          synth_rows_consumed=synth_rows_consumed)
            csv_logger.close()
            sys.stdout.flush()
            sys.exit(1)

        # -- Backward ---------------------------------------------------------
        t_bwd_start = time.perf_counter()
        loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        t_bwd_end = time.perf_counter()
        t_step_end = time.perf_counter()

        t_data_sum += (t_data_end - t_data_start)
        t_fwd_sum += (t_fwd_end - t_fwd_start)
        t_bwd_sum += (t_bwd_end - t_bwd_start)
        t_step_sum += (t_step_end - t_step_start)
        timing_count += 1

        with torch.no_grad():
            val_ff, val_fp, val_tp, val_cb = compute_metrics(f_lat, o_lat, CLD)
        gap_val = val_ff - val_fp

        if ema_loss is None:
            ema_loss = loss_val
            ema_gap = gap_val
        else:
            d = args.ema_decay
            ema_loss = d * ema_loss + (1 - d) * loss_val
            ema_gap = d * ema_gap + (1 - d) * gap_val

        csv_logger.log(step, loss_val, gap_val, val_ff, val_fp,
                       val_tp, val_cb, hf_rows_consumed, synth_rows_consumed)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step - start_step) / elapsed
            eta = (args.total_steps - step) / sps / 3600
            print(f"[{step:>7d}] loss={loss_val:.4f}  ema_loss={ema_loss:.4f}  "
                  f"gap={gap_val:.4f}  ema_gap={ema_gap:.4f}  "
                  f"{sps:.1f} sps  ETA {eta:.1f}h")
            n = timing_count
            print(f"  timing: data={t_data_sum/n*1000:.1f}ms  "
                  f"fwd={t_fwd_sum/n*1000:.1f}ms  "
                  f"bwd={t_bwd_sum/n*1000:.1f}ms  "
                  f"total={t_step_sum/n*1000:.1f}ms")
            sys.stdout.flush()
            t_data_sum, t_fwd_sum, t_bwd_sum, t_step_sum = 0.0, 0.0, 0.0, 0.0
            timing_count = 0

            if ema_gap > best_gap:
                best_gap, best_gap_step = ema_gap, step
                path = os.path.join(args.save_dir,
                                    f"{args.run_name}_best_gap.pth")
                save_snapshot(model, optimizer, path, step,
                              best_gap, best_gap_step, best_loss, best_loss_step,
                              ema_loss=ema_loss, ema_gap=ema_gap,
                              hf_rows_consumed=hf_rows_consumed,
                              synth_rows_consumed=synth_rows_consumed)

            if ema_loss < best_loss:
                best_loss, best_loss_step = ema_loss, step
                path = os.path.join(args.save_dir,
                                    f"{args.run_name}_best_loss.pth")
                save_snapshot(model, optimizer, path, step,
                              best_gap, best_gap_step, best_loss, best_loss_step,
                              ema_loss=ema_loss, ema_gap=ema_gap,
                              hf_rows_consumed=hf_rows_consumed,
                              synth_rows_consumed=synth_rows_consumed)

        if step % args.save_every == 0:
            path = os.path.join(args.save_dir,
                                f"{args.run_name}_{step // 1000}k.pth")
            save_snapshot(model, optimizer, path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step,
                          ema_loss=ema_loss, ema_gap=ema_gap,
                          hf_rows_consumed=hf_rows_consumed,
                          synth_rows_consumed=synth_rows_consumed)

    path = os.path.join(args.save_dir, f"{args.run_name}_final.pth")
    save_snapshot(model, optimizer, path, args.total_steps,
                  best_gap, best_gap_step, best_loss, best_loss_step,
                  ema_loss=ema_loss, ema_gap=ema_gap,
                  hf_rows_consumed=hf_rows_consumed,
                  synth_rows_consumed=synth_rows_consumed)

    csv_logger.close()

    total = time.time() - t0
    print(f"\nDone in {total/3600:.1f}h. "
          f"Best gap={best_gap:.4f} at step {best_gap_step}, "
          f"Best loss={best_loss:.4f} at step {best_loss_step}")


if __name__ == "__main__":
    main()
