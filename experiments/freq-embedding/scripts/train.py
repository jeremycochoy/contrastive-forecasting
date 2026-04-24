#!/usr/bin/env python3
"""
Contrastive training with frequency embedding + optional mixup.

Same as experiments/periodic-synth-mix/scripts/train.py but:
 - backbone receives a learned frequency embedding concat'd per-patch
   (see src/freq_embedding.py + src.models.ConfigurableModel.freq_emb_dim)
 - optional mixup augmentation: with probability --mixup-p, each step
   linearly interpolates two batch items (both X and the freq embedding)

Usage
-----
    # freqemb only, no mixup
    python experiments/freq-embedding/scripts/train.py \
        --device cuda --run-name freqemb_mix --total-steps 30000 \
        --batch-size 24 --lr 1e-4 --save-dir checkpoints \
        --hf-repo jeremycochoy/contrastive-training-base-bundles \
        --hf-path base_mixed_v1 --mix-ratio 0.5 \
        --freq-emb-dim 3 --mixup-p 0.0

    # freqemb + mixup
    python experiments/freq-embedding/scripts/train.py \
        --run-name freqemb_mixup_mix --freq-emb-dim 3 --mixup-p 0.3 \
        ... other args same as above ...
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
from src.dataloader import create_mixed_periodic_dataloader, create_hf_dataloader
from src.loss import contrastive_latent_loss
from src.checkpoint import save_training_state, load_training_state

# -- Tiny architecture (identical to v3c) -----------------------------------
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


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive + freq embedding training")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--run-name", default="freqemb")
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-path", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mix-ratio", type=float, default=0.5)
    p.add_argument("--synth-seed", type=int, default=None)
    # Freq embedding
    p.add_argument("--freq-emb-dim", type=int, default=3,
                   help="Frequency embedding dimension (0 disables it).")
    # Mixup on (X, freq)
    p.add_argument("--mixup-p", type=float, default=0.0,
                   help="Probability of applying mixup at each step. 0 disables.")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Beta(alpha, alpha) parameter for mixup lambda.")
    return p.parse_args()


def random_sign_flip(x):
    B, T, C = x.shape
    signs = torch.where(torch.rand(B, 1, C, device=x.device) < 0.5,
                        torch.ones(1, device=x.device),
                        -torch.ones(1, device=x.device))
    return x * signs


def forward_step(model, x, freq_ids=None, freq_embs=None):
    """Apply RevEWMNorm + transformer with optional freq embedding."""
    W = model.W
    H = model.H
    if model.rev_norm is not None:
        x = model.rev_norm(x, mode='norm')
    B, T_raw, C = x.shape
    T = T_raw // W
    xr = x.view(B, T, W, C).permute(0, 1, 3, 2)
    # Apply freq embedding to the patch
    xr = model._apply_freq_embedding(xr, freq_ids=freq_ids, freq_embs=freq_embs)
    f_flat, o_flat = model.transformer(xr)
    f_lat = f_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    o_lat = o_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    return f_lat, o_lat


def maybe_mixup(x, freq_ids, model, args):
    """With prob p, linearly interpolate X and freq embedding between
    randomly-paired batch items. Returns (x, freq_ids, freq_embs).

    When no mixup is applied, returns (x, freq_ids, None) and the caller
    should pass freq_ids to model.forward(); freq_embs=None means the
    model does its own lookup.

    When mixup IS applied, freq_ids is returned unchanged (unused) and
    freq_embs is a tensor the caller must pass via freq_embs=... so the
    model uses the mixed embedding instead of looking up by id.
    """
    if args.mixup_p <= 0 or model.freq_embedding is None:
        return x, freq_ids, None
    if torch.rand(()).item() >= args.mixup_p:
        return x, freq_ids, None

    # Draw a single alpha for the whole batch (standard mixup).
    a = float(torch.distributions.Beta(args.mixup_alpha, args.mixup_alpha).sample())
    B = x.shape[0]
    idx = torch.randperm(B, device=x.device)
    x_mix = a * x + (1 - a) * x[idx]
    # Embedding is indexed by id; compute both and mix.
    emb_a = model.freq_embedding(freq_ids)
    emb_b = model.freq_embedding(freq_ids[idx])
    emb_mix = a * emb_a + (1 - a) * emb_b
    return x_mix, freq_ids, emb_mix


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
            print(f"  [checkpoint] Branching to '{candidate}'.")
            return candidate
        n += 1


class CSVLogger:
    def __init__(self, path, flush_every=100):
        self.path = path
        self.flush_every = flush_every
        self._buffer = []
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if os.path.getsize(path) == 0:
            self._writer.writerow([
                "step", "loss", "gap", "ff", "fp", "tp", "cross_batch",
                "hf_rows_consumed", "synth_rows_consumed", "mixup_applied",
            ])
            self._file.flush()

    def log(self, step, loss, gap, ff, fp, tp, cross_batch,
            hf_rows_consumed, synth_rows_consumed, mixup_applied):
        self._buffer.append([step, loss, gap, ff, fp, tp, cross_batch,
                             hf_rows_consumed, synth_rows_consumed,
                             int(mixup_applied)])
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.resume:
        args.run_name = safe_run_name(args.save_dir, args.run_name)

    # -- Model -----------------------------------------------------------------
    model_config = dict(MODEL_CONFIG)
    model_config["freq_emb_dim"] = args.freq_emb_dim
    model = ConfigurableModel(**model_config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

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
          f"lr={args.lr}, T={T_RAW}, mix_ratio={args.mix_ratio}, "
          f"freq_emb_dim={args.freq_emb_dim}, mixup_p={args.mixup_p}")
    print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")

    csv_path = os.path.join(args.save_dir, f"{args.run_name}_losses.csv")
    csv_logger = CSVLogger(csv_path, flush_every=100)
    print(f"Loss CSV: {csv_path}")

    # -- Data -----------------------------------------------------------------
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

    data_loader = create_mixed_periodic_dataloader(
        repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
        mix_ratio=args.mix_ratio,
        path_in_repo=args.hf_path, split=args.split,
        skip_rows=hf_rows_consumed, T_raw=T_RAW, seed=synth_seed,
        emit_freq_ids=(args.freq_emb_dim > 0),
    )
    print(f"Data: MIX {(1-args.mix_ratio)*100:.0f}% HF + "
          f"{args.mix_ratio*100:.0f}% synth, hf_bs={hf_bs}, synth_bs={synth_bs}")
    data_iter = iter(data_loader)
    sys.stdout.flush()

    # -- Training loop --------------------------------------------------------
    t0 = time.time()
    t_data_sum, t_fwd_sum, t_bwd_sum, t_step_sum = 0.0, 0.0, 0.0, 0.0
    timing_count = 0
    mixup_applied_count = 0

    for step in range(start_step + 1, args.total_steps + 1):
        t_step_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        t_data_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            print(f"\n=== Epoch boundary at step {step} ===")
            sys.stdout.flush()
            data_iter = iter(data_loader)
            batch = next(data_iter)

        if args.freq_emb_dim > 0:
            x, freq_ids = batch
            freq_ids = freq_ids.to(device)
        else:
            x = batch
            freq_ids = None

        hf_rows_consumed += hf_rows_per_step
        synth_rows_consumed += synth_rows_per_step
        x = x.to(device)
        x = random_sign_flip(x)

        # Optional mixup (X + freq embedding)
        x, freq_ids, freq_embs = maybe_mixup(x, freq_ids, model, args)
        mixup_applied = (freq_embs is not None)
        if mixup_applied:
            mixup_applied_count += 1
        t_data_end = time.perf_counter()

        t_fwd_start = time.perf_counter()
        f_lat, o_lat = forward_step(model, x, freq_ids=freq_ids, freq_embs=freq_embs)
        loss = contrastive_latent_loss((f_lat, o_lat), validation=False,
                                       spec=LOSS_SPEC)
        t_fwd_end = time.perf_counter()

        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n*** NaN/Inf DETECTED at step {step} ***")
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
            ema_loss = loss_val; ema_gap = gap_val
        else:
            d = args.ema_decay
            ema_loss = d * ema_loss + (1 - d) * loss_val
            ema_gap  = d * ema_gap  + (1 - d) * gap_val

        csv_logger.log(step, loss_val, gap_val, val_ff, val_fp,
                       val_tp, val_cb, hf_rows_consumed, synth_rows_consumed,
                       mixup_applied)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            sps = (step - start_step) / elapsed
            eta = (args.total_steps - step) / sps / 3600
            print(f"[{step:>7d}] loss={loss_val:.4f}  ema_loss={ema_loss:.4f}  "
                  f"gap={gap_val:.4f}  ema_gap={ema_gap:.4f}  "
                  f"mixup={mixup_applied_count}/{timing_count}  "
                  f"{sps:.1f} sps  ETA {eta:.1f}h")
            n = timing_count
            print(f"  timing: data={t_data_sum/n*1000:.1f}ms  "
                  f"fwd={t_fwd_sum/n*1000:.1f}ms  "
                  f"bwd={t_bwd_sum/n*1000:.1f}ms  "
                  f"total={t_step_sum/n*1000:.1f}ms")
            sys.stdout.flush()
            t_data_sum, t_fwd_sum, t_bwd_sum, t_step_sum = 0.0, 0.0, 0.0, 0.0
            timing_count = 0
            mixup_applied_count = 0

            if ema_gap > best_gap:
                best_gap, best_gap_step = ema_gap, step
                path = os.path.join(args.save_dir, f"{args.run_name}_best_gap.pth")
                save_snapshot(model, optimizer, path, step,
                              best_gap, best_gap_step, best_loss, best_loss_step,
                              ema_loss=ema_loss, ema_gap=ema_gap,
                              hf_rows_consumed=hf_rows_consumed,
                              synth_rows_consumed=synth_rows_consumed)
            if ema_loss < best_loss:
                best_loss, best_loss_step = ema_loss, step
                path = os.path.join(args.save_dir, f"{args.run_name}_best_loss.pth")
                save_snapshot(model, optimizer, path, step,
                              best_gap, best_gap_step, best_loss, best_loss_step,
                              ema_loss=ema_loss, ema_gap=ema_gap,
                              hf_rows_consumed=hf_rows_consumed,
                              synth_rows_consumed=synth_rows_consumed)

        if step % args.save_every == 0:
            path = os.path.join(args.save_dir, f"{args.run_name}_{step // 1000}k.pth")
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
