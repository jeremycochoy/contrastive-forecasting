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
from src.dataloader import (
    create_hf_dataloader,
    create_mixed_periodic_dataloader,
    create_mixed_composite_dataloader,
)
from src.forecasting_head import (
    ForecastingHead,
    QuantileForecastingHead,
    QUANTILE_LEVELS,
    quantile_loss,
    W,
    FORECAST_LEN,
    extract_forecaster_latents,
    extract_encoder_latents,
    rollout_latent,
    compute_valid_targets,
    compute_reconstruction_targets,
)

# -- Backbone architecture (must match checkpoint) ---------------------------
BACKBONE_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
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
    p.add_argument("--no-resume-data-skip", action="store_true",
                   help="When resuming, DON'T compute skip_rows from start_step. "
                        "The HF skip is O(rows_to_skip) and can take an hour+ "
                        "for multi-M skips. Since our corpus is much larger than "
                        "what we train on, re-seeing some early rows is harmless.")
    p.add_argument("--quantile-head", action="store_true",
                   help="Use QuantileForecastingHead with pinball loss "
                        "(9 quantile levels) instead of the MSE point head. "
                        "Replaces all final-layer projections; rest of the GRU "
                        "trunk identical. Required by GIFT-Eval's WQL metric.")
    p.add_argument("--rev-norm-kind", default="ewma",
                   choices=["ewma", "revin", "none"],
                   help="MUST match the backbone's training-time choice "
                        "(both RevEWMNorm and RevIN have 0 params so state_dict "
                        "doesn't disambiguate). Default 'ewma'.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm for clipping (0 to disable)")
    p.add_argument("--forecast-len", type=int, default=128,
                   help="Head forecast length: 128 (default) or 16 for W-heads")
    p.add_argument("--mixed-rollout", type=int, default=0,
                   help="If >0, train on mixed real+rolled latent sequences. "
                        "Uses first 48 patches as context, rolls out N tokens, "
                        "and trains on the full [context+rolled] sequence.")
    p.add_argument("--reconstruction", default=None,
                   choices=["forecaster", "encoder"],
                   help="Train as RECONSTRUCTION head (time-aligned targets). "
                        "'forecaster': f[t]→patch t+1 values. "
                        "'encoder': e[t]→patch t values. "
                        "If not set, uses old prediction targets (head predicts future).")
    p.add_argument("--encoder-type", default=None,
                   choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv"],
                   help="Override backbone encoder type (must match checkpoint)")
    p.add_argument("--freq-emb-dim", type=int, default=None,
                   help="Frequency embedding dim in the backbone. If the "
                        "backbone was trained with freq_emb_dim=D, set the same "
                        "here so the state_dict loads. Auto-detected from the "
                        "checkpoint if omitted.")
    p.add_argument("--seasonality-emb-dim", type=int, default=None,
                   help="Seasonality embedding dim in the backbone. "
                        "Auto-detected from the checkpoint if omitted.")
    p.add_argument("--rev-norm-span", type=int, default=32,
                   help="Span used by the backbone's RevEWMNorm. MUST match "
                        "the backbone's training-time value (norm has 0 params "
                        "so state_dict doesn't disambiguate). Default 32.")
    p.add_argument("--patch-stats", default="auto",
                   choices=["auto", "none", "diff", "raw"],
                   help="Backbone patch-stats setting. 'auto' (default) "
                        "detects from the encoder's input width in the "
                        "checkpoint; pass an explicit value to override.")
    p.add_argument("--mix-ratio", type=float, default=0.0,
                   help="If >0, train on a mix of HF + on-the-fly periodic "
                        "synth (matches the backbone's MixedPeriodicLoader). "
                        "1.0 = synth-only — used for the synth-only "
                        "reconstruction-head experiment.")
    p.add_argument("--synth-seed", type=int, default=None,
                   help="Seed for the periodic synth generator when "
                        "--mix-ratio > 0. Defaults to args.seed + 20_000 to "
                        "stay separate from the backbone's synth stream.")
    p.add_argument("--synth-kind", default="periodic",
                   choices=["periodic", "composite"],
                   help="Which on-the-fly synth to mix with HF. Must match "
                        "the backbone's --synth-kind so the head trains on "
                        "the same data distribution. Default 'periodic'.")
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
    if args.encoder_type is not None:
        BACKBONE_CONFIG["encoder_type"] = args.encoder_type

    # Auto-detect freq_emb_dim and seasonality_emb_dim from the checkpoint
    # if not explicitly set.
    sd = torch.load(args.backbone_path, map_location=device, weights_only=True)
    if args.freq_emb_dim is None:
        w = sd.get("freq_embedding.embedding.weight")
        if w is not None:
            args.freq_emb_dim = w.shape[1]
            print(f"  [head-train] auto-detected freq_emb_dim={args.freq_emb_dim} "
                  f"from backbone checkpoint")
        else:
            args.freq_emb_dim = 0
    if args.seasonality_emb_dim is None:
        w = sd.get("seasonality_embedding.embedding.weight")
        if w is not None:
            args.seasonality_emb_dim = w.shape[1]
            print(f"  [head-train] auto-detected "
                  f"seasonality_emb_dim={args.seasonality_emb_dim} "
                  f"from backbone checkpoint")
        else:
            args.seasonality_emb_dim = 0
    BACKBONE_CONFIG["freq_emb_dim"] = args.freq_emb_dim
    BACKBONE_CONFIG["seasonality_emb_dim"] = args.seasonality_emb_dim
    BACKBONE_CONFIG["rev_norm_kind"] = args.rev_norm_kind
    if args.rev_norm_kind == "ewma":
        BACKBONE_CONFIG["rev_norm_span"] = args.rev_norm_span
    # Auto-detect patch_stats from the encoder's first projection input width.
    # The GRU encoder stores `encoder.skip.weight` of shape [H, encoder_input];
    # MLP-style encoders store `encoder.linear1.weight` similarly. Either way
    # the in-features tells us W + freq_emb_dim + (2 if patch_stats else 0).
    if args.patch_stats == "auto":
        from src.norm import PATCH_STATS_DIM
        W = BACKBONE_CONFIG["W"]
        skip_w = sd.get("encoder.skip.weight")
        linear1_w = sd.get("encoder.linear1.weight")
        ref = skip_w if skip_w is not None else linear1_w
        if ref is None:
            args.patch_stats = "none"
        else:
            in_features = ref.shape[1]
            extra = in_features - W - args.freq_emb_dim - args.seasonality_emb_dim
            if extra == 0:
                args.patch_stats = "none"
            elif extra == PATCH_STATS_DIM:
                # Default to 'diff' on auto — the only kind we plan to ship.
                args.patch_stats = "diff"
            else:
                raise ValueError(
                    f"Unexpected encoder in_features={in_features}: extra "
                    f"width={extra} doesn't match W ({W}) + freq_emb_dim "
                    f"({args.freq_emb_dim}) + seasonality_emb_dim "
                    f"({args.seasonality_emb_dim}) + 0 or {PATCH_STATS_DIM}.")
        print(f"  [head-train] auto-detected patch_stats={args.patch_stats}")
    BACKBONE_CONFIG["patch_stats_kind"] = args.patch_stats

    backbone = ConfigurableModel(**BACKBONE_CONFIG)
    backbone.load_state_dict(sd)
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print(f"Backbone loaded from {args.backbone_path} "
          f"({count_parameters(backbone):,} params, frozen)")

    # -- Forecasting head ------------------------------------------------------
    head_config = dict(HEAD_CONFIG)
    head_config['forecast_len'] = args.forecast_len
    if args.quantile_head:
        head = QuantileForecastingHead(**head_config).to(device)
        head_kind = "quantile (9 levels)"
    else:
        head = ForecastingHead(**head_config).to(device)
        head_kind = "MSE (point)"
    n_head_params = count_parameters(head)
    print(f"Forecasting head [{head_kind}]: {n_head_params:,} trainable params")

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
    if args.no_resume_data_skip:
        hf_rows_consumed = args.skip_rows
        print(f"  [data] --no-resume-data-skip: starting from HF offset "
              f"{args.skip_rows} (NOT {start_step * rows_per_step + args.skip_rows})")
    else:
        hf_rows_consumed = start_step * rows_per_step + args.skip_rows

    emit_labels = (args.freq_emb_dim > 0 or args.seasonality_emb_dim > 0)
    if args.mix_ratio > 0 or emit_labels:
        # Use the mixed loader when we need labels, even if mix_ratio=0
        # (it falls through to MixedPeriodicLoader with synth_bs=0 and
        # yields the (x, freq_ids, seasonality_ids) tuples extract_*_latents
        # consumes).
        synth_seed = args.synth_seed if args.synth_seed is not None else args.seed + 20_000
        if args.synth_kind == "composite":
            data_loader = create_mixed_composite_dataloader(
                repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
                mix_ratio=args.mix_ratio,
                path_in_repo=args.hf_path, skip_rows=hf_rows_consumed,
                seed=synth_seed, emit_freq_ids=emit_labels,
            )
        else:
            data_loader = create_mixed_periodic_dataloader(
                repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
                mix_ratio=args.mix_ratio,
                path_in_repo=args.hf_path, skip_rows=hf_rows_consumed,
                seed=synth_seed, emit_freq_ids=emit_labels,
            )
        synth_bs = int(round(args.batch_size * args.mix_ratio))
        hf_bs = args.batch_size - synth_bs
        print(f"Data: MIX {(1-args.mix_ratio)*100:.0f}% HF + "
              f"{args.mix_ratio*100:.0f}% synth ({args.synth_kind}), "
              f"hf_bs={hf_bs}, synth_bs={synth_bs}, synth_seed={synth_seed}, "
              f"emit_labels={emit_labels}")
    else:
        data_loader = create_hf_dataloader(
            args.hf_repo, batch_size=args.batch_size, C=C,
            path_in_repo=args.hf_path, skip_rows=hf_rows_consumed)
        print(f"Data: HF streaming from {args.hf_repo}/{args.hf_path} "
              f"(skip={hf_rows_consumed} rows)")
    data_iter = iter(data_loader)

    print(f"\nTraining for {args.total_steps} steps, bs={args.batch_size}, "
          f"lr={args.lr}, forecast_len={args.forecast_len}")
    if args.reconstruction:
        print(f"RECONSTRUCTION mode: {args.reconstruction} "
              f"(head decodes what latent represents, not future)")
    if args.mixed_rollout > 0:
        print(f"Mixed-rollout mode: {args.mixed_rollout} rolled tokens per step "
              f"(48 context patches + {args.mixed_rollout} rolled)")
    print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")
    sys.stdout.flush()

    # -- Training loop ---------------------------------------------------------
    t0 = time.time()
    ema_loss = None
    ema_decay = 0.99

    for step in range(start_step + 1, args.total_steps + 1):
        head.train()
        optimizer.zero_grad()

        # Data loading — when emit_labels is on, the dataloader yields
        # (x, freq_ids, seasonality_ids); otherwise it yields just x.
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        if isinstance(batch, tuple):
            x, freq_ids, seasonality_ids = batch
            freq_ids = freq_ids.to(device)
            seasonality_ids = seasonality_ids.to(device)
        else:
            x = batch
            freq_ids = None
            seasonality_ids = None
        hf_rows_consumed += rows_per_step
        x = x.to(device)

        if args.mixed_rollout > 0:
            # Mixed training: use first 48 patches as context, roll out N tokens
            N_roll = args.mixed_rollout
            T_ctx_raw = 48 * W  # 768 timesteps
            x_ctx = x[:, :T_ctx_raw, :]

            # Get encoder + forecaster latents from context
            e_bc, _ = extract_encoder_latents(
                backbone, x_ctx, freq_ids=freq_ids,
                seasonality_ids=seasonality_ids)
            f_ctx, x_norm_ctx = extract_forecaster_latents(
                backbone, x_ctx, freq_ids=freq_ids,
                seasonality_ids=seasonality_ids)
            T_ctx_patches = f_ctx.size(1)  # 48

            # Roll out N tokens in latent space
            future_f = rollout_latent(backbone, e_bc, N_roll)

            # Full sequence for head: [context_f, rolled_f]
            full_f = torch.cat([f_ctx, future_f], dim=1)  # (B*C, 48+N, H)

            # Targets: from full x_norm (need to normalize full sequence)
            with torch.no_grad():
                if backbone.rev_norm is not None:
                    x_norm = backbone.rev_norm(x, mode='norm')
                else:
                    x_norm = x

            # Choose target computation based on reconstruction mode
            if args.reconstruction:
                targets, T_valid_full = compute_reconstruction_targets(
                    x_norm, W=W, output_len=args.forecast_len,
                    mode=args.reconstruction)
            else:
                targets, T_valid_full = compute_valid_targets(
                    x_norm, W=W, forecast_len=args.forecast_len)
            targets = targets.to(device)

            # Take targets for our sequence positions only
            T_total = full_f.size(1)
            T_use = min(T_total, T_valid_full)
            preds = head(full_f)[:, :T_use, :]
            targets = targets[:, :T_use, :]

            if args.reconstruction and args.mixed_rollout > 0:
                # R3 mode: loss only on rolled positions
                preds = preds[:, T_ctx_patches:, :]
                targets = targets[:, T_ctx_patches:, :]

            loss = torch.nn.functional.mse_loss(preds, targets)
        elif args.reconstruction == 'encoder':
            # Encoder reconstruction: e[t] → patch t values
            e_bc, x_norm = extract_encoder_latents(
                backbone, x, freq_ids=freq_ids,
                seasonality_ids=seasonality_ids)
            targets, T_valid = compute_reconstruction_targets(
                x_norm, W=W, output_len=args.forecast_len, mode='encoder')
            targets = targets.to(device)
            preds = head(e_bc)[:, :T_valid, :]
            loss = torch.nn.functional.mse_loss(preds, targets)

        elif args.reconstruction == 'forecaster':
            # Forecaster reconstruction: f[t] → patch t+1 values
            f_bc, x_norm = extract_forecaster_latents(
                backbone, x, freq_ids=freq_ids,
                seasonality_ids=seasonality_ids)
            targets, T_valid = compute_reconstruction_targets(
                x_norm, W=W, output_len=args.forecast_len, mode='forecaster')
            targets = targets.to(device)
            preds = head(f_bc)
            if args.quantile_head:
                preds = preds[:, :T_valid, :, :]                  # (BC, T, Q, L)
                loss = quantile_loss(preds, targets, QUANTILE_LEVELS)
            else:
                preds = preds[:, :T_valid, :]
                loss = torch.nn.functional.mse_loss(preds, targets)

        else:
            # Standard prediction training (old behavior)
            f_bc, x_norm = extract_forecaster_latents(
                backbone, x, freq_ids=freq_ids,
                seasonality_ids=seasonality_ids)
            targets, T_valid = compute_valid_targets(
                x_norm, W=W, forecast_len=args.forecast_len)
            targets = targets.to(device)
            preds = head(f_bc)
            if args.quantile_head:
                preds = preds[:, :T_valid, :, :]
                loss = quantile_loss(preds, targets, QUANTILE_LEVELS)
            else:
                preds = preds[:, :T_valid, :]
                loss = torch.nn.functional.mse_loss(preds, targets)

        # NaN detection -- skip bad batches instead of crashing
        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            nan_skip_count = getattr(main, '_nan_skips', 0) + 1
            main._nan_skips = nan_skip_count
            print(f"  [step {step}] NaN/Inf loss detected, skipping batch "
                  f"(total skips: {nan_skip_count})")
            sys.stdout.flush()
            optimizer.zero_grad()  # discard any partial gradients
            continue

        # Backward + step
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
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
