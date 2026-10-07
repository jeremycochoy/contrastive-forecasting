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

import numpy as np
import torch
import torch.optim as optim

from src.models import ConfigurableModel, count_parameters
from src.checkpoint import (gru_input_bound_of, multi_patch_sizes_of,
                            prepare_backbone_state_dict, save_encoder_source)
from src.dataloader import create_hf_dataloader, create_mixed_periodic_dataloader
from src.freq_embedding import vocab_of_rows
from src.loss import masked_mean
from src.forecasting_head import (
    ForecastingHead,
    QuantileForecastingHead,
    LinearForecastingHead,
    LinearQuantileForecastingHead,
    TransformerQuantileForecastingHead,
    TransformerGaussianForecastingHead,
    ForecastingHeadBank,
    bank_quantile_loss,
    bank_training_inputs,
    QUANTILE_LEVELS,
    quantile_loss,
    gaussian_nll_loss,
    W,
    FORECAST_LEN,
    extract_forecaster_latents,
    extract_encoder_latents,
    rollout_latent,
    compute_valid_targets,
    masked_quantile_loss,
    valid_target_keep,
    compute_reconstruction_targets,
    reconstruction_quantile_loss,
)


def lr_multiplier(step: int, total_steps: int, schedule: str,
                  warmup_steps: int, decay_start_step: int,
                  final_lr_ratio: float) -> float:
    """Multiplier in [0, 1] applied to peak LR at training step ``step``.

    Schedules:
      ``constant``: 1.0 always (existing behaviour).
      ``wsd``:     warmup (linear 0→1) → stable at 1 → linear decay 1→
                   final_lr_ratio over [decay_start_step, total_steps].
      ``cosine``:  warmup → cosine from 1 → final_lr_ratio over
                   [warmup_steps, total_steps]. ``decay_start_step`` ignored.
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    if schedule == "constant":
        return 1.0
    if schedule == "wsd":
        if step < decay_start_step:
            return 1.0
        if step >= total_steps:
            return final_lr_ratio
        prog = (step - decay_start_step) / max(total_steps - decay_start_step, 1)
        return 1.0 - prog * (1.0 - final_lr_ratio)
    if schedule == "cosine":
        if step >= total_steps:
            return final_lr_ratio
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return final_lr_ratio + 0.5 * (1.0 - final_lr_ratio) * (
            1.0 + math.cos(math.pi * prog))
    raise ValueError(f"unknown schedule '{schedule}'")

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


def parse_args(argv=None):
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
    # #419: a backbone trained on GiftEvalPretrain (its checkpoint holds
    # rev_norm.leading_zero_pad) trains its head on the same stream, with no
    # flag. These two only point that stream elsewhere (tests).
    p.add_argument("--gift-pretrain-index", default=None,
                   help="GiftEvalPretrain record-batch index. Default: "
                        "src/gift_pretrain_index.json.gz.")
    p.add_argument("--gift-pretrain-root", default=None,
                   help="A local copy of the dataset repository to read "
                        "instead of Hugging Face (tests).")
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
                   choices=["ewma", "revin", "meanstd", "none"],
                   help="MUST match the backbone's training-time choice "
                        "(both RevEWMNorm and RevIN have 0 params so state_dict "
                        "doesn't disambiguate). Default 'ewma'. A checkpoint "
                        "with the mean/std scaling (#421) names it with the "
                        "buffer rev_norm.mean_std_scaling, and the head reads "
                        "that kind whatever this flag says.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm for clipping (0 to disable)")
    p.add_argument("--forecast-len", type=int, default=128,
                   help="Head forecast length: 128 (default) or 16 for "
                        "W-heads. A head bank (#412) does not read it: its "
                        "head of patch size P decodes P values.")
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
    p.add_argument("--encoder-source", default="student",
                   choices=["student", "teacher"],
                   help="Which encoder the head reads (#393). 'student' "
                        "(default) is every run before #393. 'teacher' loads "
                        "the backbone with the EMA teacher's patch embedding "
                        "and encoder stack in place of the student's, so the "
                        "head decodes teacher latents; the forecaster stays "
                        "the student's (the teacher has none). Requires a "
                        "backbone trained with --ema-embedding / "
                        "--ema-encoder. The choice is written next to every "
                        "head checkpoint and the official eval refuses to "
                        "run a head on the other encoder.")
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
                        "2026-04-17_reconstruction-head experiment.")
    p.add_argument("--synth-seed", type=int, default=None,
                   help="Seed for the periodic synth generator when "
                        "--mix-ratio > 0. Defaults to args.seed + 20_000 to "
                        "stay separate from the backbone's synth stream.")
    # Backbone architecture overrides — match the values used when training
    # the backbone (see experiments/freq-embedding/scripts/train.py CLI).
    # If omitted, the historical defaults (C=4, H=512, nhead=8, num_layers=6,
    # T_RAW=1024) are used and the state_dict load will fail with a size
    # mismatch on a non-default backbone.
    p.add_argument("--t-raw", type=int, default=None,
                   help="Backbone context window. Overrides T_RAW (default 1024).")
    p.add_argument("--n-channels", type=int, default=None,
                   help="Backbone input channel count C. Overrides "
                        "BACKBONE_CONFIG['C'] (default 4).")
    p.add_argument("--d-model", type=int, default=None,
                   help="Backbone hidden width H. Overrides "
                        "BACKBONE_CONFIG['H'] (default 512).")
    p.add_argument("--n-heads", type=int, default=None,
                   help="Backbone transformer heads. Overrides "
                        "BACKBONE_CONFIG['nhead'] (default 8).")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Backbone transformer depth. Overrides "
                        "BACKBONE_CONFIG['num_layers'] (default 6).")
    p.add_argument("--forecaster-d-model", type=int, default=None,
                   help="Backbone forecaster bottleneck dim. None=same as d-model. "
                        "Required to match v13 backbone (default 128).")
    p.add_argument("--forecaster-n-heads", type=int, default=None,
                   help="Backbone forecaster heads. None=same as n-heads. "
                        "v13 uses 4.")
    # -- LR schedule + AdamW HP --------------------------------------------
    p.add_argument("--schedule", default="constant",
                   choices=["constant", "wsd", "cosine"],
                   help="LR schedule. 'wsd': warmup→stable→linear decay "
                        "(MiniCPM/Hägele et al). 'cosine': warmup→cosine. "
                        "Default 'constant' (the legacy baseline).")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="Linear warmup from 0→peak LR over this many steps.")
    p.add_argument("--decay-start-step", type=int, default=None,
                   help="WSD only: step at which the linear cooldown begins. "
                        "Defaults to 0.8 * total_steps. Ignored for cosine.")
    p.add_argument("--final-lr-ratio", type=float, default=0.1,
                   help="Cooldown / cosine endpoint as a fraction of peak LR.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="AdamW β1.")
    p.add_argument("--beta2", type=float, default=0.999,
                   help="AdamW β2. Set 0.98 to match Moirai HP.")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay. Set 0.1 to match Moirai HP.")
    p.add_argument("--eps", type=float, default=1e-8,
                   help="AdamW eps.")
    # -- Head architecture --------------------------------------------------
    p.add_argument("--head-arch", default="gru",
                   choices=["gru", "linear", "transformer",
                            "transformer-gaussian"],
                   help="Head architecture. 'gru': default. 'linear': "
                        "diagnostic single-Linear probe. 'transformer': "
                        "causal/bidir transformer with 9-bin pinball "
                        "quantile output. 'transformer-gaussian': same "
                        "trunk but predicts (μ, log σ²) per step trained "
                        "with Gaussian NLL — closed-form quantiles via "
                        "inverse normal CDF at eval.")
    p.add_argument("--head-num-layers", type=int, default=6,
                   help="Transformer head: number of decoder layers. "
                        "Default 6 to mirror the backbone.")
    p.add_argument("--head-nhead", type=int, default=6,
                   help="Transformer head: number of attention heads. "
                        "Default 6 to mirror the backbone (H=384/64).")
    p.add_argument("--head-ffn-mult", type=float, default=4.0,
                   help="Transformer head: FFN hidden = ffn_mult * H.")
    p.add_argument("--head-dropout", type=float, default=0.1,
                   help="Dropout for transformer/GRU heads.")
    p.add_argument("--head-causal", default="true",
                   choices=["true", "false"],
                   help="Transformer head: causal mask (default true). "
                        "Set 'false' for a bidirectional head — required "
                        "when forecast_len > W=16 so the head can use "
                        "f_t..f_{t+k} to reconstruct multiple patches.")
    p.add_argument("--head-train-input", default="f_only",
                   choices=["f_only", "e_then_f"],
                   help="Training-time input layout to the head. "
                        "'f_only' (default): head sees f_0..f_{T-1} only "
                        "— the legacy training pattern. "
                        "'e_then_f': head sees [e_0..e_{T-1}, f_0..f_{T-1}] "
                        "(length 2T), loss applied at positions T..2T-1. "
                        "Matches the [e_ctx, rolled_f] input the head sees "
                        "at eval B-strategies, fixing a train-eval input "
                        "distribution mismatch.")
    p.add_argument("--amp-dtype", default="none",
                   choices=["none", "bf16", "fp16"],
                   help="Mixed-precision dtype for backbone fwd + head fwd + "
                        "loss. 'none' (default) = pure fp32 (byte-identical "
                        "to legacy). 'bf16' = autocast to bfloat16 (memory ~½) "
                        "— matches the contrastive trainer's convention. "
                        "'fp16' = float16 autocast (no GradScaler, matching "
                        "the contrastive trainer).")
    return p.parse_args(argv)


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


class _NullContext:
    """Context manager no-op — used when AMP autocast is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def build_head(args, head_config, forecast_len):
    """``(head, kind)``: the head --head-arch names, decoding
    ``forecast_len`` values per quantile."""
    H = head_config["H"]
    causal = args.head_causal == "true"
    shape = (f"({args.head_num_layers}L H{H} nhead{args.head_nhead} "
             f"{'causal' if causal else 'bidir'})")
    if args.head_arch == "linear":
        if args.quantile_head:
            return (LinearQuantileForecastingHead(H=H, forecast_len=forecast_len),
                    "linear-probe quantile (9 levels)")
        return (LinearForecastingHead(H=H, forecast_len=forecast_len),
                "linear-probe MSE (point)")
    if args.head_arch == "transformer":
        if not args.quantile_head:
            raise ValueError(
                "transformer head only implemented for quantile output; "
                "pass --quantile-head")
        return (TransformerQuantileForecastingHead(
            H=H, num_layers=args.head_num_layers, nhead=args.head_nhead,
            ffn_mult=args.head_ffn_mult, forecast_len=forecast_len,
            dropout=args.head_dropout, causal=causal), f"transformer-q {shape}")
    if args.head_arch == "transformer-gaussian":
        return (TransformerGaussianForecastingHead(
            H=H, num_layers=args.head_num_layers, nhead=args.head_nhead,
            ffn_mult=args.head_ffn_mult, forecast_len=forecast_len,
            dropout=args.head_dropout, causal=causal),
            f"transformer-gauss {shape}")
    config = dict(head_config, forecast_len=forecast_len)
    if args.quantile_head:
        return QuantileForecastingHead(**config), "quantile (9 levels)"
    return ForecastingHead(**config), "MSE (point)"


def refuse_bank_flags(args):
    """Refuse the flags a head bank (#412) cannot train with."""
    if args.reconstruction == "forecaster" or args.mixed_rollout > 0:
        raise SystemExit("A head bank (#412) trains the prediction branch "
                         "or --reconstruction encoder (#425). Drop "
                         "--reconstruction forecaster / --mixed-rollout.")
    if not args.quantile_head or args.head_arch == "transformer-gaussian":
        raise SystemExit("A head bank (#412) trains quantile heads on the "
                         "pinball loss. Pass --quantile-head, and a head "
                         "arch other than transformer-gaussian.")


def seed_everything(seed):
    """Seed torch on the CPU and on each GPU, and numpy."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def backbone_config(args, sd):
    """``(config, head_config)``: the shape of the frozen backbone that the
    checkpoint ``sd`` holds, from the defaults, the flags and what the
    checkpoint names, and the head shape that reads it. Writes the values
    it detects to ``args``. Each call starts from the defaults, so the jobs
    of one process (#425) share no config."""
    config, head_config = dict(BACKBONE_CONFIG), dict(HEAD_CONFIG)
    # -- Load frozen backbone --------------------------------------------------
    # Backbone architecture overrides (CLI > defaults). head_config['H']
    # must follow the backbone's H so the head input width matches.
    global T_RAW
    if args.t_raw is not None:
        T_RAW = args.t_raw
    if args.n_channels is not None:
        config["C"] = args.n_channels
    if args.d_model is not None:
        config["H"] = args.d_model
        head_config["H"] = args.d_model
    if args.n_heads is not None:
        config["nhead"] = args.n_heads
    if args.num_layers is not None:
        config["num_layers"] = args.num_layers
    if args.forecaster_d_model is not None:
        config["forecaster_d_model"] = args.forecaster_d_model
    if args.forecaster_n_heads is not None:
        config["forecaster_n_heads"] = args.forecaster_n_heads
    if args.encoder_type is not None:
        config["encoder_type"] = args.encoder_type

    # Auto-detect freq_emb_dim and seasonality_emb_dim from the checkpoint
    # if not explicitly set.
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
    config["freq_emb_dim"] = args.freq_emb_dim
    config["seasonality_emb_dim"] = args.seasonality_emb_dim
    config["rev_norm_kind"] = args.rev_norm_kind
    # #419: the frequency table's row count is the vocabulary (10: v1, every
    # earlier model), and a buffer marks a normaliser that skips zero
    # padding. Both rebuild the backbone the checkpoint trained.
    freq_w = sd.get("freq_embedding.embedding.weight")
    if freq_w is not None:
        config["num_freqs"] = freq_w.shape[0]
    zero_pad = "rev_norm.leading_zero_pad" in sd
    config["rev_norm_skip_leading_zeros"] = zero_pad
    if not zero_pad and (args.gift_pretrain_index or args.gift_pretrain_root):
        raise SystemExit("--gift-pretrain-index and --gift-pretrain-root read "
                         "the stream of a zero-padding backbone (#419), and "
                         "this backbone has none: its head trains on "
                         "--hf-repo. Drop them.")
    # #421: the bound of the GRU input, when the backbone trained with one.
    config["gru_input_bound"] = gru_input_bound_of(sd)
    # #412: the checkpoint names its patch sizes (#417) and the mean/std
    # scaling (#421). Such a backbone gets one head per patch size, trained
    # as the backbone trained: sizes, split and statistics per row.
    patch_sizes = multi_patch_sizes_of(sd)
    config["multi_patch_sizes"] = patch_sizes
    if "rev_norm.mean_std_scaling" in sd and args.rev_norm_kind != "meanstd":
        print(f"  [head-train] the checkpoint names the mean/std scaling "
              f"(#421); --rev-norm-kind {args.rev_norm_kind} is not read")
        args.rev_norm_kind = "meanstd"
    config["rev_norm_kind"] = args.rev_norm_kind
    use_bank = bool(patch_sizes) or args.rev_norm_kind == "meanstd"
    if use_bank:
        refuse_bank_flags(args)
    if zero_pad and (args.reconstruction == "forecaster"
                     or args.mixed_rollout > 0):
        raise SystemExit(
            "This backbone trained on zero-padded GiftEvalPretrain windows "
            "(#419). Its head masks the padded targets of the prediction "
            "branch and of --reconstruction encoder (#425) only. Drop "
            "--reconstruction forecaster / --mixed-rollout.")
    # Auto-detect CLIP-style learnable τ from the checkpoint (#28). If
    # log_inv_tau is in the state_dict, we must instantiate the backbone
    # with learnable_tau=True so load_state_dict succeeds. The head loss
    # doesn't use τ, so this is just to keep the param around.
    if "log_inv_tau" in sd:
        config["learnable_tau"] = True
        print(f"  [head-train] auto-detected learnable τ "
              f"(log_inv_tau={sd['log_inv_tau'].item():.4f}, "
              f"τ={float((-sd['log_inv_tau']).exp()):.4f}) from backbone checkpoint")
    # Auto-detect num_encoder_layers from transformer.encoder_layers.<N>.* keys.
    # Encoder-forecaster backbones (2026-05-10) prepend N causal layers before
    # the forecaster, and their state_dict has the matching keys. Old backbones
    # have an empty encoder stack and the keys are absent → defaults to 0.
    enc_layer_idxs = set()
    for k in sd:
        if k.startswith("transformer.encoder_layers."):
            try:
                enc_layer_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if enc_layer_idxs:
        config["num_encoder_layers"] = max(enc_layer_idxs) + 1
        print(f"  [head-train] auto-detected num_encoder_layers="
              f"{config['num_encoder_layers']} from backbone checkpoint")
    # Auto-detect the b1024 collapse-fix norms (#322): QK-norm (q_norm/k_norm) and
    # attention-output RMSNorm (attn_out_rms) add per-layer params to encoder +
    # forecaster layers. Build with the matching flags so _qk_aon backbones load
    # cleanly. Absent keys -> flags stay False (older backbones unaffected).
    if any(k.endswith(".q_norm.weight") for k in sd):
        config["qk_norm"] = True
        print("  [head-train] auto-detected qk_norm=True from backbone checkpoint")
    if any(k.endswith(".attn_out_rms.weight") for k in sd):
        config["attn_out_norm"] = True
        print("  [head-train] auto-detected attn_out_norm=True from backbone checkpoint")
    # Auto-detect a CPC multi-step forecaster (#316). Two families:
    #   transformer.cpc_layers.<N>.*  → 'cpc'        (K transformer-1L heads, #1)
    #   transformer.cpc_heads.<N>.*   → 'linear_cpc' (K linear heads, #2/#3)
    # Build with the matching forecaster_kind + K so load_state_dict succeeds.
    # For 'cpc' the bottleneck dim is read from cpc_down.0.weight. Either way
    # extract_forecaster_latents returns the next-step (k=1) head.
    lin_idxs = set()
    for k in sd:
        if k.startswith("transformer.cpc_heads."):
            try:
                lin_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if lin_idxs:
        config["forecaster_kind"] = "linear_cpc"
        config["cpc_k_steps"] = max(lin_idxs) + 1
        print(f"  [head-train] auto-detected linear_cpc forecaster "
              f"(K={config['cpc_k_steps']}) from checkpoint")
    cpc_head_idxs = set()
    for k in sd:
        if k.startswith("transformer.cpc_layers."):
            try:
                cpc_head_idxs.add(int(k.split(".")[2]))
            except (IndexError, ValueError):
                continue
    if cpc_head_idxs:
        config["forecaster_kind"] = "cpc"
        config["cpc_k_steps"] = max(cpc_head_idxs) + 1
        w = sd.get("transformer.cpc_down.0.weight")
        if w is not None:
            config["forecaster_d_model"] = w.shape[0]
            args.forecaster_d_model = w.shape[0]
        print(f"  [head-train] auto-detected cpc forecaster "
              f"(K={config['cpc_k_steps']}, "
              f"d={config.get('forecaster_d_model')}) from checkpoint")
    if args.rev_norm_kind == "ewma":
        config["rev_norm_span"] = args.rev_norm_span
    # Auto-detect patch_stats from the encoder's first projection input width.
    # The GRU encoder stores `encoder.skip.weight` of shape [H, encoder_input].
    # MLP-style encoders store `encoder.linear1.weight` similarly. Either way
    # the in-features tells us W + freq_emb_dim + (2 if patch_stats else 0).
    if args.patch_stats == "auto":
        from src.norm import PATCH_STATS_DIM
        W = config["W"]
        skip_w = sd.get("encoder.skip.weight")
        if skip_w is None:
            skip_w = sd.get(f"encoder.encoders.{W}.skip.weight")
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
    config["patch_stats_kind"] = args.patch_stats
    return config, head_config


def load_frozen_backbone(args, config, sd, device):
    """The backbone of ``config`` with the weights of ``sd``, frozen."""
    backbone = ConfigurableModel(**config)
    # Drops the pretraining-only branches (CPC-InfoNCE `cpc_w1.*`, the EMA
    # teacher's `teacher_*`) so the strict load matches the head-time
    # backbone, and — under --encoder-source teacher (#393) — promotes the
    # teacher's encoder weights into the student's slots first.
    sd = prepare_backbone_state_dict(sd, args.encoder_source)
    backbone.load_state_dict(sd)
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    print(f"Backbone loaded from {args.backbone_path} "
          f"({count_parameters(backbone):,} params, frozen, "
          f"encoder={args.encoder_source})")
    return backbone


def build_job_head(args, config, head_config, use_bank, device):
    """``(head, kind)``: the head of one run, on ``device``."""
    if use_bank:
        sizes = config["multi_patch_sizes"] or (config["W"],)
        built = {p: build_head(args, head_config, p) for p in sizes}
        head = ForecastingHeadBank(
            {p: h for p, (h, _) in built.items()}).to(device)
        head_kind = (f"bank of {len(sizes)} heads, sizes {sizes}, each "
                     f"{built[sizes[0]][1]} decoding its P values (#412)")
    else:
        head, head_kind = build_head(args, head_config, args.forecast_len)
        head = head.to(device)
    n_head_params = count_parameters(head)
    print(f"Forecasting head [{head_kind}]: {n_head_params:,} trainable params")
    return head, head_kind


class HeadJob:
    """One head on its frozen backbone: the state and the steps of one run
    of this script.

    :func:`main` trains one job. ``train_forecasting_heads_shared.py``
    (#425) trains several jobs on one data stream: each job keeps its own
    backbone, head, optimizer, seed and files, and takes the steps of its
    solo run.
    """

    def __init__(self, args):
        self.args = args
        device = self.device = torch.device(args.device)
        # Reproducibility
        seed_everything(args.seed)
        os.makedirs(args.save_dir, exist_ok=True)
        sd = torch.load(args.backbone_path, map_location=device, weights_only=True)
        config, head_config = backbone_config(args, sd)
        self.C = config["C"]
        self.zero_pad = config["rev_norm_skip_leading_zeros"]
        self.patch_sizes = config["multi_patch_sizes"]
        self.use_bank = bool(self.patch_sizes) or args.rev_norm_kind == "meanstd"
        freq_w = sd.get("freq_embedding.embedding.weight")
        self.freq_vocab = vocab_of_rows(freq_w.shape[0]) if freq_w is not None else "v1"
        self.emit_labels = (args.freq_emb_dim > 0 or args.seasonality_emb_dim > 0)
        self.backbone = load_frozen_backbone(args, config, sd, device)
        head, _ = build_job_head(args, config, head_config, self.use_bank, device)
        optimizer = optim.AdamW(
            head.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
        )

        # Schedule defaults: WSD → 80% stable + 20% cooldown if user didn't set.
        if args.decay_start_step is None:
            args.decay_start_step = int(0.8 * args.total_steps)
        print(f"Optimizer: AdamW lr={args.lr} betas=({args.beta1},{args.beta2}) "
              f"wd={args.weight_decay} eps={args.eps}")
        print(f"Schedule: {args.schedule} warmup={args.warmup_steps} "
              f"decay_start={args.decay_start_step} "
              f"final_ratio={args.final_lr_ratio}")

        # -- Resume --------------------------------------------------------
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

        rows_per_step = args.batch_size * self.C
        if args.no_resume_data_skip:
            hf_rows_consumed = args.skip_rows
            print(f"  [data] --no-resume-data-skip: starting from HF offset "
                  f"{args.skip_rows} (NOT {start_step * rows_per_step + args.skip_rows})")
        else:
            hf_rows_consumed = start_step * rows_per_step + args.skip_rows
        self.head, self.optimizer = head, optimizer
        self.start_step, self.best_loss = start_step, best_loss
        self.best_loss_step = best_loss_step
        self.rows_per_step, self.hf_rows_consumed = rows_per_step, hf_rows_consumed
        self.ema_loss, self.ema_decay, self.nan_skips = None, 0.99, 0
        self.csv_logger, self.t0 = None, None

    def data_key(self):
        """What decides the batches of this run. Runs with one key read the
        same batches in the same order, so one stream feeds them all (#425).
        """
        args = self.args
        stream = (("GiftEvalPretrain", args.seed, self.freq_vocab,
                   args.gift_pretrain_index, args.gift_pretrain_root)
                  if self.zero_pad else ("hf", args.hf_repo, args.hf_path))
        synth = ((args.mix_ratio, args.synth_seed if args.synth_seed is not None
                  else args.seed + 20_000) if args.mix_ratio > 0 else ())
        return (stream, synth, self.emit_labels, self.C, args.batch_size,
                self.hf_rows_consumed, self.start_step, args.total_steps)

    def data_loader(self):
        """The data stream of this run."""
        args, C, freq_vocab = self.args, self.C, self.freq_vocab
        zero_pad, emit_labels = self.zero_pad, self.emit_labels
        hf_rows_consumed = self.hf_rows_consumed
        # #419: the head of a zero-padding backbone reads the stream the
        # backbone trained on, in the vocabulary its checkpoint holds.
        real_rows = None
        if zero_pad:
            from src.gift_pretrain import stream_factory
            real_rows = stream_factory(
                [args.seed, hf_rows_consumed], C, freq_vocab,
                args.gift_pretrain_index, args.gift_pretrain_root)
            print(f"Data: the backbone trained on GiftEvalPretrain with zero "
                  f"padding (#419): the head trains on the same stream, "
                  f"vocabulary {freq_vocab}, padded targets skipped. "
                  f"--hf-repo/--hf-path are not read.")
        if args.mix_ratio > 0 or emit_labels:
            # Use the mixed loader when we need labels, even if mix_ratio=0
            # (it falls through to MixedPeriodicLoader with synth_bs=0 and
            # yields the (x, freq_ids, seasonality_ids) tuples extract_*_latents
            # consumes).
            synth_seed = args.synth_seed if args.synth_seed is not None else args.seed + 20_000
            data_loader = create_mixed_periodic_dataloader(
                repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
                mix_ratio=args.mix_ratio,
                path_in_repo=args.hf_path, skip_rows=hf_rows_consumed,
                seed=synth_seed, emit_freq_ids=emit_labels,
                real_rows=real_rows,
            )
            synth_bs = int(round(args.batch_size * args.mix_ratio))
            hf_bs = args.batch_size - synth_bs
            print(f"Data: MIX {(1-args.mix_ratio)*100:.0f}% HF + "
                  f"{args.mix_ratio*100:.0f}% synth, hf_bs={hf_bs}, "
                  f"synth_bs={synth_bs}, synth_seed={synth_seed}, "
                  f"emit_labels={emit_labels}")
        elif real_rows is not None:
            data_loader = real_rows(args.batch_size, False)
        else:
            data_loader = create_hf_dataloader(
                args.hf_repo, batch_size=args.batch_size, C=C,
                path_in_repo=args.hf_path, skip_rows=hf_rows_consumed)
            print(f"Data: HF streaming from {args.hf_repo}/{args.hf_path} "
                  f"(skip={hf_rows_consumed} rows)")
        return data_loader

    def start(self):
        """Open the loss CSV, describe the run and start its clock."""
        args = self.args
        csv_path = os.path.join(args.save_dir, f"{args.run_name}_losses.csv")
        self.csv_logger = CSVLogger(csv_path)
        print(f"Loss CSV: {csv_path}")

        print(f"\nTraining for {args.total_steps} steps, bs={args.batch_size}, "
              f"lr={args.lr}, forecast_len={args.forecast_len}")
        if self.use_bank:
            print(f"Head bank (#412): each row draws its patch size from its "
                  f"frequency's range {'and its split ' if args.rev_norm_kind == 'meanstd' else ''}"
                  f"as the backbone trained, and trains the head of that size. "
                  f"--forecast-len is not read.")
        if args.reconstruction:
            print(f"RECONSTRUCTION mode: {args.reconstruction} "
                  f"(head decodes what latent represents, not future)")
        if args.mixed_rollout > 0:
            print(f"Mixed-rollout mode: {args.mixed_rollout} rolled tokens per step "
                  f"(48 context patches + {args.mixed_rollout} rolled)")
        print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")
        sys.stdout.flush()
        self.t0 = time.time()

    def train_step(self, step, batch):
        """One optimizer step on one batch of the stream."""
        args, device = self.args, self.device
        backbone, head, optimizer = self.backbone, self.head, self.optimizer
        zero_pad, use_bank = self.zero_pad, self.use_bank
        patch_sizes = self.patch_sizes
        head.train()
        optimizer.zero_grad()

        # Apply schedule before this step's update.
        mult = lr_multiplier(
            step - 1, args.total_steps, args.schedule,
            args.warmup_steps, args.decay_start_step, args.final_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = args.lr * mult

        if isinstance(batch, tuple):
            x, freq_ids, seasonality_ids = batch
            freq_ids = freq_ids.to(device)
            seasonality_ids = seasonality_ids.to(device)
        else:
            x = batch
            freq_ids = None
            seasonality_ids = None
        self.hf_rows_consumed += self.rows_per_step
        x = x.to(device)

        # AMP autocast wraps the (frozen) backbone forward + head forward +
        # loss. No GradScaler — matches the contrastive trainer's convention
        # (bf16's range matches fp32, and for fp16 we skip the scaler too).
        # Pinball / Gaussian-NLL / MSE losses don't have F.normalize-style
        # fp32 promotion, so no `to(amp_dtype)` cast trick is needed here
        # (unlike the contrastive trainer's `f_lat.to(amp_dtype)`).
        amp_dtype = (torch.bfloat16 if args.amp_dtype == "bf16"
                     else torch.float16 if args.amp_dtype == "fp16"
                     else None)
        amp_ctx = (torch.amp.autocast('cuda', dtype=amp_dtype)
                   if amp_dtype is not None else _NullContext())
        with amp_ctx:
            if use_bank:
                # #412: each row reads its patch size and trains that
                # size's head. loc and scale read the context before the
                # split, and the loss counts the values after it. #425:
                # a reconstruction head decodes the encoder latent of each
                # patch into the values of that patch.
                x_norm, sample_sizes, keep = bank_training_inputs(
                    backbone, x, freq_ids, patch_sizes)
                loss = bank_quantile_loss(
                    backbone, head, x_norm, sample_sizes, keep,
                    freq_ids=freq_ids, seasonality_ids=seasonality_ids,
                    reconstruction=args.reconstruction == "encoder")
            elif args.mixed_rollout > 0:
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
            elif args.reconstruction == 'encoder' and args.quantile_head:
                # #425: the B4 quantile head on e[t] → patch t values, on
                # the pinball loss. Left padding (#419) counts in no term.
                e_bc, x_norm = extract_encoder_latents(
                    backbone, x, freq_ids=freq_ids,
                    seasonality_ids=seasonality_ids)
                keep = ~backbone.rev_norm.pad_mask if zero_pad else None
                loss = reconstruction_quantile_loss(
                    head(e_bc), x_norm, W, keep=keep,
                    output_len=args.forecast_len)

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
                if args.head_train_input == "e_then_f":
                    # Match the eval-time [e_ctx, rolled_f] layout: feed the
                    # head [e_0..e_{T-1}, f_0..f_{T-1}] (length 2T). The
                    # head's outputs at positions T..(T+T_valid-1) — that is the
                    # f-half — get the loss against `targets`.
                    #
                    # Custom mask prevents the head from peeking at e_{p_f+1}
                    # (the encoder latent that encodes the *target* patch
                    # for f-block position p_f). With the standard causal
                    # mask, all e-block positions are "past" relative to any
                    # f-block position, so the head can copy the target
                    # directly — that bug showed up as self.ema_loss collapsing
                    # to <0.12 vs the legitimate ~0.19 plateau.
                    e_bc, _ = extract_encoder_latents(
                        backbone, x, freq_ids=freq_ids,
                        seasonality_ids=seasonality_ids)
                    T_e = e_bc.size(1)
                    T_f = f_bc.size(1)
                    seq = torch.cat([e_bc, f_bc], dim=1)             # (BC, T_e+T_f, H)
                    from src.forecasting_head import build_e_then_f_mask
                    src_mask = build_e_then_f_mask(T_e, T_f, device=device)
                    preds = head(seq, src_mask=src_mask)
                    f_slice = slice(T_e, T_e + T_valid)
                else:
                    preds = head(f_bc)
                    f_slice = slice(0, T_valid)
                if isinstance(preds, tuple):
                    # Gaussian head: (mu, log_var) per position.
                    mu, log_var = preds
                    mu = mu[:, f_slice, :]
                    log_var = log_var[:, f_slice, :]
                    loss = gaussian_nll_loss(mu, log_var, targets)
                elif args.quantile_head:
                    preds = preds[:, f_slice, :, :]                   # (BC, T, Q, L)
                    loss = quantile_loss(preds, targets, QUANTILE_LEVELS)
                else:
                    preds = preds[:, f_slice, :]
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
                # #419: the padded targets of a zero-padding backbone.
                keep = (valid_target_keep(backbone.rev_norm.pad_mask, W,
                                          args.forecast_len)
                        if zero_pad else None)
                if args.quantile_head:
                    preds = preds[:, :T_valid, :, :]
                    loss = (quantile_loss(preds, targets, QUANTILE_LEVELS)
                            if keep is None else masked_quantile_loss(
                                preds, targets, keep, QUANTILE_LEVELS))
                else:
                    preds = preds[:, :T_valid, :]
                    loss = (torch.nn.functional.mse_loss(preds, targets)
                            if keep is None
                            else masked_mean((preds - targets) ** 2, keep))

        # NaN detection -- skip bad batches instead of crashing
        loss_val = loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            self.nan_skips += 1
            print(f"  [step {step}] NaN/Inf loss detected, skipping batch "
                  f"(total skips: {self.nan_skips})")
            sys.stdout.flush()
            optimizer.zero_grad()  # discard any partial gradients
            return

        # Backward + step
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        optimizer.step()

        # EMA tracking
        if self.ema_loss is None:
            self.ema_loss = loss_val
        else:
            self.ema_loss = (self.ema_decay * self.ema_loss
                             + (1 - self.ema_decay) * loss_val)

        # Per-step CSV logging
        self.csv_logger.log(step, loss_val, self.hf_rows_consumed)

        # Console logging
        if step % args.log_every == 0:
            elapsed = time.time() - self.t0
            sps = (step - self.start_step) / elapsed
            eta = (args.total_steps - step) / sps / 3600

            print(f"[{step:>7d}] loss={loss_val:.6f}  ema_loss={self.ema_loss:.6f}  "
                  f"{sps:.1f} sps  ETA {eta:.1f}h")
            sys.stdout.flush()

        # Best checkpoint
        if step % args.log_every == 0 and self.ema_loss < self.best_loss:
            self.best_loss = self.ema_loss
            self.best_loss_step = step
            path = os.path.join(args.save_dir, f"{args.run_name}_best.pth")
            _save_head(head, optimizer, path, step, self.best_loss,
                       self.best_loss_step, args.encoder_source)
            print(f"  -> New best: {path} (ema_loss={self.ema_loss:.6f})")

        # Periodic snapshot
        if step % args.save_every == 0:
            path = os.path.join(
                args.save_dir, f"{args.run_name}_{step // 1000}k.pth")
            _save_head(head, optimizer, path, step, self.best_loss,
                       self.best_loss_step, args.encoder_source)
            print(f"  -> Saved {path}")

    def finish(self):
        """Save the final head and close the loss CSV."""
        args, head, optimizer = self.args, self.head, self.optimizer
        path = os.path.join(args.save_dir, f"{args.run_name}_final.pth")
        _save_head(head, optimizer, path, args.total_steps, self.best_loss,
                   self.best_loss_step, args.encoder_source)
        self.csv_logger.close()

        total = time.time() - self.t0
        print(f"\nDone in {total / 3600:.1f}h. "
              f"Best loss={self.best_loss:.6f} at step {self.best_loss_step}")


def main():
    job = HeadJob(parse_args())
    data_loader = job.data_loader()
    data_iter = iter(data_loader)
    job.start()
    for step in range(job.start_step + 1, job.args.total_steps + 1):
        # Data loading — when emit_labels is on, the dataloader yields
        # (x, freq_ids, seasonality_ids). Otherwise it yields just x.
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        job.train_step(step, batch)
    job.finish()


def _save_head(head, optimizer, path, step, best_loss, best_loss_step,
               encoder_source):
    """Head weights, optimizer companion, and the encoder-source marker.

    The marker (#393) travels with every checkpoint so a head trained on the
    EMA teacher can never be evaluated through the student encoder.
    """
    torch.save(head.state_dict(), path)
    _save_optim_meta(optimizer, path, step, best_loss, best_loss_step)
    save_encoder_source(path, encoder_source)


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
