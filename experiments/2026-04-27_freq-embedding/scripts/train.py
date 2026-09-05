#!/usr/bin/env python3
"""
Contrastive training with frequency embedding + optional mixup.

Same as experiments/2026-04-27_periodic-synth-mix/scripts/train.py but:
 - backbone receives a learned frequency embedding concat'd per-patch
   (see src/freq_embedding.py + src.models.ConfigurableModel.freq_emb_dim)
 - optional mixup augmentation: with probability --mixup-p, each step
   linearly interpolates two batch items (both X and the freq embedding)

Usage
-----
    # freqemb only, no mixup
    python experiments/2026-04-27_freq-embedding/scripts/train.py \
        --device cuda --run-name freqemb_mix --total-steps 30000 \
        --batch-size 24 --lr 1e-4 --weight-decay 0.1 --save-dir checkpoints \
        --hf-repo jeremycochoy/contrastive-training-base-bundles \
        --hf-path base_mixed_v1 --mix-ratio 0.5 \
        --freq-emb-dim 3 --mixup-p 0.0

    # freqemb + mixup
    python experiments/2026-04-27_freq-embedding/scripts/train.py \
        --run-name freqemb_mixup_mix --freq-emb-dim 3 --mixup-p 0.3 \
        ... other args same as above ...
"""

import argparse
import csv
import math
import os
import shlex
import sys
import time
import torch
import torch.optim as optim
from types import SimpleNamespace

from src.models import (ConfigurableModel, compute_metrics, count_parameters,
                        ema_tau_at_step, generate_random_batch,
                        linear_schedule_at_step)
from src.blocks import ATTN_AMP_DIAG
from src.dataloader import (
    create_mixed_periodic_dataloader,
    create_mixed_composite_dataloader,
    create_mixed_forked_arma_dataloader,
    create_hf_dataloader,
)
from src.loss import (contrastive_latent_loss, cpc_infonce_aux_loss,
                      cpc_infonce_all_loss, align_loss, align_moco_loss,
                      sigreg_loss)
from src.checkpoint import save_training_state, load_training_state
from src.dist_utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    gather_latent,
    gather_latents,
    average_gradients,
    broadcast_module,
)
from src.metrics import (
    q_random,
    q_naive_latent,
    dim_usage,
    drift_pair,
    rollout_cos_error,
    u_batchtime,
    retrieval_auc_topk,
)
from src.forecasting_head import (extract_encoder_latents,
                                  extract_teacher_encoder_latents,
                                  rollout_forecaster_latents)

# -- Tiny architecture (identical to v3c) -----------------------------------
# C and T_raw can be overridden at runtime via --n-channels / --t-raw to
# accommodate datasets shaped differently from the standard
# (T_raw=1024, C=4) bundles — for example exp_realonly_4096_2arm trains at
# (T_raw=4096, C=1) on jeremycochoy/gift-pretrain-small-4096.
MODEL_CONFIG = dict(
    C=4, H=512, W=16,
    encoder_type="gru", num_layers=6, nhead=8,
    ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.1,
)

LOSS_SPEC = SimpleNamespace(train_configuration={
    "contrastive_divergence_temperature": 0.07,
    "contrastive_latent_noise": None,
    "loss_shape": "cosine_similarity_batch_no_time_neg",
    "contrastive_latent_delay": 0,
})
CLD = LOSS_SPEC.train_configuration["contrastive_latent_delay"] + 1

# The loss shapes that read `rep_loss_weight` (#382, #409). Every other shape
# ignores it, so a decay schedule on one of those would move nothing.
REP_WEIGHT_SHAPES = ("cosine_similarity_batch_split_pred_rep",
                     "cosine_similarity_batch_rep_only")

T_RAW = 1024  # Default. Overridden by --t-raw CLI flag.


def parse_args():
    p = argparse.ArgumentParser(description="Contrastive + freq embedding training")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    # Optimizer hyperparams. --adam-beta1/2 keep torch.optim.AdamW defaults
    # so prior runs that omitted them reproduce bit-identically.
    # MOIRAI Aksu and others recipe: lr=1e-3, weight_decay=0.1, betas=(0.9, 0.98).
    # --weight-decay is intentionally REQUIRED (no default): a silent 0.01
    # default previously let runs inherit the wrong decay by accident.
    # Forcing an explicit value makes every (re)training state its decay on
    # the command line — use 0.1 for new trainings.
    p.add_argument("--weight-decay", type=float, required=True,
                   help="AdamW weight decay. REQUIRED — pass it explicitly so "
                        "a run can never silently inherit a wrong value. Use "
                        "0.1 for new trainings (MOIRAI recipe).")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--run-name", default="freqemb")
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--extra-save-steps", default=None,
                   help="Comma-separated list of extra step counts at which "
                        "to snapshot on top of --save-every (e.g. "
                        "'2500,25000' when the base cadence is 10000 but "
                        "downstream eval needs off-cadence cells).")
    p.add_argument("--traj-save-every", type=int, default=0,
                   help="If > 0, also write a fine-grained trajectory "
                        "checkpoint every N steps as "
                        "`<run>_step<STEP>.pth`. Separate cadence from "
                        "--save-every: the coarse `<run>_<K>k.pth` files "
                        "are still emitted. Use for sub-1000-step "
                        "trajectories where the coarse `step // 1000` "
                        "naming collides. Default 0 = off.")
    # Latent-drift probe (#XXX). Fixed synthetic probe batch, no-grad
    # forward every N steps, dumps drift_cos / drift_cos_aligned /
    # rot_gap / cka between the current h_t and (a) the previous probe
    # and (b) the initial probe. Rank-0 only. CSV goes to
    # `<save_dir>/<run_name>_latent_drift.csv`. Default on — probe cost
    # is one no-grad forward per cadence step (≪ 1 s per probe at the
    # #374 arch), so leaving it on for every future run gives us "free"
    # drift curves.
    p.add_argument("--latent-drift-probe",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Track h_t drift on a fixed probe batch. "
                        "Writes <run>_latent_drift.csv. Default on. "
                        "Pass --no-latent-drift-probe to disable.")
    p.add_argument("--latent-drift-probe-every", type=int, default=0,
                   help="Probe cadence in steps. 0 (default) = mirror "
                        "--save-every so probes coincide with snapshots.")
    p.add_argument("--latent-drift-probe-batch-size", type=int, default=64,
                   help="Probe batch (fixed ARMA draw). Kept small — the "
                        "metric is the geometry of h_t, not throughput.")
    p.add_argument("--latent-drift-probe-seed", type=int, default=20260722,
                   help="Seed for the fixed ARMA probe batch. Held "
                        "constant across a run so every probe sees the "
                        "same input; change it to check probe-noise "
                        "sensitivity.")
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-path", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mix-ratio", type=float, default=0.5)
    p.add_argument("--crossfade-ratio", type=float, default=0.0,
                   help="Fraction of the batch built as regime-crossfade rows "
                        "(#325): a monotone blend of two distinct real windows "
                        "from the step's real sub-batch. Stacks with --mix-ratio "
                        "(forked-arma); real fraction = 1 - mix_ratio - "
                        "crossfade_ratio. Only valid with --synth-kind forked-arma.")
    p.add_argument("--crossfade-triplets", type=int, default=0,
                   help="Number of explicit (A_norm, B_norm, C) crossfade "
                        "triplets to append ON TOP of the batch (#328): both "
                        "z-normalised parents plus their monotone blend, 3 rows "
                        "per triplet, additive (total batch = batch_size + "
                        "3*crossfade_triplets). Drawn from the real sub-batch; "
                        "complements --crossfade-ratio (which adds C-only rows).")
    p.add_argument("--synth-seed", type=int, default=None)
    p.add_argument("--t-raw", type=int, default=T_RAW,
                   help="Raw window length (T) per sample. Default 1024 "
                        "(matches the standard contrastive-training bundles); "
                        "set to 4096 for gift-pretrain-small-4096.")
    p.add_argument("--n-channels", type=int, default=MODEL_CONFIG["C"],
                   help="Number of input channels (C). Default 4 (matches "
                        "base_mixed_v1 4-channel stack); set to 1 for "
                        "single-channel datasets like gift-pretrain-small-4096.")
    p.add_argument("--d-model", type=int, default=MODEL_CONFIG["H"],
                   help="Hidden / embedding dimension (H). Default 512 "
                        "(Tiny). Use 384 for the smaller-arch sweep arms.")
    p.add_argument("--n-heads", type=int, default=MODEL_CONFIG["nhead"],
                   help="Number of attention heads. Default 8 (Tiny). "
                        "Use 6 for the H=384 smaller-arch arms.")
    p.add_argument("--num-layers", type=int, default=MODEL_CONFIG["num_layers"],
                   help="Number of encoder layers. Default 6.")
    p.add_argument("--forecaster-d-model", type=int, default=None,
                   help="Width of the FORECASTER stack only (the decoder "
                        "side after the encoder boundary, i.e. "
                        "TransformerBlock.layers). When unset, inherits "
                        "--d-model (legacy: forecaster runs at the encoder's "
                        "width). Set smaller (e.g. 128) to add a JEPA "
                        "Linear bottleneck around the forecaster: encoder "
                        "output is projected H -> forecaster_d_model, the "
                        "1L (or NL) forecaster runs at the smaller width, "
                        "then projected back to H for the contrastive loss "
                        "in the encoder's d_model space. Encoder stack and "
                        "x_original (loss target) keep --d-model untouched.")
    p.add_argument("--forecaster-n-heads", type=int, default=None,
                   help="Number of attention heads in the forecaster stack. "
                        "When unset, inherits --n-heads. Required to satisfy "
                        "forecaster-d-model %% forecaster-n-heads == 0. "
                        "v13 default: --forecaster-d-model 128 "
                        "--forecaster-n-heads 4 (4 heads x 32 dim/head).")
    p.add_argument("--forecaster-kind", default="transformer",
                   choices=["transformer", "cpc", "linear_cpc"],
                   help="Forecaster variant (#316). 'transformer' (default) "
                        "is the legacy single causal forecaster (optionally "
                        "bottlenecked via --forecaster-d-model). 'cpc' uses "
                        "--cpc-k-steps independent forecaster heads, each "
                        "ARCHITECTURALLY IDENTICAL to the transformer "
                        "forecaster (down -> causal transformer -> up, same "
                        "--forecaster-d-model/-n-heads bottleneck), with head k "
                        "forecasting the encoder latent k steps ahead (h_{t+k}); "
                        "requires --loss-shape cpc_multistep. At k=1 it is "
                        "byte-identical to the transformer forecaster, so only "
                        "the number of forecast steps differs.")
    p.add_argument("--cpc-k-steps", type=int, default=12,
                   help="CPC forecast horizon K (number of transformer-1L "
                        "forecaster heads) when --forecaster-kind cpc. Default "
                        "12 (van den Oord et al. 2018). No-op otherwise.")
    p.add_argument("--num-encoder-layers", type=int, default=0,
                   help="Number of causal transformer-encoder layers inserted "
                        "between the patch encoder and the forecaster. "
                        "Default 0 = baseline (no pre-forecaster encoder). "
                        "Encoder output is the contrastive target (x_original).")
    p.add_argument("--encoder-dropkey", type=float, default=0.0,
                   help="Per-step DropKey probability on the encoder layers' "
                        "below-diagonal attention entries. 0.0 (default) = "
                        "pure causal. e.g. 0.3 drops 30%% of past-key edges "
                        "per layer per step. Forecaster mask is unaffected.")
    p.add_argument("--encoder-dropkey-share-heads", action="store_true",
                   help="If set, the per-step dropkey mask is shared across "
                        "all heads of a given (batch_row, layer). Default "
                        "False = independent per (batch_row, head). "
                        "Tying heads drops variance by ~num_heads× and "
                        "prevents heads from cooperating to count positions.")
    p.add_argument("--encoder-dropkey-share-layers", action="store_true",
                   help="If set, the per-step dropkey mask is drawn ONCE and "
                        "reused for ALL encoder layers in the current forward "
                        "pass. Default False = independent per-layer draw. "
                        "Combined with --encoder-dropkey-share-heads, only the "
                        "(batch_row, step) axes carry randomness. Makes a "
                        "given token either fully visible across all layers "
                        "(prob 1-p) or fully blocked across all layers (prob "
                        "p) — much harder for a position counter to recover "
                        "than the layer-independent case where the union of "
                        "visible-at-some-layer is large.")
    p.add_argument("--encoder-type", default=MODEL_CONFIG["encoder_type"],
                   choices=["mlp", "mlp_wide", "residual_silu", "gru", "conv",
                            "transformer"],
                   help="Patch encoder type. Default 'gru' (matches all backbone-beta runs). "
                        "'transformer' replaces GRU+skip with a small decoder-only "
                        "causal transformer (Linear(W'->H) + N layers attending over T).")
    p.add_argument("--enc-num-layers", type=int, default=4,
                   help="encoder-transformer: number of layers. Default 4.")
    p.add_argument("--enc-nhead", type=int, default=6,
                   help="encoder-transformer: attention heads. Default 6 "
                        "(head_dim=64 with H=384).")
    p.add_argument("--enc-ffn-mult", type=float, default=4.0,
                   help="encoder-transformer: FFN expansion factor. Default 4.0 "
                        "(matches the backbone's ffn_mult).")
    p.add_argument("--enc-dropout", type=float, default=0.0,
                   help="encoder-transformer: dropout. Default 0.0.")
    p.add_argument("--enc-depthwise-conv", type=int, default=3,
                   help="encoder-transformer: depthwise causal conv kernel "
                        "size. Default 3 (matches backbone). 0 disables it "
                        "(closer to a pure-residual highway at init).")
    p.add_argument("--enc-chunk-size", type=int, default=8192,
                   help="encoder-transformer: chunk size along the B*T*C "
                        "axis. With B=256, T=256, C=1 the encoder sees "
                        "65k length-22 sequences in parallel — chunking is "
                        "needed to fit in 24 GB. Default 8192. 0 disables.")
    p.add_argument("--enc-no-grad-ckpt", action="store_true",
                   help="encoder-transformer: disable activation "
                        "checkpointing on encoder layers. Default uses "
                        "checkpointing — costs ~30%% extra compute, saves "
                        "the FFN intermediates from being kept for backward.")
    p.add_argument("--residual-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp32",
                   help="Dtype of the residual stream + LayerNorm + depthwise "
                        "conv. Outer-most precision. fp32 is the safe default. "
                        "fp16/bf16 enable mixed precision but bf16 is unstable "
                        "for our high-aligned-cos-sim regime.")
    p.add_argument("--attn-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp32",
                   help="Dtype for SA block matmuls (Q/K/V proj, scores, "
                        "softmax, output proj). Independent of residual-dtype. "
                        "If different, output is cast back to residual-dtype "
                        "before the residual ADD.")
    p.add_argument("--ffn-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp32",
                   help="Dtype for FFN block (x4 expansion + projection). "
                        "Same rules as --attn-dtype.")
    p.add_argument("--conv-dtype", choices=["fp32", "fp16", "bf16"],
                   default=None,
                   help="Dtype for the depthwise causal conv. Independent "
                        "of --residual-dtype (same cast-back rules as "
                        "--attn-dtype). Default None = inherit "
                        "--residual-dtype, which is byte-identical to the "
                        "pre-conv_dtype behaviour (conv ran under the "
                        "residual-stream autocast) for every existing run / "
                        "checkpoint, including the historical residual="
                        "fp16/bf16 arms. Set bf16 to run conv in bf16 while "
                        "keeping the residual stream fp32.")
    p.add_argument("--patch-emb-dtype", choices=["fp32", "fp16", "bf16"],
                   default="fp32",
                   help="Dtype for the input pipeline: RevEWMNorm + patch "
                        "encoder (GRU). Default fp32 (safe — RevEWMNorm still "
                        "uses fp64 internal stats; only its output cast and "
                        "GRU compute follow this dtype). Set fp16 for max "
                        "speedup; revert to fp32 if training diverges.")
    p.add_argument("--depthwise-conv", type=int, default=3,
                   help="Kernel size of the per-layer depthwise causal conv "
                        "in the NEW (proper, Conformer-style) placement: "
                        "y = conv(x); x = x_res + sa(norm1(y)). Residual stream "
                        "stays clean. Default 3. Set 0 to disable. "
                        "Mutually exclusive with --deprecated-depthwise-conv.")
    p.add_argument("--deprecated-depthwise-conv", type=int, default=0,
                   help="Kernel size of the LEGACY in-place depthwise conv on "
                        "the residual stream (x = conv(x); x = x + sa(norm1(x))). "
                        "Use only when resuming a checkpoint that was trained "
                        "with this placement (all prior runs in this repo). "
                        "Default 0 = off. Mutually exclusive with --depthwise-conv.")
    p.add_argument("--log-attn-amplitude", action="store_true",
                   help="Diagnostic: every --log-attn-amplitude-every steps, "
                        "record per transformer layer the max-abs of the "
                        "pre-softmax QK^T logits, the SA-block input, the "
                        "SA-block output, and the residual stream, to a "
                        "sidecar CSV <save_dir>/<run_name>_attn_amplitude.csv. "
                        "Default off = strict no-op (zero overhead, training "
                        "math byte-identical). Used to diagnose the fresh-init "
                        "all-fp16 divergence (v11/v11b/v18/v19): does QK^T "
                        "grow past the fp16 65504 ceiling near divergence?")
    p.add_argument("--log-attn-amplitude-every", type=int, default=200,
                   help="Step interval for --log-attn-amplitude sampling. "
                        "Default 200. Only meaningful with "
                        "--log-attn-amplitude.")
    p.add_argument("--qk-norm", action="store_true",
                   help="QK-norm (PaLM/Gemma/ViT-22B): RMSNorm on Q and K per "
                        "head before the attention dot-product, to bound the "
                        "pre-softmax logits independently of q/k projection "
                        "weight magnitude. Default off = byte-identical to the "
                        "nn.MultiheadAttention path. On = an SDPA forward reusing "
                        "the same projection weights + q/k RMSNorm (same fused "
                        "kernel, ~no perf cost). Targets the batch-1024 "
                        "activation-amplitude divergence (#322).")
    p.add_argument("--attn-out-norm", action="store_true",
                   help="Sandwich norm on the ATTENTION OUTPUT only (Gemma2-style "
                        "post-sublayer RMSNorm on sa_out before the residual add). "
                        "Bounds sa_out — #322's residual-runaway driver — regardless "
                        "of V/out_proj magnitude. Attention only (FFN output does not "
                        "grow, measured). Off = byte-identical.")
    p.add_argument("--tau", type=float, default=None,
                   help="Contrastive temperature. None = use the LOSS_SPEC "
                        "default (0.07). Used by 2026-05-02_exp_realonly_4096_smaller_tau_sweep.")
    p.add_argument("--tau-rep", type=float, default=None,
                   help="Separate temperature for the L_rep term of split "
                        "loss shapes (#379 tau_rep=1.0 arms). When unset "
                        "(default) both L_pred and L_rep share --tau — "
                        "byte-for-byte identical to the historical objective. "
                        "When set, L_pred keeps --tau and L_rep (the "
                        "h-anchored family aggregate) uses --tau-rep. "
                        "Only meaningful for loss_shape in "
                        "{cosine_similarity_batch_split_pred_rep, "
                        "cosine_similarity_batch_rep_only}; ignored elsewhere.")
    p.add_argument("--learnable-tau", action="store_true",
                   help="CLIP-style learnable τ (#28). Adds log_inv_tau as "
                        "an nn.Parameter on the model; loss uses τ = "
                        "exp(-log_inv_tau). Init from --tau (default 0.07). "
                        "After each optimizer.step, log_inv_tau is clamped "
                        "to [0, log(100)] so τ ∈ [0.01, 1.0].")
    # Freq embedding
    p.add_argument("--seasonality-emb-dim", type=int, default=0,
                   help="Seasonality embedding dim (0 = disabled).")
    p.add_argument("--freq-emb-dim", type=int, default=3,
                   help="Frequency embedding dimension (0 disables it).")
    # Mixup on (X, freq)
    p.add_argument("--mixup-p", type=float, default=0.0,
                   help="Probability of applying mixup at each step. 0 disables.")
    p.add_argument("--mixup-alpha", type=float, default=0.2,
                   help="Beta(alpha, alpha) parameter for mixup lambda.")
    p.add_argument("--rev-norm-kind", default="ewma",
                   choices=["ewma", "revin", "none"],
                   help="Reversible normalization variant. 'ewma' = "
                        "RevEWMNorm(span=32) (default); 'revin' = standard "
                        "single-instance z-score; 'none' to disable.")
    p.add_argument("--rev-norm-span", type=int, default=32,
                   help="Span parameter for RevEWMNorm (ignored unless "
                        "--rev-norm-kind=ewma). Default 32 matches the "
                        "established baseline; sweep larger spans (64/128/256) "
                        "to retain more periodic amplitude.")
    p.add_argument("--patch-stats", default="none",
                   choices=["none", "diff", "raw"],
                   help="Concatenate per-patch RevEWMNorm summary stats to "
                        "the encoder input. 'none' (default) preserves the "
                        "established arms; 'diff' appends scale-free dmean "
                        "(in stdev units) and dlogstd (log ratio); 'raw' is "
                        "the centered mean + log_std ablation.")
    p.add_argument("--loss-shape",
                   default="cosine_similarity_batch_no_time_neg",
                   choices=["cosine_similarity_batch_no_time_neg",
                            "cosine_similarity_batch",
                            "cosine_similarity_batch_square",
                            "cosine_similarity_batch_add_pos_htft",
                            "cosine_similarity_batch_add_pos_htft_add_f_cross_negs",
                            "cosine_similarity_batch_add_f_cross_negs",
                            "cosine_similarity_batch_add_skip_f_negs",
                            "cosine_similarity_batch_add_neg_htft",
                            "cosine_similarity_batch_full_fh_negs",
                            "cosine_similarity_batch_full_hh_negs",
                            "cosine_similarity_batch_full_ff_negs",
                            "cosine_similarity_batch_full_fh_hh_negs",
                            "cosine_similarity_batch_full_hh_ff_negs",
                            "cosine_similarity_batch_full_fh_hh_ff_negs",
                            "cosine_similarity_batch_full_hh_negs_xbfree",
                            "cosine_similarity_batch_full_hh_negs_xshh",
                            "cosine_similarity_batch_full_hh_negs_xshh_allt",
                            "cosine_similarity_batch_split_pred_rep",
                            "cosine_similarity_batch_rep_only",
                            "cpc_multistep",
                            "cpc_multistep_cpcnegs",
                            "cosine_similarity",
                            "cosine_similarity_old"],
                   help="Contrastive loss formulation. Default 'no_time_neg' "
                        "matches the established arms. 'cosine_similarity_batch' "
                        "is the paper-described loss with cross-time negatives "
                        "(h[b,t-1,c] <-> h[b,t,c] and cross-channel time terms) "
                        "— re-introduced after being dropped during ARMA-era tuning.")
    p.add_argument("--pos-in-denominator", action="store_true",
                   help="Train with the normalized-InfoNCE objective: put the "
                        "positive in BOTH numerator and denominator → loss = "
                        "-log(e^pos / (e^pos + Σ e^neg)) ≥ 0 (vs the default "
                        "negatives-only -log(e^pos / Σ e^neg), unbounded / can "
                        "go negative). Honored by the logsumexp variants "
                        "(cosine_similarity_batch[_no_time_neg/_square/"
                        "_full_fh_negs]); a no-op default keeps every prior "
                        "run's objective unchanged.")
    p.add_argument("--stopgrad-positive-h", action="store_true",
                   help="SimSiam/BYOL-style target stop-grad on the InfoNCE "
                        "positive: sim(sg(h_{t+1}), f_{t+1}) — the encoder "
                        "side of the positive pair is detached everywhere "
                        "that term appears (numerator and, with "
                        "--pos-in-denominator, denominator). Negatives keep "
                        "gradient on h; forward loss value is unchanged. "
                        "Only the xshh_allt loss shape (raises otherwise).")
    p.add_argument("--align-loss-weight", type=float, nargs="?",
                   const=1.0, default=0.0,
                   help="λ for a BYOL/SimSiam alignment term "
                        "λ·(2−2·cos(f_t, sg(h_{t+1}))) added on top of the "
                        "training loss (#309). OPT-IN: omit ⇒ 0.0 (no align "
                        "term, objective unchanged); bare --align-loss-weight "
                        "⇒ 1.0; or pass an explicit λ. A non-saturating "
                        "positive: its per-cosine gradient is a constant −2, "
                        "independent of the negatives, vs the InfoNCE "
                        "positive's −(1−p₊)/τ which fades once the negatives "
                        "separate (p₊→1). Stop-grad on the encoder target. "
                        "Applies to ANY loss_shape.")
    p.add_argument("--align-target", choices=("student", "teacher"),
                   default="student",
                   help="Target of the L_align term (#388, #390). "
                        "'student' (default) is the student's own "
                        "sg(h_{t+1}) — what #382 and #379 trained on. "
                        "'teacher' is the EMA teacher's h_{t+1}, the BYOL "
                        "form L_align was designed for; requires "
                        "--ema-embedding / --ema-encoder. Selects the target "
                        "of the --align-loss-weight term on both of its "
                        "paths: standalone under --no-main-contrastive-loss, "
                        "and as the add-on inside the contrastive loss. "
                        "--align-moco-loss-weight is a separate term with a "
                        "teacher target of its own and is NOT affected.")
    p.add_argument("--align-moco-loss-weight", type=float, nargs="?",
                   const=1.0, default=0.0,
                   help="λ for a MoCo-style InfoNCE alignment (#374 arm 6): "
                        "positive = cos(h_{b,t}, h^T_{b,t}) / τ, denominator = "
                        "LSE over cos(h_{b,t}, h^T_{b',t}) / τ across b'. The "
                        "student encoder is the anchor (gradient flows through "
                        "h); the EMA teacher supplies both the positive and "
                        "the cross-batch keys (gradient blocked by the EMA "
                        "update path). Requires --ema-encoder to be on and "
                        "uses the loop's `tau_tensor` for τ. Applies on top of "
                        "any loss_shape; 0.0 ⇒ off (default).")
    p.add_argument("--subtract-contrastive-floor", action="store_true",
                   help="Re-base the loss by the constant normalized-InfoNCE "
                        "floor log(1+N·e^(−1/τ)) so the logged curve reads ~0 "
                        "at the uniformity floor (#309). Gradient-neutral "
                        "(a constant); needs --pos-in-denominator. N is "
                        "computed from the variant and B/T/C.")
    p.add_argument("--cpc-infonce-weight", type=float, nargs="?",
                   const=1.0, default=0.0,
                   help="λ for the CPC InfoNCE auxiliary term (#344, van den "
                        "Oord et al. 2018, Eq. 4; k=1) ADDED on top of the "
                        "training contrastive loss: predict e_{t+1} from the "
                        "AR context h_t through a new learnable log-bilinear "
                        "W_1, L = −log(e^{e_{t+1}^T W_1 h_t} / Σ_C e^{e_j^T W_1 "
                        "h_t}). NO stop-grad (paper-exact); the unbounded "
                        "bilinear carries the scale (no τ). OPT-IN: omit ⇒ 0.0 "
                        "(no term, W_1 not created, objective unchanged); bare "
                        "--cpc-infonce-weight ⇒ 1.0 (equal weight); or pass an "
                        "explicit λ. Applies to ANY 4-D loss_shape (not the "
                        "cpc_multistep stack). Cross-batch negatives are "
                        "chunked via env CPC_CB_CHUNK (default 256).")
    p.add_argument("--cpc-infonce-negs", choices=["matched", "cross", "all"], default="matched",
                   help="CPC InfoNCE candidate set (#348). 'matched' (default, "
                        "the #344 term): cross-batch at the matched next step "
                        "t+1 + same-sequence other steps. 'cross': van den Oord "
                        "Eq. 4 STRICT — {positive} ∪ every OTHER sequence b'≠b at "
                        "all steps l (marginal/context-independent negatives ⇒ "
                        "Theorem 1 / MI bound holds exactly). 'all': also include "
                        "the same sequence's other steps (literal full batch×time "
                        "grid; correlated negatives ⇒ approximate bound). "
                        "Cross-sequence Gram chunked via env CPC_ALL_CHUNK (8).")
    p.add_argument("--no-main-contrastive-loss", action="store_true",
                   help="Drop the main contrastive loss entirely; train only on "
                        "the auxiliary terms (--cpc-infonce-weight and/or "
                        "--align-loss-weight and/or --sigreg-embedding/-encoding). "
                        "Skips the contrastive_latent_loss call (no xshh_allt Gram "
                        "in the backward), and the align term is then computed "
                        "standalone (same BYOL form, encoder target stop-gradded). "
                        "Use to test whether CPC + a separate forecaster loss beats "
                        "the contrastive objective (#344), or to isolate a single "
                        "auxiliary term (SIGReg / CPC / align) end-to-end (#382). "
                        "The loss_tau_ref diagnostic is still logged as a "
                        "contrastive-reference curve. Requires at least one of "
                        "--cpc-infonce-weight / --align-loss-weight / "
                        "--sigreg-embedding / --sigreg-encoding > 0.")
    p.add_argument("--pred-loss-weight", type=float, default=1.0,
                   help="Scalar weight on L_pred inside the "
                        "cosine_similarity_batch_split_pred_rep shape (#382). "
                        "Default 1.0 = historical objective. Set to 0.0 to "
                        "isolate L_rep (e.g. the 'rep' and 'rep_moco' arms); a "
                        "no-op for every other loss_shape.")
    p.add_argument("--rep-loss-weight", type=float, default=1.0,
                   help="Scalar weight on L_rep, the h-anchored repel term. "
                        "Default 1.0 = historical objective. Set to 0.0 to "
                        "isolate L_pred (e.g. the 'pred' and 'pred_moco' arms). "
                        "Read by --loss-shape "
                        "cosine_similarity_batch_split_pred_rep (#382) and "
                        "cosine_similarity_batch_rep_only (#409), whose whole "
                        "main loss IS L_rep; a no-op for every other "
                        "loss_shape. This is the value at step 0 when "
                        "--rep-loss-weight-end sets a schedule.")
    p.add_argument("--rep-loss-weight-end", type=float, default=None,
                   help="Decay L_rep's weight linearly from --rep-loss-weight "
                        "at step 0 to this value, then hold (#409). Omit "
                        "(default) = the weight is constant, byte-for-byte "
                        "the objective of every run before #409. The ramp "
                        "spans --total-steps unless --rep-loss-weight-ramp-"
                        "steps anchors it. L_rep carries the negatives of "
                        "this objective, so at 0.0 nothing pushes the "
                        "representations apart: watch the `auc` column. "
                        "Refused for a loss_shape that reads no rep weight, "
                        "and refused at 0.0 when the run keeps no other "
                        "gradient-bearing term.")
    p.add_argument("--rep-loss-weight-ramp-steps", type=int, default=None,
                   help="Anchor the --rep-loss-weight-end ramp to a FIXED "
                        "step count instead of --total-steps (#409). The "
                        "weight reaches --rep-loss-weight-end at this step "
                        "and holds there. A ladder resumes each leg with a "
                        "new --total-steps, so without this anchor every leg "
                        "would ramp over its own budget and no two stops "
                        "would sit on one curve. Same contract as "
                        "--ema-tau-ramp-steps.")
    p.add_argument("--train-rollout-depth", type=int, default=0,
                   help="k — train the COMPOSED forecaster, not just one step "
                        "(#373). Every loss term that ties f to h is duplicated "
                        "at depth j = 1..k, copy j tying f^(j)_t to h_{t+1+j}, "
                        "where f^(j) is the forecaster re-applied to its own "
                        "output j more times (the operator the eval rollout "
                        "composes). --train-rollout-reduce says how the k + 1 "
                        "copies combine; 0 (default) is byte-for-byte today's "
                        "loss under either. Terms that carry no f (L_rep, L_rep_moco, "
                        "align_moco, SIGReg) enter the total once at any k. "
                        "Applies to the main contrastive loss, the standalone "
                        "align term and the CPC InfoNCE auxiliary. Changes the "
                        "training objective only — eval rollout is unaffected. "
                        "Not defined for --forecaster-kind cpc/linear_cpc. "
                        "Refused when NO term of the run ties f to h, since "
                        "every depth would then add exactly zero. Adds "
                        "cos_err_d0..dk to the losses CSV. Full reference: "
                        "docs/train_rollout_depth.md.")
    p.add_argument("--train-rollout-reduce", choices=["sum", "mean"],
                   default="sum",
                   help="How the k + 1 copies of every f-bearing term combine "
                        "(#401). 'sum' (default) is #373's objective: the "
                        "f-side then carries k + 1 times its k = 0 weight "
                        "against the terms that carry no f, which enter once "
                        "at any k. 'mean' divides the copies by k + 1, so the "
                        "f-side holds its k = 0 weight at every depth, and "
                        "the depth changes what the model trains on rather "
                        "than how much the f-side outweighs the rest. The "
                        "mean covers the f-bearing copies only. At k = 0 "
                        "there is one copy and the two agree exactly, so "
                        "every published run reproduces under either. "
                        "docs/train_rollout_depth.md.")
    p.add_argument("--ema-embedding", action="store_true",
                   help="BYOL/JEPA EMA-teacher copy of the patch-embedding "
                        "(--encoder-type's input_to_latent). Non-trained; "
                        "updated each step via θ_T ← τ·θ_T + (1−τ)·θ_S. "
                        "Combined with --ema-encoder, forms the teacher "
                        "representation path whose h_{t+1} replaces the "
                        "student's as the main-contrastive POSITIVE. The "
                        "forecaster, the negatives, and the CPC term stay "
                        "on the student. Mutually exclusive with "
                        "--stopgrad-positive-h (the teacher IS the target "
                        "stop-grad). Only the xshh_allt loss shape. (#353)")
    p.add_argument("--ema-encoder", action="store_true",
                   help="BYOL/JEPA EMA-teacher copy of the transformer "
                        "encoder layers (the stack between patch-embed and "
                        "forecaster). Non-trained; updated each step via "
                        "EMA. See --ema-embedding. (#353)")
    p.add_argument("--ema-tau", type=float, default=0.99,
                   help="EMA momentum α for --ema-embedding/--ema-encoder: "
                        "the weight the teacher keeps on its own previous "
                        "value in θ_T ← α·θ_T + (1−α)·θ_S. Higher α = slower "
                        "teacher. Constant unless --ema-tau-end is given. "
                        "Default 0.99 (half-life ln(0.5)/ln(α) ≈ 69 "
                        "steps). (#353)")
    p.add_argument("--ema-tau-end", type=float, default=None,
                   help="End value of a LINEAR α schedule (#388): α goes from "
                        "--ema-tau at step 0 to this value at --total-steps, "
                        "then holds. Omit (default) ⇒ α constant at --ema-tau, "
                        "so runs predating #388 are unchanged. 1.0 freezes the "
                        "teacher at the end of the budget. The live α is "
                        "written to <run>_losses.csv every step.")
    p.add_argument("--ema-tau-ramp-steps", type=int, default=None,
                   help="Anchor the --ema-tau-end ramp to a FIXED step count "
                        "instead of --total-steps (#393). α reaches "
                        "--ema-tau-end at this step and holds there. Runs that "
                        "stop at different budgets then follow one α curve, "
                        "which a budget-relative ramp cannot give them. Omit "
                        "(default) ⇒ the #388 behaviour, ramp over "
                        "--total-steps.")
    p.add_argument("--moco-negatives", action="store_true",
                   help="MoCo-style negatives (#374 arms 3+4): route the "
                        "cross-batch f↔h negatives through the EMA teacher "
                        "(hy_teacher_norm) instead of the student, so the "
                        "positive and the f-anchored cross-batch negatives "
                        "share one slowly-moving space. Requires "
                        "--ema-embedding/--ema-encoder and --loss-shape "
                        "cosine_similarity_batch_split_pred_rep or "
                        "cosine_similarity_batch_full_hh_negs_xshh_allt; "
                        "raises otherwise.")
    p.add_argument("--moco-rep-keys", action="store_true",
                   help="MoCo-style keys on L_rep (#374 arm bimoco / arm 6 "
                        "v2): route the three h-anchored families "
                        "(log_neg_xx, log_neg_hh_all, log_neg_xs_allt) "
                        "through the EMA teacher on the key side — student "
                        "anchor h_{b,t}, teacher keys h^T_{b',l}. Adds a "
                        "same-batch same-time student↔teacher positive; "
                        "L_rep becomes a normalized InfoNCE (positive-in-"
                        "denominator). Requires --ema-embedding/"
                        "--ema-encoder and --loss-shape in "
                        "{cosine_similarity_batch_split_pred_rep, "
                        "cosine_similarity_batch_rep_only}.")
    p.add_argument("--sigreg-embedding", action="store_true",
                   help="LeJEPA spherical SIGReg term on the patch-embedding "
                        "e_t (the GRU patch-embed output, [B,T,C,H] before "
                        "the encoder transformer stack). Pushes the pooled "
                        "marginal toward Unif(S^{K-1}); a principled, "
                        "isotropic anti-collapse term. Stateless (no buffers; "
                        "the M unit-direction projections are resampled every "
                        "forward), so checkpoints / strict-loading are "
                        "byte-for-byte unchanged. Total loss adds "
                        "--sigreg-embedding-weight · L_sigreg_embedding. (#355)")
    p.add_argument("--sigreg-encoding", action="store_true",
                   help="LeJEPA spherical SIGReg term on the encoding h_t "
                        "(the 3L transformer output — the codebase's "
                        "original_latent). Same statistic and stateless "
                        "contract as --sigreg-embedding. Total loss adds "
                        "--sigreg-encoding-weight · L_sigreg_encoding. (#355)")
    p.add_argument("--sigreg-post-normalization", action="store_true",
                   help="When ON, both SIGReg terms are evaluated on the "
                        "POST-F.normalize unit-sphere version of e_t / h_t "
                        "(the LeJEPA-strict, σ²=1/K placement). When OFF "
                        "(default — issue #355's arm), they are evaluated on "
                        "the raw PRE-normalisation vectors — pushes the "
                        "encoder to LAND on the sphere with a uniform "
                        "marginal, leaving the downstream L2-normalize a "
                        "near-identity. (#355)")
    p.add_argument("--sigreg-embedding-weight", type=float, default=0.1,
                   help="λ for L_sigreg_embedding (the e_t term) in the total "
                        "loss. LeJEPA default 0.1. No-op when "
                        "--sigreg-embedding is OFF. Independent from "
                        "--sigreg-encoding-weight so the two sides can be "
                        "tuned separately. (#359)")
    p.add_argument("--sigreg-encoding-weight", type=float, default=0.1,
                   help="λ for L_sigreg_encoding (the h_t term) in the total "
                        "loss. LeJEPA default 0.1. No-op when "
                        "--sigreg-encoding is OFF. Independent from "
                        "--sigreg-embedding-weight so the two sides can be "
                        "tuned separately. (#359)")
    p.add_argument("--sigreg-m", type=int, default=1024,
                   help="Number of random unit-direction projections per "
                        "SIGReg forward call. LeJEPA default 1024. (#355)")
    p.add_argument("--sigreg-t-knots", type=int, default=17,
                   help="Trapezoidal-rule knot count for the Epps–Pulley "
                        "integral. LeJEPA default 17. (#355)")
    p.add_argument("--sigreg-n-chunk", type=int, default=8192,
                   help="Chunk size along the pooled sample axis N for the "
                        "SIGReg cos/sin integrand evaluation. Each chunk's "
                        "body is gradient-checkpointed (recomputed in "
                        "backward), so this is a real memory knob — smaller "
                        "chunk → smaller peak. Default 8192. Lower (e.g. "
                        "2048-4096) when the backbone graph is already "
                        "memory-tight. (#355)")
    p.add_argument("--shard-loss-on-batch", action="store_true",
                   help="DDP only: compute the contrastive loss on each "
                        "rank's LOCAL shard instead of all-gathering latents "
                        "to the global batch. Trades correctness for memory/"
                        "speed — the negative pool shrinks to B/world_size "
                        "(e.g. 2 GPUs → HALF the negatives), so this is NOT "
                        "the same objective as single-GPU. Default OFF: the "
                        "proper gathered loss (global negatives, identical to "
                        "1-GPU @ global B). No effect single-GPU. NOTE: "
                        "loss_tau_ref is also computed shard-local under this "
                        "flag, so that baseline column is NOT comparable to "
                        "gathered/single-GPU runs — don't plot them together.")
    p.add_argument("--synth-kind", default="periodic",
                   choices=["periodic", "composite", "forked-arma"],
                   help="On-the-fly synthesizer. 'periodic' (default) is the "
                        "clean single-primitive generator from synthetic_periodic. "
                        "'composite' is the TimesFM-style stacked recipe "
                        "(trend + ARIMA + 2 free waves + 1 seas-tied wave) "
                        "from synthetic_composite.")
    p.add_argument("--enable-pulse", action="store_true",
                   help="Composite-only: enable the PULSE primitive (sparse "
                        "burst train) as a 4th option alongside sin/sq/saw. "
                        "Targets the spike-deficit identified in phase-1 "
                        "(bizitobs_application, bitbrains).")
    p.add_argument("--seas-heavy", action="store_true",
                   help="Composite-only: swap (2 free waves + 1 seas-tied) → "
                        "(1 free wave + 2 seas-tied). Boosts periodic-signal "
                        "coverage in seas-tied-on rows; targets the "
                        "wave-dilution losses identified in phase-1 (solar/H, "
                        "bizitobs_l2c/H — strongly periodic configs where "
                        "composite hurt EWMA-128).")
    p.add_argument("--more-primitives", action="store_true",
                   help="Composite-only: add TRIANGLE and HALF_SIN waveforms "
                        "to the {sin, square, saw} pool. Targets the "
                        "diversity-vs-quantity insight from phase-2: more "
                        "distinct primitives, not more copies of the same.")
    p.add_argument("--env-gain-max", type=float, default=10.0,
                   help="Composite-only: upper bound of the multiplicative "
                        "exp(λt) envelope total gain (default 10× growth or "
                        "decay across T). Set to 100 to expose covid-style "
                        "100× explosive trends; range becomes (1/max, max) so "
                        "log-symmetric around 1.")
    return p.parse_args()


def random_sign_flip(x):
    B, T, C = x.shape
    signs = torch.where(torch.rand(B, 1, C, device=x.device) < 0.5,
                        torch.ones(1, device=x.device),
                        -torch.ones(1, device=x.device))
    return x * signs


def forward_step(model, x, freq_ids=None, freq_embs=None,
                  seasonality_ids=None, seasonality_embs=None,
                  want_teacher=False, want_embed=False):
    """Apply RevEWMNorm + transformer with optional freq + seasonality
    embeddings + optional patch-stats. Routes through
    ``model.prepare_encoder_input`` so the patching path is identical to
    ``model.forward`` and to the head-trainer's ``extract_*_latents``.

    When ``want_teacher=True`` (only meaningful with --ema-embedding/
    --ema-encoder), also runs the model's teacher representation path on
    the same prepared input and returns ``teacher_o_lat`` in the same
    ``[B, T, C, H]`` layout as ``o_lat`` — the EMA-target encoder output
    the loss substitutes for the student's positive (#353).

    When ``want_embed=True`` (used by the SIGReg term, #355), also returns
    the patch-embedding output ``e_lat`` in ``[B, T, C, H]`` layout — the
    GRU patch-embed output, before the encoder transformer stack. Returned
    as the last tensor of the tuple so the existing ``f_lat, o_lat``
    (optionally ``, teacher_o_lat``) prefix is preserved.
    """
    H = model.H
    if model.rev_norm is not None:
        x = model.rev_norm(x, mode='norm')
    B, T_raw, C = x.shape
    T = T_raw // model.W
    xr = model.prepare_encoder_input(
        x, freq_ids=freq_ids, freq_embs=freq_embs,
        seasonality_ids=seasonality_ids, seasonality_embs=seasonality_embs)
    cpc = getattr(model, 'forecaster_kind', 'transformer') in ('cpc', 'linear_cpc')
    out = model.transformer(xr, return_multi=cpc, return_embed=want_embed)
    if want_embed:
        f_flat, o_flat, e_in = out
    else:
        f_flat, o_flat = out
        e_in = None
    if cpc:
        # f_flat is the [B*C, T, K, H] multi-step stack. Keep K as a 4th axis
        # so the loss sees [B, T, C, K, H]. Diagnostics in main() slice k=1.
        K = f_flat.shape[2]
        f_lat = f_flat.reshape(B, C, T, K, H).permute(0, 2, 1, 3, 4)
    else:
        f_lat = f_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    o_lat = o_flat.reshape(B, C, T, H).permute(0, 2, 1, 3)
    # e_in is captured PRE-permute, so it is already [B, T, C, H].
    e_lat = e_in
    teacher_o_lat = model.teacher_forward(xr) if want_teacher else None
    if want_teacher and want_embed:
        return f_lat, o_lat, teacher_o_lat, e_lat
    if want_teacher:
        return f_lat, o_lat, teacher_o_lat
    if want_embed:
        return f_lat, o_lat, e_lat
    return f_lat, o_lat


def maybe_mixup(x, freq_ids, seasonality_ids, model, args):
    """With prob p, linearly interpolate X and label embeddings between
    randomly-paired batch items.

    Returns ``(x, freq_ids, freq_embs, seasonality_ids, seasonality_embs)``.
    ``*_embs`` are None when no mixup applies (caller passes ids directly to
    the model so it does its own lookup); when mixup applies, ``*_embs``
    are pre-mixed tensors and ids are unused by the model lookup.
    """
    no_freq = model.freq_embedding is None
    no_seas = model.seasonality_embedding is None
    if args.mixup_p <= 0 or (no_freq and no_seas):
        return x, freq_ids, None, seasonality_ids, None
    if torch.rand(()).item() >= args.mixup_p:
        return x, freq_ids, None, seasonality_ids, None

    a = float(torch.distributions.Beta(args.mixup_alpha, args.mixup_alpha).sample())
    B = x.shape[0]
    idx = torch.randperm(B, device=x.device)
    x_mix = a * x + (1 - a) * x[idx]

    if not no_freq:
        emb_a = model.freq_embedding(freq_ids)
        emb_b = model.freq_embedding(freq_ids[idx])
        freq_emb_mix = a * emb_a + (1 - a) * emb_b
    else:
        freq_emb_mix = None

    if not no_seas:
        emb_a = model.seasonality_embedding(seasonality_ids)
        emb_b = model.seasonality_embedding(seasonality_ids[idx])
        seas_emb_mix = a * emb_a + (1 - a) * emb_b
    else:
        seas_emb_mix = None

    return x_mix, freq_ids, freq_emb_mix, seasonality_ids, seas_emb_mix


def parse_extra_save_steps(spec):
    """Parse a comma-separated list of extra checkpoint steps.

    None / empty → empty set. Used by --extra-save-steps to add off-cadence
    snapshots on top of --save-every without changing the base cadence.

    Raises SystemExit if any entry is not a positive integer, or if two
    entries fall in the same 1000-block. The snapshot filename is
    `{run}_{step // 1000}k.pth`, so two extras within the same 1000-block
    (for example 2500 and 2800) would silently overwrite each other. Reject at
    parse time rather than during training when the collision surfaces
    hours in.
    """
    if not spec:
        return frozenset()
    try:
        vals = [int(s.strip()) for s in spec.split(",") if s.strip()]
    except ValueError as e:
        raise SystemExit(
            f"--extra-save-steps: cannot parse {spec!r} as a comma-"
            f"separated list of integers ({e}).")
    if any(v <= 0 for v in vals):
        raise SystemExit(
            f"--extra-save-steps: every entry must be > 0; got {vals!r}.")
    blocks = {}
    for v in vals:
        b = v // 1000
        if b in blocks and blocks[b] != v:
            raise SystemExit(
                f"--extra-save-steps: entries {blocks[b]} and {v} share "
                f"1000-block {b} — snapshot filename `_{b}k.pth` would "
                f"overwrite. Space them into distinct 1000-blocks.")
        blocks[b] = v
    return frozenset(vals)


def should_snapshot(step, save_every, extra_steps):
    """True at step > 0 if step matches --save-every or is in the extras set."""
    if step <= 0:
        return False
    return (step % save_every == 0) or (step in extra_steps)


def save_snapshot(model, optimizer, path, step, best_gap, best_gap_step,
                  best_loss, best_loss_step, ema_loss=None, ema_gap=None,
                  hf_rows_consumed=0, synth_rows_consumed=0):
    # Rank-0 only — concurrent writers to one path corrupt the checkpoint.
    # Centralised here so every call site (NaN/periodic/best/final) is
    # covered. Params are kept in sync across ranks (broadcast at init +
    # averaged grads), so rank 0's state_dict is authoritative.
    if not is_main_process():
        return
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


def main_term_depth_gap(args):
    """Why the main contrastive term takes no rollout depth (#373), or None.

    A depth (`--train-rollout-depth k`) rides on the terms that tie f to h.
    Three settings leave the MAIN term with no such part, so every depth
    copy of it adds exactly zero. Returns the reason, naming the flag or the
    shape at fault. None means the term does take the depths.
    """
    if args.no_main_contrastive_loss:
        return "--no-main-contrastive-loss drops the main term"
    if args.loss_shape == "cosine_similarity_batch_rep_only":
        return (f"--loss-shape {args.loss_shape} is h-anchored end to end "
                "(L_rep only), so its depth copy is exactly zero")
    if (args.loss_shape == "cosine_similarity_batch_split_pred_rep"
            and args.pred_loss_weight == 0):
        return ("--pred-loss-weight 0 zeroes L_pred, the f-bearing half of "
                f"--loss-shape {args.loss_shape}")
    return None


def rollout_depth_has_no_consumer(args):
    """Whether NO term of this run can consume a rollout depth (#373).

    Three terms can: the main contrastive term (see
    :func:`main_term_depth_gap`), L_align, and the CPC auxiliary. SIGReg,
    L_rep and align_moco carry no f and enter once at any k.
    """
    return (main_term_depth_gap(args) is not None
            and args.align_loss_weight <= 0
            and args.cpc_infonce_weight <= 0)


def keeps_gradient_without_rep(args):
    """Whether this run still trains after L_rep's weight reaches 0.0 (#409).

    L_rep is the whole main loss of `cosine_similarity_batch_rep_only` and
    half of `..._split_pred_rep`. Every other term of the objective is an
    add-on the training script attaches: L_pred, L_align, the CPC auxiliary,
    align_moco and the two SIGReg terms. A run that keeps none of them at a
    weight above 0 has nothing left to differentiate at the end of the ramp.
    """
    if (args.loss_shape == "cosine_similarity_batch_split_pred_rep"
            and args.pred_loss_weight > 0):
        return True
    return (args.align_loss_weight > 0
            or args.cpc_infonce_weight > 0
            or args.align_moco_loss_weight > 0
            or (args.sigreg_embedding and args.sigreg_embedding_weight > 0)
            or (args.sigreg_encoding and args.sigreg_encoding_weight > 0))


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
    def __init__(self, path, flush_every=100, tau_ref_column=True,
                 rollout_depth=0):
        self.path = path
        self.flush_every = flush_every
        # rollout_depth (#373): k > 0 adds one `cos_err_dj` column per depth
        # j = 0..k, the per-depth forecast error 1 − cos(f^(j)_t, h_{t+1+j}).
        # A k = 0 run writes no such column — its depth-0 curve is 1 − ff.
        self.rollout_depth = int(rollout_depth)
        # tau_ref_column: when True, the CSV gains a `loss_tau_ref` column
        # (positioned right after `loss`) carrying the τ=0.07 reference loss
        # — same loss recomputed under torch.no_grad() with a fixed canonical
        # τ. Comparable across runs regardless of --tau / --learnable-tau.
        self.tau_ref_column = bool(tau_ref_column)
        self._buffer = []
        # Rank-0 only: non-main ranks must not open/write the shared CSV
        # (truncation race / duplicate rows). log/flush/close become no-ops.
        self._enabled = is_main_process()
        if not self._enabled:
            self._file = None
            self._writer = None
            return
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        header = ["step", "loss"]
        if self.tau_ref_column:
            header.append("loss_tau_ref")
        # gap_ratio = (1 - ff) / (1 - fp): forecast-vs-future gap
        # normalized by past-vs-future gap. ff -> 1 (perfect forecast)
        # and fp -> 0 (decorrelated past) drive this toward 0. Lower is
        # better. Sits next to `gap = ff - fp` so the two are read together.
        header += ["gap", "gap_ratio", "ff", "fp", "tp", "cross_batch",
                   "hf_rows_consumed", "synth_rows_consumed", "mixup_applied"]
        # Per-batch backbone diagnostic metrics (same names as the
        # post-hoc proxy CSV in 2026-05-05_exp_qhead_improvements so the
        # two can merge cleanly on column name).
        header += ["r2_random", "r2_naive", "u_temporal", "u_batch",
                   "auc", "top1", "top3"]
        # cpc_aux (#344): the CPC InfoNCE auxiliary term's value (blank
        # for runs without --cpc-infonce-weight). Trailing column so
        # name-keyed readers of older CSVs are unaffected.
        header += ["cpc_aux"]
        # SIGReg (#355): per-step term values and the e_t mirror of
        # the dim-usage diagnostic (existing u_temporal/u_batch already
        # read h_t). Blank for runs without --sigreg-embedding /
        # --sigreg-encoding. Trailing so older CSV readers ignore them.
        header += ["sigreg_e", "sigreg_h", "u_temporal_e", "u_batch_e"]
        # (#363) Cross-(batch × time) dim-usage on h_t and e_t. Pools
        # (B, T) into one sample axis — same axis SIGReg pools over for
        # its random-projection statistic, so this is the dim-usage
        # measurement that matches what SIGReg regularises.
        header += ["u_batchtime", "u_batchtime_e"]
        # ema_tau (#388): the live EMA momentum α of the step. Constant for
        # runs without --ema-tau-end, blank for runs without a teacher.
        # Trailing so name-keyed readers of older CSVs are unaffected.
        header += ["ema_tau"]
        # (#409) rep_w: the live weight on L_rep. Constant for runs without
        # --rep-loss-weight-end, blank for a loss_shape that reads no rep
        # weight. l_pred / l_rep / l_align: the UNWEIGHTED value of each
        # term the loss computed this step, blank for a term it skipped.
        # Before #409 the CSV carried the total alone, so a report had to
        # read L_rep as the residual of every other term. l_align is the
        # depth-0 copy: the `cos_err_d*` columns carry the other depths,
        # and the column is blank under --no-main-contrastive-loss, where
        # L_align is added outside the main loss.
        header += ["rep_w", "l_pred", "l_rep", "l_align"]
        # Per-depth forecast error (#373), only on --train-rollout-depth runs.
        if self.rollout_depth:
            header += [f"cos_err_d{j}"
                       for j in range(self.rollout_depth + 1)]
        if os.path.getsize(path) == 0:
            self._writer.writerow(header)
            self._file.flush()
        else:
            # Resume schema guard (#356-P5): if the existing CSV's header
            # has fewer columns than the writer expects (for example a pre-SIGReg
            # run resumed with --sigreg-embedding/--sigreg-encoding), the
            # appended rows would silently shift column meaning. Refuse
            # rather than corrupt the file. The check compares the first
            # header row's column count against `header` — we don't try
            # to handle a header that is a strict prefix. The safe action
            # is fail-loudly and let the caller pick a fresh CSV.
            try:
                with open(path, "r", newline="") as f:
                    existing_header = next(csv.reader(f), None)
            except StopIteration:
                existing_header = None
            if existing_header is not None and \
                    len(existing_header) != len(header):
                raise SystemExit(
                    f"CSV resume schema mismatch at {path}: existing "
                    f"header has {len(existing_header)} columns, writer "
                    f"expects {len(header)}. Refusing to append (would "
                    f"corrupt the file). Pick a fresh --run-name or "
                    f"move the old CSV aside.")

    def log(self, step, loss, gap, gap_ratio, ff, fp, tp, cross_batch,
            hf_rows_consumed, synth_rows_consumed, mixup_applied,
            r2_random, r2_naive, u_temporal, u_batch, auc, top1, top3,
            loss_tau_ref=None, cpc_aux=None,
            sigreg_e=None, sigreg_h=None,
            u_temporal_e=None, u_batch_e=None,
            u_batchtime=None, u_batchtime_e=None, ema_tau=None,
            rep_w=None, l_pred=None, l_rep=None, l_align=None,
            cos_err_depths=None):
        if not self._enabled:
            return
        row = [step, loss]
        if self.tau_ref_column:
            # Always write a value when the column is enabled. If the caller
            # hasn't computed it (shouldn't happen with the current trainer)
            # fall back to the unscaled loss to keep schema stable.
            row.append(loss if loss_tau_ref is None else loss_tau_ref)
        row += [gap, gap_ratio, ff, fp, tp, cross_batch,
                hf_rows_consumed, synth_rows_consumed, int(mixup_applied)]
        row += [r2_random, r2_naive, u_temporal, u_batch, auc, top1, top3]
        row.append('' if cpc_aux is None else cpc_aux)
        for extra in (sigreg_e, sigreg_h, u_temporal_e, u_batch_e,
                      u_batchtime, u_batchtime_e, ema_tau,
                      rep_w, l_pred, l_rep, l_align):
            row.append('' if extra is None else extra)
        if self.rollout_depth:
            depths = cos_err_depths or []
            row += [depths[j] if j < len(depths) else ''
                    for j in range(self.rollout_depth + 1)]
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if self._enabled and self._buffer:
            self._writer.writerows(self._buffer)
            self._file.flush()
            self._buffer = []

    def close(self):
        if not self._enabled:
            return
        self.flush()
        self._file.close()


class LatentDriftProbe:
    """Fixed-batch h_t drift probe. Rank-0-only CSV writer.

    Holds a fixed ARMA probe batch in device memory. Each ``probe(step)``
    call runs a no-grad, ``eval()``-mode forward through
    ``extract_encoder_latents`` and computes :func:`src.metrics.drift_pair`
    against (a) the previous probe's ``h`` (``kind="adjacent"``) and
    (b) the initial probe's ``h`` (``kind="vs_initial"``). Both rows are
    written per probe step. The initial probe fires on the first
    ``probe`` call and writes no comparison row.

    When the model carries an EMA teacher (#388), every probe also runs the
    teacher path and writes the same two rows for it, tagged ``teacher_h``
    in the ``latent`` column. Both curves then come from one probe batch at
    one cadence, so they are directly comparable.

    CSV columns are ``COLUMNS`` below. A resume starts the probe over: the
    first ``probe`` call of the resumed leg becomes that leg's initial
    snapshot, so its ``vs_initial`` rows reference the resume step, not step
    0. ``step_ref`` records which, and readers must group on it — see
    ``read_drift`` in the #388 experiment's ``make_plots.py``.

    Cached ``h`` tensors are kept on CPU as fp16 so memory stays trivial
    (~10 MB per snapshot at ``B=64, T_lat=256, H=384``). The metric
    itself runs in fp32 on ``device`` after cast-back.
    """

    COLUMNS = ["step", "latent", "kind", "step_ref", "delta_step",
               "drift_cos", "drift_cos_aligned", "rot_gap", "cka"]

    def __init__(self, csv_path, probe_x, device):
        self.csv_path = csv_path
        self.probe_x = probe_x.to(device)
        self.device = device
        self.initial_h = {}
        self.initial_step = None
        self.prev_h = {}
        self.prev_step = None
        self._enabled = is_main_process()
        if not self._enabled:
            self._file = None
            self._writer = None
            return
        new_file = (not os.path.exists(csv_path)
                    or os.path.getsize(csv_path) == 0)
        if not new_file:
            self.assert_csv_schema(csv_path)
        self._file = open(csv_path, "a", newline="")
        self._writer = csv.writer(self._file)
        if new_file:
            self._writer.writerow(self.COLUMNS)
            self._file.flush()

    @classmethod
    def assert_csv_schema(cls, csv_path):
        """Refuse to append rows written under a different schema.

        ``latent`` sits at column 2 (#388), not at the end, so a drift CSV
        written before #388 would swallow 9-field rows under its 8-field
        header without an error and every column after ``step`` would read
        shifted. Runs are in flight against ``COLUMNS``, so the order is
        fixed. On a mismatch the safe action is to stop and let the caller
        move the old CSV aside.
        """
        with open(csv_path, "r", newline="") as fh:
            existing = next(csv.reader(fh), None)
        if existing is not None and existing != cls.COLUMNS:
            raise SystemExit(
                f"CSV resume schema mismatch at {csv_path}: existing header "
                f"{existing} != writer header {cls.COLUMNS}. Refusing to "
                "append (would corrupt the file). Move the old CSV aside or "
                "pick a fresh --run-name.")

    @torch.no_grad()
    def extract_h(self, model):
        """{latent_name: h} on CPU fp16 — the student always, the EMA
        teacher too when the model has one.

        Teacher presence is read off ``ConfigurableModel.ema_embedding`` /
        ``.ema_encoder``; both are pinned by
        ``tests/test_388_align_teacher_ema_schedule.py`` so a rename cannot
        quietly drop every ``teacher_h`` row.
        """
        was_training = model.training
        model.eval()
        try:
            h, _ = extract_encoder_latents(model, self.probe_x)
            out = {"student_h": h.detach().cpu().to(torch.float16)}
            if getattr(model, "ema_embedding", False) or \
                    getattr(model, "ema_encoder", False):
                t, _ = extract_teacher_encoder_latents(model, self.probe_x)
                out["teacher_h"] = t.detach().cpu().to(torch.float16)
        finally:
            if was_training:
                model.train()
        return out

    def _write(self, step, latent, kind, step_ref, m):
        self._writer.writerow([
            step, latent, kind, step_ref, step - step_ref,
            f"{m['drift_cos'].item():.6f}",
            f"{m['drift_cos_aligned'].item():.6f}",
            f"{m['rot_gap'].item():.6f}",
            f"{m['cka'].item():.6f}",
        ])

    def probe(self, model, step: int):
        if not self._enabled:
            return
        h = self.extract_h(model)
        if not self.initial_h:
            self.initial_h = h
            self.initial_step = step
            self.prev_h = h
            self.prev_step = step
            return
        for latent, snapshot in h.items():
            cur = snapshot.to(self.device).float()
            adj = drift_pair(self.prev_h[latent].to(self.device).float(), cur)
            self._write(step, latent, "adjacent", self.prev_step, adj)
            # vs_initial only when the adjacent-pair isn't already the
            # initial-pair. A second write would be a redundant duplicate row.
            if self.prev_step != self.initial_step:
                ini = drift_pair(
                    self.initial_h[latent].to(self.device).float(), cur)
                self._write(step, latent, "vs_initial", self.initial_step, ini)
        self._file.flush()
        self.prev_h = h
        self.prev_step = step

    def close(self):
        if self._file is not None:
            self._file.close()


class AttnAmplitudeCSV:
    """Sidecar CSV for the attention-amplitude diagnostic.

    One row per (logged step, transformer layer). Columns:
        step, layer_idx, block(enc|fcst),
        qk_logit_maxabs, sa_in_maxabs, sa_out_maxabs,
        resid_post_sa_maxabs, resid_post_ffn_maxabs

    resid_post_sa_maxabs  = residual max-abs right after the SA add.
    resid_post_ffn_maxabs = residual max-abs after the FFN add (end-of-layer
                            value fed to the next layer) — this is where
                            cross-depth residual growth is most visible.

    Append-mode (resume-safe): header only written when the file is empty.
    """

    def __init__(self, path):
        self.path = path
        # Rank-0 only (shared sidecar file); no-op on other ranks.
        self._enabled = is_main_process()
        if not self._enabled:
            self._file = None
            self._writer = None
            return
        self._file = open(path, "a", newline="")
        self._writer = csv.writer(self._file)
        if os.path.getsize(path) == 0:
            self._writer.writerow([
                "step", "layer_idx", "block",
                "qk_logit_maxabs", "sa_in_maxabs", "sa_out_maxabs",
                "resid_post_sa_maxabs", "resid_post_ffn_maxabs"])
            self._file.flush()

    def write_rows(self, step, rows):
        if not self._enabled:
            return
        # rows: (layer_idx, block, qk, sa_in, sa_out, resid_sa, resid_ffn)
        for (layer_idx, block, qk, sa_in, sa_out,
             resid_sa, resid_ffn) in rows:
            self._writer.writerow(
                [step, layer_idx, block, qk, sa_in, sa_out,
                 resid_sa, resid_ffn])
        self._file.flush()

    def close(self):
        if not self._enabled:
            return
        self._file.close()


def main():
    args = parse_args()

    # The run's own command line, first line of its log. A wrapper builds this
    # command line from a cell name and an environment, so the log is the only
    # place a reader (or a check) can see which flags actually reached the
    # trainer. #401 reads the rollout reduction off this line after the first
    # window of a leg: the objective it selects leaves no other trace — the
    # same file names, the same CSV columns and the same log lines under either
    # word. `shlex.quote` keeps a value with a space readable as one argument.
    print("Command line: " + " ".join(shlex.quote(a) for a in sys.argv),
          flush=True)

    # Distributed (opt-in, env-driven): launch with
    #   torchrun --nproc_per_node=N experiments/.../train.py ...
    # WORLD_SIZE<=1 (the default, no torchrun) → (0,1,0,False) and every
    # dist_utils helper is a strict no-op, so the single-GPU path is
    # byte-identical. --batch-size is PER RANK. The loss sees the global
    # W*B batch via gather_latents (2-GPU @ B/2 == 1-GPU @ B).
    rank, world_size, local_rank, distributed = setup_distributed()
    if distributed:
        args.device = f"cuda:{local_rank}"

    device = torch.device(args.device)
    # Identical model init on every rank (also broadcast post-build); data
    # RNG is offset per rank below so each rank streams DIFFERENT samples.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as _np
    _np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.resume:
        args.run_name = safe_run_name(args.save_dir, args.run_name)

    # -- Model -----------------------------------------------------------------
    model_config = dict(MODEL_CONFIG)
    model_config["C"] = args.n_channels
    model_config["H"] = args.d_model
    model_config["nhead"] = args.n_heads
    model_config["num_layers"] = args.num_layers
    model_config["encoder_type"] = args.encoder_type
    model_config["enc_transformer_num_layers"] = args.enc_num_layers
    model_config["enc_transformer_nhead"] = args.enc_nhead
    model_config["enc_transformer_ffn_mult"] = args.enc_ffn_mult
    model_config["enc_transformer_dropout"] = args.enc_dropout
    model_config["enc_transformer_depthwise_conv"] = args.enc_depthwise_conv
    model_config["enc_transformer_chunk_size"] = args.enc_chunk_size
    model_config["enc_transformer_use_grad_checkpoint"] = not args.enc_no_grad_ckpt
    model_config["freq_emb_dim"] = args.freq_emb_dim
    model_config["seasonality_emb_dim"] = args.seasonality_emb_dim
    model_config["rev_norm_kind"] = args.rev_norm_kind
    if args.rev_norm_kind == "ewma":
        model_config["rev_norm_span"] = args.rev_norm_span
    model_config["patch_stats_kind"] = args.patch_stats
    # CLIP-style learnable τ (#28). When --learnable-tau is set, the model
    # registers `log_inv_tau` as an nn.Parameter (init from args.tau or 0.07).
    # The trainer passes `model.tau()` (a 0-d tensor) as `tau_override` to
    # the loss so gradient flows. After each optimizer.step we clamp.
    tau_init = args.tau if args.tau is not None else 0.07
    model_config["learnable_tau"] = bool(args.learnable_tau)
    model_config["tau_init"] = tau_init
    model_config["num_encoder_layers"] = args.num_encoder_layers
    model_config["encoder_dropkey"] = args.encoder_dropkey
    model_config["encoder_dropkey_share_heads"] = args.encoder_dropkey_share_heads
    model_config["encoder_dropkey_share_layers"] = args.encoder_dropkey_share_layers
    model_config["residual_dtype"] = args.residual_dtype
    model_config["attn_dtype"] = args.attn_dtype
    model_config["ffn_dtype"] = args.ffn_dtype
    model_config["conv_dtype"] = args.conv_dtype
    model_config["patch_emb_dtype"] = args.patch_emb_dtype
    # Depthwise-conv placement (mutually exclusive integer kernel sizes).
    # --depthwise-conv N (default 3) → NEW (Conformer-style) placement.
    # --deprecated-depthwise-conv N (default 0) → LEGACY in-place on residual,
    #   for resuming pre-refactor checkpoints. Both 0 = no conv.
    if args.depthwise_conv > 0 and args.deprecated_depthwise_conv > 0:
        raise ValueError(
            "--depthwise-conv and --deprecated-depthwise-conv are mutually "
            "exclusive (got --depthwise-conv {}, --deprecated-depthwise-conv {})"
            .format(args.depthwise_conv, args.deprecated_depthwise_conv))
    model_config["depthwise_conv"] = args.depthwise_conv
    model_config["deprecated_depthwise_conv"] = args.deprecated_depthwise_conv
    # Forecaster bottleneck (#286 follow-up, v13). When None on both the
    # ConfigurableModel side defaults them to (H, nhead) — full no-op for
    # all pre-v13 runs / checkpoints. v13: 128 + 4 (vs encoder H=384 / 6).
    model_config["forecaster_d_model"] = args.forecaster_d_model
    model_config["forecaster_n_heads"] = args.forecaster_n_heads
    # CPC multi-step linear forecaster (#316). Defaults ("transformer", 12)
    # are a no-op for every legacy run/checkpoint.
    model_config["forecaster_kind"] = args.forecaster_kind
    model_config["cpc_k_steps"] = args.cpc_k_steps
    # CPC InfoNCE auxiliary (#344): create the learnable W_1 only when the
    # term is enabled (weight > 0), so disabled runs keep an unchanged
    # state_dict / param count.
    if args.cpc_infonce_weight > 0 and args.forecaster_kind in ("cpc", "linear_cpc"):
        raise SystemExit(
            "--cpc-infonce-weight is the single-step CPC InfoNCE auxiliary (#344) "
            "and operates on a 4-D forecaster latent; it is not defined for the "
            f"cpc_multistep forecaster (--forecaster-kind {args.forecaster_kind}). "
            "Use --forecaster-kind transformer, or drop --cpc-infonce-weight.")
    if args.no_main_contrastive_loss and not (
            args.cpc_infonce_weight > 0 or args.align_loss_weight > 0
            or args.sigreg_embedding or args.sigreg_encoding):
        raise SystemExit(
            "--no-main-contrastive-loss drops the only contrastive term, so the "
            "objective would be empty; pass --cpc-infonce-weight and/or "
            "--align-loss-weight > 0 and/or --sigreg-embedding / "
            "--sigreg-encoding.")
    model_config["cpc_infonce"] = args.cpc_infonce_weight > 0
    model_config["qk_norm"] = bool(args.qk_norm)
    model_config["attn_out_norm"] = bool(args.attn_out_norm)
    model_config["log_attn_amplitude"] = bool(args.log_attn_amplitude)
    # EMA-target teacher (#353). Two independent flags so the embedding-only
    # and encoder-only ablations are reachable. Arm 1 of the issue sets both.
    # Combination with --stopgrad-positive-h is rejected here (the teacher
    # IS the target stop-grad — passing both is intent-conflicting).
    if (args.ema_embedding or args.ema_encoder) and args.stopgrad_positive_h:
        raise SystemExit(
            "--ema-embedding/--ema-encoder are mutually exclusive with "
            "--stopgrad-positive-h: the teacher path replaces the stop-grad. "
            "Drop one.")
    _EMA_LOSS_SHAPES = (
        "cosine_similarity_batch_full_hh_negs_xshh_allt",
        "cosine_similarity_batch_split_pred_rep",
        "cosine_similarity_batch_rep_only",
    )
    # The shape restriction is about the teacher's h_{t+1} REPLACING the
    # student's positive inside contrastive_latent_loss. Under
    # --no-main-contrastive-loss that call never happens, so the teacher only
    # feeds the auxiliary terms (--align-target teacher, #388) and any shape
    # is fine.
    if (args.ema_embedding or args.ema_encoder) and \
            not args.no_main_contrastive_loss and \
            args.loss_shape not in _EMA_LOSS_SHAPES:
        raise SystemExit(
            "--ema-embedding/--ema-encoder only implemented for "
            f"--loss-shape in {_EMA_LOSS_SHAPES}.")
    if (args.ema_embedding or args.ema_encoder) and not (0.0 < args.ema_tau < 1.0):
        raise SystemExit("--ema-tau must be in (0, 1); got "
                         f"{args.ema_tau!r}.")
    # Parse --extra-save-steps at validation time (not deep in the training
    # loop) so a malformed value fails immediately instead of after model +
    # dataloader construction — the parser raises SystemExit on bad input.
    _extra_save_steps = parse_extra_save_steps(args.extra_save_steps)
    # α = 1.0 is a legal END value (a teacher frozen at the end of the
    # budget); it is not a legal start value, since a teacher that never
    # moves at all is a plain frozen init.
    if args.ema_tau_end is not None and not (0.0 < args.ema_tau_end <= 1.0):
        raise SystemExit("--ema-tau-end must be in (0, 1]; got "
                         f"{args.ema_tau_end!r}.")
    if args.ema_tau_end is not None and not (args.ema_embedding
                                             or args.ema_encoder):
        raise SystemExit("--ema-tau-end schedules the EMA teacher's momentum "
                         "but no teacher exists; pass --ema-embedding "
                         "and/or --ema-encoder.")
    # #393: the anchor only means something for a schedule that ramps.
    if args.ema_tau_ramp_steps is not None:
        if args.ema_tau_ramp_steps <= 0:
            raise SystemExit("--ema-tau-ramp-steps must be positive; got "
                             f"{args.ema_tau_ramp_steps!r}.")
        if args.ema_tau_end is None:
            raise SystemExit("--ema-tau-ramp-steps anchors the α ramp but no "
                             "ramp is configured; pass --ema-tau-end.")
    # #409: the L_rep weight schedule. Every refusal below stops a run whose
    # command line states a decay the objective would not carry.
    if args.rep_loss_weight_end is not None:
        if args.rep_loss_weight_end < 0.0:
            raise SystemExit("--rep-loss-weight-end must be ≥ 0; got "
                             f"{args.rep_loss_weight_end!r}.")
        if args.loss_shape not in REP_WEIGHT_SHAPES:
            raise SystemExit(
                "--rep-loss-weight-end decays the weight on L_rep, and "
                f"loss_shape {args.loss_shape!r} reads no rep weight. The "
                f"shapes that do are {REP_WEIGHT_SHAPES}.")
        if args.no_main_contrastive_loss:
            raise SystemExit(
                "--rep-loss-weight-end decays a term of the main "
                "contrastive loss, and --no-main-contrastive-loss drops "
                "that loss whole. The decay would move nothing.")
        if args.rep_loss_weight_end == 0.0 and not keeps_gradient_without_rep(args):
            raise SystemExit(
                "--rep-loss-weight-end 0.0 removes L_rep, and this run keeps "
                "no other term that carries a gradient. The backward pass "
                "would have nothing to differentiate at the end of the ramp. "
                "Add --align-loss-weight, --cpc-infonce-weight, "
                "--align-moco-loss-weight or a SIGReg term with a weight "
                "above 0, or decay to a value above 0.")
    if args.rep_loss_weight_ramp_steps is not None:
        if args.rep_loss_weight_ramp_steps <= 0:
            raise SystemExit("--rep-loss-weight-ramp-steps must be positive; "
                             f"got {args.rep_loss_weight_ramp_steps!r}.")
        if args.rep_loss_weight_end is None:
            raise SystemExit(
                "--rep-loss-weight-ramp-steps anchors the L_rep weight ramp "
                "but no ramp is configured; pass --rep-loss-weight-end.")
    # #388: L_align's teacher target. Both preconditions are hard errors —
    # silently falling back to the student target is the #382 bug this flag
    # exists to fix. The target applies to both L_align paths (#390): the
    # standalone align_loss() under --no-main-contrastive-loss, and the
    # --align-loss-weight add-on inside contrastive_latent_loss.
    if args.align_target == "teacher":
        if not (args.ema_embedding or args.ema_encoder):
            raise SystemExit(
                "--align-target teacher needs an EMA teacher; pass "
                "--ema-embedding and/or --ema-encoder.")
        if args.align_loss_weight <= 0:
            raise SystemExit(
                "--align-target teacher picks the target of L_align, but "
                "this run has no L_align term; pass --align-loss-weight.")
    model_config["ema_embedding"] = bool(args.ema_embedding)
    model_config["ema_encoder"] = bool(args.ema_encoder)
    # LeJEPA SIGReg (#355): the term contributes nothing to the model's
    # state_dict (no buffers. M projections resampled per forward), so
    # there is no model_config to thread through — only the run-level
    # knobs are read from args at the loss-call site below. Reject
    # nonsense combinations here.
    if args.sigreg_embedding_weight < 0:
        raise SystemExit(
            f"--sigreg-embedding-weight must be ≥ 0; got "
            f"{args.sigreg_embedding_weight}.")
    if args.sigreg_encoding_weight < 0:
        raise SystemExit(
            f"--sigreg-encoding-weight must be ≥ 0; got "
            f"{args.sigreg_encoding_weight}.")
    if args.sigreg_m <= 0:
        raise SystemExit(f"--sigreg-m must be > 0; got {args.sigreg_m}.")
    if args.sigreg_t_knots < 3:
        raise SystemExit("--sigreg-t-knots must be ≥ 3 (trapezoidal rule "
                         f"needs at least 3 knots); got {args.sigreg_t_knots}.")
    if args.train_rollout_depth < 0:
        raise SystemExit("--train-rollout-depth must be ≥ 0; got "
                         f"{args.train_rollout_depth}.")
    if args.train_rollout_depth > 0 and args.forecaster_kind != "transformer":
        # The depth applies the forecaster to its OWN output, so it needs one
        # operator whose output lives in its own input space. Only
        # --forecaster-kind transformer is that. The guard rejects every
        # other kind rather than only today's two, so a kind added later
        # cannot pick up the flag by default. The CPC families are named
        # because they are the ones that exist and the reason they cannot
        # compose is instructive.
        why = (" (K parallel heads, f^(k)_t = W_k h_t, each predicting its "
               "own horizon straight from h_t)"
               if args.forecaster_kind in ("cpc", "linear_cpc") else "")
        raise SystemExit(
            "--train-rollout-depth applies the forecaster to its own output, "
            "so it needs the single transformer forecaster. "
            f"--forecaster-kind {args.forecaster_kind}{why} composes no such "
            "operator. Use --forecaster-kind transformer, or drop "
            "--train-rollout-depth (#373).")
    if args.train_rollout_depth > 0 and rollout_depth_has_no_consumer(args):
        # No term of this run ties f to h, so every depth copy adds exactly
        # zero: the run trains at k = 0 and the CSV still writes k + 1
        # plausible cos_err_dj curves, because the diagnostic reads the depth
        # tensors and not the loss. Refuse it here rather than let the
        # diagnostic pass for the objective.
        raise SystemExit(
            f"--train-rollout-depth {args.train_rollout_depth} has no term to "
            "enter: the depths ride on the terms that tie f to h, and this "
            f"run keeps none of them. {main_term_depth_gap(args)}; "
            "--align-loss-weight and --cpc-infonce-weight are both 0; SIGReg, "
            "L_rep and align_moco carry no f. Add --align-loss-weight or "
            "--cpc-infonce-weight, change the main term, or drop "
            "--train-rollout-depth (#373).")
    # Override the loss_shape from CLI (LOSS_SPEC is a module-level default).
    LOSS_SPEC.train_configuration["loss_shape"] = args.loss_shape
    LOSS_SPEC.train_configuration["include_positive_in_denominator"] = args.pos_in_denominator
    LOSS_SPEC.train_configuration["stopgrad_positive_h"] = args.stopgrad_positive_h
    LOSS_SPEC.train_configuration["align_loss_weight"] = args.align_loss_weight
    LOSS_SPEC.train_configuration["align_target"] = args.align_target
    LOSS_SPEC.train_configuration["subtract_contrastive_floor"] = args.subtract_contrastive_floor
    LOSS_SPEC.train_configuration["moco_negatives"] = args.moco_negatives
    LOSS_SPEC.train_configuration["moco_rep_keys"] = args.moco_rep_keys
    LOSS_SPEC.train_configuration["pred_loss_weight"] = args.pred_loss_weight
    LOSS_SPEC.train_configuration["rep_loss_weight"] = args.rep_loss_weight
    LOSS_SPEC.train_configuration["train_rollout_depth"] = args.train_rollout_depth
    LOSS_SPEC.train_configuration["train_rollout_reduce"] = args.train_rollout_reduce
    if args.tau is not None:
        LOSS_SPEC.train_configuration["contrastive_divergence_temperature"] = args.tau
    if args.tau_rep is not None:
        # #379 — separate temperature for the L_rep term of split shapes.
        # When unset the loss code falls back to `tau` (see src/loss.py's
        # split_pred_rep / rep_only branches), preserving historical
        # objectives byte-for-byte.
        LOSS_SPEC.train_configuration["contrastive_divergence_temperature_rep"] = args.tau_rep
    model = ConfigurableModel(**model_config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )

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
                # load_training_state calls torch.load(..., map_location=device)
                # with device=cuda, which moves the saved CPU ByteTensor onto
                # the GPU. torch.set_rng_state requires a CPU ByteTensor, so
                # we must explicitly .cpu() it back before casting / restoring.
                # Reference: May 3 2026 #10-resume incident — silent failure
                # ("RNG state must be a torch.ByteTensor") caused the resumed
                # run's per-batch loss std to jump +52% at step 30k because
                # torch RNG was effectively re-seeded from clock.
                rng = restored["rng_state_torch"].cpu()
                if rng.dtype != torch.uint8:
                    rng = rng.byte()
                torch.set_rng_state(rng)
            if restored.get("rng_state_numpy") is not None:
                _np.random.set_state(restored["rng_state_numpy"])
        except Exception as e:
            print(f"  [checkpoint] WARNING: Could not restore RNG state: {e}")
        print(f"Resumed from {args.resume} at step {start_step}")

    # Every rank must start from byte-identical weights/buffers (fresh
    # init or resumed). No-op when not distributed.
    broadcast_module(model)

    _loss_mode = ""  # only used in the distributed arm of the print below.
    # hoisted so a future edit can't turn the lazy reference into an
    # UnboundLocalError on the single-GPU path.
    if distributed:
        _loss_mode = ("SHARDED loss (local negatives only, B/world_size — "
                      "NOT single-GPU-equivalent)" if args.shard_loss_on_batch
                      else "gathered loss (global negatives, == 1-GPU @ global B)")
    print(f"Device: {device} | Params: {count_parameters(model):,}"
          + (f" | DDP rank {rank}/{world_size} (per-rank bs={args.batch_size}, "
             f"global bs={args.batch_size * world_size}) | {_loss_mode}"
             if distributed else ""))
    print(f"Training for {args.total_steps} steps, bs={args.batch_size}, "
          f"lr={args.lr}, T={args.t_raw}, C={args.n_channels}, "
          f"mix_ratio={args.mix_ratio}, "
          f"freq_emb_dim={args.freq_emb_dim}, "
          f"seasonality_emb_dim={args.seasonality_emb_dim}, "
          f"mixup_p={args.mixup_p}, "
          f"rev_norm_kind={args.rev_norm_kind}"
          + (f"(span={args.rev_norm_span})" if args.rev_norm_kind == 'ewma' else "")
          + f", patch_stats={args.patch_stats}"
          + f", encoder_type={args.encoder_type}")
    print(f"Checkpoints: {args.save_dir}/{args.run_name}_*.pth")

    csv_path = os.path.join(args.save_dir, f"{args.run_name}_losses.csv")
    csv_logger = CSVLogger(csv_path, flush_every=100,
                           rollout_depth=args.train_rollout_depth)
    print(f"Loss CSV: {csv_path}")

    # Latent-drift probe (rank-0 only). Fixed ARMA batch drawn once and
    # kept for the whole run. Cadence defaults to --save-every.
    latent_drift_probe = None
    if args.latent_drift_probe:
        drift_every = (args.latent_drift_probe_every
                       if args.latent_drift_probe_every > 0
                       else args.save_every)
        drift_csv_path = os.path.join(
            args.save_dir, f"{args.run_name}_latent_drift.csv")
        probe_x = generate_random_batch(
            batch_size=args.latent_drift_probe_batch_size,
            T_raw=args.t_raw, C=args.n_channels,
            seed=args.latent_drift_probe_seed, dimension=4)
        latent_drift_probe = LatentDriftProbe(
            drift_csv_path, probe_x, device)
        print(f"Latent-drift CSV: {drift_csv_path} "
              f"(every {drift_every} steps, "
              f"probe_bs={args.latent_drift_probe_batch_size}, "
              f"probe_seed={args.latent_drift_probe_seed})")
        # Initial probe at start_step so the first adjacent-pair row
        # covers step_ref=start_step.
        latent_drift_probe.probe(model, start_step)
    else:
        drift_every = 0

    # Sidecar attention-amplitude diagnostic CSV (opt-in). Only created when
    # --log-attn-amplitude is set. Otherwise None and the per-step hook is a
    # strict no-op (the global ATTN_AMP_DIAG.active stays False forever).
    attn_amp_csv = None
    if args.log_attn_amplitude:
        attn_amp_path = os.path.join(
            args.save_dir, f"{args.run_name}_attn_amplitude.csv")
        attn_amp_csv = AttnAmplitudeCSV(attn_amp_path)
        print(f"Attn-amplitude CSV: {attn_amp_path} "
              f"(every {args.log_attn_amplitude_every} steps)")

    # -- Data -----------------------------------------------------------------
    C = args.n_channels
    if (args.crossfade_ratio > 0 or args.crossfade_triplets > 0) and args.synth_kind != "forked-arma":
        raise ValueError(
            "--crossfade-ratio / --crossfade-triplets require --synth-kind forked-arma")
    synth_bs = int(round(args.batch_size * args.mix_ratio))
    # Crossfade rows are blended FROM the real rows (they consume no extra HF
    # rows), so they shrink hf_bs but not hf_rows_per_step's per-real accounting.
    cross_bs = int(round(args.batch_size * args.crossfade_ratio))
    hf_bs = args.batch_size - synth_bs - cross_bs
    hf_rows_per_step = hf_bs * C
    synth_rows_per_step = synth_bs * C
    if args.resume and restored.get("hf_rows_consumed", 0) > 0:
        hf_rows_consumed = restored["hf_rows_consumed"]
        synth_rows_consumed = restored.get("synth_rows_consumed", 0)
    else:
        hf_rows_consumed = start_step * hf_rows_per_step
        synth_rows_consumed = start_step * synth_rows_per_step
    synth_seed = args.synth_seed if args.synth_seed is not None else args.seed + 10_000
    # Per-rank data offset: each rank MUST stream different samples, else
    # the gathered global batch is just W identical shards and the larger
    # negative pool is fake. Distinct synth seed + HF skip stride per rank
    # (rank 0 offset = 0, so the restored rank-0 counter stays the base).
    if distributed:
        synth_seed += rank * 1_000_003
        hf_rows_consumed += rank * max(1, hf_rows_per_step) * 100_003

    if args.synth_kind == "composite":
        synth_kwargs = {}
        if args.enable_pulse:
            synth_kwargs["enable_pulse"] = True
        if args.seas_heavy:
            synth_kwargs["n_free_waves"] = 1
            synth_kwargs["n_seas_tied_waves"] = 2
        if args.more_primitives:
            synth_kwargs["enable_more_primitives"] = True
        if args.env_gain_max != 10.0:
            synth_kwargs["env_gain_range"] = (1.0 / args.env_gain_max,
                                               args.env_gain_max)
        data_loader = create_mixed_composite_dataloader(
            repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
            mix_ratio=args.mix_ratio,
            path_in_repo=args.hf_path, split=args.split,
            skip_rows=hf_rows_consumed, T_raw=args.t_raw, seed=synth_seed,
            emit_freq_ids=(args.freq_emb_dim > 0 or args.seasonality_emb_dim > 0),
            synth_kwargs=synth_kwargs or None,
        )
    elif args.synth_kind == "forked-arma":
        data_loader = create_mixed_forked_arma_dataloader(
            repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
            mix_ratio=args.mix_ratio, crossfade_ratio=args.crossfade_ratio,
            cross_triplets=args.crossfade_triplets,
            path_in_repo=args.hf_path, split=args.split,
            skip_rows=hf_rows_consumed, T_raw=args.t_raw, seed=synth_seed,
            emit_freq_ids=(args.freq_emb_dim > 0 or args.seasonality_emb_dim > 0),
        )
    else:
        data_loader = create_mixed_periodic_dataloader(
            repo_id=args.hf_repo, batch_size=args.batch_size, C=C,
            mix_ratio=args.mix_ratio,
            path_in_repo=args.hf_path, split=args.split,
            skip_rows=hf_rows_consumed, T_raw=args.t_raw, seed=synth_seed,
            emit_freq_ids=(args.freq_emb_dim > 0 or args.seasonality_emb_dim > 0),
        )
    real_frac = (1 - args.mix_ratio - args.crossfade_ratio) * 100
    trip_rows = 3 * args.crossfade_triplets
    print(f"Data: MIX {real_frac:.0f}% HF + {args.mix_ratio*100:.0f}% synth "
          f"({args.synth_kind}) + {args.crossfade_ratio*100:.0f}% crossfade "
          f"+ {args.crossfade_triplets} triplet(s)={trip_rows} rows, "
          f"hf_bs={hf_bs}, synth_bs={synth_bs}, cross_bs={cross_bs}, "
          f"total_bs={args.batch_size + trip_rows}")
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

        if args.freq_emb_dim > 0 or args.seasonality_emb_dim > 0:
            x, freq_ids, seasonality_ids = batch
            freq_ids = freq_ids.to(device)
            seasonality_ids = seasonality_ids.to(device)
        else:
            x = batch
            freq_ids = None
            seasonality_ids = None

        hf_rows_consumed += hf_rows_per_step
        synth_rows_consumed += synth_rows_per_step
        x = x.to(device)
        x = random_sign_flip(x)

        # Optional mixup (X + freq + seasonality embeddings)
        (x, freq_ids, freq_embs,
         seasonality_ids, seasonality_embs) = maybe_mixup(
            x, freq_ids, seasonality_ids, model, args)
        mixup_applied = (freq_embs is not None or seasonality_embs is not None)
        if mixup_applied:
            mixup_applied_count += 1
        t_data_end = time.perf_counter()

        # Attention-amplitude diagnostic: arm the per-layer hook for THIS
        # forward only, every N steps. When --log-attn-amplitude is off,
        # log_attn_now is always False and set_active is never called with
        # True, so the hook stays a strict no-op (the layers also gate on
        # their own log_attn_amplitude=False). Drained right after the
        # forward (rows are recorded during forward, before backward).
        log_attn_now = (
            args.log_attn_amplitude
            and step % args.log_attn_amplitude_every == 0)
        if log_attn_now:
            ATTN_AMP_DIAG.set_active(True)

        t_fwd_start = time.perf_counter()
        use_ema = args.ema_embedding or args.ema_encoder
        use_sigreg = args.sigreg_embedding or args.sigreg_encoding
        # e_lat is captured under either SIGReg flag: it powers the u_*_e
        # mirror diagnostics (per-rank read) regardless of which term is
        # active. The all-gather below is gated narrowly on the SIGReg-on-e
        # term so a run with only --sigreg-encoding doesn't pay for a
        # gather it never consumes (#356-P4).
        want_embed = use_sigreg
        if use_ema:
            res = forward_step(
                model, x,
                freq_ids=freq_ids, freq_embs=freq_embs,
                seasonality_ids=seasonality_ids,
                seasonality_embs=seasonality_embs,
                want_teacher=True, want_embed=want_embed)
            if want_embed:
                f_lat, o_lat, teacher_o_lat, e_lat = res
            else:
                f_lat, o_lat, teacher_o_lat = res
                e_lat = None
        else:
            res = forward_step(
                model, x,
                freq_ids=freq_ids, freq_embs=freq_embs,
                seasonality_ids=seasonality_ids,
                seasonality_embs=seasonality_embs,
                want_embed=want_embed)
            if want_embed:
                f_lat, o_lat, e_lat = res
            else:
                f_lat, o_lat = res
                e_lat = None
            teacher_o_lat = None
        if log_attn_now:
            ATTN_AMP_DIAG.set_active(False)
            attn_amp_csv.write_rows(step, ATTN_AMP_DIAG.take_rows())
        # Loss + compute_metrics ALWAYS see fp32 latents. When
        # residual-dtype is fp16/bf16 the model returns latents in that
        # dtype. Cast back here so downstream cos-sim arithmetic (loss,
        # loss_tau_ref, compute_metrics, q_random, q_naive_latent,
        # dim_usage, retrieval_auc_topk) runs in fp32. The contrastive
        # logsumexp(sims/τ) with small τ loses precision in bf16 — root
        # cause of v4/v5/v6/v9b divergences.
        f_lat = f_lat.float()
        o_lat = o_lat.float()
        if teacher_o_lat is not None:
            teacher_o_lat = teacher_o_lat.float()
        if e_lat is not None:
            e_lat = e_lat.float()
        # Rollout depth (#373): f^(1)..f^(k), the forecaster re-applied to
        # its own output. Built from the LOCAL f_lat — the operator runs per
        # sequence — then gathered like f_lat below so every depth pools the
        # same global batch. Gradient flows through the chain (no detach).
        #
        # Same numeric path as depth 0, on purpose: the call sits at the same
        # scope as `forward_step` above (this trainer opens no outer autocast
        # — mixed precision is per-layer, `_autocast_ctx(residual_dtype)`
        # inside each decoder layer), it feeds an fp32 sequence the way the
        # encoder boundary feeds the depth-0 pass, and it runs the same
        # `forecaster_forward` under the same fp32-tail policy.
        rollout_lats = rollout_forecaster_latents(
            model, f_lat, args.train_rollout_depth)
        # DDP: gather latents across ranks so the contrastive loss pools
        # negatives over the GLOBAL (W*B) batch — 2-GPU @ B/2 == 1-GPU @ B.
        # Strict no-op single-GPU. Done on the fp32 latents so loss,
        # loss_tau_ref and compute_metrics all see the same global set.
        #
        # --shard-loss-on-batch: SKIP the gather → each rank's loss sees
        # only its local shard (negatives = B/world_size). Cheaper / no
        # O(global_B²) loss memory, but a DIFFERENT, weaker objective —
        # opt-in only. The default keeps the proper gathered loss.
        # average_gradients() below still averages param grads across
        # ranks (standard DDP) so the sharded objective is the mean of
        # the per-rank local losses.
        #
        # The flag picks the GATHER and nothing else. The rollout depths are
        # built above this branch and reach the loss on both paths (#373), so
        # a sharded run trains at the k it was given, on its local shard.
        if not args.shard_loss_on_batch:
            f_lat, o_lat = gather_latents(f_lat, o_lat)
            # Same global pooling for every rollout depth (#373); no-op
            # single-GPU, ONE all-gather per depth under torchrun (a depth is
            # a lone tensor, so it takes the single-tensor form).
            rollout_lats = [gather_latent(f_j) for f_j in rollout_lats]
            if teacher_o_lat is not None:
                # Same global pooling as o_lat. The gather_latents call is a no-op
                # single-GPU. Teacher carries no grad either way.
                _dummy, teacher_o_lat = gather_latents(teacher_o_lat, teacher_o_lat)
            if e_lat is not None and args.sigreg_embedding:
                # SIGReg-on-e pools its statistic over the global batch —
                # same gather contract as the contrastive loss. Skip the
                # gather when only --sigreg-encoding is on (no SIGReg
                # consumer for e_lat); the u_*_e diagnostics are per-rank
                # marginal reads and work fine without a gather (#356-P4).
                _dummy, e_lat = gather_latents(e_lat, e_lat)
        # CPC multi-step (#316): f_lat is [B,T,C,K,H]. The loss / loss_tau_ref
        # consume the full stack. The per-batch diagnostics (compute_metrics,
        # q_*, retrieval_auc) want a single [B,T,C,H] forecaster latent, so
        # use the next-step (k=1) head — the analogue of the legacy 1-step
        # forecaster. For the transformer forecaster f1_lat IS f_lat (4-D),
        # so the diagnostic path stays byte-identical.
        f1_lat = f_lat[:, :, :, 0, :] if f_lat.dim() == 5 else f_lat
        # When learnable_tau is on, pass model.tau() (0-d tensor with grad)
        # as tau_override so gradient reaches log_inv_tau. Otherwise the
        # loss uses LOSS_SPEC.train_configuration's scalar.
        tau_tensor = model.tau() if args.learnable_tau else None
        # #409: L_rep's weight for THIS step. Constant at --rep-loss-weight
        # unless --rep-loss-weight-end sets a decay, which spans
        # --total-steps unless --rep-loss-weight-ramp-steps anchors it to a
        # fixed step. It travels as a function argument, not through
        # LOSS_SPEC, so the run's fixed base weight stays what the
        # `loss_tau_ref` diagnostic below reads. The value used here is the
        # one logged to the losses CSV.
        rep_w_now = linear_schedule_at_step(
            step, args.total_steps, args.rep_loss_weight,
            args.rep_loss_weight_end, args.rep_loss_weight_ramp_steps)
        # Per-term readout for the losses CSV: the loss fills this dict with
        # the UNWEIGHTED L_pred / L_rep / L_align it computes, and leaves the
        # key out for a term it skips.
        loss_terms = {}
        with torch.amp.autocast('cuda', enabled=False):
            tau_tensor_loss = (tau_tensor.float()
                               if tau_tensor is not None else None)
            if args.no_main_contrastive_loss:
                # #344 follow-up arm: drop the main contrastive loss. Train only
                # on the auxiliary terms. Skip contrastive_latent_loss (no
                # xshh_allt Gram backward) and add the BYOL align term standalone
                # (same form, encoder target stop-gradded). L_cpc is added below.
                loss = f_lat.new_zeros(())
                if args.align_loss_weight > 0:
                    # #388: --align-target teacher swaps the target for the
                    # EMA teacher's h_{t+1} (the BYOL form). Argparse rejects
                    # `teacher` without a teacher — re-check it here too,
                    # where the value is used: a None target silently falls
                    # back to the student, which is the #382 bug this flag
                    # fixes. `raise`, not `assert`: `python -O` strips
                    # asserts and would reinstate that exact fallback.
                    if args.align_target == "teacher" and teacher_o_lat is None:
                        raise SystemExit(
                            "--align-target teacher but no teacher latents "
                            "at the loss call. Falling back to the student "
                            "target is the #382 defect this flag removes.")
                    align_target = (teacher_o_lat
                                    if args.align_target == "teacher" else None)
                    loss = loss + align_loss(
                        f_lat, o_lat, args.align_loss_weight,
                        target_latent=align_target,
                        rollout_latents=rollout_lats,
                        depth_reduce=args.train_rollout_reduce)
            else:
                loss = contrastive_latent_loss(
                    (f_lat, o_lat), validation=False,
                    spec=LOSS_SPEC, tau_override=tau_tensor_loss,
                    teacher_original_latent=teacher_o_lat,
                    rollout_latents=rollout_lats,
                    rep_loss_weight=rep_w_now,
                    term_out=loss_terms)
            # #374 arm 6: MoCo-style alignment on the encoder side (student
            # query, teacher key). Requires teacher_o_lat.
            align_moco_val = float('nan')
            if args.align_moco_loss_weight > 0:
                if teacher_o_lat is None:
                    raise SystemExit(
                        "--align-moco-loss-weight > 0 requires an EMA teacher "
                        "(--ema-encoder). teacher_o_lat is None.")
                _tau_am = (float(tau_tensor.detach()) if tau_tensor is not None
                           else args.tau)
                align_moco = align_moco_loss(
                    o_lat, teacher_o_lat, tau=_tau_am,
                    weight=args.align_moco_loss_weight)
                loss = loss + align_moco
                align_moco_val = align_moco.item()
            # CPC InfoNCE auxiliary (#344): total = contrastive + λ·L_cpc,
            # equal weight at λ=1. f_lat is the AR context h_t (4-D here. The
            # cpc_multistep stack is 5-D and returns earlier in the loss), o_lat
            # the encoder embeddings e. Same fp32 block as the contrastive loss.
            cpc_aux_val = float('nan')
            if args.cpc_infonce_weight > 0:
                if args.cpc_infonce_negs == "matched":
                    cpc_aux = cpc_infonce_aux_loss(
                        f_lat, o_lat, model.cpc_w1,
                        rollout_latents=rollout_lats,
                        depth_reduce=args.train_rollout_reduce)
                else:  # "cross" (strict marginal) or "all" (full batch×time grid)
                    cpc_aux = cpc_infonce_all_loss(
                        f_lat, o_lat, model.cpc_w1,
                        marginal_only=(args.cpc_infonce_negs == "cross"),
                        rollout_latents=rollout_lats,
                        depth_reduce=args.train_rollout_reduce)
                loss = loss + args.cpc_infonce_weight * cpc_aux
                cpc_aux_val = cpc_aux.item()
            # LeJEPA SIGReg (#355): regularise the pooled marginal of e_t
            # (patch-embed) and/or h_t (encoding) toward Unif(S^{K-1}). The
            # statistic is stateless (no buffers. M projections resampled
            # every forward); λ is per-term (#359) so the two sides can be
            # tuned independently.
            sigreg_e_val = float('nan')
            sigreg_h_val = float('nan')
            if args.sigreg_embedding:
                sigreg_e = sigreg_loss(
                    e_lat, M=args.sigreg_m, T_knots=args.sigreg_t_knots,
                    post_normalize=args.sigreg_post_normalization,
                    n_chunk=args.sigreg_n_chunk)
                loss = loss + args.sigreg_embedding_weight * sigreg_e
                sigreg_e_val = sigreg_e.item()
            if args.sigreg_encoding:
                sigreg_h = sigreg_loss(
                    o_lat, M=args.sigreg_m, T_knots=args.sigreg_t_knots,
                    post_normalize=args.sigreg_post_normalization,
                    n_chunk=args.sigreg_n_chunk)
                loss = loss + args.sigreg_encoding_weight * sigreg_h
                sigreg_h_val = sigreg_h.item()
        # Diagnostic: same loss with fixed τ=0.07 (no gradient). Comparable
        # across runs regardless of --tau / --learnable-tau, useful as a
        # cross-experiment baseline curve. Re-uses the already-forwarded
        # latents so the only extra cost is one similarity-matrix softmax
        # under no_grad (target <5% step-time overhead).
        # include_positive_in_denominator=True makes this a proper
        # normalized InfoNCE (positive in both numerator and denominator)
        # so the column is always ≥ 0 — unlike the training `loss` above,
        # whose negatives-only objective is intentionally unchanged and
        # goes negative once positives separate. ONLY this diagnostic
        # call passes the flag. The training loss keeps the default.
        with torch.no_grad():
            # `cosine_similarity_batch_split_pred_rep` (#374) is L_pred +
            # L_rep where L_pred is ALREADY normalized-InfoNCE. The shape
            # rejects `include_positive_in_denominator` as a semantic no-op.
            # Its own default at τ=0.07 IS the correct reference.
            _pos_in_denom_ref = (
                args.loss_shape not in (
                    "cosine_similarity_batch_split_pred_rep",
                    "cosine_similarity_batch_rep_only"))
            loss_tau_ref = contrastive_latent_loss(
                (f_lat.detach(), o_lat.detach()),
                validation=False, spec=LOSS_SPEC,
                tau_override=torch.tensor(
                    0.07, device=f_lat.device, dtype=f_lat.dtype),
                include_positive_in_denominator=_pos_in_denom_ref,
                # Keep this a PURE contrastive reference regardless of the
                # run's --align-loss-weight / --subtract-contrastive-floor
                # / --moco-negatives (the diagnostic doesn't have a teacher
                # to route through anyway. Force off to keep it a fixed
                # student-side reference).
                align_loss_weight=0.0,
                subtract_contrastive_floor=False,
                moco_negatives=False,
                moco_rep_keys=False,
                # Depth-0 reference on a --train-rollout-depth run too (#373):
                # one curve comparable across k, and the extra depths are the
                # thing the run varies.
                train_rollout_depth=0,
            )
        loss_tau_ref_val = loss_tau_ref.item()
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
            if attn_amp_csv is not None:
                attn_amp_csv.close()
            sys.stdout.flush()
            sys.exit(1)

        t_bwd_start = time.perf_counter()
        loss.backward()
        # DDP: all_reduce(SUM)/W the param grads (== DDP's averaging). With
        # the W× from DifferentiableAllGather.backward this yields exactly
        # the single-GPU full-batch gradient. No-op single-GPU.
        average_gradients(model)
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        # Clamp learnable τ after the step (CLIP convention). No-op when
        # learnable_tau=False.
        if args.learnable_tau:
            with torch.no_grad():
                model.clamp_log_inv_tau()
        # EMA-teacher update (#353): pull teacher params one fraction of the way
        # toward the just-stepped student, θ_T = α·θ_T + (1−α)·θ_S. α is
        # constant unless --ema-tau-end sets a linear schedule (#388), which
        # spans --total-steps unless --ema-tau-ramp-steps anchors it to a
        # fixed step (#393). The value used here is the one logged to the
        # losses CSV below.
        ema_tau_now = None
        if use_ema:
            ema_tau_now = ema_tau_at_step(
                step, args.total_steps, args.ema_tau, args.ema_tau_end,
                args.ema_tau_ramp_steps)
            model.update_teacher(ema_tau_now)
        t_bwd_end = time.perf_counter()
        t_step_end = time.perf_counter()

        t_data_sum += (t_data_end - t_data_start)
        t_fwd_sum += (t_fwd_end - t_fwd_start)
        t_bwd_sum += (t_bwd_end - t_bwd_start)
        t_step_sum += (t_step_end - t_step_start)
        timing_count += 1

        with torch.no_grad():
            val_ff, val_fp, val_tp, val_cb = compute_metrics(f1_lat, o_lat, CLD)
            # Per-batch backbone diagnostic metrics. Convention matches
            # experiments/2026-05-05_exp_qhead_improvements/scripts/eval_backbone_metrics.py:
            # f_lat is forecaster output, o_lat is encoder ("h"). Same shapes
            # (B, T, C, H) so the slicing here mirrors the eval script.
            f_det = f1_lat.detach()
            o_det = o_lat.detach()
            T_lat = f_det.shape[1]
            q_r = q_random(f_det[:, :T_lat - 1], o_det[:, 1:T_lat]).item()
            q_n = q_naive_latent(
                f_det[:, :T_lat - 1], o_det[:, 1:T_lat],
                o_det[:, :T_lat - 1]).item()
            u_t = dim_usage(o_det, axis=1).item()
            u_b = dim_usage(o_det, axis=0).item()
            # Cross-(batch × time) dim-usage on h_t (#363). Pools (B, T)
            # into one sample axis — SIGReg's own pooling — so the dim-
            # usage panel reports what SIGReg actually regularises.
            u_bt = u_batchtime(o_det).item()
            # Mirror metrics on the patch-embedding e_t (#355). Without
            # --sigreg-embedding / --sigreg-encoding the trainer never
            # captures e_lat, so leave the CSV columns blank (None).
            if e_lat is not None:
                e_det = e_lat.detach()
                u_t_e = dim_usage(e_det, axis=1).item()
                u_b_e = dim_usage(e_det, axis=0).item()
                u_bt_e = u_batchtime(e_det).item()
            else:
                u_t_e = None
                u_b_e = None
                u_bt_e = None
            ret = retrieval_auc_topk(f_det[:, :T_lat - 1], o_det)
            # Per-depth forecast error (#373): does the composed forecaster
            # improve with training, and does depth 0 pay for it? Blank
            # (None) on a k = 0 run, whose depth-0 curve is 1 − ff.
            cos_err_depths = (
                rollout_cos_error(f_det, o_det,
                                  [f_j.detach() for f_j in rollout_lats])
                if args.train_rollout_depth else None)
            r2_random_val = 1.0 - q_r
            r2_naive_val = 1.0 - q_n
            auc_val = ret["auc"].item()
            top1_val = ret["top1"].item()
            top3_val = ret["top3"].item()
        gap_val = val_ff - val_fp
        # (1 - ff) / (1 - fp). Lower is better: ff -> 1 makes the numerator
        # vanish, fp -> 0 pushes the denominator toward 1. Clamp denominator
        # away from 0 so a one-shot val_fp ≈ 1 (degenerate state) doesn't
        # produce inf in the CSV.
        gap_ratio_val = (1.0 - val_ff) / max(1e-6, 1.0 - val_fp)

        if ema_loss is None:
            ema_loss = loss_val; ema_gap = gap_val
        else:
            d = args.ema_decay
            ema_loss = d * ema_loss + (1 - d) * loss_val
            ema_gap  = d * ema_gap  + (1 - d) * gap_val

        csv_logger.log(step, loss_val, gap_val, gap_ratio_val, val_ff, val_fp,
                       val_tp, val_cb, hf_rows_consumed, synth_rows_consumed,
                       mixup_applied,
                       r2_random_val, r2_naive_val, u_t, u_b,
                       auc_val, top1_val, top3_val,
                       loss_tau_ref=loss_tau_ref_val,
                       cpc_aux=(cpc_aux_val if args.cpc_infonce_weight > 0
                                else None),
                       sigreg_e=(sigreg_e_val
                                 if args.sigreg_embedding else None),
                       sigreg_h=(sigreg_h_val
                                 if args.sigreg_encoding else None),
                       u_temporal_e=u_t_e, u_batch_e=u_b_e,
                       u_batchtime=u_bt, u_batchtime_e=u_bt_e,
                       ema_tau=ema_tau_now,
                       rep_w=(rep_w_now
                              if args.loss_shape in REP_WEIGHT_SHAPES
                              and not args.no_main_contrastive_loss
                              else None),
                       l_pred=loss_terms.get('l_pred'),
                       l_rep=loss_terms.get('l_rep'),
                       l_align=loss_terms.get('l_align'),
                       cos_err_depths=cos_err_depths)

        if step % args.log_every == 0 and is_main_process():
            elapsed = time.time() - t0
            sps = (step - start_step) / elapsed
            eta = (args.total_steps - step) / sps / 3600
            tau_str = ""
            if args.learnable_tau:
                tau_str = f"  τ={float(model.tau().detach()):.4f}"
            if args.cpc_infonce_weight > 0:
                tau_str += f"  cpc={cpc_aux_val:.4f}"
            print(f"[{step:>7d}] loss={loss_val:.4f}  ema_loss={ema_loss:.4f}  "
                  f"gap={gap_val:.4f}  ema_gap={ema_gap:.4f}  "
                  f"mixup={mixup_applied_count}/{timing_count}  "
                  f"{sps:.1f} sps  ETA {eta:.1f}h{tau_str}")
            print(f"              R²_rand={r2_random_val:.4f}  "
                  f"R²_naive={r2_naive_val:.4f}  "
                  f"U_t={u_t:.4f}  U_b={u_b:.4f}  "
                  f"AUC={auc_val:.4f}  Top1={top1_val:.4f}")
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

        if should_snapshot(step, args.save_every, _extra_save_steps):
            path = os.path.join(args.save_dir, f"{args.run_name}_{step // 1000}k.pth")
            save_snapshot(model, optimizer, path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step,
                          ema_loss=ema_loss, ema_gap=ema_gap,
                          hf_rows_consumed=hf_rows_consumed,
                          synth_rows_consumed=synth_rows_consumed)

        if args.traj_save_every > 0 and step % args.traj_save_every == 0:
            path = os.path.join(args.save_dir, f"{args.run_name}_step{step}.pth")
            save_snapshot(model, optimizer, path, step,
                          best_gap, best_gap_step, best_loss, best_loss_step,
                          ema_loss=ema_loss, ema_gap=ema_gap,
                          hf_rows_consumed=hf_rows_consumed,
                          synth_rows_consumed=synth_rows_consumed)

        if latent_drift_probe is not None and drift_every > 0 \
                and step % drift_every == 0:
            latent_drift_probe.probe(model, step)

    path = os.path.join(args.save_dir, f"{args.run_name}_final.pth")
    save_snapshot(model, optimizer, path, args.total_steps,
                  best_gap, best_gap_step, best_loss, best_loss_step,
                  ema_loss=ema_loss, ema_gap=ema_gap,
                  hf_rows_consumed=hf_rows_consumed,
                  synth_rows_consumed=synth_rows_consumed)
    if latent_drift_probe is not None:
        # One final probe at total_steps so the CSV covers the run's tail.
        if latent_drift_probe.prev_step != args.total_steps:
            latent_drift_probe.probe(model, args.total_steps)
        latent_drift_probe.close()
    csv_logger.close()
    if attn_amp_csv is not None:
        attn_amp_csv.close()
    total = time.time() - t0
    if is_main_process():
        print(f"\nDone in {total/3600:.1f}h. "
              f"Best gap={best_gap:.4f} at step {best_gap_step}, "
              f"Best loss={best_loss:.4f} at step {best_loss_step}")
    # Barrier + tear down the process group so all ranks exit cleanly
    # (no-op single-GPU).
    cleanup_distributed()


if __name__ == "__main__":
    main()
