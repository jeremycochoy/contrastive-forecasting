# Architecture Search Plan — Contrastive Forecasting

**Duration**: 48 hours (March 18–20, 2026)
**GPU**: Single RTX 4090 (24GB) on elisa, GPU 1 (`CUDA_VISIBLE_DEVICES=1`)
**Branch**: `architecture-search` (based on `experiments`)
**Goal**: Find a better encoder and/or transformer architecture for the contrastive forecasting model

---

## Current Baseline Architecture

| Component | Config | Notes |
|-----------|--------|-------|
| **Encoder** | Linear(32→64)→ReLU→Linear(64→H) + skip Linear(32→H) + LayerNorm | Very simple, intermediate_dim=64 is tiny vs H=1024 |
| **Transformer** | 12 layers, H=1024, nhead=4, ffn_mult=2, GELU, norm_first, causal conv k=3, dropout=0.1 | head_dim=256 (unusually large), FFN=2048 |
| **Channel mixing** | Kronecker product with learnable R (self) and Q (cross) matrices | Unchanged in experiments |
| **Training** | AdamW, lr=1e-4, bs=16, T_raw=4096 (128 patches of W=32) | Loss: cosine_similarity_batch_no_time_neg, τ=0.07 |
| **Baseline metrics** | val FF≈0.65, FP≈0.55, TP≈0.41, CB≈0.06 (at convergence) | |
| **Param count** | ~100M | |

## Reference: TimesFM 200M Architecture

| Parameter | Value |
|-----------|-------|
| num_layers | 20 |
| model_dims | 1280 |
| num_heads | 16 (head_dim=80) |
| intermediate_size | 1280 (FFN mult=1x!) |
| patch_length | 32 (same as ours) |
| norm | RMSNorm |
| positional_embedding | None |
| attention_dropout | 0.0 |

**Key takeaways from TimesFM**: Many heads with small head_dim (80), FFN is same size as model dim (1x), more layers (20), slightly larger hidden (1280), RMSNorm, no positional embeddings.

---

## Experimental Plan

### Phase 1: Quick Encoder Comparison (est. 6–8 hours)

**Setup**: Small transformer (6 layers, H=512, nhead=8, ffn_mult=2) to isolate encoder effects.
**Training**: 50k steps each, bs=16, lr=1e-4. Estimated ~45-60 min each.
**Metric**: val FF, val (FF - FP) gap, val CB.

| ID | Encoder | Description |
|----|---------|-------------|
| E1 | `mlp_baseline` | Current: Linear→ReLU→Linear + skip + LN (intermediate=64) |
| E2 | `mlp_wide` | Same structure but intermediate=256 (wider bottleneck) |
| E3 | `residual_silu` | Residual MLP: Linear(W→H)→SiLU→Linear(H→H) + skip(W→H) + LN (inspired by TimeFM) |
| E4 | `gru_encoder` | GRU(input=1, hidden=128, 2 layers) → final hidden → Linear→H + LN |
| E5 | `conv_encoder` | Conv1d(1→64, k=5)→SiLU→Conv1d(64→128, k=3)→SiLU→flatten→Linear→H + LN |

### Phase 2: Transformer Config Comparison (est. 8–10 hours)

**Setup**: Use best encoder from Phase 1. Small transformer (6 layers, H=512). 50k steps each.

| ID | Config | Description |
|----|--------|-------------|
| T1 | `baseline` | nhead=8, ffn_mult=2, GELU, LayerNorm (same as Phase 1 best encoder) |
| T2 | `more_heads` | nhead=16 (head_dim=32), rest same |
| T3 | `ffn_4x` | ffn_mult=4, rest same |
| T4 | `ffn_1x` | ffn_mult=1 (like TimesFM), rest same |
| T5 | `silu_activation` | SiLU instead of GELU |
| T6 | `no_conv` | Remove depthwise causal conv (depthwise_conv=0) to measure its contribution |
| T7 | `larger_conv` | depthwise_conv=5 instead of 3 |

### Phase 3: Scaling Experiments (est. 12–16 hours)

**Setup**: Use best encoder + best transformer config. 100k steps each.

| ID | Config | Est. time |
|----|--------|-----------|
| S1 | 12L, H=1024, nhead=16 (head_dim=64), bs=16 | ~4h |
| S2 | 16L, H=1024, nhead=16, bs=16 | ~5h |
| S3 | 20L, H=1280, nhead=16 (head_dim=80), bs=8 | ~8h |
| S4 | 12L, H=1280, nhead=16, bs=12 | ~5h |

### Phase 4: Full Training of Best Model (est. 12–20 hours)

Train the best architecture from Phase 3 for 500k–1M steps.
Monitor convergence and early-stop if metrics plateau.

### Phase 5: Parameter Recovery Validation (est. 2–3 hours)

Train DeepGRU recovery head (20k epochs) on the best fully-trained model.
Compare recovery metrics (total error, sign agreement, correlation) vs the original H1024 baseline.

---

## Training Script Design

Create `train_contrastive_v2.py` that extends `train_contrastive.py` with:
- `--encoder-type`: mlp (default), mlp_wide, residual_silu, gru, conv
- `--nhead`: number of attention heads
- `--ffn-mult`: feedforward multiplier
- `--activation`: gelu, silu
- `--depthwise-conv`: kernel size (0 to disable)
- `--intermediate-dim`: encoder intermediate dimension
- Better logging: save metrics to a JSON log file for easy comparison
- Track best val FF and save best checkpoint separately

The new encoder variants will be defined in a new `encoders.py` file.

---

## Metrics Interpretation

- **val FF** (forecast-future cosine sim): HIGHER is better. The model's forecast matches actual future.
- **val FP** (forecast-past cosine sim): LOWER is better. The forecast isn't just copying the past.
- **val TP** (true future-past cosine sim): Reference. How similar consecutive patches naturally are.
- **val CB** (cross-batch cosine sim): LOWER is better. Different ARMA processes have distinct embeddings.
- **FF - FP gap**: HIGHER is better. The forecast is meaningfully closer to the future than to the past.

---

## Decision Framework

At each phase boundary:
1. Compare metrics across experiments
2. Pick the best performer (prioritize val FF, then FF-FP gap, then CB)
3. If two configs are within noise, prefer the simpler/faster one
4. Move to next phase with the best config

---

## Files on Elisa

All training runs save to:
- Model: `arch_search_{experiment_id}.pth` and `arch_search_{experiment_id}_best.pth`
- Logs: `arch_search_{experiment_id}.log`
- Metrics: `arch_search_{experiment_id}_metrics.json`

---

## Key Code References

- `network.py`: SimpleModel, Simple_encoder (the current encoder)
- `blocks.py`: TransformerBlock, DecoderOnlyTransformerLayer, CausalConv, Simple_channel_mixing_module
- `loss.py`: contrastive_latent_loss (uses cosine_similarity_batch_no_time_neg)
- `train_contrastive.py`: Current training pipeline
- `arma.py`: ARMA data generation (dimension=4 for this experiment)
- `train_parameter_recovery.py`: Recovery head training (DeepGRU model definition here)

## Important Notes

- GPU 1 only. Some other processes on GPU 1 use ~4.6GB, leaving ~20GB.
- The existing H1024 2M-step model is at `trained_simple_model_H1024.pth` — do NOT overwrite.
- Do NOT commit training artifacts (.pth, logs). Only commit code changes.
- Use `python3 -u` for unbuffered output when running with nohup.
- Monitor GPU memory on first run of each new architecture to catch OOM early.
