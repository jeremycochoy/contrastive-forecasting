# Backbone Training Summary Table

Quick reference: every contrastive backbone training run with architecture, hyperparameters, duration, and results.

## Backbone Training Runs

| Run | L | H | nhead | ffn | LR | BS | Steps | Params | Duration | step/s | Peak gap |
|-----|---|---|-------|-----|-----|-----|-------|--------|----------|--------|----------|
| **Phase 1: Encoder search** (6L H=512, 50k steps) | | | | | | | | | | | |
| E1 mlp | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.2M | 30.5 min | 27.3 | 0.073 |
| E2 mlp_wide | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.3M | 29.8 min | 27.9 | 0.075 |
| E3 residual_silu | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.7M | 29.9 min | 27.9 | 0.084 |
| **E4 gru** | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.7M | 61.6 min | 13.5 | **0.115** |
| E5 conv | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.2M | 33.1 min | 25.2 | 0.075 |
| **Phase 2: Transformer config** (6L H=512 gru, 50k steps) | | | | | | | | | | | |
| T1 baseline | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.7M | 61.6 min | 13.5 | 0.119 |
| T2 nhead=16 | 6 | 512 | 16 | 2x | 1e-4 | 16 | 50k | 13.7M | 62.7 min | 13.3 | 0.110 |
| **T3 ffn=4x** | 6 | 512 | 8 | **4x** | 1e-4 | 16 | 50k | 20.0M | 69.2 min | 12.0 | **0.125** |
| T4 ffn=1x | 6 | 512 | 8 | 1x | 1e-4 | 16 | 50k | 10.5M | 57.8 min | 14.4 | 0.115 |
| T5 silu | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.7M | 61.6 min | 13.5 | 0.106 |
| T6 no_conv | 6 | 512 | 8 | 2x | 1e-4 | 16 | 50k | 13.7M | 64.3 min | 13.0 | 0.104 |
| T7 nhead=16+silu | 6 | 512 | 16 | 2x | 1e-4 | 16 | 50k | 13.7M | 62.6 min | 13.3 | 0.118 |
| **Phase 4: Full training of best config** | | | | | | | | | | | |
| phase4_gru_ffn4x | 12 | 1024 | 8 | 4x | 7e-5 | 8 | 500k | 153.8M | 1211 min (20.2 h) | 6.9 | **0.186** |
| **12L extension to ~2M** (resume segments) | | | | | | | | | | | |
| v2_2M_resumed (seg B) | 12 | 1024 | 8 | 4x | 7e-5 | 8 | 1.97M | 153.8M | ~4750 min (79 h) | 6.9 | 0.2015 |
| v2_2M_final (seg C) | 12 | 1024 | 8 | 4x | 7e-5 | 8 | 50k | 153.8M | 121 min (2.0 h) | 6.9 | **0.2028** |
| **Scaling search: 200k comparison** | | | | | | | | | | | |
| scaling_12L | 12 | 1024 | 8 | 4x | 7e-5 | 8 | 200k | 153.8M | 482 min (8.0 h) | 6.9 | 0.162 |
| **scaling_16L** | 16 | 1024 | 8 | 4x | 7e-5 | 8 | 200k | 204.2M | 621 min (10.4 h) | 5.4 | **0.166** |
| scaling_20L | 20 | 1024 | 8 | 4x | 7e-5 | 8 | 200k | 254.6M | 749 min (12.5 h) | 4.5 | 0.154 |
| scaling_12L_H1280 | 12 | 1280 | 10 | 4x | 7e-5 | 8 | 3 (aborted) | 240.0M | <1 min | — | — |
| **20L full training attempts** | | | | | | | | | | | |
| 20L from-scratch (FAILED, collapse) | 20 | 1024 | 8 | 4x | 7e-5 | 8 | 127k | 254.6M | ~470 min | 4.4 | 0.00 |
| 20L resumed (lost ckpt) | 20 | 1024 | 8 | 4x | 7e-5 | 8 | 2M | 254.6M | **7555 min (125.9 h)** | 4.4 | **0.2019** |
| **20L retraining (ongoing)** | 20 | 1024 | 8 | 4x | **5.4e-5** | 8 | 2M target | 254.6M | ~126 h (ongoing) | 4.4 | 0.2019+ (climbing) |

## Key architectural constants

All backbone runs share:
- **Encoder** (since Phase 1 winner): bidirectional GRU, 2-layer, hidden=128
- **Activation**: GELU
- **Depthwise causal conv**: kernel_size=3
- **Dropout**: 0.1
- **Temperature**: 0.07
- **ARMA dimension**: 4 (p,q ∈ {1..4} per channel)
- **T_raw**: 4096 (→ 128 patches of W=32)
- **Channels**: 4

## Phase 1/2 summary by variable

### Effect of encoder type (Phase 1, 6L H=512, 50k steps)
| Encoder | Peak gap | Δ vs MLP |
|---------|----------|----------|
| MLP | 0.073 | — |
| MLP wide | 0.075 | +2.7% |
| Conv | 0.075 | +2.7% |
| Residual SiLU | 0.084 | +15% |
| **GRU** | **0.115** | **+58%** |

### Effect of FFN multiplier (Phase 2, 6L H=512 GRU, 50k steps)
| FFN | Peak gap | Params |
|-----|----------|--------|
| 1x | 0.115 | 10.5M |
| **2x** | **0.119** | 13.7M |
| **4x** | **0.125** | 20.0M |

### Effect of depth (200k steps, H=1024, lr=7e-5)
| Layers | Params | Peak gap @ 200k | step/s | Speed cost |
|--------|--------|-----------------|--------|------------|
| 12L | 153.8M | 0.162 | 6.9 | baseline |
| 16L | 204.2M | **0.166** | 5.4 | 1.28x slower |
| 20L | 254.6M | 0.154 (slow warmup) | 4.5 | 1.53x slower |

### Effect of training length (12L H=1024)
| Steps | Peak gap |
|-------|----------|
| 50k (Phase 2 T3) | 0.125 |
| 500k (Phase 4) | 0.186 |
| ~2M (v2_2M full) | 0.203 |

### Effect of training length (20L H=1024)
| Steps | Peak gap | LR | Notes |
|-------|----------|----|-------|
| 200k | 0.154 | 7e-5 | Scaling search |
| 2M (lost run) | 0.202 | 7e-5 | Checkpoint lost |
| 2.3M (final) | **0.203** | 5.4e-5 | Resumed from 200k, stopped |

### Effect of depth on recovery (GRU h128 l2 head, MSE, 20k epochs, 4x2 coefs)

| Backbone | Peak gap | Recovery | Sign AR | Sign MA | Corr AR | Corr MA |
|----------|----------|----------|---------|---------|---------|---------|
| **12L H=1024** | **0.203** | **6.96x** | **92.4%** | **90.9%** | **0.934** | **0.931** |
| 20L H=1024 | 0.203 | 6.77x | 92.0% | 90.8% | 0.929 | 0.929 |
| 12L H=1024 (V1, MLP enc) | 0.105 | 6.59x | — | — | — | — |

Key finding: at matched gap (0.203), 12L recovers better than 20L (6.96x vs 6.77x). Higher gap correlates with better recovery across architectures (V1 0.105→6.59x, V2 0.203→6.96x), but adding depth alone at same gap does not help.

## Training time cost ladder (wall clock on RTX 4090)

| Config | Steps | Duration | Notes |
|--------|-------|----------|-------|
| 6L H=512 (any encoder) | 50k | ~30-70 min | Phase 1/2 ablation |
| 12L H=1024 | 200k | 8.0 h | Quick comparison |
| 12L H=1024 | 500k | 20.2 h | Phase 4 full |
| 12L H=1024 | 2M | ~101 h (4.2 d) | v2_2M total |
| 16L H=1024 | 200k | 10.4 h | +30% over 12L |
| 20L H=1024 | 200k | 12.5 h | +56% over 12L |
| 20L H=1024 | 2M | ~126 h (5.2 d) | Lost + retraining |

## Notes for LR tuning

- Phase 1/2 used **lr=1e-4** with bs=16 (H=512, 6L)
- Phase 4+ used **lr=7e-5** with bs=8 (H=1024, 12L)
- For scaling to 20L: depth scaling heuristic suggests **lr ~= 7e-5 × sqrt(12/20) = 5.4e-5**
- Standard PyTorch init (Kaiming/Xavier), NOT muP — so width scaling would require its own LR sweep
- Deeper models showed training collapse at 7e-5 when trained from scratch (20L attempt 1) — resume-from-smaller-checkpoint was required

## Lost/failed runs log

| Run | Cause | Lesson |
|-----|-------|--------|
| scaling_12L_H1280 | Killed intentionally (kept H=1024) | — |
| 20L from-scratch @ 7e-5 | Gap collapse, stayed at 0 for 127k steps | Resume from pre-trained checkpoint for deeper models |
| 20L 2M (lost ckpt) | Save path overwritten by continuation | `safe_save_path()` added, never reuse save-path |
| 20L lowlr resume | Wrong `_best.pth` was from early step (high FF, zero gap) | best-ckpt selection uses raw FF, not gap — be aware |
