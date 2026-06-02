# Head/Rollout Experiment — Execution Plan

## Goal

Compare 6 rollout strategies on GIFT-Eval. V2 MASE is flat at ~1.275 regardless of backbone training (30k→112k). The head/rollout architecture is the bottleneck.

## Existing Assets

- Backbone: `checkpoints/vast_tiny_v2/tiny_v2_best_gap.pth` (76MB, gap 0.428)
- 128-head (real): `checkpoints/head_v2_bbfinal/head_v2_bbfinal_best.pth` (2.4MB, 113k steps)
- **A1 baseline already evaluated: MASE = 1.275**

## Why 3 Heads, Not 1

The head is a **bidirectional GRU** that processes the full latent sequence. B-variants feed mixed sequences (real context + rolled latents). The backward GRU for rolled positions only sees other rolled tokens — a distribution never seen during standard training. So:

| Head | forecast_len | Training data | Serves | Steps |
|------|-------------|---------------|--------|-------|
| W=16 (real) | 16 | Real context latents | A2 | 30k |
| 128 (mixed) | 128 | Real context + rolled latents | B1, B2, B3 | 30k |
| W=16 (mixed) | 16 | Real context + rolled latents | B4 | 30k |

"Mixed" training: roll out N latent tokens during training, concatenate with real context forecaster latents, compute loss on the rolled positions.

## Infrastructure: 3x RTX 4090 datacenter in parallel

RTX 4090 at $0.35/hr. Datacenter only, reliability >95%.

### Machine A (A-variants + B2)
| t | Task | Duration |
|---|------|----------|
| 0h | Setup (deps, data, checkpoints) | 0.5h |
| 0.5h | Train W=16 real head (30k steps) | 1.5h |
| 2h | Eval A2 (value-space, W=16) | 1.5h |
| 3.5h | Copy 128 mixed head from Machine B (ready at t=2h) | ~1min |
| 3.5h | Eval B2 (latent, crop 16) | 3h |
| **6.5h** | **Done — destroy** | |

### Machine B (128-mixed training + B1, B3)
| t | Task | Duration |
|---|------|----------|
| 0h | Setup | 0.5h |
| 0.5h | Train 128 mixed head (30k steps) | 1.5h |
| 2h | Eval B1 (latent, decode end) | 3h |
| 5h | Eval B3 (latent, non-overlapping) | 2h |
| **7h** | **Done — destroy** | |

### Machine C (W=16 mixed + B4)
| t | Task | Duration |
|---|------|----------|
| 0h | Setup | 0.5h |
| 0.5h | Train W=16 mixed head (30k steps) | 1.5h |
| 2h | Eval B4 (latent, W=16) | 3h |
| **5h** | **Done — destroy** | |

### Cost

| | Wall time | Machine-hours | Cost @ $0.35/hr |
|--|-----------|---------------|-----------------|
| 1 machine | 18h | 18h | $6.30 |
| **3 machines** | **7h** | **18.5h** | **$6.50** |

Budget: $8 (with margin). **Vast.ai credit: $11.57 — sufficient.**

## Code Changes Required

1. **Fix B-variant sequence context** (`src/forecasting_head.py`): feed `[context_f, rolled_f]` as one sequence to head
2. **Mixed training script** (`experiments/2026-04-16_head-rollout-comparison/scripts/train_head_variant.py`): `--mixed-rollout` flag
3. **Eval script** (`experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py`): add `--strategy`, `--forecast-len`
4. **Training script** (`experiments/2026-04-13_gift-eval/scripts/train_forecasting_head.py`): add `--forecast-len`

## Periodic Sync

Every 15 min from each machine: rsync checkpoints, results, logs locally. Protects against preemption.

## Deliverables

- 5 new GIFT-Eval scores (A2, B1, B2, B3, B4) + existing A1 (1.275)
- Comparison table (overall + per-domain GM-Relative MASE)
- Winner → longer training in follow-up experiment
