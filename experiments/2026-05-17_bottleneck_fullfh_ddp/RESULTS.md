# Bottleneck + full-fh-negs (normalized InfoNCE) + 2-GPU DDP

**Q.** On the v13 forecaster-bottleneck (6-layer encoder @ d384/6h → d128/4h
forecaster), does the new all-(fₜ,hₗ)-negatives loss (every
forecast×future-step pair as a negative) under **normalized
InfoNCE** (`--pos-in-denominator`, loss ≥ 0), dropkey 0.70 (per-step
attention key-dropout), AdamW β2 0.95, 2-GPU DDP (256 global, full
cross-rank negatives) (a) train stably, and (b) beat prior backbones on
held-out GM-Relative MASE (geo-mean of model÷seasonal-naive MASE over
GIFT-Eval; 1.0 = seasonal naive, lower better)?

**A. Stability depends on depth+precision; accuracy is not competitive.**

| Arm | forecaster | body precision | outcome | min loss |
|---|---|---|---|---|
| 1 | **2L** | attn/ffn/conv **bf16**, resid fp32 | **diverged ~step 1.1k** | 2.80 then →10 |
| 2 | **1L** | attn/ffn/conv **fp16**, resid fp32 | **stable, full 50k** | **2.174** |

The only change that mattered was depth+precision: 2-layer/bf16 blew up;
**1-layer/fp16 trained cleanly to 50k** (top-1 → 1.0, no divergence).

![Arm 1 — divergence (log-log)](plots/divergence_loglog.png)
*Arm 1: forecaster residual max-abs explodes 65 → 3.0e5 while **forecaster**
QKᵀ stays O(10–100) (the value/residual path overflows, not the softmax;
encoder QKᵀ also blows up) — the documented fresh-init failure.*

![Arm 2 — successful run (log-log)](plots/success_curves_loglog.png)
*Arm 2: (A) loss 13→2.18, `loss_tau_ref` 16→0.21; (B) 1−AUC → ~0 by step
~30 (contrastive task trivially separable); (C) amplitudes **bounded** —
fcst residual post-FFN 65 → 7 (vs Arm 1's 3.0e5): the stability
mechanism; (D) embedding dimension-usage (higher = less collapse) rises
0.01 → 0.20.*

**Held-out eval.** Standard 2L causal-transformer q-head (30k, `e_then_f`,
bf16) on the Arm-2 50k backbone, official GIFT-Eval:

| backbone (full GM-MASE, 97 cfg) | value |
|---|---|
| v11c (best prior, no bottleneck, dk0.9) | 1.292 |
| v16 (no bottleneck, dk0.7) | 1.335 |
| v13 (bottleneck d128, dk0.9, 1L) | 1.451 |
| **this run** (bottleneck d128, dk0.70, full_fh_negs+normInfoNCE, fp16) | **1.4377** |
| seasonal-naive gate | 1.000 |

Triage (11 cfg) 1.5611 → full 1.4377. That is 1.438 vs v13's 1.451
(within the prior experiment's stated ±~10% eval noise — not a real
gain); it does **not** beat the non-bottleneck v16/v11c, and no arm
beats seasonal naive (1.0).

**Takeaway.** The new loss + normalized InfoNCE + dk0.70 is a **stability**
result (1-layer fp16 trains where 2-layer bf16 diverges; amplitudes stay
bounded), **not** a held-out-accuracy gain — the bottleneck still trails
the plain dk0.7/dk0.9 backbones and the seasonal-naive ceiling persists.
Recipe in [`RUN_PLAN.md`](RUN_PLAN.md).

---
*Operational (not science): a spurious orchestrator "FAILED" was a
status-file race; training was clean to 50k. A continuous-optimizer
50k→100k extension is running.*
