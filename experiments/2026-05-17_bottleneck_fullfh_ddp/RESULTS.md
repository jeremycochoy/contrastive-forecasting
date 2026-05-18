# Bottleneck + full-fh-negs (normalized InfoNCE) + 2-GPU DDP

**Q.** On the v13 forecaster-bottleneck (6-layer encoder @ d384/6h → d128/4h
forecaster), does the new all-(fₜ,hₗ)-negatives loss (every
forecast×future-step pair as a negative) under **normalized
InfoNCE** (`--pos-in-denominator`, loss ≥ 0), dropkey 0.70 (per-step
attention key-dropout), AdamW β2 0.95, 2-GPU DDP (256 global, full
cross-rank negatives) (a) train stably, and (b) beat prior backbones on
held-out GM-Relative MASE (geo-mean of model÷seasonal-naive MASE over
GIFT-Eval; 1.0 = seasonal naive, lower better)?

**A. The 1L/fp16 config trains stably; accuracy is not competitive. Causes
are confounded — see caveats (the stability vs the diverged arm, and the
accuracy gap vs prior backbones, are each not isolated).**

| Arm | forecaster | body precision | outcome | min loss |
|---|---|---|---|---|
| 1 | **2L** | attn/ffn/conv **bf16**, resid fp32 | **diverged ~step 1.1k** | 2.80 then →10 |
| 2 | **1L** | attn/ffn/conv **fp16**, resid fp32 | **stable, full 50k** | **2.174** |

Arm 2 changed **both** depth (2L→1L) **and** precision (bf16→fp16) vs
Arm 1, so which one prevented divergence is **not isolated** (single run
each). The 1L/fp16 combination trained cleanly to 50k (top-1 → 1.0, no
divergence).

![Arm 1 — divergence (log-log)](plots/divergence_loglog.png)
*Arm 1: forecaster residual max-abs explodes 65 → 3.0e5 while **forecaster**
QKᵀ stays O(10–100) (the value/residual path overflows, not the softmax;
encoder QKᵀ also blows up) — the documented fresh-init failure.*

**Held-out eval & training horizon.** Standard 2L causal-transformer
q-head (30k, `e_then_f`, bf16) on the Arm-2 backbone at three
continuous-optimizer checkpoints, official GIFT-Eval:

| backbone | triage (11) | **full GM-MASE (97)** |
|---|---|---|
| 50k | 1.5611 | 1.4377 |
| **100k** | 1.5492 | **1.3936** |
| 150k | 1.5740 | 1.4090 |
| v11c (prior best, no bottleneck) | — | 1.292 |
| seasonal-naive gate | — | 1.000 |

Full GM-MASE across 50k/100k/150k is 1.438 / 1.394 / 1.409 — the
spread (≈3%) is within the prior experiment's stated ±~7–10% eval
noise, so **no clear horizon trend** can be claimed (3 noisy points,
single seed). None approaches v11c (1.292) or seasonal-naive (1.0).

![Continuous 0→150k trajectory (log-log)](plots/trajectory_0_150k_loglog.png)
*Across the full 150k the contrastive loss & `loss_tau_ref` keep
dropping and forecaster amplitudes stay bounded (65→6, no divergence),
while held-out GM-MASE stays within noise. Observation: more contrastive
training of **this** config did not improve held-out accuracy.
**Hypothesis (not tested here):** the gap is backbone-side rather than
horizon — no architecture variable was isolated this session.*

**Takeaway.** Supported: the 1L/fp16 recipe trains stably (amplitudes
bounded, no divergence) and underperforms the prior v11c/v16 backbones
(1.29/1.34) on held-out GM-MASE, with no clear 50k→150k trend.
**Caveat — not isolated:** this config differs from v11c/v16 in ≥4
confounded variables (bottleneck, dropkey 0.70 vs 0.90, the new loss,
precision); "the cause is backbone-side / architectural" is a
**hypothesis**, not a result of this experiment. Exact recipe: the run
script [`scripts/run_ddp.sh`](scripts/run_ddp.sh).
