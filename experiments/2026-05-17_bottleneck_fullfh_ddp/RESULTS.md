# Does the new full-(fₜ,hₗ) negatives loss improve forecasting?

**Verdict.** No improvement demonstrated — and **not isolable**: the new
loss ran bundled with ≥4 other changes vs the baselines, so this run can
neither credit nor clear it. The one robust result is a **stable fp16
training recipe** (1-layer forecaster, residual fp32 / attn-ffn-conv
fp16), where the 2-layer/bf16 variant diverged.

![Continuous 0→150k trajectory (log-log)](plots/trajectory_0_150k_loglog.png)
*150k clean: the contrastive loss and the fixed-τ=0.07 `loss_tau_ref`
diagnostic keep dropping, forecaster amplitudes stay bounded (65→6, no
divergence) — yet held-out GM-MASE stays flat within noise. Training the
contrastive objective harder did not transfer to forecasting.*

**Held-out GM-MASE** (standard 2L causal q-head 30k; three
continuous-optimizer checkpoints; GIFT-Eval, 97 cfg; 1.0 = seasonal
naive, lower better):

| backbone | full GM-MASE |
|---|---|
| 50k / **100k** / 150k | 1.4377 / **1.3936** / 1.4090 |
| v11c (prior best) / seasonal-naive | 1.292 / 1.000 |

≈3% spread, within the prior experiment's ±~7–10% eval noise → no
horizon trend (single seed); none approaches v11c or the 1.0 ceiling.

## Question

Can fₜ↔hₗ negatives — every forecast contrasted against the encoder
latent at *every* future step, not just the next — significantly lower
held-out GM-Relative MASE?

## Stability finding (the solid result)

1-layer forecaster + residual fp32 / attn-ffn-conv fp16 → clean training
through 150k (top-1 → 1.0, amplitudes bounded, no divergence). The
2-layer/bf16 variant diverged at ~step 1.1k. *Caveat: depth and
precision changed together (single run each) — not isolated.*

![Arm 1 — divergence (log-log)](plots/divergence_loglog.png)
*Arm 1: forecaster residual max-abs 65 → 3.0e5 while QKᵀ stays O(10–100)
— the value/residual path overflows, not softmax (encoder QKᵀ also
blows up); matches the prior v25 fresh-init failure.*

## What was tested, and why the loss is not isolated

The new loss was run *bundled* with normalized-InfoNCE
(`--pos-in-denominator`), the v13 forecaster-bottleneck (6-layer d384
encoder → d128/4-head forecaster), dropkey 0.70, AdamW β2 0.95, and
2-GPU DDP — then compared to *prior* backbones (v11c, v16) that differ
in **≥4 of these at once**. So this run evaluates the bundle, not the
loss term; "the gap is architectural / backbone-side" is an explicit
**untested hypothesis**, not a result. Exact recipe:
[`scripts/run_ddp.sh`](scripts/run_ddp.sh).
