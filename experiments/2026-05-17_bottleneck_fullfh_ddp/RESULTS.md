# Bottleneck + full-fh-negs (normalized InfoNCE) + 2-GPU DDP — DIVERGED

## Question
Does the v13-style forecaster-bottleneck backbone train stably under the
new all-(f_t,h_l)-negatives loss with **normalized InfoNCE**
(pos in num+denom), dropkey 0.70, 2-layer forecaster, β2=0.95, on 2-GPU
DDP (global batch 256, full cross-rank negatives), with residual fp32 and
attn/ffn/conv in **bf16**?

## Result: diverged at ~step 1.1 k; did not recover

| phase | steps | loss | gap | top1 |
|---|---|---|---|---|
| healthy descent | 1 → ~1104 | 13 → **2.796** (min @1104) | → ~1.0 | → ~1.0 |
| collapse | ~1100 → ~6000 | 2.8 → ~10 (>6 @ step 1413) | → ~0.4 | → ~0.02 |
| collapsed plateau | 6 k → 50 k | 10–11 (flat) | ~0.4 | ~0.02 |

Best loss 2.796 @ step 1104; ran all 50 k but collapsed for ~49 k of them.

**Mechanism — fresh-init bf16-body residual explosion** (the documented
failure mode; amplitude CSV, forecaster layer 1):

| step | qk_logit | sa_out | resid_post_ffn |
|---|---|---|---|
| 200–1200 | 3–19 | 1.5–3 | **63–93 (stable)** |
| 6000 | 115 | 4 832 | **160 642** |
| 12000 | 73 | 9 600 | **201 378** |

Residual max-abs blows up ~2500× once the bf16 attn/ffn intermediates
overflow. fp32 residual held the residual *cast* but not the bf16
attn/ffn products — i.e. this **reproduces the prior v25 finding**
(residual-fp32 alone does not save a bf16 body at fresh init;
`../2026-05-11_exp_encoder_forecaster/RESULTS.md`).

## Protocol
Code: worktree @ `origin/experiments` 6fdfe89 + PR #294 (`--conv-dtype`).
Recipe + exact flags: [`RUN_PLAN.md`](RUN_PLAN.md). torchrun 2-GPU,
128/GPU = 256 global, gathered loss (full cross-rank negatives) — DDP and
loss wiring verified by a 40-step smoke and the live "global bs=256 |
gathered loss" banner. Artifacts in `runs/` (5 k-spaced checkpoints +
optimizer companions through 45 k, `_losses.csv`, `_attn_amplitude.csv`),
logs in `results/`.

## What we learned
1. The well-established **fresh-init partial-bf16 divergence reproduces**
   here even with conv added to the bf16 set and residual kept fp32 —
   consistent with v18/v22–v25. Amplitude signature is textbook.
2. **Confound (bounded as hypothesis):** five variables changed at once
   vs the proven fp32 v16 family (bf16 body, `full_fh_negs`, normalized
   InfoNCE, 2-L forecaster, β2 0.95). The amplitude evidence points at
   the bf16 body as the cause; a pure-fp32 rerun is required to isolate
   it and to evaluate the new loss on its own.

## Operational (outside the science thread)
The first divergence watcher anchored its blow-up baseline in a
post-warmup window that coincided with the collapse, so it false-reported
DONE instead of stopping early (~1.75 h of 2-GPU compute wasted; no data
lost). Fixed in `scripts/watch_divergence.sh` (global-best baseline +
absolute ceiling + exact-step completion); replay over this run's CSV
confirms it would fire at step ~2.4 k.

## Status
Stopped; GPUs freed; other sessions' GPU0 notebooks untouched. PR #294
(`--conv-dtype`, byte-identical default) is valid independent of this
outcome. Next step deferred to user (no rerun launched).
