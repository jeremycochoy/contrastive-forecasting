# encoder-forecaster v2 — backbone sweep + fp16/bf16 stability

## Goal

Find the best JEPA-style encoder-forecaster backbone (GRU patch encoder
→ causal encoder → causal forecaster) by full GIFT-Eval, and determine
whether partial-fp16 training is a safe speedup.

**Metric.** GM-Relative MASE = geometric mean over configs of
(model MASE ÷ seasonal-naive MASE). Lower is better; 1.0 = seasonal
naive. "Full" = 97 configs, "triage" = 11-config screen. Held-out
forecasting accuracy on GIFT-Eval; says nothing about contrastive
in-training fit.

## Backbone sweep — verdict

Full GIFT-Eval (97 configs), 2L transformer q-head:

| Backbone | Full GM-MASE | Note                                  |
|----------|-------------:|---------------------------------------|
| **v11c** |    **1.292** | best — enc6/fcst1, dropkey 0.9 shared |
| v16      |        1.335 | enc6/fcst1, dropkey 0.7               |
| v17      |        1.409 | enc6/fcst1, dropkey 0.95              |
| v13      |        1.451 | fcst-bottleneck d=128, 4 heads        |
| v15      |        1.558 | enc6/fcst4, dropkey 0.9               |
| v14      |        1.661 | enc6/fcst6, dropkey 0.9               |

Triage-only reference (11 configs, no full eval run): v7 baseline
1.512, v10 (JEPA) 1.437, v12 (residual-SiLU) 1.514.

**v11c wins decisively.** Best forecaster depth = 1 layer; deeper
forecasters (v15 fcst4, v14 fcst6) monotonically worsen. dropkey
0.9 > 0.7 > 0.95. None beat the GIFT-Eval gate (1.0) or seasonal naive.

**Triage is an optimistic-direction-but-noisy proxy.** Triage(11) ran
~7% pessimistic vs full(97) for v11c/v15/v16, only +4% for v13, and
+22% for v17 — ranking is preserved at the top but the gap to mid-pack
is compressed. Trust full eval for any decision finer than ~10%.

## Reproducibility & robustness

- **v11c is not a lucky-init outlier.** A fresh q-head retrain on the
  same frozen v11c backbone reproduced triage **1.388 exactly**.
  v11c at a 50k backbone snapshot scored **1.365** (better than the
  earlier snapshot) — v11c keeps improving with more backbone training,
  so it is a real effect, not seed luck.
- **2L q-head beats 12L.** A 12L transformer head *hurts* well-trained
  backbones (v11c 1.388 → 1.519) and only *helps* over-constrained ones
  (v17 1.718 → 1.576, v15 1.671 → 1.602). 2L is the right head.

## fp16 / bf16 stability — verdict

Goal: replace the fp32 GRU/RevEWMNorm path with fp16/bf16 for a
~25-30% speedup. Tested on the v11c/v16 recipes.

**Fresh-init partial-fp16 diverges in every tested combination:**
all-fp16-body, attn-fp32-rest-fp16, and residual-fp32-rest-fp16 all
blow up (loss → 10+) within 1k–3k steps. dropkey 0.7 only *delays*
divergence (v19: ~38k vs v18: ~2.8k) — it does not prevent it.

**The only robust speedup is fp32 warmup (~5k steps) → fp16.** v20
(fresh seed, 5k fp32 warmup then fp16 body) is healthy and stable past
41k steps at report time (loss ~2.11). A separate warm-resume
"precision-envelope" sweep — resuming the *trained* v11c_5k checkpoint
under five fp16/bf16 axes — was stable on all five to 15k. So the
fragility is specific to **fresh-init** fp16; once the residual stream
is warmed up, fp16 holds.

**Mechanism.** Instrumentation shows the residual-stream max-abs
amplitude grows unbounded with depth and training (forecaster block:
~80 at step 200 → ~1070 by step 2800, an >8× blowup), while attention
QK logits stay bounded (~30-60). fp16's narrow mantissa cannot
represent the growing residual magnitudes; fp32 warmup lets the
network settle into a lower-amplitude regime before the cast.

Run-by-run divergence steps, the full amplitude tables, the
precision-envelope and at40k/12L apples-to-apples sets, and the git
branch-divergence note are in
[`EXPERIMENT_LOG_2026-05-15_fp16_precision.md`](EXPERIMENT_LOG_2026-05-15_fp16_precision.md).

## What we learned

1. **Best backbone = v11c** (enc6 / fcst1 / dropkey 0.9 shared),
   full GM-MASE **1.292**. Shallow forecaster + high shared dropkey
   wins; it improves with more training and reproduces exactly.
2. **No arm beats seasonal naive** on full GIFT-Eval — the
   encoder-forecaster architecture as configured is not competitive
   yet; this is an architecture ceiling, not undertraining or seed.
3. **Fresh-init fp16 is unsafe; fp32-warmup→fp16 is safe.** Caused by
   unbounded residual-amplitude growth, not attention. Use the v20
   recipe for any future fp16 speedup.
4. **Use a 2L q-head and trust full (not triage) eval** for decisions
   finer than ~10%.
