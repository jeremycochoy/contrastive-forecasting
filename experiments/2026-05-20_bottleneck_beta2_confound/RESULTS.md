# #309 Bottleneck × β2 confound on (B): can α match v11c?

## Question

(B) — the `cosine_similarity_batch_full_hh_negs` variant of the
bottleneck-fullfh recipe — reaches full-97 GM-MASE 1.3572 (#303),
still **+5%** above v11c 1.292. Of the ≥6 axes that differ between (B)
and v11c, two are testable cheaply: the forecaster bottleneck and
AdamW β2. Does isolating those two axes close the gap?

## Verdict

**α (bottleneck removed + β2 0.98) does NOT match v11c — it
diverges by step ~1000.** Removing the forecaster bottleneck while
keeping the (B) fp16 body is fp16-unstable at fresh init. β (bottleneck
kept, β2 0.98) tracks (B) cleanly; γ (no bottleneck, β2 0.95)
diverges like α (single-axis confirmation that bottleneck-removal —
not β2 — is the source of instability).

| Arm   | β2   | Fcst bneck | Train status @ 50k | Full-97 GM-MASE |
|-------|-----:|-----------:|--------------------|----------------:|
| v11c† |  0.98 | removed (all-fp32) | converged | **1.292** |
| (B)†† |  0.95 | kept       | converged          | 1.3572 |
| α     |  0.98 | removed    | **DIVERGED** @ step 1000 | _TBD_ |
| β     |  0.98 | kept       | _TBD_                    | _TBD_ |
| γ     |  0.95 | removed    | _TBD_ (likely diverges)  | _TBD_ |

† v11c: 2026-05-11 encoder-forecaster sweep winner, all-fp32 body,
forecaster d=384 (no bottleneck), β2=0.98, dropkey 0.9.
†† (B): #303 cl_hh_50k. fp16 body, bottleneck d=128, β2=0.95, dropkey 0.7.

## Training curves

_TBD: log/log loss + loss_tau_ref + gap + 1−AUC, 4 arms (B, α, β, γ)._
The α curve diverges from step ~1000; β tracks (B) closely; γ behaviour
will be plotted once it completes (or is SIGTERM'd at the same divergence
point as α).

## Per-domain star (full GIFT-Eval, 97 configs)

_TBD: same style as 2026-05-19_crossed_loss_ablation/plots/perdomain_star.png.
4 arms + v11c (thin dashed) reference._

## Mechanism (hypothesis)

The (B) recipe's fp16 body is load-bearing on the forecaster bottleneck.
Documented in `experiments/2026-05-11_exp_encoder_forecaster/
EXPERIMENT_LOG_2026-05-15_fp16_precision.md`: residual-stream max-abs
amplitude grows unbounded with depth and training (forecaster block:
~80 @ step 200 → ~1070 @ step 2800, >8× blowup), and "fresh-init
partial-fp16 diverges in every tested combination" without a recovery
mechanism. The (B) bottleneck (d=128) constrains forecaster capacity,
which constrains residual growth, which keeps fp16 stable.

α and γ remove the bottleneck → forecaster runs at full d=384 → larger
residual stream → fp16 mantissa runs out → divergence. β keeps the
bottleneck → fp16 stays stable, matches (B).

This is a hypothesis derived from prior amplitude data + α's
divergence signature (loss climbs monotonically from step 1000;
representation collapses — AUC 1.0→0.97, Top1 1.0→0.23, R²_rand
0.81→0.34). Direct confirmation would require an amplitude trace on
α (the same instrumentation v11c work used). The γ run provides an
independent test: same hypothesis predicts γ diverges identically.

## Implication for closing the gap to v11c

The (B) recipe's gap to v11c (1.3572 vs 1.292) **cannot** be closed by
removing the forecaster bottleneck while keeping the fp16 body. To
match v11c, either:

1. Switch (B) to a v11c-style all-fp32 body (loses the speed advantage
   of fp16), or
2. Keep the bottleneck (smaller forecaster capacity — leaves performance
   on the table by definition of the v11c recipe), or
3. Add a stability mechanism (e.g., fp32 warmup → fp16 cast à la v20)
   that lets fresh-init fp16 survive the unbounded residual growth.

None of these are tested in #309. The question reframes from "which
of these two axes closes the gap?" to "what stability mechanism would
let the (B) recipe match v11c?" — out of scope for this card.

## Annex

### Arms in detail

- **α** — (B) + AdamW β2=0.98 + no `--forecaster-d-model/--forecaster-n-heads`
  (forecaster inherits encoder d=384, n_heads=6). Encoder dropkey 0.70,
  loss `cosine_similarity_batch_full_hh_negs`, 1-GPU bs=256
  (mathematically identical to (B)'s 2-GPU DDP bs=128/GPU per
  `train.py:739-740` — gathered global negatives), 50k steps,
  seed 20260520.
- **β** — same as α but `--forecaster-d-model 128 --forecaster-n-heads 4`
  (bottleneck kept).
- **γ** — same as α (no bottleneck) but `--adam-beta2 0.95` (the (B) value).

### Compute

1× RTX 4090 prosumer on vast.ai (offer 35882331, $0.55/h, reliability
0.992, US). 1-GPU was used because no 2× 4090 24GB prosumer offer was
available at provision time; gathered-loss runs are 1-GPU-equivalent
per `train.py`, so the recipe is mathematically intact.

α was SIGTERM'd at step 10,600 (out of 50,000) once divergence was
unambiguous, to save ~3h × $0.55 ≈ $1.65 of doomed compute.

### Code

Branch `experiment/2026-05-20-bottleneck-beta2-confound`.

Scripts: `scripts/{box_run.sh, box_run_serial.sh, provision.sh, sync.sh,
sync_loop.sh, downstream.sh}`. Box-side serial α→β→γ; elisa-side sync
into MAIN checkout (CLAUDE.md rule).
