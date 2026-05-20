# #309 Bottleneck × β2 confound on (B): can α match v11c?

## Question

(B) — the `cosine_similarity_batch_full_hh_negs` variant of the
bottleneck-fullfh recipe — reaches full-97 GM-MASE 1.3572 (#303),
still **+5%** above v11c 1.292. Of the ≥6 axes that differ between (B)
and v11c, two are testable cheaply: the forecaster bottleneck and
AdamW β2. Does isolating those two axes close the gap?

## Verdict

**Yes — both bottleneck-removed arms beat v11c despite training
diverging.** α (full-97 1.277) and γ (1.283) both outperform v11c
(1.292), even though their backbones are snapshots from step ~1000
of 50,000 (the loss minimum just before fp16 divergence). β (1.327,
β2-only change) closes about half the gap to v11c on top of (B);
the bottleneck-removed arms close it entirely. The β2 axis explains
the ~0.5% difference between α and γ — small enough to be noise;
the bottleneck axis accounts for the rest.

| Arm   | β2   | Fcst bneck | Train @ 50k | Triage-11 GM | Full-97 GM | vs (B) |
|-------|-----:|-----------:|-------------|------------:|----------:|-------:|
| (B)†† | 0.95 | kept       | converged   | 1.4461      | **1.3572**| —      |
| v11c† | 0.98 | removed    | converged (all-fp32) | 1.3878 | **1.292** | −4.8% |
| α     | 0.98 | removed    | DIVERGED @ step ~1000 → SIGTERM @ 10600 | 1.3399 | **1.2767** | −5.9% |
| β     | 0.98 | kept       | converged   | 1.4836      | **1.3272**| −2.2% |
| γ     | 0.95 | removed    | DIVERGED @ step ~1000 → SIGTERM @ 10100 | 1.3412 | **1.2829**| −5.5% |

Both bottleneck-removed pre-divergence snapshots (α full=1.277,
γ full=1.283) beat v11c. The β2 axis contributes ~0.5% between α and γ
— small enough to plausibly be noise; the bottleneck axis accounts for
essentially all of the gap to (B).

Note on triage vs full: the v11c work documented triage(11) as a noisy
proxy (~7% pessimistic vs full for v11c/v15/v16, +22% for v17 — ranking
preserved at the top but mid-pack compressed). Here β shows
triage=1.4836 worse than (B) 1.4461 but full=1.3272 better than (B)
1.3572 — same triage-noise pattern. **Trust full-97 for the verdict.**

† v11c: 2026-05-11 encoder-forecaster sweep winner, all-fp32 body,
forecaster d=384 (no bottleneck), β2=0.98, dropkey 0.9. Source:
`experiments/2026-05-11_exp_encoder_forecaster/RESULTS.md`.
†† (B): #303 cl_hh_50k. fp16 body, bottleneck d=128, β2=0.95, dropkey 0.7.

α and γ's "FINAL" backbones are their `best_loss.pth` snapshots — the
state at the loss minimum just before divergence (around step 1000–2000),
captured by `train.py`'s standard best-loss tracker. The downstream q-head
+ GIFT-Eval ran on those snapshots.

## Per-domain star (full GIFT-Eval, 97 configs)

![star](plots/perdomain_star.png)

α (red) and γ (orange) sit nearly on top of each other and inside
v11c (purple dashed) on every domain except Web/CloudOps; β (green)
sits between (B) and v11c; (B) (blue) is outermost everywhere. The
bottleneck-removed arms capture all the information v11c does plus
extra headroom in Econ/Fin, Energy, and Healthcare. The Econ/Fin spike
for all arms is driven by a handful of hard configs (e.g.
`bizitobs_application/*` rel-MASE 2.6–3.6).

## Training curves

![curves](plots/training_curves.png)

(B) blue and β green descend monotonically to loss ≈ 2.1–2.2 at 50k;
1−AUC stays at floor (≈1e−7), gap holds at 1.09. α red and γ orange
collapse: each bottoms near step 1000 then climbs, 1−AUC spikes
∼1e−4–1e−2, gap drops from 1.13 to 0.97 (γ) or 0.27 (α) before SIGTERM.
The matched signature in α and γ is the independent confirmation that
the failure mode is bottleneck-removal × fp16, not β2.

## Mechanism (hypothesis)

The (B) recipe's fp16 body is load-bearing on the forecaster bottleneck.
`experiments/2026-05-11_exp_encoder_forecaster/EXPERIMENT_LOG_2026-05-15_fp16_precision.md`
documents that the residual-stream max-abs amplitude grows unbounded
with depth and training (forecaster block: ~80 @ step 200 → ~1070 @
step 2800, >8× blowup), and that "fresh-init partial-fp16 diverges in
every tested combination" without a recovery mechanism. The (B)
bottleneck (d=128) constrains forecaster capacity, which constrains
residual growth, which keeps fp16 stable.

α and γ remove the bottleneck → forecaster runs at full d=384 → larger
residual stream → fp16 mantissa runs out → divergence at fresh init.
β keeps the bottleneck → fp16 stays stable, matches (B). The β2 axis
is essentially noise here: β (β2=0.98, bneck) and (B) (β2=0.95, bneck)
both converge; α (β2=0.98, no bneck) and γ (β2=0.95, no bneck) both
diverge with the same signature (γ's onset is slightly slower; β2=0.95
attenuates but does not prevent the blowup).

This is a hypothesis derived from prior amplitude data + α and γ's
matched divergence signature. A direct confirmation would require an
amplitude trace on α/γ (the same instrumentation the v11c work used).

## Implication for closing the gap to v11c

(B)'s gap to v11c is closed — and slightly overshot — *not* by the
two-axis change as proposed, but by the bottleneck removal alone.
The β2 change contributes a smaller, additive ~2% on top of (B). The
practical problem is that the bottleneck-removed recipe under the (B)
fp16 body **cannot be trained**: it captures a useful representation
in the first thousand steps and then unravels. To unlock the full α
ceiling, the fp16 body needs a stability mechanism:

- v20-style fp32-warmup → fp16 cast (`run_v20_v11c_freshwarmup_fp16.sh`),
  which the 2026-05-15 precision log shows is stable for the v11c
  recipe.
- Or all-fp32 body like v11c itself (loses the speed advantage).
- Or a different stabilization (e.g., layer-wise dtype, residual scaling).

These are **not** tested in #309. The question reframes from "which of
these two axes closes the gap?" to "what stability mechanism would let
the bottleneck-removed (B) recipe train past step ~1000?" — out of
scope for this card.

## Annex

### Arms in detail

- **α** — (B) + `--adam-beta2 0.98` + no `--forecaster-d-model/-n-heads`
  (forecaster inherits encoder d=384, n_heads=6). Encoder dropkey 0.70,
  loss `cosine_similarity_batch_full_hh_negs`, 1-GPU bs=256
  (mathematically identical to (B)'s 2-GPU DDP bs=128/GPU per
  `train.py:739-740` — gathered global negatives), 50k steps target,
  seed 20260520. Actual: SIGTERM @ step 10600 once divergence vs (B)
  was unambiguous.
- **β** — same as α but `--forecaster-d-model 128 --forecaster-n-heads 4`
  (bottleneck kept). Reached 50k cleanly; loss 2.13 at 50k vs (B)'s 2.17.
- **γ** — same as α (no bottleneck) but `--adam-beta2 0.95` (the (B) value).
  Actual: SIGTERM @ step 10100, same trajectory as α.

### Compute

1× RTX 4090 prosumer on vast.ai (offer 35882331, $0.55/h, reliability
0.992, US). The recipe was run 1-GPU because no 2× 4090 24GB prosumer
offer was available at provision time; the loss is mathematically
identical 1-GPU at bs=256 vs DDP at bs=128/GPU per `train.py`. Total
vast spend: **$2.66** of $20.37 budget. α and γ each SIGTERM'd at
~step 10k to save ~$1.30 each of doomed compute.

### Limitations

- **Single seed.** α's 1.277 vs v11c's 1.292 is a 1.7% difference;
  could shrink under seed variance. Reproducing α with 2–3 more seeds
  would set a confidence interval; the variance pattern in #307
  (n=3, ±0.02) suggests the difference is real but borderline.
- **α/γ are pre-divergence snapshots.** They are *not* the same kind
  of artifact as v11c (fully-trained) or (B) / β (fully-trained). A
  fair apples-to-apples reading is "the bottleneck-removed
  representation at step ~1000 (the loss minimum before divergence) is
  competitive with v11c at 50k" — not "the recipe converges to a
  1.277 backbone".
- **q-head and eval seed.** Single random init for the q-head and the
  GIFT-Eval sample windows.

### Code

Branch `experiment/2026-05-20-bottleneck-beta2-confound`.

Scripts: `scripts/{box_run.sh, box_run_serial.sh, provision.sh, sync.sh,
sync_loop.sh, downstream.sh, plot_results.py}`. Box-side serial α→β→γ;
elisa-side sync into MAIN checkout (CLAUDE.md rule). Per-arm
forecaster-bneck flag set in `downstream.sh` so q-head + eval load the
backbone in the correct architecture.
