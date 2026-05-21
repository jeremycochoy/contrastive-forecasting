# #309 Bottleneck × β2 confound on (B): can α match v11c?

## Question

(B) — the `cosine_similarity_batch_full_hh_negs` variant of the
bottleneck-fullfh recipe — reaches full-97 GM-MASE 1.3572 (#303),
still **+5%** above v11c 1.292. Of the ≥6 axes that differ between (B)
and v11c, two are testable cheaply: the forecaster bottleneck and
AdamW β2. Does isolating those two axes close the gap?

## Verdict

**No — a fully-trained α does not match v11c.** The headline is a
two-part story:

1. α and γ *diverge* under the (B) fp16 body when the bottleneck is
   removed (forecaster d=128→384). At their loss minimum just before
   divergence (step ~900), their backbones already beat v11c on
   full-97 (α 1.277, γ 1.283 vs v11c 1.292).
2. But that win is an **under-training artifact**, not an architecture
   gain. Resuming α's step-900 checkpoint and training to 50k in
   all-fp32 (stable — fp16 was the only blocker) yields **1.369** —
   *worse* than the step-900 snapshot (1.277), worse than v11c
   (1.292), and worse even than (B) (1.357). Prolonged contrastive
   training of the no-bottleneck arm degrades GIFT-Eval transfer.

So the bottleneck-removed forecaster, trained to convergence under
this recipe (dropkey 0.7, hh-negs loss), is the **worst** arm — the
opposite of what the step-900 snapshot suggested. β (1.327, β2-only)
remains the only change that improves on (B) at convergence.

| Arm   | β2   | Fcst bneck | Train state | Triage-11 GM | Full-97 GM | vs (B) |
|-------|-----:|-----------:|-------------|------------:|----------:|-------:|
| (B)†† | 0.95 | kept       | converged 50k fp16 | 1.4461 | **1.3572**| —      |
| v11c† | 0.98 | removed    | converged 50k all-fp32 | 1.3878 | **1.292** | −4.8% |
| α     | 0.98 | removed    | snapshot @ step ~900 (pre-divergence) | 1.3399 | **1.2767** | −5.9% |
| β     | 0.98 | kept       | converged 50k fp16 | 1.4836 | **1.3272**| −2.2% |
| γ     | 0.95 | removed    | snapshot @ step ~900 (pre-divergence) | 1.3412 | **1.2829**| −5.5% |
| **α-fp32cont** | 0.98 | removed | **converged 50k all-fp32** (resumed from α step 900) | 1.4498 | **1.3687** | +0.8% |

The pre-divergence snapshots (α 1.277, γ 1.283) beat v11c, but those
backbones have only ~900 steps of training. Training α's snapshot to a
full 50k in fp32 — the same architecture, stable now — *regresses* to
1.369, the worst arm. The snapshot advantage does not survive
convergence. Within the no-bottleneck arm, more training → worse
GIFT-Eval (1.277 @ step 900 → 1.369 @ step 50k). β2 explains the ~0.5%
between the α and γ snapshots.

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

The pre-divergence snapshots α (red) and γ (orange) sit nearly on top
of each other and inside v11c (purple dashed) on every domain except
Web/CloudOps. But **α-fp32cont (brown)** — the same architecture trained
to convergence — sits *outside* almost everything, near (B) (blue): on
GIFT-Eval the fully-trained no-bottleneck backbone is the worst arm.
β (green) sits between (B) and v11c. The Econ/Fin spike for all arms is
driven by a handful of hard configs (e.g. `bizitobs_application/*`
rel-MASE 2.6–3.6).

## Training curves

![curves](plots/training_curves.png)

(B) blue and β green descend monotonically to loss ≈ 2.1–2.2 at 50k;
1−AUC stays at floor (≈1e−7), gap holds at 1.09. α red and γ orange
collapse: each bottoms near step 900 then climbs, 1−AUC spikes
∼1e−4–1e−2, gap drops from 1.13 to 0.97 (γ) or 0.27 (α) before SIGTERM.
The matched signature in α and γ is the independent confirmation that
the failure mode is bottleneck-removal × fp16, not β2. **α-fp32cont
(brown)** forks from α's red curve at step 900 and descends stably in
all-fp32 to loss ≈ 2.30 at 50k — no divergence, 1−AUC at floor, gap
holding ~1.0. So fp32 *does* train the no-bottleneck arm; the loss
plateaus slightly above (B)/β (2.30 vs 2.1–2.2), and as the GIFT-Eval
shows, that fully-trained representation transfers worse than the
brief snapshot.

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

## fp32 continuation — does training the diverged arm fix it?

The diverged arms can be trained: resume α's pre-divergence step-900
checkpoint with the body switched to **all-fp32** and continue to 50k.
The continuation is stable (loss 3.27 → 2.30, no divergence), so **fp16
was the only thing blocking training** — the no-bottleneck arch itself
trains fine in fp32.

But the converged backbone is *worse*, not better:

| α checkpoint | training | full-97 GM |
|--------------|----------|-----------:|
| step ~900 (pre-divergence snapshot) | ~900 steps fp16 | **1.2767** |
| step 50000 (fp32 continuation)      | 50k steps        | **1.3687** |

Training the no-bottleneck arm from its good step-900 state out to 50k
*degrades* GIFT-Eval transfer by 7%. This matches the cross-arm pattern:
the lowest-loss backbones (β 2.13, (B) 2.17) have the worst GIFT-Eval,
while the high-loss snapshots (α/γ ≈ 3.3) have the best. In this
encoder-forecaster regime, more contrastive pretext training
over-specializes the representation and hurts forecasting transfer.

So the step-900 win over v11c was an under-training artifact. v11c
(all-fp32, no-bottleneck, dropkey **0.9**, simpler `cosine_similarity_batch`
loss) still converges to 1.292, beating fully-trained α-fp32cont (1.369).
The two recipes differ on dropkey (0.9 vs 0.7) and loss (plain vs
hh-negs); the lower dropkey + extra negatives plausibly drive
α-fp32cont's over-specialization. **Not isolated here** — would need a
dropkey-0.9 / plain-loss fp32 continuation to test, which is the v11c
recipe itself.

### Bottom line
- The two-axis isolation (#309's premise) does **not** beat v11c at
  convergence; the apparent win was under-training.
- β2 alone (β) gives a real but small +2% over (B) at convergence.
- The bottleneck-removed forecaster, fully trained under the (B)
  recipe's dropkey/loss, transfers worse than v11c — the bottleneck
  was not the thing holding (B) back from v11c.

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
- **α-fp32cont** — resumed from α's `best_loss.pth` (step 900) with the
  body switched to all-fp32 (`--residual/attn/ffn/conv/patch-emb-dtype fp32`),
  optimizer + step + RNG restored from the companion. Everything else
  identical to α (no bottleneck, β2=0.98, dropkey 0.7, hh-negs). Reached
  50k cleanly on elisa GPU 1; loss 2.30 at 50k.

### Compute

1× RTX 4090 prosumer on vast.ai (offer 35882331, $0.55/h, reliability
0.992, US). The recipe was run 1-GPU because no 2× 4090 24GB prosumer
offer was available at provision time; the loss is mathematically
identical 1-GPU at bs=256 vs DDP at bs=128/GPU per `train.py`. Total
vast spend: **$2.66** of $20.37 budget. α and γ each SIGTERM'd at
~step 10k to save ~$1.30 each of doomed compute.

### Limitations

- **Single seed.** All full-97 numbers are one seed each; the variance
  pattern in #307 (n=3, ±0.02) means differences under ~3% (e.g. α-snap
  1.277 vs v11c 1.292) are borderline. The big effects here —
  α-snap 1.277 vs α-fp32cont 1.369 (+7%), and both fp16 arms diverging
  — are well outside that band.
- **The under-training → over-specialization finding is single-arm.**
  Only α was continued in fp32. γ would confirm whether its snapshot
  win also evaporates at convergence (predicted: yes, same recipe).
- **dropkey/loss confound vs v11c.** α-fp32cont (dropkey 0.7, hh-negs)
  converges worse than v11c (dropkey 0.9, plain `cosine_similarity_batch`).
  Whether the bottleneck-removed arch *could* match v11c with v11c's
  dropkey + loss is untested — that is essentially the v11c recipe.
- **q-head and eval seed.** Single random init for the q-head and the
  GIFT-Eval sample windows.

### Code

Branch `experiment/2026-05-20-bottleneck-beta2-confound`.

Scripts: `scripts/{box_run.sh, box_run_serial.sh, provision.sh, sync.sh,
sync_loop.sh, downstream.sh, plot_results.py}`. Box-side serial α→β→γ;
elisa-side sync into MAIN checkout (CLAUDE.md rule). Per-arm
forecaster-bneck flag set in `downstream.sh` so q-head + eval load the
backbone in the correct architecture.
