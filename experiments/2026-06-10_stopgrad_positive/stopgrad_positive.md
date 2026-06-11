# Stop-gradient on the encoder side of the positive: better transfer, earlier, without collapse

**Question.** The strongest backbone recipe so far (#328's L3 + no-bottleneck + crossfade-triplet,
PR #336) reliably beats its base on the downstream forecast at full training. In SimSiam/BYOL,
stopping the gradient through the *target* branch of the positive pair is load-bearing. Does the
analogous cut here — training with `sim(stopgrad(h_{t+1}), f_{t+1})` as the positive, everywhere
that term appears (numerator and denominator) — change the learning dynamics and the downstream
transfer of that recipe?

**Result.** Yes, both. The dynamics change qualitatively (alignment slows ~10×, dimension usage
stays ~4× higher, nothing collapses), and the downstream forecast is **reliably better at the
best-loss checkpoint on both heads** (2-layer −0.043, 6-layer −0.052; both 90% intervals fully
below zero) and statistically tied at the last checkpoint. The stop-grad arm is never worse in
any cell, and its 6-layer best-loss score (1.159) is the lowest GM-Relative MASE measured in this
project so far.

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97 tasks, of a
model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is
seasonal-naive.*

![Left: forecast error per head and checkpoint — the stop-grad arm (red) is below the reference
(blue) in every cell. Right: the paired-bootstrap change with its 90% interval — green intervals
sit fully below zero (reliable improvement), grey straddle it.](plots/gm_summary.png)

| forecasting head | checkpoint | reference | stop-grad | change | 90% interval |
|---|---|--:|--:|--:|:--:|
| 2-layer | best-loss | 1.220 | **1.177** | **−0.043** | (−0.071, −0.016) |
| 2-layer | last (12.5k) | 1.181 | 1.180 | −0.000 | (−0.024, +0.023) |
| 6-layer | best-loss | 1.211 | **1.159** | **−0.052** | (−0.079, −0.025) |
| 6-layer | last (12.5k) | 1.169 | 1.163 | −0.007 | (−0.028, +0.015) |

Each change carries a **paired bootstrap** 90% interval (resample the 97-task list with repeats,
score both models on each resample so per-task difficulty cancels). Both best-loss improvements
are reliable; both last-checkpoint changes are within task-set noise. The reference's transfer
peaks late (its last checkpoint beats its best-loss by ~0.04); the stop-grad arm is already at
that level at its best-loss checkpoint — step ~6.1k, half the training budget. Comparing each
arm's strongest cell directly (stop-grad best-loss vs reference last), the stop-grad arm is
nominally ahead on both heads (−0.004 and −0.011) but within noise.

## Training dynamics: slower alignment, no dimension collapse

![Training metrics, log-log, stop-grad (solid red) vs the reference (dashed blue), from step 100.
Top row, lower is better: floor-subtracted contrastive loss, the ratio gap (1−ff)/(1−fp), and
1−R² against naive and random baselines. Bottom row, higher is better: U_batch and U_temporal,
the fraction of embedding dimensions in use.](plots/training_metrics.png)

The single change splits the dynamics into two regimes:

- **Alignment slows by an order of magnitude.** With the encoder no longer pulled toward the
  forecast, the forecast must do all the closing: the forecast-to-future cosine reaches only
  0.45 by 12.5k steps (reference: 0.99), the floor-subtracted loss plateaus near 5.9 from ~5k
  steps on (reference: 1.04), and the ratio gap and skill metrics stall correspondingly
  (1−R²_naive 0.66 vs 0.011).
- **Dimension usage stays high instead of collapsing.** U_batch — the fraction of embedding
  dimensions that vary across the batch — holds at ~0.50 throughout, against the reference's
  early drop to ~0.13. The embedding stays ~4× higher-rank.
- **Discrimination is unaffected.** Both runs separate positives from negatives essentially
  perfectly from early on (AUC and Top-1 ≈ 1.0), and different series stay near-orthogonal
  (cross-series cosine 0.024 vs 0.002). The stop-grad changes only how fast the positive pair
  is pulled together, not whether the model can tell pairs apart.

Both arms log the same floor-subtracted loss (identical constant; the stop-grad does not change
the forward value, verified bit-equal in tests), so the loss curves are directly comparable.

## How the arm works

The training loss is a normalized InfoNCE: each forecast f_{t+1} should be similar to its own
future encoding h_{t+1} (the positive) and dissimilar from everything else (negatives across
time, series, and the batch). By default the positive's gradient flows into *both* branches —
the forecaster chases the encoder, and the encoder is simultaneously pulled back toward the
forecast. The single change here detaches h_{t+1} in the positive term wherever it appears
(numerator and denominator), so the encoder receives gradient only from the *negative* terms —
it is trained to spread series apart, never to make its own representation easier to forecast.
This is the same asymmetry SimSiam/BYOL apply to their target branch. Everything else — data
mix, crossfade triplet, architecture, floor subtraction, temperature, seed, batch, step count —
is identical to the reference run.

## Protocol

One backbone per arm, single seed (20260520), 12,500 steps at batch 1024 on one RTX 4090. The
reference is #328's best arm (L3 + no-bottleneck + triplet) unchanged; the stop-grad arm differs
by the single flag described above (`--stopgrad-positive-h`, PR #336 follow-up). Each finished
backbone is frozen and scored by training a fresh quantile forecasting head on top — once with
two transformer layers, once with six — and evaluating on GIFT-Eval's 97 tasks, at two backbone
checkpoints: **best-loss** (the step with the lowest contrastive loss; ~6.1k for the stop-grad
arm, ~6.4k for the reference) and **last** (the full 12,500 steps; the regime where the
reference's downstream advantage shows). Head training and evaluation use the same
hyperparameters and eval data as the reference's runs. Intervals are paired bootstrap over
tasks; one backbone per arm, so they quantify task-set noise, not seed noise.

## What we learned

- The encoder-side stop-grad on the positive **does not break training** — despite the pretext
  loss sitting ~5× higher, the representation transfers better.
- **Lower contrastive loss does not imply better transfer** — this experiment is the starkest
  instance yet: the arm whose loss plateaus at 5.9 beats the arm that reaches 1.04, at every
  matched checkpoint, on both heads (reliably at best-loss). #328's disentanglement had already
  shown this decoupling in both directions.
- **The transfer peak arrives at half the budget.** The reference needs all 12.5k steps for its
  best downstream score; the stop-grad arm matches that level by ~6.1k steps and gains nothing
  reliable afterwards.
- *Hypothesis (consistent with the curves, untested causally):* the gain comes from avoiding
  dimension collapse — without the positive pulling h toward the lower-rank forecast, the
  encoder keeps ~4× more dimensions in use, and the higher-rank embedding is what the
  forecasting heads exploit. Disentangling rank from alignment speed would need a separate
  intervention (e.g. a dimension-decorrelation regularizer on the reference recipe).

## Follow-up

The natural next card: **stop-grad + shorter training** (the peak is at ~6k steps — half the
compute for the same transfer), and **multi-seed confirmation** of the best-loss gain, which is
the project's largest single-change improvement so far on a one-seed basis.
