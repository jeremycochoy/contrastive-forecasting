# A stop-gradient on the encoder side of the positive improves downstream transfer

**Question.** The strongest backbone recipe in this line (#328's L3 + no-bottleneck +
crossfade-triplet, PR #336) reliably beats its base on the downstream forecast at full training.
In SimSiam/BYOL, stopping the gradient through the *target* branch of the positive pair is
load-bearing. Does the analogous cut — training with `sim(stopgrad(h_{t+1}), f_{t+1})` as the
positive, everywhere that term appears (numerator and denominator) — change the learning
dynamics and the downstream transfer of that recipe?

![The backbone (identical in both arms) and the single change. The encoder is trained to
forecast its own future representation: the InfoNCE positive pulls the forecast f_{t+1} toward
the future encoding h_{t+1}, the negatives push everything else apart. The stop-grad arm
detaches h_{t+1} in the positive (numerator and denominator), so the encoder is never pulled
toward the forecast — it receives gradient only from the negatives.](plots/arch_stopgrad.png)

**Result.** Yes, both. The positive alignment stalls while batch-wise dimension usage settles
~4× higher, and the downstream forecast is **reliably better at the best-loss checkpoint on
both heads** (both 90% intervals fully below zero), tied at the last checkpoint and nominally
better in all four cells. Its best 6-layer score is the lowest of the eight cells measured here
and edges the reference's strongest cell, though deeper or longer-trained heads elsewhere in the
project have scored lower.

*Forecast error is **GM-Relative MASE**: the geometric mean, over the GIFT-Eval benchmark's 97
forecasting tasks, of a model's error divided by the seasonal-naive forecast's error. Lower is
better; 1.0 is seasonal-naive.*

![Left: forecast error per head and checkpoint — the stop-grad arm (red) is below the reference
(blue) in every cell. Right: the paired-bootstrap change with its 90% interval — green intervals
sit fully below zero (reliable improvement), grey straddle it.](plots/gm_summary.png)

**Lower is better; bold = best score per head.** Δ = stop-grad − reference at the same head and
checkpoint, with a **paired bootstrap** 90% interval (resample the 97-task list with repeats,
score both arms on each resample so per-task difficulty cancels).

| backbone | checkpoint | 2-layer head | Δ (90% interval) | 6-layer head | Δ (90% interval) |
|---|---|--:|:--:|--:|:--:|
| reference | best-loss (~6.4k) | 1.220 | — | 1.211 | — |
| reference | last (12.5k) | 1.181 | — | 1.170 | — |
| stop-grad | best-loss (~6.6k) | **1.177** | −0.043 (−0.071, −0.016) | **1.159** | −0.052 (−0.079, −0.025) |
| stop-grad | last (12.5k) | 1.180 | −0.001 (−0.024, +0.023) | 1.163 | −0.007 (−0.028, +0.015) |

The reference needs all 12.5k steps to reach its best (its last checkpoint beats its own
best-loss by ~0.04 on both heads); the stop-grad arm is already at that level at ~6.6k — half
the budget. Comparing each arm's strongest cell directly (stop-grad best-loss vs reference
last), the stop-grad arm is nominally ahead but within noise (2L −0.004 (−0.028, +0.022); 6L
−0.011 (−0.032, +0.011)).

## Training dynamics: alignment stalls, no low-rank collapse

![Training metrics, log-log, stop-grad (solid red) vs the reference (dashed blue), from step
100. Top row, lower is better: contrastive loss minus the analytic InfoNCE floor (the loss's
lower bound for this negative count), the ratio gap (1−ff)/(1−fp), and 1−R²_naive. Bottom row:
1−R²_random (lower is better), then U_batch and U_temporal (higher = more embedding dimensions
in use).](plots/training_metrics.png)

Here **ff** is the forecast-to-future cosine (the positive pair's similarity), **fp** the
forecast-to-present cosine, **R²_naive / R²_random** the variance in the future embedding
explained by the forecast, relative to a copy-the-present / random-embedding baseline, and
**U_batch / U_temporal** the fraction of embedding dimensions that vary across the batch /
across time. The single change produces three clear differences:

- **The positive alignment stalls.** ff plateaus near 0.44 within ~800 steps (the reference
  climbs to 0.99); the floor-subtracted loss settles around ≈5.5–6.1 — its minimum, the
  evaluated best-loss checkpoint, at step ~6.6k — and ends at 5.88 vs 1.06, about 5.5× higher.
  The skill metrics sit correspondingly higher (1−R²_naive 0.67 vs 0.012).
- **Batch-wise dimension usage settles ~4× higher.** U_batch climbs to ~0.5 by ~3k steps and
  holds; the reference's never rises above ~0.14 — a much lower-rank regime, not a fall from a
  high value in either arm. U_temporal is slightly *lower* (0.100 vs 0.115), so the extra rank
  is across series, not across time.
- **Discrimination is barely affected.** Ranking the positive against the negatives, the
  stop-grad arm scores AUC 0.998 with the positive ranked first (Top-1) ~92% of the time late
  in training, vs the reference's 1.000/1.000; cross-series cosine 0.022 vs 0.003,
  near-orthogonal in both. The cut changes how
  far the positive pair is pulled together, not whether pairs are told apart.

Both arms subtract the same floor constant, and the stop-grad leaves the forward loss value
bit-identical (unit-tested), so the loss curves are directly comparable.

## Protocol

One backbone per arm, single seed (20260520), 12,500 steps at batch 1024 on one RTX 4090: the
reference is #328's best arm unchanged, the stop-grad arm adds the single flag
(`--stopgrad-positive-h`). Each finished backbone is frozen and scored by training a fresh
quantile forecasting head on top — once with two transformer layers, once with six; identical
head hyperparameters and eval data across arms — on GIFT-Eval's 97 tasks, at two backbone
checkpoints: **best-loss** (the step with the lowest smoothed contrastive loss) and **last**
(12,500). One backbone per arm, so the bootstrap intervals quantify task-set noise, not seed
noise.

## What we learned

- **Training does not break, and the higher pretext loss is not worse transfer.** The stop-grad
  arm's contrastive loss stays about 5.5× higher than the reference's, yet it transfers better.
  A lower contrastive loss therefore does not imply better transfer: the higher-loss arm wins
  every matched cell, reliably so at the best-loss checkpoint.
- **The transfer peak arrives at about half the budget**: the reference needs all 12.5k steps
  for its best downstream score; the stop-grad arm matches that level by ~6.6k, and its two
  measured checkpoints differ by less than task-set noise.
- *Hypothesis (consistent with the curves, untested causally):* the gain comes from avoiding
  the low-rank regime — with the encoder never pulled toward the forecast, ~4× more batch-wise
  dimensions stay in use, and the heads exploit the higher-rank embedding. Disentangling rank
  from alignment needs a separate intervention (e.g. dimension decorrelation on the reference
  recipe).

## Follow-up

**Stop-grad + shorter training** (the measured peak at ~6.6k ≈ half the compute for the same
transfer), and **multi-seed confirmation** of the best-loss gain — on this seed the largest
improvement a single change has produced in this recipe line.
