# Splitting the main loss into L_pred + L_rep does not improve GM-Relative MASE at 6L / last (+2.3 to +2.7 % vs the pooled champion, single seed)

**Question.** The champion backbone of the [SIGReg (λ_e, λ_h) × EMA-τ
sweep](../2026-06-28_sigreg_lambda_tau_cross/sigreg_lambda_tau_cross.md)
trains with the pooled loss `cosine_similarity_batch_full_hh_negs_xshh_allt`:
one softmax denominator holds both the f-anchored (prediction) and the
h-anchored (repulsion) negative families. Does splitting them into two
independent terms — `L_pred` (positive against the f-anchored negatives)
and `L_rep` (pooled logsumexp of the h-anchored negatives, no positive) —
improve the full-97 GM-Relative MASE?

**Answer.** No at the 6L / last cell (arm 1 +2.7 %, arm 3 +2.3 %,
champion-denominated). The two "best-loss" cells where arm 3 sits below
the champion (2L / best 1.1548 vs 1.1682; 6L / best 1.1338 vs 1.1561) are
**not compute-matched**: arm 3's "best" backbone is at step 11,800 while
the champion's `best_loss` step is not on this branch, so those margins
mix backbone-step differences into the ranking and cannot be read as a
win. The card's primary success criterion — paired-bootstrap CI vs
arm C's per-task MASE — cannot be computed here because arm C's
per-task `all_results.csv` was not carried over onto this branch.
All quoted margins are single-seed point differences (see Caveat).

![GM-Relative MASE bars for the split arms across the four (head, checkpoint) cells; arm 4 downstream cells are pending.](plots/headline_relmase.png)

## Downstream GM-Relative MASE

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo) | **1.1548** | 1.1683 | **1.1338** | 1.1511 |
| arm 4 (pooled + MoCo) | **1.1602** | 1.1546 | 1.1603 | 1.1405 |
| arm 5 (`L_align` + `L_rep`) | TODO | TODO | 1.2554 | TODO |
| arm C ref (champion) | 1.1682 | 1.1491 | 1.1561 | **1.1254** |

*GM-Relative MASE: geometric mean, over GIFT-Eval's 97 evaluation configs,
of model MASE divided by seasonal-naive MASE; 1.0 = seasonal-naive, lower
is better. One cell = one (head depth, checkpoint) evaluation of one arm.
Split-arm values are the "Aggregate GM-Relative MASE (97 configs)" line
of each `summary.txt` under `experiments/2026-07-10_split_pred_rep/results/`;
champion values are the four `cross_C` (λ_e = 1, λ_h = 1, τ = 0.90) rows of
`experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`.
Boldface marks cells where the split arm's point value sits below the
champion's at the same head / checkpoint. Arm 4's head training and
evaluation are still running; its cells will be filled in when they land.*

**Backbone step behind each cell.** The head-training protocol trains on
the arm's `FINAL.pth` for the `best` cell, then resumes on `final.pth`
for the `last` cell. `FINAL.pth` is a symlink-of-content: the launcher
copies `best_loss.pth` into it, so the `best` cell head-trains on the
step where each arm's total loss was minimum, not on a fixed early
checkpoint:

| arm | `best` cell backbone step | `last` cell backbone step |
| --- | --: | --: |
| arm 1 (split) | 12,500 (`best_loss.pth` at step 600 was superseded by a re-run — `FINAL.pth` matches `final.pth`) | 12,500 |
| arm 3 (split + MoCo) | 11,800 (`best_loss.pth = FINAL.pth`) | 12,500 |
| arm 4 (pooled + MoCo) | 600 (`best_loss.pth = FINAL.pth`) | 12,500 |
| arm C ref (champion) | not on this branch | not on this branch |

Arm 4's `best_loss` at step 600 is a heavily underfit backbone, so its
`best` cell is not comparable to arm 1's `best` cell (backbone step
12,500), and neither is comparable to arm 3's `best` cell (backbone step
11,800). The `last` cells (backbone step 12,500 for every arm) are the
only rows compute-matched across arms.

## Gradient share

![Stacked per-family shares of each term's denominator at each arm's step-12,500 backbone snapshot; mixed and periodic batches.](plots/gradient_share_stack.png)

In the split shape at step 12,500, `log_neg_cross_batch` holds 0.99 of
`L_pred`'s denominator on the periodic batch (0.90 mixed) in arm 1, and
0.997 (periodic) / 0.937 (mixed) in arm 3. In the pooled shape (arm 4)
the same family holds 0.003 on both batches; the h-anchored families
`log_neg_hh_all` and `log_neg_xs_allt` together hold 0.877 (periodic) /
0.860 (mixed).

*Measurement (`scripts/gradient_share_measurement.py`; full table
`results/gradient_share_measurement.csv`, 132 rows). Each backbone
checkpoint runs in `.eval()` mode on two fixed batches of B = 64, T = 4096:
"mixed" is the training HF stream, "periodic" is solar/H + electricity/H
windows from GIFT-Eval. Each family's share of its own term's denominator
is exp(mean(log-family − log-denominator)) over anchors at τ = 0.10, so
segments in one bar need not sum to exactly 1. Caveats relative to the
training loop: measurement batch is B = 64 (training used B = 512, and the
`log_neg_cross_batch` count scales with B); `.eval()` disables the 0.70
encoder dropkey and dropout that reshape h at training time; for the
MoCo arms (3, 4) the keys are student-side here, while training routes
them through the EMA teacher. The measurement therefore characterises
the loss landscape as seen by a frozen student — not the training-time
gradient shares.*

## Training loss

![Total training loss for the three arms, tail-aligned to zero, log-x from step 100.](plots/loss_curves.png)

Each curve is the run's total training loss (contrastive + CPC + SIGReg),
tail-aligned by subtracting the mean of steps 12,401 – 12,500 so the
three runs sit at zero at the right edge. The split-shape and pooled-shape
totals are not the same functional (pooled arm 4 keeps the champion's
`--pos-in-denominator --subtract-contrastive-floor`, both dropped by the
split shape), so their absolute levels are not directly comparable.

## Arms

| arm | loss shape | `--moco-negatives` | defining feature |
| --- | --- | :-: | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | off | split objective `L = L_pred + L_rep`, equal weight |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | on | split objective; cross-batch f ↔ h keys come from the EMA teacher (MoCo-style) instead of the student |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | on | pooled champion shape with teacher keys |
| arm 5 | `cosine_similarity_batch_rep_only` + `--align-loss-weight 1.0` | off | replace `L_pred` with BYOL-style alignment: `L = L_align + L_rep` (no InfoNCE denominator on the f side) |
| arm C ref | `cosine_similarity_batch_full_hh_negs_xshh_allt` | off | champion (λ_e = 1, λ_h = 1, τ = 0.90) of the earlier sweep, reused without retraining |

Arm 2 was reserved in the issue-card follow-up list (λ-weighted variant
of the split, `α L_pred + β L_rep`) and was not run in this experiment;
it stays a follow-up card.

Negative families (tensor names from the measurement CSV): the two
f-anchored families are `log_neg_cross_batch` (cross-batch f_t ↔ h′_{t+1})
and `log_neg_zy` (adjacent f_{t+1} ↔ f_t); the three h-anchored families
are `log_neg_hh_all` (within-series all-time h ↔ h), `log_neg_xs_allt`
(cross-series all-time h ↔ h′), and cross-channel `log_neg_xx`, which is
empty at C = 1. f is the forecaster's predicted latent, h the encoder
latent; primes mark other series of the batch. The pooled shape puts all
five families into one denominator; the split routes the two f-anchored
families to `L_pred` and the three h-anchored families to `L_rep`.

## Method

Each arm trains one backbone with the champion recipe (12,500 steps,
B = 512, T = 4096, C = 1, lr 1e-3, seed 20260520, dataset
`gift-pretrain-full-4096 / small_v1`, EMA teacher τ = 0.90, contrastive
τ = 0.10, SIGReg λ_e = λ_h = 1, CPC auxiliary loss). The arms differ in
`--loss-shape` and `--moco-negatives`; the pooled arm additionally keeps
the champion's `--pos-in-denominator --subtract-contrastive-floor`, which
the split shape does not define. For each backbone a quantile probe head
(2 or 6 layers) is trained for 30,000 steps on `FINAL.pth` (`best-loss`
copied), then for 10,000 more steps — resuming the same head — on
`final.pth` (step 12,500). Each head is evaluated on GIFT-Eval's 97
configs against the same seasonal-naive reference as the champion.

## Caveat — single seed

Every evaluation is N = 1. Matched-cell point differences vs the champion
span −1.9 % (arm 3, 6L / best: 1.1338 vs 1.1561, champion-denominated) to
+2.7 % (arm 1, 6L / last: 1.1557 vs 1.1254) across the eight scored cells;
two of the eight are within ±0.6 % and four are within ±1.6 %. The spread
between the `best` and `last` cells of the same (arm, head) pair (last
relative to best) spans −0.2 % to +1.5 % here and −3.4 % to +4.0 % in
the champion's own sweep. Under the "no compute-matched best-loss" note
above, only the four `last` cells (all at backbone step 12,500) are a
matched-backbone comparison. The card's primary success criterion —
paired-bootstrap CI vs arm C's per-task MASE — remains unmet on this
branch. A multi-seed replicate would be needed to call any ordering real.
