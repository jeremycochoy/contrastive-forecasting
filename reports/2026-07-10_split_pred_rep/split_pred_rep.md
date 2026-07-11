# Splitting the main loss into L_pred + L_rep does not improve GM-Relative MASE at 6L/last (+2.3–2.7 % vs the pooled champion, single seed)

**Question.** The champion backbone of the [SIGReg (λ_e, λ_h) × EMA-τ
sweep](../2026-06-28_sigreg_lambda_tau_cross/sigreg_lambda_tau_cross.md)
trains with the pooled loss `cosine_similarity_batch_full_hh_negs_xshh_allt`:
one softmax denominator holds both the f-anchored (prediction) and the
h-anchored (repulsion) negative families. Does splitting them into two
independent terms — `L_pred` (the positive against the f-anchored
negatives) and `L_rep` (pooled logsumexp of the h-anchored negatives, no
positive) — improve the full-97 GM-Relative MASE?

**Answer.** The result is cell-dependent. At 6L/last, arm 1 and arm 3 land
at +2.7 % and +2.3 % vs the champion's 1.1254. At the "best-loss" cells
(30k head-train on `best_loss.pth`, no 10k re-adapt), arm 3 beats the
champion at 2L/best by 1.2 % (1.1548 vs 1.1682) and at 6L/best by 1.9 %
(1.1338 vs 1.1561); at the "last" cells the champion wins in every row.
All margins are single-seed (see Caveat).

![GM-Relative MASE bars for the split arms across the four (head, checkpoint) cells; arm 4 pending.](plots/headline_relmase.png)

## Arms

Arm names follow the experiment's internal numbering.

| arm | loss shape | `--moco-negatives` | defining feature |
| --- | --- | :-: | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | off | split objective `L = L_pred + L_rep`, equal weight |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | on | split objective; cross-batch f↔h keys come from the EMA teacher (MoCo-style) instead of the student |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | on | pooled champion shape with teacher keys — downstream scores pending |
| arm C ref | `cosine_similarity_batch_full_hh_negs_xshh_allt` | off | champion (λ_e = 1, λ_h = 1, τ = 0.90) of the earlier sweep, reused without retraining |

Negative families (tensor names from the measurement CSV): the two
f-anchored families are `log_neg_cross_batch` (cross-batch f_t ↔ h′_{t+1})
and `log_neg_zy` (adjacent f_{t+1} ↔ f_t); the three h-anchored families are
`log_neg_hh_all` (within-series all-time h ↔ h), `log_neg_xs_allt`
(cross-series all-time h ↔ h′), and cross-channel `log_neg_xx`, which is
empty at C = 1. f is the forecaster's predicted latent, h the encoder
latent; primes mark other series of the batch. The pooled shape puts all
families into one denominator; the split gives the two f-anchored families
to `L_pred` and the three h-anchored families to `L_rep`.

## Downstream GM-Relative MASE

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo) | **1.1548** | 1.1683 | **1.1338** | 1.1511 |
| arm 4 (pooled + MoCo) | TODO | TODO | TODO | TODO |
| arm C ref (champion) | 1.1682 | 1.1491 | 1.1561 | **1.1254** |

*GM-Relative MASE: geometric mean, over GIFT-Eval's 97 evaluation configs,
of model MASE divided by seasonal-naive MASE; 1.0 = seasonal-naive, lower is
better. One cell = one (head depth, checkpoint) evaluation of an arm.
Split-arm values are the "Aggregate GM-Relative MASE (97 configs)" line
of each `summary.txt` under `experiments/2026-07-10_split_pred_rep/results/`;
champion values are the four `cross_C` (λ_e = 1, λ_h = 1, τ = 0.90) rows of
`experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`.
Boldface marks cells where the split arm beats the champion at the same
head / checkpoint. Arm 4's head training and evaluation are still running;
its cells will be filled in when they finish.*

## Gradient share

![Stacked per-family shares of each term's denominator at each arm's step-12,500 backbone snapshot, mixed and periodic batches.](plots/gradient_share_stack.png)

In the split shape at step 12,500, `log_neg_cross_batch` holds 0.99 of the
`L_pred` denominator on the periodic batch (0.90 mixed) in arm 1, and 0.997
(periodic) / 0.937 (mixed) in arm 3. In the pooled shape (arm 4) the same
family holds 0.003 on both batches; the h-anchored families `log_neg_hh_all`
and `log_neg_xs_allt` together hold 0.877 (periodic) / 0.860 (mixed).

*Measurement (`scripts/gradient_share_measurement.py`; full table
`results/gradient_share_measurement.csv`, 132 rows): each backbone
checkpoint runs in eval mode on two fixed batches (B = 64, T = 4096) —
"mixed" = training HF stream, "periodic" = solar/H + electricity/H windows
from GIFT-Eval — and each family's share of its own term's denominator is
exp(mean(log-family − log-denominator)) over anchors at τ = 0.10, so the
segments of one bar need not sum to exactly 1. Keys are student-side for
all arms; the EMA teacher is not run at measurement time.*

## Training loss

![Total training loss for the three arms, shifted by each run's final level, log-x from step 100.](plots/loss_curves.png)

Each curve is the run's total training loss (contrastive + CPC + SIGReg
terms) shifted by its own final level (mean over steps 12,401–12,500), so
the three runs align at 0. Between steps 10,000 and 12,500 the endpoint
levels of the 101-step rolling means differ by 0.034 (arm 1), 0.063
(arm 3) and 0.097 nats (arm 4); the max − min over the same window is
0.138 / 0.103 / 0.310 nats (arm 4's spike at step ≈12,264 accounts for
its larger max − min).

## Method

Each arm trains one backbone with the champion recipe (12,500 steps,
B = 512, T = 4096, C = 1, lr 1e-3, seed 20260520, dataset
`gift-pretrain-full-4096 / small_v1`, EMA teacher τ = 0.90, contrastive
τ = 0.10, SIGReg λ_e = λ_h = 1, CPC auxiliary loss). The arms differ in
`--loss-shape` and `--moco-negatives`; the pooled arm additionally keeps the
champion's `--pos-in-denominator --subtract-contrastive-floor`, which the
split shape does not define. For each backbone a quantile probe head (2 or
6 layers) is trained for 30,000 steps on the `best-loss` checkpoint, then
for 10,000 more steps — resuming the same head — on the `last` checkpoint
(step 12,500). Each head is evaluated on GIFT-Eval's 97 configs against the
same seasonal-naive reference as the champion.

## Caveat — single seed

Every evaluation is N = 1. Matched-cell gaps to the champion span
−1.9 % (arm 3, 6L / best: 1.1338 vs 1.1561) to +2.7 % (arm 1, 6L / last:
1.1557 vs 1.1254) across the eight scored cells; four of the eight are
within ±0.6 %. The spread between best and last checkpoints of the same
(arm, head) pair (last relative to best) spans −0.2 % to +1.5 % here and
−3.4 % to +4.0 % in the champion's own sweep. A multi-seed replicate
would be needed to call any ordering real.
