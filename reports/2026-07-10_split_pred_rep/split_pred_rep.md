# At the four compute-matched cells no split-vs-pooled or MoCo contrast separates from zero at 95 % CI; `L_align + L_rep` is a real regression (+5 % to +15 %)

**Question.** The champion backbone of the [SIGReg (λ_e, λ_h) × EMA-τ
sweep](../2026-06-28_sigreg_lambda_tau_cross/sigreg_lambda_tau_cross.md)
trains with the pooled loss `cosine_similarity_batch_full_hh_negs_xshh_allt`:
one softmax denominator holds both the f-anchored (prediction) and the
h-anchored (repulsion) negative families. Does splitting them into two
independent terms — `L_pred` (positive against the f-anchored negatives)
and `L_rep` (pooled logsumexp of the h-anchored negatives, no positive) —
improve the full-97 GM-Relative MASE? Two side arms probe alternative
f-side objectives: arm 4 keeps the pooled shape and adds MoCo teacher keys
to isolate the MoCo axis, and arm 5 drops the InfoNCE denominator on the
f side entirely, replacing `L_pred` with a BYOL-style alignment
(`L = L_align + L_rep`).

**Answer.** At the four `last` cells (all four arms at backbone step
12,500 — the only cross-arm compute-matched read), a 20 000-resample
paired bootstrap over the 97 configs gives every pairwise ratio between
arms 1, 3 and 4 as a 95 % CI that straddles 1 (see the *Paired-bootstrap*
subsection below). The single-axis split-vs-pooled contrast (arm 3 vs
arm 4) is 1.0119 [0.9970, 1.0267] at 2L and 1.0093 [0.9960, 1.0269] at
6L; the MoCo axis (arm 1 vs arm 3) is 0.9988 [0.9834, 1.0158] at 2L and
1.0039 [0.9902, 1.0195] at 6L. The point-difference ranking the champion
enjoys in every `last` row (arm 1 +2.7 %, arm 3 +2.3 %, arm 4 +1.3 % at
6L) is therefore not separated from noise at N = 1. Dropping the InfoNCE
denominator on the f side (arm 5, `L_align + L_rep`) is a real
regression: arm 5 vs arm 1 is 1.0557 [1.0220, 1.0891] at 6L / last and
1.1041 [1.0632, 1.1473] at 2L / last (both intervals lie above 1),
so arm 5 is worse than every other arm on every scored cell. Note: this
report has no per-task `all_results.csv` for arm C on this branch, so
paired CIs vs the champion are computed against arm 1 instead of arm C;
the two arms are at matched backbone step 12,500 and their
point-difference at 6L / last is 0.0303 (arm 1 minus champion), so an
arm 5-vs-champion 6L / last interval would sit slightly higher than the
arm 5-vs-arm 1 interval — still above 1.

![GM-Relative MASE across arms and (head, checkpoint) cells.](plots/headline_relmase.png)

## Downstream GM-Relative MASE

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo) | **1.1548** | 1.1683 | **1.1338** | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | 1.1546 | 1.1603 | 1.1405 |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm C ref (champion) | 1.1682 | **1.1491** | 1.1561 | **1.1254** |

*GM-Relative MASE: geometric mean, over GIFT-Eval's 97 evaluation configs,
of model MASE divided by seasonal-naive MASE; 1.0 = seasonal-naive, lower
is better. One cell = one (head depth, checkpoint) evaluation of one arm.
Arms 1 / 3 values are `Aggregate GM-Relative MASE (97 configs)` in each
`summary.txt` under `experiments/2026-07-10_split_pred_rep/results/`;
arms 4 / 5 values are the same line under `results_arm4/` and
`results_arm5/`. Champion values are the four `cross_C` (λ_e = 1, λ_h = 1,
τ = 0.90) rows of `experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`.
Boldface marks the column minimum across all arms. Point-difference
rankings by themselves cannot be read as evidence at this seed count —
see the paired-bootstrap subsection below and the Caveat.*

**Backbone step behind each cell.** The head-training protocol trains on
the arm's `FINAL.pth` for the `best` cell, then resumes on `final.pth`
for the `last` cell. Each launcher's end-of-training block is
`cp best_loss.pth → FINAL.pth`, falling through to `final.pth` if
`best_loss.pth` is absent. The actual step behind each arm's `FINAL.pth`
is verified below by md5-comparing `FINAL.pth` to each of
`best_loss.pth`, `final.pth` and the intermediate `Xk.pth` snapshots:

| arm | `best` cell backbone step | `last` cell backbone step | `FINAL.pth` md5-matches |
| --- | --: | --: | --- |
| arm 1 (split) | 12,500 | 12,500 | `final.pth` (weights also equal to `12k.pth` — see below) |
| arm 3 (split + MoCo) | 504 | 12,500 | `best_loss.pth` |
| arm 4 (pooled + MoCo) | 494 | 12,500 | `best_loss.pth` |
| arm 5 (`L_align` + `L_rep`) | 11,809 | 12,500 | `best_loss.pth` |
| arm C ref (champion) | *n/a on this branch* | 12,500 | (per sweep protocol; `best_loss.pth` step not exported here) |

For arm 1, `FINAL.pth`, `final.pth` and `12k.pth` are byte-different
files but the 193 tensors in each state-dict are identical
(`torch.equal` across all keys), so arm 1's backbone did not update in
the final 500 steps and the `best` and `last` cells are on the same
weights. For arms 3, 4 and 5, `FINAL.pth` is a byte-identical copy of
`best_loss.pth`; the reported step is the losses-CSV `argmin(loss)`
step. Only the four `last` cells (arm 1 through arm 5 all at step
12,500 for the head's second training phase) are cross-arm
compute-matched; every `best` cell mixes objective differences with
backbone-step differences.

## Paired-bootstrap 95 % CI on GM-Relative MASE ratios

20 000 resamples over the 97 configs, seed 42, using each arm's
per-task `MASE[0.5]` normalised by the shared seasonal-naive reference.
Ratio `A/B < 1` means arm A beats arm B. `arm C` is absent because its
per-task `all_results.csv` is not on this branch (`paired_bootstrap.py`
requires it); the arm 5 read vs the champion falls out of the arm 1
column plus the +0.0303 point-difference between arm 1 and the champion
at 6L / last. Full 24-row table:
`experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci.csv`.
Cells at compute-matched `last` (step 12,500 across all four arms):

| cell | contrast | axis toggled | ratio A/B | 95 % CI | separates from 1? |
| --- | --- | --- | --: | --- | :-: |
| 2L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 1.0119 | [0.9970, 1.0267] | no |
| 6L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 1.0093 | [0.9960, 1.0269] | no |
| 2L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 0.9988 | [0.9834, 1.0158] | no |
| 6L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 1.0039 | [0.9902, 1.0195] | no |
| 2L / last | arm 1 vs arm 4 | joint (split + no-MoCo ↔ pooled + MoCo) | 1.0107 | [0.9963, 1.0262] | no |
| 6L / last | arm 1 vs arm 4 | joint | 1.0133 | [0.9957, 1.0344] | no |
| 2L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 1.1041 | [1.0632, 1.1473] | **yes (arm 5 worse)** |
| 6L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 1.0557 | [1.0220, 1.0891] | **yes (arm 5 worse)** |

None of the arm 1 / 3 / 4 pairwise contrasts separate from zero at
compute-matched cells. Arm 5's regression separates from zero on every
scored cell (12 CIs; see `pairwise_bootstrap_ci.csv`), and also
separates when read against arms 3 / 4 (ratios ≥ 1.055 with lower CI
bounds ≥ 1.019 across all four cells).

## Denominator share

![Stacked per-family shares of each term's denominator at each arm's `FINAL.pth` snapshot (arm 1 = step 12,500 weights, arm 3 = step 504, arm 4 = step 494); mixed and periodic batches.](plots/gradient_share_stack.png)

`log_neg_cross_batch` holds 0.90–0.99 of `L_pred`'s denominator in the
split shape (arms 1 and 3, periodic and mixed batches). The same tensor
holds 0.003 in arm 4's pooled denominator on both batches, while the
h-anchored families (`log_neg_hh_all` + `log_neg_xs_allt`) hold 0.877
(periodic) / 0.860 (mixed). The measurement thus confirms the pooled
shape's motivating hypothesis — the cross-batch f-anchored family is
crowded out by the two h-anchored families — on arm 4's step-494
backbone. This is the arm the report ships; per issue #374, the card
asks for the measurement to run on arm C, which is a follow-up.

*Measurement (`scripts/gradient_share_measurement.py`; full table
`results/gradient_share_measurement.csv`, 132 rows). Each backbone
checkpoint runs in `.eval()` mode on two fixed batches of B = 64,
T = 4096: "mixed" is the training HF stream, "periodic" is solar / H +
electricity / H windows from GIFT-Eval. Each family's share of its own
term's denominator is `exp(mean(log-family − log-denominator))` over
anchors at τ = 0.10, so segments in one bar need not sum to exactly 1.
Read the reported quantities as the loss landscape a frozen student sees,
not the training-time gradient shares of the MoCo arms: measurement
batch is B = 64 (training used B = 512, and the `log_neg_cross_batch`
count scales with B); `.eval()` disables the 0.70 encoder dropkey and
dropout that reshape h at training time; for the MoCo arms (3, 4) the
keys are student-side at measurement, while training routes them through
the EMA teacher.*

## Training loss

![Total training loss per arm, tail-aligned to zero, log-x from step 100.](plots/loss_curves.png)

Each curve is the run's total training loss, shifted by the mean of
steps 12 401–12 500 so the tails meet at zero. The three InfoNCE arms
(1, 3, 4) and the BYOL-alignment arm (5) optimise different functionals
and their absolute loss levels are not directly comparable; the shape
axis this figure captures is *time to reach the tail* on each objective.
Arms 3 and 4 drop below their own tail level by step ~500 and stay
essentially flat afterwards — consistent with their `best_loss.pth` at
step 504 / 494 in the backbone-step table. Arm 1 keeps a small monotonic
drift after step ~1 000. Arm 5's `L_align + L_rep` reaches its tail
around step 4 000 and continues to fluctuate around it, with the
`best_loss.pth` step at 11 809.

## Arms

| arm | loss shape | `--moco-negatives` | defining feature |
| --- | --- | :-: | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | off | split objective `L = L_pred + L_rep`, equal weight |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | on | split objective; cross-batch f ↔ h keys come from the EMA teacher (MoCo-style) instead of the student |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | on | pooled champion shape with teacher keys |
| arm 5 | `cosine_similarity_batch_rep_only` + `--align-loss-weight 1.0` | off | replace `L_pred` with BYOL-style alignment: `L = L_align + L_rep` (no InfoNCE denominator on the f side) |
| arm C ref | `cosine_similarity_batch_full_hh_negs_xshh_allt` | off | champion (λ_e = 1, λ_h = 1, τ = 0.90) of the earlier sweep, reused without retraining |

Arm 2 was reserved in the issue-card follow-up list (a λ-weighted
variant of the split, `α L_pred + β L_rep`) and was not run in this
experiment; it stays a follow-up card.

**Multi-axis confound.** Arm 1 vs arm C ref changes three things at
once: the loss functional (split vs pooled), `--pos-in-denominator`
(dropped by the split), and `--subtract-contrastive-floor` (dropped by
the split; both flags are derived for the pooled shape and are not
defined for the split). Arm 3 vs arm 4 is the single-axis contrast that
holds the loss functional's shape flags fixed and toggles only split vs
pooled at matched MoCo — that pair is the clean split-vs-pooled read.

Negative families (tensor names from the measurement CSV): the two
f-anchored families are `log_neg_cross_batch` (cross-batch f_t ↔ h′_{t+1})
and `log_neg_zy` (adjacent f_{t+1} ↔ f_t); the three h-anchored families
are `log_neg_hh_all` (within-series all-time h ↔ h), `log_neg_xs_allt`
(cross-series all-time h ↔ h′), and cross-channel `log_neg_xx`, which is
empty at C = 1. f is the forecaster's predicted latent, h the encoder
latent; primes mark other series of the batch. The pooled shape puts all
five families into one denominator; the split routes the two f-anchored
families to `L_pred` and the three h-anchored families to `L_rep`.

Glossary of specialised vocabulary used above: **MoCo** — replaces the
student `h` keys in the cross-batch f ↔ h′ negative with an EMA teacher
`h^T` (slow-moving copy of the encoder). **EMA teacher** — an
exponentially-moving-average shadow of the student encoder with decay
τ = 0.90 that supplies stable positive / key latents.
**BYOL-style alignment** — a negative-free InfoNCE-adjacent objective
that maximises cosine similarity between the student's forecaster latent
and the (teacher-side or stopgrad) encoder latent. **SIGReg** — a
regulariser that pushes the marginal of pooled `e` and pooled `h` toward
uniform on the sphere. **CPC** — a batch-cross InfoNCE auxiliary that
predicts `e` from `h` at matched (b, t) indices.

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
configs against the same seasonal-naive reference file
(`/home/jupyter/workspaces/gift-eval/results/seasonal_naive/all_results.csv`,
also used for the paired-bootstrap CI computation).

## Caveat — single seed, one uncheckable cell

Every evaluation is N = 1. The paired bootstrap above measures within-run
across-task variability; between-seed variance is not measured on this
branch and would need a replicate run to bound. `results_arm4/…_last_6L/all_results.csv`
was on-disk truncated (the first 10 244 bytes were NUL-filled by a lost
write); the file was reconstructed from `summary.txt`, so its
`MASE[0.5]` column exactly reproduces the summary's aggregate 1.1405
but its non-MASE columns are `NaN`. All paired-bootstrap ratios above
depend only on `MASE[0.5]` and are unaffected. Matched-cell point
differences vs the champion, over the sixteen scored cells of arms 1 /
3 / 4 / 5, span −1.9 % (arm 3, 6L / best, best-cell not compute-matched)
to +14.5 % (arm 5, 2L / best); on the four compute-matched `last` cells
alone, point differences vs the champion span +0.5 % (arm 4, 2L / last:
1.1546 vs 1.1491) to +2.7 % (arm 1, 6L / last: 1.1557 vs 1.1254) for
arms 1 / 3 / 4 and reach +12.1 % (arm 5, 2L / last: 1.2883 vs 1.1491)
for arm 5. The best → last spread of the same (arm, head) pair (last
relative to best) at compute-matched `last` step 12 500 ranges from
−3.7 % (arm 5, 2L: 1.2883 vs 1.3374) to +1.5 % (arm 3, 6L: 1.1511 vs
1.1338 = +1.53 %); the +1.17 % at arm 3, 2L (1.1683 vs 1.1548) is
smaller. Arm 4's underfit-backbone `best` cells (backbone step 494)
sit within 0.5 % of arm 1's fully-trained `best` cells at both head
depths, which independently indicates the downstream metric is
insensitive to backbone step at the ±1–3 % margins this experiment
chases; the underfit-backbone control asked for on the review pass
(one head-train + full-97 eval from a randomly-initialised or
early-step backbone) is a follow-up card that would close the
sensitivity question directly.
