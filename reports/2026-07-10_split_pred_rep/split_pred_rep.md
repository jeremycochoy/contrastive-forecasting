# Splitting L_pred + L_rep does not improve GM-Rel MASE at any compute-matched arm 1 / 3 / 4 comparison (arm 3 vs arm 4: 1.0093 [0.9960, 1.0269] at 6L / last); MoCo separates from 1 at the near-compute-matched 6L / best pair (arm 1 vs arm 3: 1.0209 [1.0039, 1.0404]); `L_align + L_rep` regresses across every scored evaluation

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

**Answer.** Take one arm's evaluation at (head depth, checkpoint) as
one "scored evaluation" of that arm. The four scored evaluations of arm
C ref (2L / best, 2L / last, 6L / best, 6L / last) and the sixteen of
arms 1 / 3 / 4 / 5 make twenty in total. At compute-matched pairings —
arm 1, arm 3, arm 4 and arm 5 all at backbone step 12,500 in their
`last` scored evaluation, and arm 1 vs arm 3 at 6L / best sit 700 steps
apart (12,500 vs 11,800) — a 20 000-resample paired bootstrap over the
97 GIFT-Eval configs gives every arm 1 / arm 3 / arm 4 pairwise ratio
at `last` as a 95 % CI that straddles 1: split-vs-pooled (arm 3 vs arm 4)
= 1.0119 [0.9970, 1.0267] at 2L and 1.0093 [0.9960, 1.0269] at 6L; MoCo
axis (arm 1 vs arm 3) = 0.9988 [0.9834, 1.0158] at 2L and 1.0039
[0.9902, 1.0195] at 6L. The near-compute-matched 6L / best pair
arm 1 vs arm 3 separates: 1.0209 [1.0039, 1.0404], `p_a_beats_b` =
0.0068 — MoCo beats no-MoCo by roughly two percent at 6L / best with
the split shape fixed. The 6L / best arm 3 vs arm 4 = 0.9771
[0.9571, 0.9951] also separates but sits at 11,800 vs 600 backbone
steps, so the split axis is confounded with an 11 200-step backbone
gap. Dropping the InfoNCE denominator on the f side (arm 5,
`L_align + L_rep`) is a real regression: arm 5 vs arm 1 = 1.0557
[1.0220, 1.0891] at 6L / last, 1.1041 [1.0632, 1.1473] at 2L / last,
and every one of the twelve arm 5 CIs in the committed table sits
above 1. Champion CIs are absent — arm C's per-task
`all_results.csv` is not on this branch — so the point differences vs
arm C in the table are bare margins, not intervals.

![GM-Relative MASE across arms and (head, checkpoint) scored evaluations.](plots/headline_relmase.png)

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
is better. Arms 1 / 3 values are the `Aggregate GM-Relative MASE (97 configs)`
line of each `summary.txt` under `experiments/2026-07-10_split_pred_rep/results/`;
arms 4 / 5 values are the same line under `results_arm4/` and
`results_arm5/`. Champion values are the four `cross_C` (λ_e = 1,
λ_h = 1, τ = 0.90) rows of
`experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`.
Boldface marks the column minimum across all arms. Point-difference
rankings by themselves cannot be read as evidence at this seed count —
see the paired-bootstrap subsection below.*

**Backbone step behind each cell.** The head-training protocol trains on
the arm's `FINAL.pth` for the `best` cell, then resumes on `final.pth`
for the `last` cell. Each launcher's end-of-training block is
`cp best_loss.pth → FINAL.pth`, falling through to `final.pth` if
`best_loss.pth` is absent (or if `final.pth` overwrites in a later run).
The step behind each arm's `FINAL.pth` is determined by (a) md5 which
file `FINAL.pth` is a copy of, and (b) for arms whose `FINAL.pth`
matches `best_loss.pth`, the last `Saved …_best_loss.pth` event in
the arm's run log (`best_loss.pth` saves on smoothed loss on 100-step
boundaries, so `argmin` of the raw `_losses.csv` `loss` column does
not identify the file's step). All four backbone logs are committed
under `results/` and `results_arm4/` / `results_arm5/`; the
verification script `scripts/verify_backbone_steps.sh` re-runs both
checks and writes `results/backbone_step_verification.log`:

| arm | `best` cell backbone step | `last` cell backbone step | source |
| --- | --: | --: | --- |
| arm 1 (split) | 12,500 | 12,500 | `FINAL.pth` md5 = `final.pth`; `torch.equal` across all 193 tensors also holds vs `12k.pth`, so arm 1's backbone did not update in the last 500 steps. Arm 1's `best_loss.pth` on disk has no post-resume save event, so it is a pre-resume artefact and was not the source of `FINAL.pth`. |
| arm 3 (split + MoCo) | 11,800 | 12,500 | `FINAL.pth` md5 = `best_loss.pth`; run log's last `_best_loss.pth` save is step 11,800 (15 saves total, ending at step 11,800). |
| arm 4 (pooled + MoCo) | 600 | 12,500 | `FINAL.pth` md5 = `best_loss.pth`; run log's last `_best_loss.pth` save is step 600 (6 saves total, all in [100, 600]). |
| arm 5 (`L_align` + `L_rep`) | 11,800 | 12,500 | `FINAL.pth` md5 = `best_loss.pth`; run log's last `_best_loss.pth` save is step 11,800 (40 saves total, ending at step 11,800). |
| arm C ref (champion) | *not exported to this branch* | 12,500 | sweep protocol; `best_loss.pth` step not in `gm_table.csv`. |

Consequences for reading the table:

- The four arms' `last` cells (arm 1, arm 3, arm 4, arm 5 all at step
  12,500) are the only cross-arm compute-matched read for the split /
  MoCo axes.
- The 6L / best arm 1 vs arm 3 pair (12,500 vs 11,800) sits 700 steps
  apart, i.e. near-compute-matched; the arm 4 `best` cells (step 600)
  are heavily underfit and their 1.1602 / 1.1603 downstream — within
  0.5 % of arm 1's step-12,500 `best` cells — is measured but not
  interpreted here; the random-init or early-step underfit-backbone
  control is a follow-up card that would say whether the metric is
  resolving backbones at the 1–3 % margins this experiment chases.

## Paired-bootstrap 95 % CI on GM-Relative MASE ratios

20 000 resamples over the 97 configs, seed 42, using each arm's per-task
`MASE[0.5]` normalised by the shared seasonal-naive reference file
(`results/seasonal_naive_all_results.csv`, sha256
`d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949`,
copied into the branch for reproducibility). The seasonal-naive divisor
cancels in the paired *ratio* `GM(a)/GM(b)`, so the CI is unchanged if
any 97-config baseline is substituted; the divisor is used only to
report per-arm `GM-Rel MASE` levels in the CSV. Ratio `A/B < 1` means
arm A beats arm B. `arm C` is absent because its per-task
`all_results.csv` is not on this branch. Compute-matched or near-
compute-matched pairs, plus every row where the CI excludes 1:

| cell | contrast | axis toggled | backbone steps (A, B) | ratio A/B | 95 % CI | separates from 1? |
| --- | --- | --- | --- | --: | --- | :-: |
| 2L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 12,500, 12,500 | 1.0119 | [0.9970, 1.0267] | no |
| 6L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 12,500, 12,500 | 1.0093 | [0.9960, 1.0269] | no |
| 2L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 12,500, 12,500 | 0.9988 | [0.9834, 1.0158] | no |
| 6L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 12,500, 12,500 | 1.0039 | [0.9902, 1.0195] | no |
| 2L / last | arm 1 vs arm 4 | joint (split + no-MoCo ↔ pooled + MoCo) | 12,500, 12,500 | 1.0107 | [0.9963, 1.0262] | no |
| 6L / last | arm 1 vs arm 4 | joint | 12,500, 12,500 | 1.0133 | [0.9957, 1.0344] | no |
| 6L / **best** | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 12,500, 11,800 | 1.0209 | [1.0039, 1.0404] | **yes (arm 3 better)** — near-compute-matched |
| 6L / **best** | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 11,800, 600 | 0.9771 | [0.9571, 0.9951] | yes — but step-confounded (11,200-step gap) |
| 2L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 12,500, 12,500 | 1.1041 | [1.0632, 1.1473] | **yes (arm 5 worse)** |
| 6L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 12,500, 12,500 | 1.0557 | [1.0220, 1.0891] | **yes (arm 5 worse)** |

All 24 pairwise rows are committed in
`experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci.csv`.
Every arm 5-vs-anyone row separates (arm 5 worse); the eight arm 5
`last` and `best` cells against arms 1 / 3 / 4 give ratios in
[1.0557, 1.1476] with lower CI bounds in [1.0220, 1.0953].

## Denominator share

![Stacked per-family shares of each term's denominator at each arm's `FINAL.pth` snapshot (arm 1: step 12,500 weights; arm 3: step 11,800; arm 4: step 600); mixed and periodic batches.](plots/gradient_share_stack.png)

`log_neg_cross_batch` (cross-batch f_t ↔ h'_{t+1}) holds 0.90–1.00 of
`L_pred`'s denominator on arm 1 / arm 3's `_FINAL.pth` snapshots and
across arm 1's whole trajectory 2k / 5k / 12k / FINAL. The same tensor
holds 0.003 in arm 4's pooled denominator at step 600, while the
h-anchored families (`log_neg_hh_all` + `log_neg_xs_allt`) hold 0.877
(periodic) / 0.860 (mixed). The gradient_share CSV also carries arm 4
at 2k / 5k / 10k, and the pattern is stable across those trained
checkpoints too — arm 4 at step 10 000: cross_batch 0.004, hh_all + xs_allt
= 0.867 (periodic) / 0.913 (mixed) — so the split shape's motivating
hypothesis (that pooling leaves the cross-batch f-anchored family at a
sub-percent share of the pooled denominator) is measured on both an
underfit and a trained pooled backbone.

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
the EMA teacher. The card also asks for this measurement on arm C —
that is a follow-up.*

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
experiment.

**Confound.** The split shape's `L_pred` is normalized InfoNCE by
construction (positive in denominator), so `--pos-in-denominator` is
rejected as a no-op for the split; `--subtract-contrastive-floor` is
supported by the split (it subtracts `f_pred + f_rep`, a constant, and
is gradient-neutral). Arm 1 vs arm C ref therefore differs on one
effective axis — the loss functional (split vs pooled). Arm 3 vs arm 4
is the same functional axis with MoCo held fixed on both sides; arm 1
vs arm 3 is the MoCo axis with the split shape held fixed on both
sides. Arm 1 vs arm 4 changes both.

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
the champion's `--pos-in-denominator --subtract-contrastive-floor` (the
split accepts only the latter and it is gradient-neutral). For each
backbone a quantile probe head (2 or 6 layers) is trained for 30,000
steps on `FINAL.pth`, then for 10,000 more steps — resuming the same
head — on `final.pth` (step 12,500). Each head is evaluated on
GIFT-Eval's 97 configs against the same seasonal-naive reference file,
committed to this branch at `results/seasonal_naive_all_results.csv`
(sha256 `d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949`).

## Caveat — single seed

Every evaluation is N = 1. The paired bootstrap above measures
within-run across-task variability; between-seed variance is not
measured on this branch and would need a replicate run to bound. Arm
4's step-600 `best` cells score 1.1602 / 1.1603 — within 0.5 % of arm
1's step-12,500 `best` cells; the underfit-backbone control that would
say whether the metric is insensitive to backbone step or whether
arm 4's objective simply reaches useful representations early is a
follow-up. `results_arm4/…_last_6L/all_results.csv` was rewritten
after the on-disk file was found NUL-truncated: its `MASE[0.5]` column
reproduces `summary.txt`'s aggregate 1.1405 exactly, all other columns
are `NaN`; all paired-bootstrap ratios depend only on `MASE[0.5]` and
are unaffected. Arm-vs-champion deltas in the table remain point
differences because arm C's per-task `all_results.csv` is not on this
branch.
