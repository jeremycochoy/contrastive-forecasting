# Splitting L_pred + L_rep does not change GM-Relative MASE at any properly controlled comparison of arms 1 / 3 / 4; replacing L_pred with a BYOL alignment (arm 5) makes it worse on every scored evaluation

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

**Answer.** At every arm 1 / 3 / 4 pairwise contrast whose backbones
are matched (both arms at step 12,500, four `last` cells) — split
vs pooled at matched MoCo (arm 3 vs arm 4), MoCo axis at matched split
(arm 1 vs arm 3), joint (arm 1 vs arm 4) — the 20 000-resample
task-level paired-bootstrap 95 % CI on GM-Rel MASE ratio straddles 1
(worst-case task-level lower bound 0.9834, worst-case task-level upper
bound 1.0344); the dataset-clustered bootstrap (resampling 28 base
datasets rather than 97 tasks) widens each interval slightly but does
not move any of the six across 1. Arm 5 (`L_align + L_rep`) is worse
than every other arm on every one of its twelve pairwise contrasts:
ratios in [1.0557, 1.1581] with task-level lower bounds in
[1.0220, 1.1116], and all twelve intervals remain above 1 under the
clustered bootstrap. The one apparent MoCo signal in the panel — 6L / best
arm 1 vs arm 3 = 1.0209 [1.0039, 1.0404] — is not a compute-matched
read: arm 3's `best` checkpoint is its own `best_loss.pth` selection at
backbone step 11,800, and arm 3's own `best → last` swing at 6L
(1.1338 → 1.1511 = +1.53 %) shows that arm 3 at step 11,800 is roughly
1.5 % better than arm 3 at step 12,500 on this head protocol; arm 1's
`best → last` swing on identical weights (see the backbone-step
verification) is −0.16 %, so the head-training-length component is
below 0.2 %. The 2.09 % ratio therefore mixes ≈1.4 % of arm-3-specific
checkpoint-selection benefit with any remaining MoCo effect. At the
same two arms' compute-matched 6L / last row, the CI is 1.0039
[0.9902, 1.0195] — no separation. Twenty-four contrasts tested at 95 %
without a multiplicity correction give ≈1.2 false positives in
expectation; the 6L / best MoCo row's one-sided `p_a_beats_b` = 0.0068
does not survive Bonferroni at 0.05 / 24 = 0.0021. Champion CIs are
absent from this branch (arm C's per-task `all_results.csv` was not
exported into the sweep tree), so the card's primary success criterion
(paired bootstrap vs arm C) is unmet; the arm C row of the table below
is a point reference, not a ranking.

![GM-Relative MASE across arms and (head, checkpoint) scored evaluations.](plots/headline_relmase.png)

![Paired-bootstrap 95 % CIs on GM-Rel MASE ratios across every contrast the report reads. Task-level bootstrap (top per row) and dataset-clustered bootstrap (bottom per row) shown together; * marks checkpoint-selection or step-confounded rows.](plots/ci_forest.png)

## Downstream GM-Relative MASE

*One "scored evaluation" = one arm at one (head depth, checkpoint) —
twenty in total (five arms × four (head, ckpt) pairs). Arm C ref is a
point reference and does not carry a CI on this branch.*

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo) | **1.1548** | 1.1683 | **1.1338** | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | 1.1546 | 1.1603 | 1.1405 |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm C ref (champion, point reference) | 1.1682 | 1.1491 | 1.1561 | 1.1254 |

*GM-Relative MASE: geometric mean, over GIFT-Eval's 97 evaluation configs,
of model MASE divided by seasonal-naive MASE; 1.0 = seasonal-naive, lower
is better. Arms 1 / 3 values are the `Aggregate GM-Relative MASE (97 configs)`
line of each `summary.txt` under `experiments/2026-07-10_split_pred_rep/results/`;
arms 4 / 5 values are the same line under `results_arm4/` and
`results_arm5/`. Arm C values are the four `cross_C` (λ_e = 1,
λ_h = 1, τ = 0.90) rows of
`experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`.
Boldface marks the column minimum across arms 1 / 3 / 4 / 5; no CI
against arm C is available on this branch.*

**Backbone step behind each cell.** The head-training protocol trains
on the arm's `FINAL.pth` for the `best` cell, then resumes on
`final.pth` for the `last` cell. Each launcher's end-of-training block
is `cp best_loss.pth → FINAL.pth`, falling through to `final.pth` if
`best_loss.pth` is absent (or if `final.pth` was written later without
a matching `best_loss.pth` save). The step behind each arm's
`FINAL.pth` is determined by (a) md5 which file `FINAL.pth` is a copy
of, and (b) for arms whose `FINAL.pth` matches `best_loss.pth`, the
last `Saved …_best_loss.pth` event in the run log
(`best_loss.pth` saves on smoothed loss on 100-step boundaries, so
`argmin` of the raw `_losses.csv` `loss` column does not identify the
file's step). All four backbone logs are committed; the
verification script `scripts/verify_backbone_steps.sh` re-runs both
checks and writes `results/backbone_step_verification.log`:

| arm | `best` cell backbone step | `last` cell backbone step | source |
| --- | --: | --: | --- |
| arm 1 (split) | 12,500 | 12,500 | `FINAL.pth` md5 = `final.pth`; `torch.equal` across all 193 tensors also holds vs `12k.pth`, so arm 1's backbone did not update in the last 500 steps. Arm 1's `best_loss.pth` on disk is a pre-resume artefact (0 post-resume saves in the run log) and was not the cp source. |
| arm 3 (split + MoCo) | 11,800 | 12,500 | `FINAL.pth` md5 = `best_loss.pth`; run log's last `_best_loss.pth` save is step 11,800 (15 saves total). |
| arm 4 (pooled + MoCo) | 600 | 12,500 | `FINAL.pth` md5 = `best_loss.pth`; run log's last `_best_loss.pth` save is step 600 (6 saves total, all in [100, 600]). |
| arm 5 (`L_align` + `L_rep`) | 11,800 | 12,500 | `FINAL.pth` md5 = `best_loss.pth`; run log's last `_best_loss.pth` save is step 11,800 (40 saves total). |
| arm C ref (champion) | *not exported to this branch* | 12,500 | sweep protocol; `best_loss.pth` step not in `gm_table.csv`. |

**Head-adaptation asymmetry across the `last` column.** The head
trains 30 000 steps on `FINAL.pth`, then 10 000 more on `final.pth`.
For arm 1, `FINAL.pth` == `final.pth` (weight-identical), so arm 1's
`last` cell head-trained 40 000 steps on the evaluated backbone. For
arms 3 / 4 / 5, whose `FINAL.pth` is `best_loss.pth`, the `last` cell
head-trained 30 000 steps on a different backbone (step 11 800 / 600 /
11 800) and only 10 000 on the evaluated one. `last`-cell contrasts are
therefore backbone-step-matched but head-adaptation-asymmetric.

## Paired-bootstrap 95 % CI on GM-Relative MASE ratios

Panel of contrasts, task-level bootstrap and dataset-clustered
bootstrap side by side. 20 000 resamples, seed 42, seasonal-naive
divisor at `results/seasonal_naive_all_results.csv` (sha256
`d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949`);
the divisor cancels in the paired ratio. Driver:
`scripts/build_ci_panel.py`. Full 24-row output CSVs:
`pairwise_bootstrap_ci.csv` (task), `pairwise_bootstrap_ci_clustered.csv`
(dataset-clustered), `pairwise_bootstrap_ci_periodic.csv` (37-config
periodic subset). Ratio `A/B < 1` means arm A beats arm B.

| cell | contrast | axis toggled | backbone steps (A, B) | ratio A/B | 95 % CI task | 95 % CI clustered |
| --- | --- | --- | --- | --: | --- | --- |
| 2L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 12,500, 12,500 | 1.0119 | [0.9970, 1.0267] | [0.9939, 1.0294] |
| 6L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 12,500, 12,500 | 1.0093 | [0.9960, 1.0269] | [0.9956, 1.0275] |
| 2L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 12,500, 12,500 | 0.9988 | [0.9834, 1.0158] | [0.9801, 1.0176] |
| 6L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 12,500, 12,500 | 1.0039 | [0.9902, 1.0195] | [0.9890, 1.0198] |
| 2L / last | arm 1 vs arm 4 | joint | 12,500, 12,500 | 1.0107 | [0.9963, 1.0262] | [0.9957, 1.0260] |
| 6L / last | arm 1 vs arm 4 | joint | 12,500, 12,500 | 1.0133 | [0.9957, 1.0344] | [0.9935, 1.0356] |
| 6L / best* | arm 1 vs arm 3 | MoCo — ckpt-selection confound | 12,500, 11,800 | 1.0209 | [1.0039, 1.0404] | [1.0051, 1.0393] |
| 6L / best* | arm 3 vs arm 4 | split — 11,200-step gap | 11,800, 600 | 0.9771 | [0.9571, 0.9951] | [0.9553, 0.9948] |
| 2L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 12,500, 12,500 | 1.1041 | [1.0632, 1.1473] | [1.0615, 1.1447] |
| 6L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 12,500, 12,500 | 1.0557 | [1.0220, 1.0892] | [1.0203, 1.0894] |

Rows marked `*` are `best` cells with a checkpoint-selection or
step confound and are shown for completeness; no compute-matched
arm 1 / 3 / 4 contrast separates from 1 at 95 % task-level or
dataset-clustered bootstrap. Arm 5 vs arms 1 / 3 / 4 (12 rows in the
full CSV) span task-level ratios [1.0557, 1.1581] with lower bounds
[1.0220, 1.1116], all above 1 under both resampling schemes.
Twenty-four contrasts tested at 95 % without a multiplicity control:
Bonferroni-adjusted α = 0.05 / 24 = 0.0021 leaves every arm 5 row
(`p_a_beats_b` ∈ {0.0000, 0.0009, 0.9991, 1.0000} two-tailed) but
kills the 6L / best arm 1 vs arm 3 row (`p_a_beats_b` = 0.0068
one-sided).

**Periodic-subset secondary read** (37 configs — `solar/`, `electricity/`,
`ett1/`, `m4_hourly/`, `bizitobs_*` — the cluster the card names as
the loss-shape hypothesis' target). Every arm 1 / 3 / 4 pairwise
contrast on this subset gives a task-level CI that straddles 1
(worst-case lower bound 0.9803, worst-case upper bound 1.0871); the
only row above 1 is 6L / last arm 1 vs arm 4 = 1.0381 [1.0010, 1.0871],
whose lower bound is 1.0010 and does not survive multiplicity. Arm 5
regresses on all four periodic contrasts vs arm 4 and vs arm 3 (four
rows above 1), and the arm-5-vs-arm-1 6L cells are on the edge:
2L / last 0.9143 [0.8565, 0.9799], 6L / last 0.9829 [0.9277, 1.0492]
— 6L / last arm 5 vs arm 1 straddles 1 on the periodic subset, unlike
the full-97 case. The full 24-row periodic CSV is
`pairwise_bootstrap_ci_periodic.csv`.

## Denominator share

![Stacked per-family shares of each term's denominator at each arm's `FINAL.pth` snapshot (arm 1: step 12,500 weights; arm 3: step 11,800; arm 4: step 600); mixed and periodic batches.](plots/gradient_share_stack.png)

`log_neg_cross_batch` (cross-batch f_t ↔ h'_{t+1}) holds 0.86–1.00 of
`L_pred`'s denominator on arm 1's trajectory (2k / 5k / 12k /
FINAL — mixed: 0.858 → 0.873 → 0.901 → 0.901; periodic: 0.979 → 0.984
→ 0.991 → 0.991) and arm 3's `_FINAL.pth` snapshot (mixed 0.937,
periodic 0.997). The same tensor holds 0.003 in arm 4's pooled
denominator at step 600 while the h-anchored families
(`log_neg_hh_all` + `log_neg_xs_allt`) hold 0.877 (periodic) / 0.860
(mixed); the pattern is stable across arm 4's earlier saves (at
step 10 000: cross_batch 0.004, hh_all + xs_allt 0.867 periodic /
0.913 mixed), so the split shape's motivating hypothesis (the
cross-batch f-anchored family sits at a sub-percent share of the
pooled denominator) is measured on both an underfit and a trained
pooled backbone.

*Measurement (`scripts/gradient_share_measurement.py`; full table
`results/gradient_share_measurement.csv`, 132 rows). Each backbone
checkpoint runs in `.eval()` mode on two fixed batches of B = 64,
T = 4096: "mixed" is the training HF stream, "periodic" is solar / H +
electricity / H windows from GIFT-Eval. Each family's share of its own
term's denominator is `exp(mean(log-family − log-denominator))` over
anchors at τ = 0.10, so segments in one bar need not sum to exactly 1.
Read the reported quantities as the loss landscape a frozen student
sees, not the training-time gradient shares of the MoCo arms:
measurement batch is B = 64 (training used B = 512, and the
`log_neg_cross_batch` count scales with B); `.eval()` disables the
0.70 encoder dropkey and dropout that reshape h at training time; for
the MoCo arms (3, 4) the keys are student-side at measurement, while
training routes them through the EMA teacher. The card also asks for
this measurement on arm C — that is a follow-up.*

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

**Confound.** The split's `L_pred` is normalized InfoNCE by
construction (positive in denominator), so `--pos-in-denominator` is a
no-op for the split; `--subtract-contrastive-floor` is supported by
the split (it subtracts `f_pred + f_rep`, a constant, and is
gradient-neutral). Arm 1 vs arm C ref therefore differs on one
effective axis — the loss functional (split vs pooled). Arm 3 vs arm 4
is the same functional axis with MoCo held fixed on both sides; arm 1
vs arm 3 is the MoCo axis with the split shape held fixed on both
sides.

Negative families (tensor names from the measurement CSV): the two
f-anchored families are `log_neg_cross_batch` (cross-batch f_t ↔ h′_{t+1})
and `log_neg_zy` (adjacent f_{t+1} ↔ f_t); the three h-anchored families
are `log_neg_hh_all` (within-series all-time h ↔ h), `log_neg_xs_allt`
(cross-series all-time h ↔ h′), and cross-channel `log_neg_xx`, which is
empty at C = 1. f is the forecaster's predicted latent, h the encoder
latent; primes mark other series of the batch. The pooled shape puts
all five families into one denominator; the split routes the two
f-anchored families to `L_pred` and the three h-anchored families to
`L_rep`.

Glossary of specialised vocabulary used above: **MoCo** — replaces the
student `h` keys in the cross-batch f ↔ h′ negative with an EMA teacher
`h^T` (slow-moving copy of the encoder). **EMA teacher** — an
exponentially-moving-average shadow of the student encoder with decay
τ = 0.90 that supplies stable positive / key latents.
**BYOL-style alignment** — a negative-free InfoNCE-adjacent objective
that maximises cosine similarity between the student's forecaster
latent and the (teacher-side or stopgrad) encoder latent.
**SIGReg** — a regulariser that pushes the marginal of pooled `e` and
pooled `h` toward uniform on the sphere. **CPC** — a batch-cross
InfoNCE auxiliary that predicts `e` from `h` at matched (b, t)
indices.

## Method

Each arm trains one backbone with the champion recipe (12,500 steps,
B = 512, T = 4096, C = 1, lr 1e-3, seed 20260520, dataset
`gift-pretrain-full-4096 / small_v1`, EMA teacher τ = 0.90,
contrastive τ = 0.10, SIGReg λ_e = λ_h = 1, CPC auxiliary loss). The
arms differ in `--loss-shape` and `--moco-negatives`; the pooled arm
additionally keeps the champion's
`--pos-in-denominator --subtract-contrastive-floor`. For each backbone
a quantile probe head (2 or 6 layers) is trained for 30 000 steps on
`FINAL.pth`, then for 10 000 more steps — resuming the same head — on
`final.pth` (step 12,500). Each head is evaluated on GIFT-Eval's 97
configs against the same seasonal-naive reference file, committed to
this branch at `results/seasonal_naive_all_results.csv`.

## Caveat — single seed

Every evaluation is N = 1. The paired bootstrap measures within-run
across-task variability; between-seed variance is not measured on this
branch and would need a replicate run to bound. `paired_bootstrap.py`
runs one pair at a time; the twenty-four rows in the panel come from
`scripts/build_ci_panel.py`, which drives it. Arm 1's `best` and `last`
cells run on identical weights (md5 + `torch.equal` across all 193
tensors); the resulting best → last swing (2L: 1.1654 → 1.1669 = +0.13
%; 6L: 1.1575 → 1.1557 = −0.16 %) bounds the head-training-length
noise floor at ≤ 0.2 % — smaller than every non-arm-5 point difference
in the table. Arm 4's step-600 `best` cells scoring 1.1602 / 1.1603 —
within 0.5 % of arm 1's step-12,500 `best` cells — is either a
property of the objective at step 600 or evidence that the downstream
metric does not resolve backbones at these margins; the random-init or
early-step underfit-backbone control is a follow-up.
`results_arm4/…_last_6L/all_results.csv` carries `MASE[0.5]` only
(reconstructed from `summary.txt`; other columns `NaN`); paired
bootstrap uses `MASE[0.5]` alone, so the CIs are unaffected.
