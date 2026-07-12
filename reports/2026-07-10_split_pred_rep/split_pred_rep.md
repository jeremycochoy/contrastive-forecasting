# On the full 97-config GM-Relative MASE no split-vs-pooled or MoCo contrast between arms 1 / 3 / 4 separates from 1 under Bonferroni control; on the medium+long horizon subset the compute-matched arm 3 vs arm 4 contrast is nominally in the direction of pooled better than split (2L: 1.0228, 6L: 1.0140; neither survives Bonferroni); replacing L_pred with a BYOL alignment (arm 5) regresses on every scored evaluation

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

**Answer.** All CIs below are 20 000-resample paired bootstraps over the
97 GIFT-Eval configs, seed 42, task-level unless a dataset-clustered
one is quoted alongside; nominal 95 % coverage per contrast, Bonferroni
threshold α = 0.05 / 24 = 0.0021 across the 24 pairwise contrasts (see
panel below). At the four compute-matched (arms 1 / 3 / 4 / 5 all at
backbone step 12,500) `last` cells, every arm-1 / 3 / 4 pairwise CI on
the full 97 configs straddles 1 under both the task-level and the
28-dataset-clustered bootstrap; worst-case task-level lower bound
0.9834, worst-case upper bound 1.0344. On the 42-config medium+long
horizon subset — the card's secondary read — the single-axis
split-vs-pooled contrast (arm 3 vs arm 4) points to pooled better than
split at both head depths: 2L / last 1.0228 [1.0059, 1.0403]
(one-sided `p_a_beats_b` = 0.0042), 6L / last 1.0140 [1.0031, 1.0252]
(one-sided `p_a_beats_b` = 0.0064); the direction is consistent with
the point estimates on the full-97 and periodic subsets (arm 3 vs
arm 4 point ratio > 1 in all six cases). Neither medium+long row
survives multiplicity control: two-sided p = 0.0084 / 0.0128 vs
Bonferroni α = 0.0021. The `L_align + L_rep` arm (arm 5) is worse than
every other arm on every one of its twelve full-97 pairwise contrasts:
ratios in [1.0557, 1.1581] with task-level lower bounds in
[1.0220, 1.1116]. The one MoCo-looking row in the panel — 6L / best
arm 1 vs arm 3 = 1.0209 [1.0039, 1.0404] — is checkpoint-selection
confounded: arm 3's `best` cell is its own `best_loss.pth` selection at
step 11,800, and arm 3's own `best → last` swing at 6L on a matched
head protocol is +1.53 %, so ≈1.4 % of the 2.09 % ratio is
arm-3-specific checkpoint selection and only the residual is a
candidate MoCo effect (whose sign is not measurable in this contrast).
Champion CIs are absent from this branch (arm C's per-task
`all_results.csv` was not exported into the sweep tree), so the card's
primary success criterion (paired bootstrap vs arm C) is unmet; the
arm C row in the table below is a point reference, not a ranking.

![GM-Relative MASE across arms and (head, checkpoint) scored evaluations.](plots/headline_relmase.png)

![Paired-bootstrap 95 % CIs on GM-Rel MASE ratios across every contrast the report reads. Task-level bootstrap (top per row) and dataset-clustered bootstrap (bottom per row) shown together; * marks checkpoint-selection or step-confounded rows.](plots/ci_forest.png)

## Downstream GM-Relative MASE

*One "scored evaluation" = one arm at one (head depth, checkpoint) —
twenty in total (five arms × four (head, ckpt) pairs). Arm C ref is a
point reference and carries no CI on this branch.*

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo) | **1.1548** | 1.1683 | **1.1338** | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | **1.1546** | 1.1603 | **1.1405** |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm C ref (champion, point reference) | 1.1682 | 1.1491 | 1.1561 | 1.1254 |

*Boldface marks the column minimum across arms 1 / 3 / 4 / 5 at each
(head, checkpoint). Arms 1 / 3 values are the `Aggregate GM-Relative
MASE (97 configs)` line of each `summary.txt` under
`experiments/2026-07-10_split_pred_rep/results/`; arms 4 / 5 values are
the same line under `results_arm4/` / `results_arm5/`. Arm C values are
the four `cross_C` (λ_e = 1, λ_h = 1, τ = 0.90) rows of
`experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`.*

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
checks and writes `results/backbone_step_verification.log`, including a
`torch.equal` cross-check on arm 1's 193-tensor state dict:

| arm | `best` cell backbone step | `last` cell backbone step | source |
| --- | --: | --: | --- |
| arm 1 (split) | 12,500 | 12,500 | `FINAL.pth` md5 = `final.pth`; the verification log records `torch.equal(FINAL, 12k) = True`, so arm 1's backbone did not update in the last 500 steps. Arm 1's `best_loss.pth` on disk is a pre-resume artefact (0 post-resume saves in the run log) and was not the cp source. |
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
11 800) and only 10 000 on the evaluated one. `last`-cell contrasts
are therefore backbone-step-matched but head-adaptation-asymmetric —
except **arm 3 vs arm 4**, which is matched on both axes (both
head-trained 30 000 on their own `best_loss.pth` at step 11 800 / 600
and then 10 000 on step 12 500).

## Paired-bootstrap 95 % CI on GM-Relative MASE ratios

Panels of contrasts, task-level bootstrap plus dataset-clustered
bootstrap side by side. 20 000 resamples, seed 42, seasonal-naive
divisor at `results/seasonal_naive_all_results.csv` (sha256
`d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949`);
the divisor cancels in the paired ratio. Driver:
`scripts/build_ci_panel.py`. Full 24-row output CSVs:
`pairwise_bootstrap_ci.csv` (task-level over 97 configs),
`pairwise_bootstrap_ci_clustered.csv` (28 base datasets),
`pairwise_bootstrap_ci_periodic.csv` (37-config periodic subset,
selected by family prefix), and
`pairwise_bootstrap_ci_medlong.csv` (42-config medium+long horizon
subset). Ratio `A/B < 1` means arm A beats arm B.

### Full-97 configs

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
step confound. None of the six full-97 compute-matched arm-1 / 3 / 4
CIs separates from 1 at nominal 95 % under either scheme. Arm 5 vs
arms 1 / 3 / 4 spans all twelve rows above 1: task-level ratios
[1.0557, 1.1581], task-level lower bounds [1.0220, 1.1116], and the
same twelve rows above 1 under the clustered scheme.

### Periodic-cluster subset (37 configs — `solar/`, `electricity/`, `ett1/`, `m4_hourly/`, `bizitobs_*`)

Selected by family prefix (not by `rel_MASE ≥ 1.25`, so the subset
does not condition on the outcome). All twelve arm-1 / 3 / 4 CIs
straddle 1 under both schemes; worst-case task-level lower bound
**0.9493** (6L / best, arm 3 vs arm 4), worst-case task-level upper
bound **1.0871** (6L / last, arm 1 vs arm 4). Six of the eight
arm-5-vs-arm-3 / arm-4 CIs sit above 1; two straddle:
6L / best arm 5 vs arm 4 = 1.0785 [0.9830, 1.1655] and
6L / last arm 5 vs arm 3 = 1.0315 [0.9735, 1.0859]. Full 24 rows in
`pairwise_bootstrap_ci_periodic.csv`.

### Medium+long horizon subset (42 configs — every `dataset/*/{medium,long}`)

The card's secondary read. Every arm-1 / 3 / 4 compute-matched
contrast on this subset:

| cell | contrast | ratio A/B | 95 % CI task | one-sided `p_a_beats_b` |
| --- | --- | --: | --- | --: |
| 2L / last | arm 3 vs arm 4 | 1.0228 | [1.0059, 1.0403] | 0.0042 |
| 6L / last | arm 3 vs arm 4 | 1.0140 | [1.0031, 1.0252] | 0.0064 |
| 2L / last | arm 1 vs arm 3 | 0.9717 | [0.9521, 0.9926] | 0.9951 |
| 6L / last | arm 1 vs arm 3 | 0.9833 | [0.9668, 1.0009] | 0.9690 |
| 2L / last | arm 1 vs arm 4 | 0.9939 | [0.9757, 1.0132] | 0.7381 |
| 6L / last | arm 1 vs arm 4 | 0.9971 | [0.9799, 1.0150] | 0.6232 |

Two rows separate at nominal 95 % — arm 3 vs arm 4 at both head
depths, in the direction pooled better than split (arm 3 = split
worse). Neither survives Bonferroni at α = 0.05 / 24 = 0.0021 (two-sided
p = 0.0084 and 0.0128). Arm 1 vs arm 3 at 2L / last separates in the
direction no-MoCo better than MoCo (0.9717 [0.9521, 0.9926]); at 6L / last
the same contrast straddles 1 by 0.0009 (0.9833 [0.9668, 1.0009]).
Full 24 rows in `pairwise_bootstrap_ci_medlong.csv`.

### Multiplicity control

The 24 contrasts tested at nominal 95 % give ≈1.2 false positives in
expectation. Bonferroni-adjusted α = 0.05 / 24 = 0.0021. The stored
`p_a_beats_b` column in every CSV is the one-sided quantity `P(ratio < 1
under bootstrap)`; the two-sided p is `2 × min(p, 1 − p)`. The twelve
arm-5 rows carry stored one-sided values in {0.9991, 0.99955, 0.99995,
1.0000}; the corresponding two-sided p values are {0.0018, 0.0009,
0.0001, 0.0000}, so all twelve arm-5 contrasts clear Bonferroni — the
smallest margin is 0.0003 (0.0018 vs α = 0.0021). No other contrast
in any of the four panels clears Bonferroni.

## Denominator share

![Stacked per-family shares of each term's denominator at each arm's `FINAL.pth` snapshot (arm 1: step 12,500 weights; arm 3: step 11,800; arm 4: step 600); mixed and periodic batches.](plots/gradient_share_stack.png)

`log_neg_cross_batch` (cross-batch f_t ↔ h'_{t+1}) holds 0.86–1.00 of
`L_pred`'s denominator on arm 1's trajectory (2k / 5k / 12k /
FINAL — mixed: 0.858 → 0.873 → 0.901 → 0.901; periodic: 0.979 → 0.984
→ 0.991 → 0.991) and arm 3's `_FINAL.pth` snapshot (mixed 0.937,
periodic 0.997). The same tensor holds 0.003 in arm 4's pooled
denominator at step 600 while the h-anchored families
(`log_neg_hh_all` + `log_neg_xs_allt`) hold 0.877 (periodic) / 0.860
(mixed). The pattern is stable on a trained pooled backbone as well
— arm 4's step-10 000 checkpoint gives cross_batch 0.004,
hh_all + xs_allt 0.867 (periodic) / 0.913 (mixed); the split shape's
motivating hypothesis (that the cross-batch f-anchored family sits at
a sub-percent share of the pooled denominator) is measured on both an
underfit and a trained pooled backbone.

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
branch and would need a replicate run to bound. Arm 1's `best` and
`last` cells run on identical weights (md5 + `torch.equal` across all
193 tensors, from `backbone_step_verification.log`); its best → last
swing (2L: 1.1654 → 1.1669 = +0.13 %; 6L: 1.1575 → 1.1557 = −0.16 %)
bounds the head-training-length component at ≤ 0.2 % under an
identical backbone. Arm 4's step-600 `best` cells score 1.1602 / 1.1603
— within 0.5 % of arm 1's step-12,500 `best` cells; the random-init or
early-step underfit-backbone control that would resolve whether that is
an objective-early-fit effect or a metric-insensitivity effect is a
follow-up. `results_arm4/…_last_6L/all_results.csv` carries `MASE[0.5]`
only (reconstructed from `summary.txt`; other columns `NaN`); paired
bootstrap uses `MASE[0.5]` alone, so the CIs are unaffected. The
card's primary success criterion (paired bootstrap vs arm C) is unmet
on this branch: arm C's per-task `all_results.csv` was not exported
into the sweep tree, and computing a champion-denominated CI needs
exactly that file.
