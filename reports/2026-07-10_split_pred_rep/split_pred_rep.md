# Downstream GM-Relative MASE does not resolve arm 4's step-600 backbone from arm 1 / arm 3's step-12,500 backbones on medium+long horizons; no compute-matched arm 1 / 3 / 4 contrast on the full 97 configs clears Bonferroni; arm 5 (`L_align + L_rep`) regresses on every scored evaluation

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

**Answer.** At the compute-matched arms 1 / 3 / 4 `last` cells (all four
backbones at step 12 500) no pairwise CI on the full 97 configs separates
from 1 under the report's Bonferroni threshold (α = 0.05 / 24 = 0.0021
across the 24 arm-pair × 4 (head, ckpt) contrasts in
`pairwise_bootstrap_ci.csv`), and none separates at nominal 95 % under
the 28-dataset-clustered bootstrap either. On the 42-config
medium+long horizon subset — the card's secondary read — the
single-axis arm 3 vs arm 4 contrast points to pooled better than split
at both head depths at nominal 95 % (2L / last 1.0228 [1.0059, 1.0403]
one-sided p = 0.0042; 6L / last 1.0140 [1.0031, 1.0252] one-sided p =
0.0064); neither survives Bonferroni. At the same medium+long subset's
`best` cells, arm 4's step-600 backbone downstream is nominally better
than trained backbones' downstream on three of four contrasts, and two
of those three clear Bonferroni: 2L / best arm 3 (11 800) vs arm 4 (600)
= 1.0296 [1.0158, 1.0427] (one-sided p = 0.0000), 6L / best arm 3 vs
arm 4 = 1.0154 [1.0058, 1.0257] (one-sided p = 0.00025), 2L / best
arm 1 (12 500) vs arm 4 (600) = 1.0185 [1.0015, 1.0370] (one-sided p =
0.0158; does not clear Bonferroni). Since 11 900 additional backbone
steps do not resolve on this metric at these margins, the report cannot
distinguish between "the split-vs-pooled and MoCo axes do not change
GM-Rel MASE" and "the readout does not resolve backbones at ±1–3 %"
without the random-init or early-step underfit backbone control the
card names as a follow-up. Arm 5 (`L_align + L_rep`) is worse than
every other arm on every one of its twelve full-97 pairwise contrasts,
ratios in [1.0557, 1.1581] with task-level lower bounds in
[1.0220, 1.1116], and every arm 5 CI is above 1 on the dataset-clustered
scheme as well; the smallest two-sided p is 0.0018 (Bonferroni margin
0.0003 at α = 0.0021). Champion CIs are absent from this branch —
arm C's per-task `all_results.csv` was not exported into the sweep
tree — so the card's primary success criterion (paired bootstrap vs
arm C) is unmet. On point estimates alone, arm C beats every new arm
in both compute-matched `last` cells (2L / last 1.1491 vs best new
1.1546 = arm 4; 6L / last 1.1254 vs best new 1.1405 = arm 4).

![GM-Relative MASE across arms and (head, checkpoint) scored evaluations.](plots/headline_relmase.png)

![Paired-bootstrap 95 % CIs on GM-Rel MASE ratios. Task-level bootstrap (top per row) and dataset-clustered bootstrap (bottom per row); * marks step- or checkpoint-selection-confounded rows.](plots/ci_forest.png)

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

*Boldface marks the column minimum across arms 1 / 3 / 4 / 5.
Arms 1 / 3 aggregates are the `Aggregate GM-Relative MASE (97 configs)`
line of each `summary.txt` under
`experiments/2026-07-10_split_pred_rep/results/`; arms 4 / 5 are the
same line under `results_arm4/` / `results_arm5/`. Arm C values are
the four `cross_C` (λ_e = 1, λ_h = 1, τ = 0.90) rows of
`experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`.
Arm C's per-task file was not exported to this branch, so no CI is
computable against it.*

**Backbone step behind each cell.** The head-training protocol trains
on the arm's `FINAL.pth` for the `best` cell, then resumes on
`final.pth` for the `last` cell. Steps below come from
`results/backbone_step_verification.log` (md5 of each `FINAL.pth`
against `best_loss.pth`, `final.pth` and intermediate `Xk.pth`; run
log's last `Saved …_best_loss.pth` event for arms whose `FINAL.pth`
matches `best_loss.pth`; `torch.equal` cross-check on arm 1's 193
tensors):

| arm | `best` cell backbone step | `last` cell backbone step |
| --- | --: | --: |
| arm 1 (split) | 12,500 | 12,500 |
| arm 3 (split + MoCo) | 11,800 | 12,500 |
| arm 4 (pooled + MoCo) | 600 | 12,500 |
| arm 5 (`L_align` + `L_rep`) | 11,800 | 12,500 |
| arm C ref (champion) | not exported to this branch | 12,500 |

For arm 1, `FINAL.pth` md5 = `final.pth`, and `torch.equal(FINAL, 12k)
= True` — arm 1's backbone did not update in the last 500 steps, so its
`best` and `last` cells run on identical weights. For arms 3 / 4 / 5,
`FINAL.pth` md5 = `best_loss.pth`; the run log records 15, 6 and 40
`best_loss.pth` saves ending at steps 11,800 / 600 / 11,800.

**Head-adaptation asymmetry across the `last` column.** The head
trains 30 000 steps on `FINAL.pth`, then 10 000 more on `final.pth`.
For arm 1, `FINAL.pth` == `final.pth`, so arm 1's `last` head-trained
40 000 steps on the evaluated backbone. For arms 3 / 4 / 5, the `last`
head-trained 30 000 steps on a different backbone (step 11 800 / 600 /
11 800) and only 10 000 on the evaluated one. `last`-cell contrasts
are backbone-step-matched but head-adaptation-asymmetric — except
**arm 3 vs arm 4**, which is matched on both axes (both head-trained
30 000 steps on their own `best_loss.pth` step-11 800 / step-600 and
then 10 000 on step-12 500).

## Paired-bootstrap 95 % CI on GM-Relative MASE ratios

Panels of contrasts, task-level bootstrap plus (for the full-97 panel)
dataset-clustered bootstrap side by side. 20 000 resamples, seed 42,
seasonal-naive divisor at `results/seasonal_naive_all_results.csv`
(sha256
`d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949`);
the divisor cancels in the paired ratio. Driver:
`scripts/build_ci_panel.py`. Four output CSVs of 24 rows each:
`pairwise_bootstrap_ci.csv` (task-level 97-config panel),
`pairwise_bootstrap_ci_clustered.csv` (task-level clustered by base
dataset — 28 clusters — over the 97 configs),
`pairwise_bootstrap_ci_periodic.csv` (task-level on the 37-config
periodic subset selected by family prefix), and
`pairwise_bootstrap_ci_medlong.csv` (task-level on the 42-config
medium+long horizon subset). Ratio `A/B < 1` means arm A beats arm B.
The Bonferroni threshold α = 0.05 / 24 = 0.0021 is applied to the
24-contrast full-97 panel that the card names as the primary read; the
other panels are read at nominal 95 % as diagnostics.

### Full-97 configs

| cell | contrast | axis toggled | backbone steps (A, B) | ratio A/B | 95 % CI task | 95 % CI clustered |
| --- | --- | --- | --- | --: | --- | --- |
| 2L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 12,500, 12,500 | 1.0119 | [0.9970, 1.0267] | [0.9939, 1.0294] |
| 6L / last | arm 3 vs arm 4 | split ↔ pooled (MoCo fixed) | 12,500, 12,500 | 1.0093 | [0.9960, 1.0269] | [0.9956, 1.0275] |
| 2L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 12,500, 12,500 | 0.9988 | [0.9834, 1.0158] | [0.9801, 1.0176] |
| 6L / last | arm 1 vs arm 3 | MoCo off ↔ on (split fixed) | 12,500, 12,500 | 1.0039 | [0.9902, 1.0195] | [0.9890, 1.0198] |
| 2L / last | arm 1 vs arm 4 | joint | 12,500, 12,500 | 1.0107 | [0.9963, 1.0262] | [0.9957, 1.0260] |
| 6L / last | arm 1 vs arm 4 | joint | 12,500, 12,500 | 1.0133 | [0.9957, 1.0344] | [0.9935, 1.0356] |
| 6L / best* | arm 1 vs arm 3 | MoCo — checkpoint-selection confound | 12,500, 11,800 | 1.0209 | [1.0039, 1.0404] | [1.0051, 1.0393] |
| 6L / best* | arm 3 vs arm 4 | split — 11,200-step gap | 11,800, 600 | 0.9771 | [0.9571, 0.9951] | [0.9553, 0.9948] |
| 2L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 12,500, 12,500 | 1.1041 | [1.0632, 1.1473] | [1.0615, 1.1447] |
| 6L / last | arm 5 vs arm 1 | `L_align + L_rep` ↔ split | 12,500, 12,500 | 1.0557 | [1.0220, 1.0892] | [1.0203, 1.0894] |

Rows marked `*` are `best` cells with a step or checkpoint-selection
confound (arm 1's `best` cell has identical weights to its `last`
cell, so arm-1-`best` is compared against arm-3-`best` at 11 800 and
arm-4-`best` at 600 — the arm-3 row confounds MoCo with 700 backbone
steps of checkpoint selection, and the arm-3 vs arm-4 row confounds
split-vs-pooled with an 11 200-step backbone gap). None of the six
compute-matched arm-1 / 3 / 4 rows separates from 1 at nominal 95 %
under either scheme; arm 5 vs arms 1 / 3 / 4 spans all twelve rows
above 1 with task-level ratios in [1.0557, 1.1581] and lower bounds in
[1.0220, 1.1116].

### Periodic-cluster subset (37 configs — `solar/`, `electricity/`, `ett1/`, `m4_hourly/`, `bizitobs_*`)

Selected by family prefix, not by `rel_MASE ≥ 1.25`, so the subset
does not condition on the outcome. Eleven of twelve arm-1 / 3 / 4
task-level CIs straddle 1; the one that does not is 6L / last arm 1
vs arm 4 = 1.0381 [1.0010, 1.0871], one-sided p = 0.0208 — nominally
separates at 95 %, does not clear Bonferroni. Worst-case straddling
lower bound is 0.9493 (6L / best arm 3 vs arm 4), worst-case upper is
1.0871. Eight of twelve arm-5 task-level CIs sit above 1; four
straddle: 6L / best arm 5 vs arm 4 = 1.0785 [0.9830, 1.1655],
6L / last arm 5 vs arm 3 = 1.0315 [0.9735, 1.0859],
6L / best arm 5 vs arm 1 = 1.0585 [0.9725, 1.1385],
6L / last arm 5 vs arm 1 = 1.0174 [0.9531, 1.0779]. Full 24 rows in
`pairwise_bootstrap_ci_periodic.csv`.

### Medium+long horizon subset (42 configs — every `dataset/*/{medium,long}`)

The card's secondary read. Every arm-1 / 3 / 4 pairwise contrast at
the compute-matched `last` cells:

| cell | contrast | ratio A/B | 95 % CI task | one-sided `p_a_beats_b` |
| --- | --- | --: | --- | --: |
| 2L / last | arm 3 vs arm 4 | 1.0228 | [1.0059, 1.0403] | 0.0042 |
| 6L / last | arm 3 vs arm 4 | 1.0140 | [1.0031, 1.0252] | 0.0064 |
| 2L / last | arm 1 vs arm 3 | 0.9717 | [0.9521, 0.9926] | 0.9951 |
| 6L / last | arm 1 vs arm 3 | 0.9833 | [0.9668, 1.0009] | 0.9690 |
| 2L / last | arm 1 vs arm 4 | 0.9939 | [0.9757, 1.0132] | 0.7381 |
| 6L / last | arm 1 vs arm 4 | 0.9971 | [0.9799, 1.0150] | 0.6232 |

Three rows separate at nominal 95 %: arm 3 vs arm 4 at both head depths
(split shape worse than pooled) and arm 1 vs arm 3 at 2L / last (split
without MoCo better than split with MoCo). The arm 1 vs arm 3 row is
head-adaptation-asymmetric (arm 1 = 40 k on the evaluated backbone;
arm 3 = 10 k), so its direction cannot be attributed to MoCo alone;
the arm 3 vs arm 4 rows are matched on both backbone step and head
adaptation. None of the three clears Bonferroni α = 0.05 / 24 = 0.0021
(two-sided p = 0.0084, 0.0128, 0.0099).

**Medium+long `best` cells and the readout-sensitivity finding.** On
these cells arm 4's `FINAL.pth` points at step 600, arm 1's at step
12 500 and arm 3's at step 11 800. The 42-config compute margin thus
loads arm 4 with 11 900 fewer training steps than arms 1 / 3. Three of
the four arm-4-vs-trained comparisons point in the direction *arm 4
better than trained*, and two of the three clear Bonferroni:

| cell | contrast | ratio A/B | 95 % CI task | two-sided p | vs α = 0.0021 |
| --- | --- | --: | --- | --: | :-: |
| 2L / best | arm 3 (11 800) vs arm 4 (600) | 1.0296 | [1.0158, 1.0427] | < 1 × 10⁻⁴ | clears |
| 6L / best | arm 3 (11 800) vs arm 4 (600) | 1.0154 | [1.0058, 1.0257] | 0.0005 | clears |
| 2L / best | arm 1 (12 500) vs arm 4 (600) | 1.0185 | [1.0015, 1.0370] | 0.0316 | no |
| 6L / best | arm 1 (12 500) vs arm 4 (600) | 1.0104 | [0.9958, 1.0264] | 0.1729 | — |

Because more backbone training would be expected to help (or at worst
not hurt), the observed reversal indicates the 30 000-step probe head
does not resolve arm 4's step-600 backbone from arms 1 / 3's
step-12 500 backbones on medium+long horizons at the ±1–3 % margins
this experiment measures. The random-init or early-step underfit
backbone control that would disambiguate an objective-early-fit effect
from a metric-insensitivity effect is a follow-up card. Under this
constraint the nominal medium+long arm 3 vs arm 4 `last`-cell rows
above are consistent with either "pooled better than split" or "the
readout is at its resolution floor".

## Denominator share

![Stacked per-family shares of each term's denominator at each arm's `FINAL.pth` snapshot (arm 1: step 12,500 weights; arm 3: step 11,800; arm 4: step 600); mixed and periodic batches.](plots/gradient_share_stack.png)

`log_neg_cross_batch` (cross-batch f_t ↔ h'_{t+1}) holds the four
per-checkpoint numbers 0.858 / 0.873 / 0.901 / 0.901 (arm 1, mixed
batch, at steps 2 k / 5 k / 12 k / FINAL — recall arm 1's FINAL and 12 k
are the same weights) and 0.979 / 0.984 / 0.991 / 0.991 (arm 1,
periodic batch, same checkpoints); arm 3's `FINAL.pth` (step 11 800)
gives 0.937 mixed / 0.997 periodic. The same tensor holds 0.003 in
arm 4's pooled denominator at step 600 while the h-anchored families
(`log_neg_hh_all` + `log_neg_xs_allt`) hold 0.877 (periodic) / 0.860
(mixed); arm 4's step-10 000 checkpoint gives cross_batch 0.004,
hh_all + xs_allt 0.867 (periodic) / 0.913 (mixed), so the pattern is
measured on both a step-600 and a step-10 000 pooled backbone. The
card's requirement to measure this on arm C (pooled with MoCo off) is
not met on this branch — arm 4 replaces the cross-batch keys with the
EMA teacher's, which is precisely the tensor whose share is being
read, so this measurement is not a substitute for the champion's
denominator.

*Measurement (`scripts/gradient_share_measurement.py`; full table
`results/gradient_share_measurement.csv`, 132 rows). Each backbone
checkpoint runs in `.eval()` mode on two fixed batches of B = 64,
T = 4096: "mixed" is the training HF stream, "periodic" is solar / H +
electricity / H windows from GIFT-Eval. Each family's share of its
own term's denominator is `exp(mean(log-family − log-denominator))`
over anchors at τ = 0.10, so segments in one bar need not sum to
exactly 1. Read the reported quantities as the loss landscape a
frozen student sees, not the training-time gradient shares of the
MoCo arms: measurement batch is B = 64 (training used B = 512, and
the `log_neg_cross_batch` count scales with B, which raises the
training-time share of that tensor above the measured 0.003–0.004);
`.eval()` disables the 0.70 encoder dropkey and dropout that reshape
h at training time; for the MoCo arms (3, 4) the keys are
student-side at measurement, while training routes them through the
EMA teacher.*

## Arms

| arm | loss shape | `--moco-negatives` | defining feature |
| --- | --- | :-: | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | off | split objective `L = L_pred + L_rep`, equal weight |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | on | split objective; cross-batch f ↔ h keys come from the EMA teacher (MoCo-style) instead of the student |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | on | pooled champion shape with teacher keys |
| arm 5 | `cosine_similarity_batch_rep_only` + `--align-loss-weight 1.0` | off | replace `L_pred` with BYOL-style alignment: `L = L_align + L_rep` (no InfoNCE denominator on the f side) |
| arm C ref | `cosine_similarity_batch_full_hh_negs_xshh_allt` | off | champion (λ_e = 1, λ_h = 1, τ = 0.90) of the earlier sweep, reused without retraining |

**Confound.** The split's `L_pred` is normalized InfoNCE by
construction (positive in denominator), so `--pos-in-denominator` is a
no-op for the split; `--subtract-contrastive-floor` is supported by
the split (it subtracts `f_pred + f_rep`, a constant, and is
gradient-neutral). Arm 1 vs arm C ref therefore differs on one
effective axis — the loss functional (split vs pooled). Arm 3 vs arm 4
is the same functional axis with MoCo held fixed on both sides; arm 1
vs arm 3 is the MoCo axis with the split shape held fixed on both
sides.

Negative families: the two f-anchored families are `log_neg_cross_batch`
(cross-batch f_t ↔ h′_{t+1}) and `log_neg_zy` (adjacent f_{t+1} ↔ f_t);
the three h-anchored families are `log_neg_hh_all` (within-series
all-time h ↔ h), `log_neg_xs_allt` (cross-series all-time h ↔ h′), and
cross-channel `log_neg_xx`, which is empty at C = 1. f is the
forecaster's predicted latent, h the encoder latent; primes mark other
series of the batch. The pooled shape puts all five families into one
denominator; the split routes the two f-anchored families to `L_pred`
and the three h-anchored families to `L_rep`.

Glossary: **MoCo** — replaces the student `h` keys in the cross-batch
f ↔ h′ negative with an EMA teacher `h^T`. **EMA teacher** — an
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
contrastive τ = 0.10, SIGReg λ_e = λ_h = 1, CPC auxiliary loss).
Arms differ in `--loss-shape` and `--moco-negatives`; the pooled arm
additionally keeps `--pos-in-denominator --subtract-contrastive-floor`.
For each backbone a quantile probe head (2 or 6 layers) is trained
for 30 000 steps on `FINAL.pth`, then for 10 000 more steps —
resuming the same head — on `final.pth` (step 12 500). Each head is
evaluated on GIFT-Eval's 97 configs against the seasonal-naive
reference file committed to this branch at
`results/seasonal_naive_all_results.csv`.

## Caveat — single seed, single-panel Bonferroni

Every evaluation is N = 1. The paired bootstrap measures within-run
across-task variability; between-seed variance is not measured on
this branch. Bonferroni α = 0.05 / 24 is applied to the 24-contrast
full-97 panel; nominal 95 % on the other panels is diagnostic — the
card's single-seed noise band ±0.02 is comparable to every
non-arm-5 point difference in the tables. Arm 1's `best` and `last`
cells run on identical weights (md5 + `torch.equal` on 193 tensors);
its best → last spread of ±0.16 % bounds the head-training-length
component under an identical backbone but does not bound backbone-step
or seed variance. `results_arm4/…_last_6L/all_results.csv` carries
`MASE[0.5]` only (reconstructed from `summary.txt`); the CIs use
`MASE[0.5]` alone. Arm C's per-task `all_results.csv` is not on this
branch, so the card's primary success criterion (paired bootstrap vs
arm C) is unmet; arm C's point estimates lead every new arm at both
compute-matched `last` cells.
