# No arm 1 / 3 / 4 contrast on the full 97-config GM-Relative MASE panel clears Bonferroni α = 0.05 / 24 = 0.0021 (best- and last-cell contrasts alike; smallest two-sided p = 0.0099); arm 5 (`L_align + L_rep`) regresses on every scored evaluation (all 12 rows clear α at `n_boot` = 200 000); the card's primary criterion — paired bootstrap vs the pooled champion — is unmet on this branch and champion `last`-cell point-estimate gaps sit inside the card's ±0.02 single-seed band

**Definitions.** *GM-Relative MASE* is the geometric mean, over the 97
GIFT-Eval forecasting configs, of `(model MASE) / (seasonal-naive
MASE)` — a per-task ratio geometrically averaged across tasks. Lower
is better; 1.0 = seasonal-naive. *MASE* is the mean absolute scaled
error at quantile 0.5 (`MASE[0.5]`, the median forecast). "Config"
refers to a (dataset, term-length, horizon) evaluation triple defined
by GIFT-Eval; the 97 in the panel is the full public set.
"Compute-matched" means both arms' `FINAL.pth` or `final.pth` come from
the same backbone training step (12,500 in this report unless flagged).
"Card" is issue #374 in this repo.

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
(`L = L_align + L_rep`). Arm 2 in the card's numbering (a λ-weighted
split, `α L_pred + β L_rep`) is a follow-up not run here. Arm 6
(`L_align` + MoCo) is part of this experiment and is still training
at the time of writing; its cells appear as *pending* in the
downstream table.

**Answer.** No arm-1 / 3 / 4 pairwise contrast in the 12-row full-97
panel clears Bonferroni α = 0.05 / 24 = 0.0021 (the *smallest* two-sided
p on that panel is 0.0099, 6L / best arm 3 vs arm 4, which is
11,200-step confounded); no compute-matched `last`-cell CI separates
from 1 at nominal 95 % under either the task-level or the
28-dataset-clustered bootstrap. Arm 5 vs arms 1 / 3 / 4 (12 rows,
task-level ratios [1.0557, 1.1581], lower bounds [1.0220, 1.1116])
sits above 1 on every row; at `n_boot` = 200 000 all twelve rows
clear α = 0.0021 (all two-sided p ≤ 0.0018 and margin ≥ 2.3 × MC SE
on the worst row, `pairwise_bootstrap_ci_arm5_nboot200k.csv`). On
point estimates the pooled champion (arm C) leads every new arm at
the `last` cells (2L 1.1491 vs the best new arm's 1.1546; 6L 1.1254
vs 1.1405); arm C's per-task file is on the sweep tree only, so
neither gap has a CI on this branch, and both gaps sit inside the
card's ±0.02 single-seed noise band. Arm 4's `best` cells run on a
step-600 backbone (600 / 12,500 = 4.8 % of training) and score at or
below arms 1 / 3's step-12,500 / step-11,800 `best` cells on the
medium+long horizon subset (three of four ratios above 1, i.e. arm 4
better, three separate at nominal 95 %). The card's canonical
split-vs-pooled contrast — arm 3 vs arm 4 with MoCo held fixed on
both sides — is 1.0119 [0.9970, 1.0267] at 2L / last and 1.0093
[0.9960, 1.0269] at 6L / last on the full-97 panel; on the medium+long
subset it separates at nominal 95 % (2L / last 1.0228 [1.0059,
1.0403]; 6L / last 1.0140 [1.0031, 1.0252]) in the direction of pooled
better than split, but the medium+long panel is not in the Bonferroni
family so no ranking is claimed from those rows. Neither this
contrast nor arm 1 vs arm 3 is matched on head-adaptation content
(arms 3 / 4 head-trained 30 000 steps on their own `best_loss.pth`
step-11,800 / step-600 backbones and only 10 000 on the evaluated
step-12,500), so the compute-matched panel is on backbone step but
not on head adaptation.

![GM-Relative MASE across arms and (head, checkpoint) scored evaluations.](plots/headline_relmase.png)

![Paired-bootstrap 95 % CIs on GM-Relative MASE ratios. Task-level bootstrap (top per row) and dataset-clustered bootstrap (bottom per row); * marks step- or checkpoint-selection-confounded rows.](plots/ci_forest.png)

## Downstream GM-Relative MASE

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo) | **1.1548** | 1.1683 | **1.1338** | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | **1.1546** | 1.1603 | **1.1405** |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm 6 (`L_align` + MoCo) | *pending* | *pending* | *pending* | *pending* |
| arm C ref (champion, point reference) | 1.1682 | 1.1491 | 1.1561 | 1.1254 |

*Boldface = column minimum across arms 1 / 3 / 4 / 5. Arm C values
are from `experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`,
row `arm == "cross_C"` (λ_e = 1, λ_h = 1, τ = 0.90); per-task file is
on the sweep tree only, so no CI against arm C is computable here.*

**Backbone step behind each cell.** Sources: md5 of each `FINAL.pth`
against `best_loss.pth` / `final.pth` / intermediate `Xk.pth` and the
last `Saved …_best_loss.pth` event in each run log, both re-runnable
via `scripts/verify_backbone_steps.sh` →
`results/backbone_step_verification.log`. `best_loss.pth` is saved on
smoothed loss on 100-step boundaries, so `argmin` of the raw
`_losses.csv` `loss` column does not identify the file's step.

| arm | `best` cell step | `last` cell step |
| --- | --: | --: |
| arm 1 (split) | 12,500 (`FINAL.pth` md5 = `final.pth`; no post-resume `best_loss.pth` save in the run log) | 12,500 |
| arm 3 (split + MoCo) | 11,800 (15 saves, ending at step 11,800) | 12,500 |
| arm 4 (pooled + MoCo) | 600 (6 saves, all in [100, 600]) | 12,500 |
| arm 5 (`L_align` + `L_rep`) | 11,800 (40 saves, ending at step 11,800) | 12,500 |
| arm C ref (champion) | not exported to this branch | 12,500 |

**Head-adaptation asymmetry across the `last` column.** The head
trains 30 000 steps on `FINAL.pth`, then 10 000 on `final.pth`. For
arm 1, `FINAL.pth` == `final.pth` (md5 match), so arm 1's `last` head
trained 40 000 steps on the evaluated backbone. For arms 3 / 4 / 5,
whose `FINAL.pth` is `best_loss.pth`, the `last` head trained 30 000
on a different backbone (step 11 800 / 600 / 11 800) and 10 000 on
step 12 500. `last`-cell contrasts are therefore
backbone-step-matched (all four eval on step 12,500) but
head-adaptation-asymmetric. Arm 3 vs arm 4 is matched on head-step
count (30 k + 10 k both) but the head warmed up on very different
backbones (step 11 800 vs step 600), so it is not matched on
adaptation content either.

## f-anchored retrieval saturation

`auc` and `top1` are the batch-cross InfoNCE retrieval diagnostics
logged next to `loss` in every backbone losses CSV; they score the
f-anchored prediction task that `L_pred` optimises (retrieval of the
positive `h'_{t+1}` against the cross-batch f ↔ h′ candidates) and
they do not score `L_rep`, which has no positive. Arm 5 has no
`L_pred`, so those diagnostics do not have their nominal meaning for
arm 5; its row below is included for completeness. Sampled step values
from arm 1's `..._losses_full.csv` (arm 1 was resumed at step 900 and
the post-resume `losses.csv` starts at step 901; the full CSV keeps
the pre-resume steps too) and from arms 3 / 4 / 5's `..._losses.csv`:

| arm | step 600 | step 2,000 | step 6,000 | step 12,500 | `top1` min at step ≥ 600 |
| --- | --- | --- | --- | --- | --- |
| arm 1 | auc 1.0000 / top1 0.9998 | 0.9999 / 0.9835 | 1.0000 / 0.9952 | 1.0000 / 0.9926 | 0.8348 (step 3,343) |
| arm 3 | 1.0000 / 0.9998 | 1.0000 / 0.9992 | 1.0000 / 0.9996 | 1.0000 / 0.9993 | 0.9825 (step 3,538) |
| arm 4 | 1.0000 / 0.9993 | 1.0000 / 0.9995 | 1.0000 / 0.9994 | 1.0000 / 0.9974 | 0.9505 (step 934) |
| arm 5 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 |

Arm 1's `top1` at step 600 is 0.9998 (0.02 % error), dips to 0.8348 at
step 3 343 and sits below 0.99 at 5 479 of 11 901 logged steps ≥ 600
(46.0 %); arm 4's `top1` dips to 0.9505 at step 934. Total training
`loss` rises after step 600 on both arms (arm 1: 24.05 → 25.36;
arm 4: 3.26 → 3.61) and never dips below the step-600 value at any
later logged step; the run log's last `best_loss.pth` save is at
step 600 for arm 4 and there is no post-resume `best_loss.pth` save at
all for arm 1.

## Paired-bootstrap 95 % CI on GM-Relative MASE ratios

20 000 resamples, seed 42, seasonal-naive divisor at
`experiments/2026-07-10_split_pred_rep/results/seasonal_naive_all_results.csv`
(sha256
`d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949`);
the divisor cancels in the paired ratio. Driver:
`experiments/2026-07-10_split_pred_rep/scripts/build_ci_panel.py`.
Output CSVs live in the same `results/` directory. Ratio `A/B < 1`
means arm A beats arm B. Bonferroni family: the 24-contrast full-97
panel (6 arm pairs × 4 (head, ckpt) cells) at α = 0.05 / 24 = 0.0021;
the periodic and medium+long panels are read at nominal 95 % as
diagnostics and no "Bonferroni" claim is made about them.

### Full-97 (`pairwise_bootstrap_ci.csv`, 24 rows; `_clustered.csv` for 28-dataset resample)

Every arm-1 / 3 / 4 pairwise contrast:

| cell | contrast | axis toggled | backbone steps (A, B) | ratio A/B | 95 % CI task | 95 % CI clustered |
| --- | --- | --- | --- | --: | --- | --- |
| 2L / best* | arm 1 vs arm 3 | MoCo (split fixed) | 12,500, 11,800 | 1.0092 | [0.9925, 1.0286] | [0.9933, 1.0260] |
| 2L / best* | arm 1 vs arm 4 | joint | 12,500, 600 | 1.0045 | [0.9858, 1.0251] | [0.9860, 1.0209] |
| 2L / best* | arm 3 vs arm 4 | split (MoCo fixed) | 11,800, 600 | 0.9953 | [0.9727, 1.0158] | [0.9727, 1.0138] |
| 2L / last | arm 1 vs arm 3 | MoCo (split fixed) | 12,500, 12,500 | 0.9988 | [0.9834, 1.0158] | [0.9801, 1.0176] |
| 2L / last | arm 1 vs arm 4 | joint | 12,500, 12,500 | 1.0107 | [0.9963, 1.0262] | [0.9957, 1.0260] |
| 2L / last | arm 3 vs arm 4 | split (MoCo fixed) | 12,500, 12,500 | 1.0119 | [0.9970, 1.0267] | [0.9939, 1.0294] |
| 6L / best* | arm 1 vs arm 3 | MoCo — checkpoint-selection confound | 12,500, 11,800 | 1.0209 | [1.0039, 1.0404] | [1.0051, 1.0393] |
| 6L / best* | arm 1 vs arm 4 | joint — 11,900-step gap | 12,500, 600 | 0.9976 | [0.9845, 1.0112] | [0.9833, 1.0097] |
| 6L / best* | arm 3 vs arm 4 | split — 11,200-step gap | 11,800, 600 | 0.9771 | [0.9571, 0.9951] | [0.9553, 0.9948] |
| 6L / last | arm 1 vs arm 3 | MoCo (split fixed) | 12,500, 12,500 | 1.0039 | [0.9902, 1.0195] | [0.9890, 1.0198] |
| 6L / last | arm 1 vs arm 4 | joint | 12,500, 12,500 | 1.0133 | [0.9957, 1.0344] | [0.9935, 1.0356] |
| 6L / last | arm 3 vs arm 4 | split (MoCo fixed) | 12,500, 12,500 | 1.0093 | [0.9960, 1.0269] | [0.9956, 1.0275] |

Rows marked `*` are `best` cells with step or checkpoint-selection
confounds. Of the six compute-matched `last` rows, none separates
from 1 at nominal 95 % under either scheme. Of the six `best` rows,
two separate — both at 6L (arm 1 vs arm 3 and arm 3 vs arm 4) — and
both are step-confounded (700-step gap for arm 1 vs arm 3;
11 200-step gap for arm 3 vs arm 4).

Arm 5 vs arms 1 / 3 / 4 (12 rows, in `pairwise_bootstrap_ci.csv`
at `n_boot` = 20 000; re-run at `n_boot` = 200 000 in
`pairwise_bootstrap_ci_arm5_nboot200k.csv`): task-level ratios
[1.0557, 1.1581], lower bounds [1.0220, 1.1116]; all twelve above 1
under both the task-level and clustered schemes. At `n_boot` = 200 000
all twelve rows clear Bonferroni α = 0.05 / 24 = 0.0021. Nine rows
carry a zero-event count out of 200 000 (two-sided p < 5 × 10⁻⁶); of
the three rows with non-zero counts the worst is 6L / last arm 5 vs
arm 1 (one-sided p = 0.00089, two-sided p = 0.00178, Monte-Carlo
standard error on the one-sided proportion
SE₁ = √(p₁(1 − p₁) / B) = 0.000067, SE on the two-sided p is
2 × SE₁ = 0.000133; distance to α is 0.00032, so the row clears at
2.4 × MC SE). Other non-zero rows: 6L / best arm 5 vs arm 4 = 0.00093
(margin/SE 12.0), 6L / best arm 5 vs arm 1 = 0.00032 (margin/SE 31.2),
6L / last arm 5 vs arm 3 = 0.00012 (margin/SE 56.7).

### Periodic-cluster subset (37 configs — `solar/`, `electricity/`, `ett1/`, `m4_hourly/`, `bizitobs_*`)

Family-prefix selection, so the subset does not condition on the
outcome. Eleven of twelve arm-1 / 3 / 4 task-level CIs straddle 1;
the exception is 6L / last arm 1 vs arm 4 = 1.0381 [1.0010, 1.0871]
(one-sided p = 0.0208 — nominally separates). Eight of twelve arm-5
CIs sit above 1; four straddle. Full 24 rows in
`pairwise_bootstrap_ci_periodic.csv`.

### Medium+long horizon subset (42 configs — every `dataset/*/{medium,long}`)

The card's secondary read. Compute-matched (`last`, all arms at step
12 500):

| cell | contrast | ratio A/B | 95 % CI task | one-sided `p_a_beats_b` |
| --- | --- | --: | --- | --: |
| 2L / last | arm 3 vs arm 4 | 1.0228 | [1.0059, 1.0403] | 0.0042 |
| 6L / last | arm 3 vs arm 4 | 1.0140 | [1.0031, 1.0252] | 0.0064 |
| 2L / last | arm 1 vs arm 3 | 0.9717 | [0.9521, 0.9926] | 0.9951 |
| 6L / last | arm 1 vs arm 3 | 0.9833 | [0.9668, 1.0009] | 0.9690 |
| 2L / last | arm 1 vs arm 4 | 0.9939 | [0.9757, 1.0132] | 0.7381 |
| 6L / last | arm 1 vs arm 4 | 0.9971 | [0.9799, 1.0150] | 0.6232 |

Three rows separate at nominal 95 %. The two arm 3 vs arm 4 rows
(split ↔ pooled) are matched on backbone step but the heads warmed
up on very different backbones (step 11 800 vs step 600) — the head
adaptation content is asymmetric across this pair too. The arm 1 vs
arm 3 2L / last row's direction (no-MoCo better than MoCo) coincides
with the arm-1 40 k-vs-arm-3 10 k head-adaptation asymmetry disclosed
in §Backbone step, so it is not attributable to MoCo alone.

**Medium+long `best` cells.** All four arm-4-vs-trained-arms point
ratios are above 1 (i.e. arm 4's step-600 backbone scores lower
GM-Rel MASE than arms 1 / 3's step-12 500 / step-11 800 backbones on
this subset); three of the four separate at nominal 95 %.

| cell | contrast | ratio A/B | 95 % CI task | one-sided `p_a_beats_b` |
| --- | --- | --: | --- | --: |
| 2L / best | arm 3 (11,800) vs arm 4 (600) | 1.0296 | [1.0158, 1.0427] | < 1 × 10⁻⁴ |
| 6L / best | arm 3 (11,800) vs arm 4 (600) | 1.0154 | [1.0058, 1.0257] | 0.00025 |
| 2L / best | arm 1 (12,500) vs arm 4 (600) | 1.0185 | [1.0015, 1.0370] | 0.0158 |
| 6L / best | arm 1 (12,500) vs arm 4 (600) | 1.0104 | [0.9958, 1.0264] | 0.0865 |

On this subset arm 4's step-600 backbone gives lower GM-Relative
MASE than arm 3's step-11 800 and arm 1's step-12 500 at both head
depths; the underfit-random-init backbone control is a follow-up (see
§Caveat).

## Denominator share

![Stacked per-family shares of each term's denominator at each arm's `FINAL.pth` snapshot (arm 1: step 12,500 weights; arm 3: step 11,800; arm 4: step 600); mixed and periodic batches.](plots/gradient_share_stack.png)

`log_neg_cross_batch` (cross-batch f_t ↔ h'_{t+1}) holds essentially
all of `L_pred`'s denominator on arms 1 / 3's `FINAL.pth` snapshots
(arm 1: mixed 0.90, periodic 0.99; arm 3: mixed 0.94, periodic 1.00
— all four checkpoints in the plot). The same tensor holds 0.003 in
arm 4's pooled denominator at step 600 while the two h-anchored
families (`log_neg_hh_all` + `log_neg_xs_allt`) hold ≥ 0.86 on both
batches; arm 4's step-10 000 snapshot gives the same pattern. The
card asks for the measurement on arm C — a pooled backbone trained
with MoCo **off**. The probe at measurement time uses student-side
keys (identical to arm C's training regime), so the *tensor form*
measured on arm 4 is the pooled-MoCo-off denominator; what is different
from an arm-C measurement is the weights (arm 4 was trained under MoCo,
i.e. with teacher-side keys, so its checkpoint reflects a training-time
gradient distribution arm C's would not). The card's specific
prediction (`share(hh + xs) ≈ 1`, `share(cross_batch) ≈ 0` on the
periodic batch under the pooled shape at C = 1) reads correctly on
arm 4's weights at every measured step, but a re-run on arm C's own
checkpoint remains a follow-up.

*Method (`scripts/gradient_share_measurement.py`; full table
`results/gradient_share_measurement.csv`, 132 rows). Each backbone
checkpoint runs in `.eval()` on two fixed batches of B = 64, T = 4096:
mixed = training HF stream; periodic = solar / H + electricity / H
windows from GIFT-Eval. Each family's share of its own term's
denominator is `exp(mean(log-family − log-denominator))` at τ = 0.10.
Read as the loss landscape a frozen student sees, not the training-time
gradient shares of the MoCo arms — measurement B = 64 vs training
B = 512 (the training-time share at B = 512 was not measured; the
count of cross-batch negatives scales with B); `.eval()` disables the
0.70 encoder dropkey; for the MoCo arms (3, 4) keys are student-side
at measurement, teacher-side at training.*

## Arms

| arm | loss shape | `--moco-negatives` | defining feature |
| --- | --- | :-: | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | off | split objective `L = L_pred + L_rep`, equal weight |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | on | split objective; cross-batch f ↔ h keys come from the EMA teacher |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | on | pooled champion shape with EMA-teacher keys |
| arm 5 | `cosine_similarity_batch_rep_only` + `--align-loss-weight 1.0` | off | replace `L_pred` with BYOL alignment to the EMA-teacher latent: `L = L_align + L_rep`; `--pos-in-denominator` and `--subtract-contrastive-floor` drop out |
| arm C ref | `cosine_similarity_batch_full_hh_negs_xshh_allt` | off | champion (λ_e = 1, λ_h = 1, τ = 0.90) of the earlier sweep, reused without retraining |

**Confound.** The split's `L_pred` is normalized InfoNCE by
construction (positive in denominator), so `--pos-in-denominator` is
a no-op for the split; `--subtract-contrastive-floor` is supported by
the split (subtracts `f_pred + f_rep`, a constant, gradient-neutral).
Arm 1 vs arm C ref therefore differs on one effective loss-shape
axis (split vs pooled) — arm C's head protocol is documented on the
sweep branch and not re-verified here, so any head-adaptation
difference between arm 1 and arm C is unmeasured. Arm 3 vs arm 4 is
the same functional axis with MoCo on both sides (backbone-step-
matched but head warm-up backbone unmatched — 11 800 vs 600); arm 1
vs arm 3 is the MoCo axis with the split shape held fixed
(backbone-step-matched, head-adaptation-asymmetric — 40 k on the
evaluated backbone vs 30 k + 10 k). Arm 5 vs arm 1 changes two axes at
once: `L_pred` (InfoNCE) → `L_align` (BYOL alignment) plus the
`--align-loss-weight 1.0` addition.

Negative families (measurement CSV tensor names): f-anchored are
`log_neg_cross_batch` (cross-batch f_t ↔ h′_{t+1}) and `log_neg_zy`
(adjacent f_{t+1} ↔ f_t); h-anchored are `log_neg_hh_all`
(within-series all-time h ↔ h), `log_neg_xs_allt` (cross-series
all-time h ↔ h′), and cross-channel `log_neg_xx` (empty at C = 1).
Split routes the two f-anchored families to `L_pred` and the three
h-anchored to `L_rep`.

Glossary. **MoCo** — cross-batch f ↔ h′ negatives sourced from an EMA
teacher `h^T` instead of the student. **EMA teacher** — decay
τ = 0.90 shadow of the student encoder. **BYOL alignment** — negative-
free objective maximising cosine similarity between the student's
forecaster latent and the EMA teacher's encoder latent (`--ema-encoder`
routes the target). **SIGReg** — pushes the marginal of pooled `e` and
pooled `h` toward uniform on the sphere. **CPC** — batch-cross InfoNCE
auxiliary predicting `e` from `h` at matched (b, t) indices.

## Method

Backbone training: 12,500 steps, B = 512, T = 4096, C = 1, lr 1e-3,
seed 20260520, dataset `gift-pretrain-full-4096 / small_v1`, EMA
teacher τ = 0.90, contrastive τ = 0.10, SIGReg λ_e = λ_h = 1, CPC
auxiliary. Arms differ in `--loss-shape` and `--moco-negatives`; the
pooled arm additionally keeps
`--pos-in-denominator --subtract-contrastive-floor`. Downstream: a
quantile probe head (2 or 6 layers) trains 30 000 steps on
`FINAL.pth`, then 10 000 more on `final.pth` (step 12 500). Each head
is evaluated on GIFT-Eval's 97 configs against the branch-committed
seasonal-naive reference. Arm C ref's head protocol is documented on
the sweep branch; this branch does not re-verify it.

## Caveat

N = 1. The paired bootstrap measures within-run across-task
variability; between-seed variance is not measured here and the
card's single-seed noise band ±0.02 is comparable to every
non-arm-5 point difference in the tables. `results_arm4/…_last_6L/all_results.csv`
carries `MASE[0.5]` only (reconstructed from `summary.txt`); the CIs
use `MASE[0.5]` alone. Deferred to follow-up cards, all needed to
close the card fully: (i) arm C per-task `all_results.csv` +
paired CI vs arm C (the card's primary criterion), (ii) denominator
share measured on arm C (the card's required measurement), (iii)
random-init or early-step underfit backbone control to disambiguate
whether the medium+long `best`-cell result reflects arm 4's step-600
objective quality or the readout's resolution at ±1–3 %.
