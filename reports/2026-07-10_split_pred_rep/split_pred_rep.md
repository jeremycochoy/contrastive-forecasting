# Splitting the contrastive loss into prediction and repulsion terms does not improve full-97 GM-Relative MASE at any properly controlled comparison; replacing the prediction term with BYOL alignment is worse on the full-97 aggregate and on medium+long horizons but level or slightly ahead on short horizons; the primary paired-CI-vs-champion criterion is unmet on this branch

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
replaces arm 5's BYOL alignment `L_align` with a MoCo-style
contrastive alignment `L_align_moco` (student encoder anchor
`h_{b,t}` against teacher-encoder keys `h^T_{b',t}` at the same
timestep, cross-batch negatives — same-timestep, no forecaster-latent
term), while keeping `L_rep`: `L = L_align_moco + L_rep`.

**Answer.** No arm-1 / 3 / 4 full-97 pairwise contrast clears
Bonferroni α = 0.05 / 40 = 0.00125 (smallest two-sided p = 0.0099,
6L / best arm 3 vs arm 4, 11,200-step confounded), and no arm-1 / 3 / 4
compute-matched `last` CI on the full-97 panel separates from 1 at
nominal 95 % under either bootstrap scheme (see §Full-97). Arm 5 vs
arms 1 / 3 / 4 sits above 1 on all twelve rows; at
`n_boot` = 200 000 eleven of the twelve clear α = 0.00125 (worst-row
6L / last arm 5 vs arm 1 has two-sided p = 0.00178 — fails α by 4
× MC SE; the other eleven clear either at zero events or well below
threshold). Arm 6 sits below arm 5 and above arms 1 / 3 / 4 on every
cell; all twelve arm-6-vs-arms-1/3/4 rows separate at nominal 95 %
(ratio < 1 direction), all four arm-6-vs-arm-5 rows separate at
2L / best / last and straddle at 6L / best / last (details in
§Full-97).
On point estimates the pooled champion (arm C) still leads every new
arm at the `last` cells, but the card's primary criterion — a paired
bootstrap of each arm against arm C — was not run because arm C's
per-task file is not available.

The card's canonical arm 3 vs arm 4 (split ↔ pooled, MoCo fixed)
sits at 1.0119 / 1.0093 (2L / 6L `last`, full-97) — direction pooled
better, but not resolved by either scheme (§Full-97). On the
subset panels (read at nominal 95 % as diagnostics, outside the
Bonferroni family), seven arm-1 / 3 / 4 `last` rows separate across
three axes:
- **arm 3 vs arm 4 (split ↔ pooled, MoCo fixed):** three rows in the
  pooled-better direction (medium+long 2L task, medium+long 6L both
  schemes, periodic 6L clustered).
- **arm 1 vs arm 3 (MoCo off ↔ on, split fixed):** direction reverses
  across horizons (medium+long 2L MoCo-off better; short 6L MoCo-on
  better), but head-adaptation content differs on both rows so the
  reversal is not attributable to the MoCo axis alone.
- **arm 1 vs arm 4 (joint):** periodic 6L (task) and short 2L (both
  schemes) point arm-4-better; medium+long `last` rows point the other
  way and both straddle — direction inconsistent across subsets.

![GM-Relative MASE across arms and (head, checkpoint) scored evaluations.](plots/headline_relmase.png)

![Paired-bootstrap 95 % CIs on GM-Relative MASE ratios — all six `last` and all six `best` rows of the three compute-matched full-97 axes (arm 3 vs arm 4, arm 1 vs arm 3, arm 1 vs arm 4), plus the two arm 5 vs arm 1 `last` rows (14 of 24 full-97 rows total). The remaining ten undrawn full-97 rows are all arm-5 pairwise rows: two arm 5 vs arm 1 `best` rows plus all four arm 3 vs arm 5 rows and all four arm 4 vs arm 5 rows. `n_boot` = 20 000, seed 42. Task-level bootstrap (top per row) and 28-dataset-clustered bootstrap (bottom per row); * marks step- or checkpoint-selection-confounded rows.](plots/ci_forest.png)

## Downstream GM-Relative MASE

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo) | **1.1548** | 1.1683 | **1.1338** | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | **1.1546** | 1.1603 | **1.1405** |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm 6 (`L_align_moco` + `L_rep`) | 1.2188 | 1.2133 | 1.1963 | 1.2033 |
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
| arm 6 (`L_align_moco` + `L_rep`) | 10,100 (55 saves, ending at step 10,100) | 12,500 |
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

**Selection rule for the `best` column.** The rule as documented is
"downstream head-train on each backbone's `best_loss.pth` (argmin of
its own smoothed training loss)". No held-out backbone-validation
loss enters the protocol, and the five arms optimise different
objectives on different scales (arm 1 ≈ 24 → 25; arm 3 ≈ 23; arm 4
≈ 3.3 → 3.6; arm 6 ≈ 31 → 17). As shipped, the rule fires
inconsistently across the five arms because of a
checkpoint-promotion gap on arm 1: arm 1's `FINAL.pth` equals
`final.pth` (step 12,500) — its post-resume `best_loss.pth` was never
saved (the run log has zero `Saved …_best_loss.pth` events after
resume at step 900) so the shipped `best` artefact is the final
checkpoint, not the argmin. Arms 3 / 4 / 5's `FINAL.pth` all equal
their own `best_loss.pth` (steps 11,800 / 600 / 11,800). So the
`best` column scores 12,500 / 11,800 / 600 / 11,800 across the four
arms: arm 1's is the final checkpoint (rule-off); arm 4's is the
argmin of a curve that never returns below step 600 (rule-on but
early-fit); arm 3 / 5's are argmins of curves that keep improving
(rule-on late). Six of the twelve arm-1 / 3 / 4 rows in the
Bonferroni family are `best` rows, and they mix the loss-shape axis
under test with two separate checkpoint-selection axes (arm 1's
promotion gap; arm 4's early-fit curve). Reading a `best`-row
separation as evidence about the loss shape confounds the three.

## f-anchored retrieval saturation

`auc` and `top1` are the batch-cross InfoNCE retrieval diagnostics
logged next to `loss` in every backbone losses CSV; they score the
f-anchored prediction task that `L_pred` optimises (retrieval of the
positive `h'_{t+1}` against the cross-batch f ↔ h′ candidates) and
they do not score `L_rep`, which has no positive; arm 5 is
therefore not reported below. Sampled step values:

| arm | step 600 | step 2,000 | step 6,000 | step 12,500 | `top1` min at step ≥ 600 |
| --- | --- | --- | --- | --- | --- |
| arm 1 | auc 1.0000 / top1 0.9998 | 0.9999 / 0.9835 | 1.0000 / 0.9952 | 1.0000 / 0.9926 | 0.8348 (step 3,343) |
| arm 3 | 1.0000 / 0.9998 | 1.0000 / 0.9992 | 1.0000 / 0.9996 | 1.0000 / 0.9993 | 0.9825 (step 3,538) |
| arm 4 | 1.0000 / 0.9993 | 1.0000 / 0.9995 | 1.0000 / 0.9994 | 1.0000 / 0.9974 | 0.9505 (step 934) |

Arm 1's `top1` sits below 0.99 at 5,479 of 11,901 logged steps ≥ 600
(46.0 %). Total training `loss` rises after step 600 on both arms
(arm 1: 24.05 → 25.36; arm 4: 3.26 → 3.61) and never dips below the
step-600 value at any later logged step; the run log's last
`best_loss.pth` save is at step 600 for arm 4 and there is no
post-resume `best_loss.pth` save for arm 1.

## Paired-bootstrap 95 % CI on GM-Relative MASE ratios

20 000 resamples, seed 42, seasonal-naive divisor at
`experiments/2026-07-10_split_pred_rep/results/seasonal_naive_all_results.csv`
(sha256
`d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949`);
the divisor cancels in the paired ratio. Driver:
`experiments/2026-07-10_split_pred_rep/scripts/build_ci_panel.py`.
Output CSVs live in the same `results/` directory. Ratio `A/B < 1`
means arm A beats arm B. The one-sided `p_a_beats_b` column stored in
every CSV is the bootstrap proportion `P(ratio A/B < 1)`; the
two-sided p we quote is `2 · min(p, 1 − p)`. Bonferroni family: the 40-contrast full-97
panel (10 arm pairs × 4 (head, ckpt) cells) at α = 0.05 / 40 = 0.00125;
note that arm 1's `best` and `last` cells share md5-identical backbone
weights (`FINAL.pth` = `final.pth`, no post-resume `best_loss.pth` save)
and differ only in head-training length (30k vs 40k head steps), so
the sixteen arm-1-involving rows (arm 1 vs 3 / 4 / 5 / 6: 4 pairs of
`best`/`last`) are eight backbone contrasts, not sixteen, doubled by
head-adaptation length. The periodic, medium+long and short panels are read at
nominal 95 % as diagnostics and no "Bonferroni" claim is made about
them.

### Full-97 (`pairwise_bootstrap_ci.csv`, 40 rows; `_clustered.csv` for 28-dataset resample)

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

Arm 5 vs arms 1 / 3 / 4 (12 rows, in `pairwise_bootstrap_ci.csv` at
`n_boot` = 20 000; re-run at `n_boot` = 200 000 in
`pairwise_bootstrap_ci_arm5_nboot200k.csv`): quoted in the arm 5 /
arm X direction throughout (same "> 1 → worse" convention as the
other pairs), task-level ratios [1.0557, 1.1581], lower bounds
[1.0220, 1.1116]; all twelve above 1 under both task-level and
clustered schemes. At `n_boot` = 200 000
eleven of the twelve rows clear Bonferroni α = 0.05 / 40 = 0.00125;
the exception is 6L / last arm 5 vs arm 1 (two-sided p = 0.00178,
MC SE 0.000133; fails α by ~4 × MC SE). Eight of
the twelve rows carry a zero-event count out of 200 000
(rule-of-three upper bound: one-sided p < 3 / 200 000 = 1.5 × 10⁻⁵,
two-sided < 3 × 10⁻⁵); of the four rows with non-zero counts the
worst is 6L / last arm 5 vs
arm 1 (one-sided p = 0.00089, two-sided p = 0.00178, Monte-Carlo
standard error on the one-sided proportion
SE₁ = √(p₁(1 − p₁) / B) = 0.000067, SE on the two-sided p is
2 × SE₁ = 0.000133; distance to α is 0.000303, so the row clears at
2.3 × MC SE). The three other non-zero rows have two-sided p = 0.00093
(6L / best arm 5 vs arm 4), 0.00032 (6L / best arm 5 vs arm 1) and
0.00012 (6L / last arm 5 vs arm 3), all well below α.

### Periodic-cluster subset (37 configs — `solar/`, `electricity/`, `ett1/`, `m4_hourly/`, `bizitobs_*`)

Family-prefix selection, so the subset does not condition on the
outcome. Eleven of twelve arm-1 / 3 / 4 task-level CIs straddle 1;
the exception is 6L / last arm 1 vs arm 4 = 1.0381 [1.0010, 1.0871]
(one-sided p = 0.0208 — nominally separates). Under the
dataset-clustered bootstrap (7 datasets in the periodic subset)
eleven of twelve straddle 1; the exception under clustering is the
card's canonical single-axis contrast — 6L / last arm 3 vs arm 4 =
1.0239 [1.0003, 1.0847] (one-sided p = 0.0208), pointing pooled
better than split at compute-matched step 12 500 on the exact cluster
the card's mechanism is about. The lone task-level exception
6L / last arm 1 vs arm 4 widens to [0.9961, 1.1029] under clustering.
Eight of twelve arm-5 CIs sit above 1; four straddle (arm 5 / arm X
convention throughout). Full 40 rows in
`pairwise_bootstrap_ci_periodic.csv` (task) and
`pairwise_bootstrap_ci_periodic_clustered.csv` (clustered).

### Medium+long horizon subset (42 configs — every `dataset/*/{medium,long}`)

The card's secondary read. Compute-matched (`last`, all arms at step
12 500):

| cell | contrast | ratio A/B | 95 % CI task | 95 % CI clustered (14 datasets) | one-sided `p_a_beats_b` (task / clustered) |
| --- | --- | --: | --- | --- | --: |
| 2L / last | arm 3 vs arm 4 | 1.0228 | [1.0059, 1.0403] | [0.9960, 1.0490] | 0.0042 / 0.0461 |
| 6L / last | arm 3 vs arm 4 | 1.0140 | [1.0031, 1.0252] | [1.0044, 1.0251] | 0.0064 / 0.0016 |
| 2L / last | arm 1 vs arm 3 | 0.9717 | [0.9521, 0.9926] | [0.9473, 0.9975] | 0.9951 / 0.9837 |
| 6L / last | arm 1 vs arm 3 | 0.9833 | [0.9668, 1.0009] | [0.9658, 1.0022] | 0.9690 / 0.9607 |
| 2L / last | arm 1 vs arm 4 | 0.9939 | [0.9757, 1.0132] | [0.9816, 1.0034] | 0.7381 / 0.8813 |
| 6L / last | arm 1 vs arm 4 | 0.9971 | [0.9799, 1.0150] | [0.9836, 1.0113] | 0.6232 / 0.6685 |

Task-level: three rows separate at nominal 95 %. Under the more
conservative dataset-clustered bootstrap (14 base datasets in the
subset — `bitbrains_fast_storage`, `bitbrains_rnd`,
`bizitobs_application`, `bizitobs_l2c`, `bizitobs_service`,
`electricity`, `ett1`, `ett2`, `jena_weather`, `kdd_cup_2018`,
`loop_seattle`, `m_dense`, `solar`, `sz_taxi`), 2L / last
arm 3 vs arm 4 falls to [0.9960, 1.0490] and straddles 1 (p = 0.046,
just above the 0.025 one-sided threshold); only 6L / last arm 3 vs
arm 4 and 2L / last arm 1 vs arm 3 stay separated under both schemes.
Head-adaptation asymmetry: arm 3 vs arm 4 warmed up on very different
backbones (step 11 800 vs step 600), and arm 1 vs arm 3 mixes the MoCo
axis with the 40 k-vs-10 k head-adaptation asymmetry disclosed in
§Backbone step. Full 40 rows in
`pairwise_bootstrap_ci_medlong.csv` (task) and
`pairwise_bootstrap_ci_medlong_clustered.csv` (clustered).

**Medium+long `best` cells.** All four arm-4-vs-trained-arms point
ratios are above 1 (i.e. arm 4's step-600 backbone scores lower
GM-Rel MASE than arms 1 / 3's step-12 500 / step-11 800 backbones on
this subset); three of the four separate at nominal 95 %.

| cell | contrast | ratio A/B | 95 % CI task | one-sided `p_a_beats_b` |
| --- | --- | --: | --- | --: |
| 2L / best | arm 3 (11,800) vs arm 4 (600) | 1.0296 | [1.0158, 1.0427] | < 1 × 10⁻⁴ |
| 6L / best | arm 3 (11,800) vs arm 4 (600) | 1.0154 | [1.0058, 1.0257] | 0.00025 |
| 2L / best | arm 1 (12,500) vs arm 4 (600) | 1.0185 | [1.0015, 1.0370] | 0.0158 |
| 6L / best | arm 1 (12,500) vs arm 4 (600) | 1.0104 | [0.9957, 1.0264] | 0.0865 |

### Short-horizon subset (55 configs — every `dataset/*/short`, the disjoint complement of medium+long)

Every trained-vs-step-600 backbone-amount contrast on `best` cells:

| cell | contrast (backbone steps A / B) | ratio A/B | 95 % CI task | 95 % CI clustered (28 datasets) | verdict |
| --- | --- | --: | --- | --- | --- |
| 2L / best | arm 1 (12,500) vs arm 4 (600) | 0.9939 | [0.9650, 1.0281] | [0.9660, 1.0229] | straddles |
| 6L / best | arm 1 (12,500) vs arm 4 (600) | 0.9878 | [0.9692, 1.0084] | [0.9676, 1.0084] | straddles |
| 2L / best | arm 3 (11,800) vs arm 4 (600) | 0.9699 | [0.9346, 1.0032] | [0.9375, 1.0001] | straddles |
| 6L / best | arm 3 (11,800) vs arm 4 (600) | **0.9489** | [0.9176, 0.9778] | [0.9201, 0.9759] | **separates** at nominal 95 % under both schemes (task p = 1 × 10⁻⁴, `p_a_beats_b` = 0.99995 → 1 event / 20,000); this is a `best`-cell row outside the Bonferroni family (see §Selection rule for the 11,200-step confound) |

Only the last row separates. The arm 1 vs arm 4 pair — an 11,900-step
backbone gap on the same MoCo axis (both no-MoCo vs pooled+MoCo) —
straddles under both schemes at both head depths. The separating row
moves two variables at once (11,200 backbone steps AND split ↔
pooled), so it cannot isolate backbone-training amount from loss
shape. What the short subset shows is that this readout **can**
resolve some 2–5 % differences between backbones on short
horizons; it does not establish that the readout resolves
backbone-training amount as a single axis. The random-init /
early-step underfit backbone control that would give a single-axis
measurement (arm 1's committed `_2k` / `_5k` / `_10k` intermediate
checkpoints head-trained under the identical two-stage protocol on
step-12,500 weights, evaluated on the 97 configs) remains the pivotal
open item — it is one head-train + one eval, no new backbone.

Other short-subset separators at nominal 95 % (arm-1/3/4):

| cell | contrast | ratio A/B | 95 % CI task | 95 % CI clustered (28 datasets) | two-sided p task / clustered |
| --- | --- | --: | --- | --- | --: |
| 6L / best | arm 1 (12,500) vs arm 3 (11,800) | 1.0410 | [1.0147, 1.0725] | [1.0159, 1.0680] | 0.0008 / 0.0007 |
| 6L / last | arm 1 vs arm 3 (both 12,500) | 1.0200 | [1.0008, 1.0432] | [1.0021, 1.0387] | 0.040 / 0.028 |
| 2L / last | arm 1 vs arm 4 (both 12,500) | 1.0237 | [1.0039, 1.0472] | [1.0009, 1.0476] | 0.015 / 0.040 |
| 2L / best\* | arm 1 (12,500) vs arm 3 (11,800) | 1.0247 | [0.9997, 1.0556] straddles | [1.0019, 1.0483] separates | 0.053 / 0.033 |

*\* 2L / best arm 1 vs arm 3 straddles under the task scheme and
separates under clustering only; `best`-cell rows also hit §Selection
rule.*

The MoCo direction reverses across horizons at compute-matched `last`
cells: short 6L / last arm 1 vs arm 3 = 1.0200 [1.0008, 1.0432] task,
[1.0021, 1.0387] clustered (MoCo on better; both schemes) against
medium+long 2L / last 0.9717 [0.9521, 0.9926] task, [0.9473, 0.9975]
clustered (MoCo off better; both schemes). Both rows are at backbone
step 12,500 for both arms. Head adaptation is asymmetric (arm 1: 40 k
on the evaluated backbone; arm 3: 30 k on step-11,800 + 10 k on
step-12,500), so the reversal is not attributable to MoCo alone.

**Arm 5 on the short subset.** Eleven of the twelve arm-5 contrasts
here straddle 1; the one that separates is 6L / best arm 4 vs arm 5
= 1.0498 [1.0055, 1.1076] task, [1.0097, 1.1033] clustered — arm 5
scores 5 % lower GM-Relative MASE than arm 4 on that row. This is
a `best`-cell row: arm 5's `best` cell is step-11 800 and arm 4's is
step-600 (11 200-step gap), so the row cannot be read as evidence
about the loss shape (same confound as arm 3 vs arm 4 6L / best;
§Selection rule). On the short subset arm 5's
GM-Relative MASE is 0.975 – 0.997 across the four cells and it holds
the lowest level in two of them (2L / last 0.9947, 6L / last 0.9785).
The title's medium+long-vs-short scope reflects this: the deficit is
concentrated on medium+long (arm 5 = 1.63 – 1.97 there vs 1.36 – 1.43
for arms 1 / 3 / 4). Full 40 rows in
`pairwise_bootstrap_ci_short.csv` (task) and
`pairwise_bootstrap_ci_short_clustered.csv` (clustered).

## Denominator share

![Stacked per-family shares of each term's denominator at each arm's `FINAL.pth` snapshot (arm 1: step 12,500 weights; arm 3: step 11,800; arm 4: step 600); mixed and periodic batches.](plots/gradient_share_stack.png)

`log_neg_cross_batch` (cross-batch f_t ↔ h'_{t+1}) holds essentially
all of `L_pred`'s denominator on arms 1 / 3's `FINAL.pth` snapshots
(arm 1: mixed 0.90, periodic 0.99; arm 3: mixed 0.94, periodic 1.00
— the plot shows one snapshot per arm on both batches). The same
tensor holds 0.003 in arm 4's pooled denominator at step 600 while
the two h-anchored families (`log_neg_hh_all` + `log_neg_xs_allt`)
together hold 0.877 (periodic) and 0.860 (mixed); arm 4's step-10 000
snapshot gives 0.867 (periodic) / 0.913 (mixed) for the same combined
family. The card's directional prediction (h-anchored much larger
than cross-batch on the periodic batch under the pooled shape at
C = 1) reads correctly on arm 4's weights at every measured step: on
the periodic batch h-anchored sits at 0.867 – 0.901 across the four
measured checkpoints while cross-batch sits at 0.0026 – 0.0050. The
periodic-**specific** half of the prediction is not supported by the
same measurement: on the mixed batch h-anchored = 0.860 – 0.914 and
cross-batch = 0.0032 – 0.0036 — approximately the same magnitude on
both batch types. Downstream GM-Relative MASE does not improve on
the split. The probe ran on arm 4's checkpoint, not arm C's;
a re-run on arm C is the follow-up.

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
at measurement, teacher-side at training. The step at which each arm
is probed is `FINAL.pth`: arm 1 = step 12,500, arm 3 = step 11,800,
arm 4 = step 600 (see §Backbone step). Every compute-matched `last`
contrast is evaluated on step 12,500 backbones instead, so for arm 3
and arm 4 the probe reports the loss landscape at a different step
than the one the downstream `last` cells score. `final.pth` at step
12,500 exists on the branch for both arms; the follow-up is one
forward pass per arm on that file plus a re-run on arm C's own
checkpoint.*

## Arms

| arm | loss shape | `--moco-negatives` | defining feature |
| --- | --- | :-: | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | off | split objective `L = L_pred + L_rep`, equal weight |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | on | split objective; cross-batch f ↔ h keys come from the EMA teacher |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | on | pooled champion shape with EMA-teacher keys |
| arm 5 | `cosine_similarity_batch_rep_only` + `--align-loss-weight 1.0` | off | replace `L_pred` with BYOL alignment to the EMA-teacher latent: `L = L_align + L_rep`; `--pos-in-denominator` and `--subtract-contrastive-floor` drop out |
| arm 6 | `cosine_similarity_batch_rep_only` + `--align-loss-weight 0` + `--align-moco-loss-weight 1.0` | n/a | replace arm 5's BYOL `L_align` with a MoCo-style contrastive `L_align_moco` (`src/loss.py:align_moco_loss`): student encoder anchor `h_{b,t}`, teacher-encoder key `h^T_{b',t}` at the same timestep, cross-batch negatives, τ = 0.10, positive at b' = b. No forecaster-latent term in either summand (`L_rep` has no positive; `L_align_moco`'s positive is same-timestep encoder alignment, not next-step prediction). `L = L_align_moco + L_rep` |
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
auxiliary. Arms differ in `--loss-shape`, `--moco-negatives`,
`--align-loss-weight`, and `--align-moco-loss-weight` (see §Arms);
the pooled arm additionally keeps
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
was reconstructed from `summary.txt`. Deferred to follow-up cards, needed to close the card fully:
(i) arm C per-task `all_results.csv` + paired CI vs arm C (the card's
primary criterion); (ii) denominator share measured on arm C (the
card's required measurement); (iii) a single-axis underfit backbone
control — arm 1's committed `_2k` / `_5k` / `_10k` intermediate
checkpoints head-trained under the identical two-stage protocol on
step-12,500 weights, evaluated on the 97 configs, giving a
backbone-amount measurement with the loss shape held fixed. The
short-horizon subset already shows the readout can resolve some
5 % differences between backbones (arm 3 vs arm 4 at 6L / best,
task p ≈ 1 × 10⁻⁴), but the row that separates moves two variables at
once (backbone step and split ↔ pooled), so it does not establish
single-axis backbone-amount resolution.
