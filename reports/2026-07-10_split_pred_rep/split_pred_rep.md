# Bimoco at 12,500 steps and pooled + MoCo (arm 4) from 25,000 onward are the lowest arms, and both beat the SIGReg champion with a 95 % paired-bootstrap CI.

**Question.** The champion loss puts every negative under a single log-sum-exp denominator. Does splitting it into `L_pred` (f-anchored) and `L_rep` (h-anchored) improve GM-Relative MASE — the geometric mean over the 97 GIFT-Eval configs of per-task (model MASE) / (seasonal-naive MASE), lower is better and 1.0 matches seasonal-naive — and does adding EMA-teacher MoCo keys or replacing `L_pred` with BYOL alignment change the answer? (arms and other terms defined in annex)

**Answer.** Which arm is lowest depends on the backbone step at which the arms are compared.

At 12,500 steps the lowest GM-Relative MASE in all four (head, checkpoint) cells belongs to `L_pred_moco + L_rep_moco` — *bimoco*, the split loss with EMA-teacher MoCo keys on both terms. It is 95 %-separated from every other sibling arm in every cell, and on the `last` cells also 95 %-separated from arm C (representative: 6L ratio 0.980, CI [0.963, 0.994]).

That ordering does not survive further training. At 25,000 steps arm 4 (pooled + MoCo) takes the lead in both panels, and holds it at 50,000; bimoco has no 50,000-step cell. Arm 4 is 95 %-separated from arm C at 6L / 25k and at both heads at 50k, and arm 3 joins at 50k on both heads (ratios and CIs in the paired-bootstrap annex).

The lowest cell measured anywhere is arm 4, 6L, 25,000 steps, at 1.1073.

Arm C's per-task file is a same-recipe seed-2 retrain; the sibling arms train at seed 20260520, so each vs-arm-C CI absorbs one seed of noise on top of the loss-shape difference. Every other separation claim in this report is a same-seed contrast. On the two arm-4 `best` rows the compared backbones are 11,800 steps apart (arm 4 step 600 vs bimoco step 12,400), so those two rows mix loss shape with backbone step.

## Result

![GM-Relative MASE per arm across backbone step](plots/gm_curve_per_arm.png)

*GM-Relative MASE per arm across backbone step, on a shared y-axis. Arm C — the SIGReg champion recipe — is plotted as a solid dark-grey curve with diamond markers at its evaluated snapshots (step 12,500 / 25,000 / 50,000, seed-2 retrain). For each sibling arm one line joins every evaluated cell of that arm across backbone step; marker shape marks the head protocol on that cell (solid disks = fresh 40k-step head, used for 2k / 25k / 50k; hollow circles = 30k-step best-loss head, used for `best` and `last`, resumed a further 10k steps for `last`). Not every arm is evaluated at every step: bimoco has no 50,000-step cell, so its line stops at 25,000. Digits in the trajectory annex.*

![Backbone training loss aligned with evaluated GM-Relative MASE snapshots](plots/loss_vs_gm_snapshots.png)

*Backbone training loss on the left axis, as a 100-step rolling mean concatenated across the 1–12,500, 12,500–25,000 and 25,000–50,000 training segments. The right axis carries the arm's evaluated GM-Relative MASE cells: 2L as circles on a thin dotted line, 6L as triangles on a thin dashed line. A vertical guide marks each evaluated backbone step. `loss` is not comparable across arms, because the arms optimise different loss shapes, different negative counts, and in arm 4 a subtracted contrastive floor. Each panel is therefore read within itself. Sources in the trajectory annex.*

![Retrieval error (1 − ff) aligned with evaluated GM-Relative MASE snapshots, per arm](plots/ff_vs_gm_snapshots.png)

*Same alignment as the previous figure, but with `1 − ff` in the top rectangle instead of the training loss, and the downstream GM-Relative MASE at each snapshot step in the bottom rectangle (shared x-axis per arm). `ff = cos(f̂_t, f_true_{t+1})` is the training-time positive-pair cosine similarity, read from the `ff` column of each arm's losses CSV. Unlike raw `loss`, `1 − ff` is a common training-time diagnostic on the same numeric scale for every arm; interpretation still differs (arms 1, 3, 4, bimoco push it down through InfoNCE on that pair, arm 5 through BYOL alignment, arm 6 through BYOL alignment with teacher-side representation-side MoCo). 2L cells are circles on a thin dotted line, 6L cells triangles on a thin dashed line. Sources in the trajectory annex.*

### 12,500-step cells

![Downstream GM-Relative MASE per arm](plots/headline_relmase.png)

*Downstream GM-Relative MASE per arm at each (head, checkpoint) cell. These are point estimates from N = 1 seed. The dashed line marks seasonal-naive at 1.0. The hatched bars are arm C, the SIGReg-cross champion recipe (`cross_C`: λ_e = 1, λ_h = 1, τ = 0.90); the value shown is the same-recipe seed-2 retrain at backbone step 12,500, which matches the sibling `last` cells' step. Arm C has no best-loss save so no `best`-cell analog is plotted. Arm separation is the paired task-bootstrap in the paired-bootstrap annex, not a bar-to-bar comparison here.*

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo on `L_pred`) | 1.1548 | 1.1683 | 1.1338 | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | 1.1546 | 1.1603 | 1.1405 |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm 6 (`L_align` + `L_rep_moco`) | 1.1771 | 1.1712 | 1.1768 | 1.1767 |
| arm bimoco (`L_pred_moco` + `L_rep_moco`) | **1.1225** | **1.1180** | **1.1138** | **1.1087** |
| arm C ref (seed 2, step 12,500) | — | 1.1441 | — | 1.1318 |

Per-arm result-directory paths are in the Method annex.

## Trajectory cells (annex)

Aggregate GM-Relative MASE (97 configs) at every evaluated (arm, head, backbone-step) cell. A blank means the cell was not evaluated. `best` is the arm's own best-loss step, given in the backbone-step annex. The `2k` / `25k` / `50k` cells use a fresh 40k-step head; `best` and `last` use a 30k-step best-loss head, resumed a further 10k steps for `last`.

| arm | 2L 2k | 2L best | 2L last (12,500) | 2L 25k | 2L 50k | 6L 2k | 6L best | 6L last (12,500) | 6L 25k | 6L 50k |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| arm 1 | 1.2005 | 1.1654 | 1.1669 | 1.1761 | 1.2334 | 1.1938 | 1.1575 | 1.1557 | 1.1500 | 1.2129 |
| arm 3 | 1.2075 | 1.1548 | 1.1683 | 1.1484 | 1.1461 | 1.1825 | 1.1338 | 1.1511 | 1.1170 | 1.1221 |
| arm 4 | 1.1874 | 1.1602 | 1.1546 | **1.1332** | 1.1414 | 1.1564 | 1.1603 | 1.1405 | **1.1073** | 1.1199 |
| arm 5 | 1.2208 | 1.3374 | 1.2883 | 1.2279 | 1.3357 | 1.1829 | 1.2554 | 1.2201 | 1.1889 | 1.2279 |
| arm 6 | 1.1601 | 1.1771 | 1.1712 | 1.1714 | 1.1907 | 1.1604 | 1.1768 | 1.1767 | 1.1655 | 1.1823 |
| arm bimoco | 1.1438 | **1.1225** | **1.1180** | 1.1339 | — | 1.1337 | **1.1138** | **1.1087** | 1.1319 | — |
| arm C ref (seed 2) | — | — | 1.1441 | 1.1415 | 1.1768 | — | — | 1.1318 | 1.1325 | 1.1510 |

Bold marks the lowest value in that column. The directory layout is the one given in the 12,500-step-cells section. The `2k` / `25k` / `50k` cells carry the matching `_2k` / `_25k` / `_50k` suffix, `last` carries `_last`, and `best` carries no suffix.

The backbone loss curves in the second figure are concatenated per arm from:

- arm 1 — `runs/…_losses_full.csv` + `…_r2_losses.csv` + `…_r3_losses.csv`
- arm 3 — `runs/…_moco_…_losses.csv` + `…_ext25k_losses.csv` + `…_r3_losses.csv`. The two resume runs re-index their step counter from 1 and are offset by +12,500.
- arms 4, 5, 6 — `runs_arm4/`, `runs_arm5/`, `runs_arm6_v2/`, each base + `_r2` + `_r3`
- bimoco — `runs_bimoco_v2/` base + `_r2`; there is no 25k–50k segment.

## Method (annex)

All sibling backbones (arms 1, 3, 4, 5, 6, bimoco): B = 512, T = 4096, C = 1, τ = 0.10, `lr = 1e-3`, seed 20260520, dataset `gift-pretrain-full-4096 / small_v1`, EMA teacher τ = 0.90, SIGReg λ_e = λ_h = 1, CPC auxiliary, 12,500 steps, with prolongations to 25,000 and (five arms) 50,000 steps.

Arm C is the SIGReg-cross champion recipe (`cross_C`: λ_e = 1, λ_h = 1, τ = 0.90). The per-task file used here is a same-recipe seed-2 retrain in `experiments/2026-07-10_split_pred_rep/results_armC_seed2/`, evaluated at backbone steps 12,500 / 25,000 / 50,000 on both head depths. The original seed-20260520 arm C run's per-task file was not committed with `2026-06-28_sigreg_lambda_tau_cross` and is no longer on disk; only its aggregate row in `results/gm_table.csv` (arm `cross_C`) survives.

Two backbone snapshots per arm feed the 12,500-step downstream cells. The **best-cell backbone**, file `bb_<run>_FINAL.pth`, is a copy of the arm's `_best_loss.pth` save. Arm 1 recorded no `best_loss` save, so its best-cell backbone is byte-identical to its step-12,500 backbone (`experiments/2026-07-10_split_pred_rep/results/backbone_step_verification.log`). The **step-12,500 backbone**, file `bb_<run>_final.pth`, is the end-of-training checkpoint.

The `best` cell trains a fresh 30k-step quantile head, 2L or 6L, on the best-cell backbone. The `last` cell then resumes that head for a further 10k steps on the step-12,500 backbone. The 2k / 25k / 50k trajectory cells each train a fresh 40k-step head on the corresponding backbone snapshot.

Eval: GIFT-Eval 97 configs, strategy B4, quantile-median MASE divided by the seasonal-naive reference in `experiments/2026-07-10_split_pred_rep/results/seasonal_naive_all_results.csv`.

Every sibling-arm cell is the `Aggregate GM-Relative MASE (97 configs)` line of `summary.txt`, under `experiments/2026-07-10_split_pred_rep/<dir>/gift_eval_full_<arm base name>[_suffix]_<2L|6L>/`. The directory `<dir>` differs per arm:

- arm 1 — `results/`, base name `…_split_pred_rep_xftrip_…`
- arm 3 — `results/`, base name `…_split_pred_rep_moco_xftrip_…`
- arm 4 — `results_arm4/`
- arm 5 — `results_arm5/`
- arm 6 — `results_arm6_v2/`
- bimoco — `results_bimoco_v2/`
- arm C ref — `results_armC_seed2/gift_eval_full_armC_seed2_step{12500,25000,50000}_{2L,6L}/`

The superseded `results_arm6/` and `results_bimoco/` directories are not used in this report.

f-anchored retrieval is saturated across all arms after step 600. `auc` stays at or above 0.9975, and the lowest `top1` is 0.8348, at arm 1 step 3,343, with every other arm above 0.95 (`auc` / `top1` columns of `experiments/2026-07-10_split_pred_rep/runs*/bb_*_losses*.csv`). This diagnostic therefore does not separate the arms.

## Denominator share (annex)

![Per-family denominator share](plots/gradient_share_stack.png)

*Per-family denominator share at each arm's best-cell backbone snapshot — step 12,500 for arm 1, step 11,800 for arm 3, step 600 for arm 4. Each snapshot is probed on a mixed batch and on a periodic-only batch of solar and electricity windows, at τ = 0.10 and probe B = 64. `share_i = exp(mean(logit_i − log-denominator))` is a per-anchor geometric mean, so the families need not sum to 1. Each bar's column sum Σ is printed above it.*

Measured share of the cross-batch `f ↔ h′` family (`log_neg_cross_batch`) in the term carrying the prediction pairs (`experiments/2026-07-10_split_pred_rep/results/gradient_share_measurement.csv`):

| arm | term | mixed batch | periodic batch |
| --- | --- | --: | --: |
| arm 1 (split, step 12,500) | `L_pred` | 0.901 | 0.991 |
| arm 3 (split + MoCo, step 11,800) | `L_pred` | 0.937 | 0.997 |
| arm 4 (pooled, step 600) | pooled | 0.003 | 0.003 |

The three probed snapshots span 11,900 steps, so this split-vs-pooled difference is not separated from backbone step. bimoco was not probed. The probe runs at B = 64, whereas training ran at B = 512; cross-batch share depends on B, so the absolute shares above are indicative rather than identical to the training-time split.

## Arms (annex)

| arm | loss shape | flags | key structure |
| --- | --- | --- | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | — | `L = L_pred + L_rep`, no MoCo |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | `--moco-negatives` | `L_pred` uses EMA-teacher keys on cross-batch f ↔ h |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | `--moco-negatives --pos-in-denominator --subtract-contrastive-floor` | pooled champion shape with EMA-teacher keys |
| arm 5 | `cosine_similarity_batch_rep_only` | `--align-loss-weight 1.0` | `L = L_align + L_rep`; `L_align` is BYOL to teacher-encoder next-step latent |
| arm 6 | `cosine_similarity_batch_rep_only` | `--align-loss-weight 1.0 --moco-rep-keys` | arm 5's `L_align` + `L_rep_moco`: h-family keys routed through EMA teacher |
| arm bimoco | `cosine_similarity_batch_split_pred_rep` | `--moco-negatives --moco-rep-keys` | `L = L_pred_moco + L_rep_moco`; MoCo + positive-in-denominator on both terms |
| arm C ref | `cosine_similarity_batch_full_hh_negs_xshh_allt` | — | SIGReg-cross champion (λ_e = 1, λ_h = 1, EMA τ = 0.90); per-task file is the seed-2 retrain from `2026-07-07_b512_armC_seed2_traj` |

## Backbone step (annex)

| arm | `best` cell step | `last` cell step | best-cell backbone = |
| --- | --: | --: | --- |
| arm 1 | 12,500 | 12,500 | the step-12,500 checkpoint, not a loss-argmin checkpoint |
| arm 3 | 11,800 | 12,500 | `_best_loss.pth` at step 11,800 |
| arm 4 | 600 | 12,500 | `_best_loss.pth` at step 600 |
| arm 5 | 11,800 | 12,500 | `_best_loss.pth` at step 11,800 |
| arm 6 | 8,700 | 12,500 | `_best_loss.pth` at step 8,700 |
| arm bimoco | 12,400 | 12,500 | `_best_loss.pth` at step 12,400 |
| arm C ref | — (no best-loss save) | 12,500 | seed-2 retrain at steps 12,500 / 25,000 / 50,000; step 12,500 is the `last`-cell analog |

For arms 1 and 4 the `best`-cell backbone is not the argmin of a comparable curve. Arm 1's shipped checkpoint is step 12,500, and arm 4's loss never returns below its step-600 value. Six of the twelve arm-1/3/4 pairwise rows are `best` rows, as are six of the twelve arm-1/3/4-vs-bimoco rows. Those rows mix the loss-shape axis with this checkpoint-selection axis.

## Definitions (annex)

- *f* / *h* — *f* is the forecaster's next-step latent, that is, the forecast. *h* is the encoder's original latent. An *f-anchored* family takes *f* as the query, an *h-anchored* family takes *h*.
- *2L / 6L* — the depth of the downstream quantile forecasting head: a 2-layer and a 6-layer head, each trained on the same backbone snapshot.
- *`L_pred`* — normalized InfoNCE, f-anchored, with positive `cos(f_t, h'_{t+1})`. Its denominator holds the adjacent `f_{t+1} ↔ f_t` negatives plus the cross-batch `f_t ↔ h'_{t+1}` negatives.
- *`L_rep`* — pooled log-sum-exp of three h-anchored families (cross-channel `h ↔ h`, within-series all-time `h ↔ h`, cross-series all-time `h ↔ h'`), with no positive term.
- *GM-Relative MASE* — geometric mean over the 97 GIFT-Eval configs of `(model MASE) / (seasonal-naive MASE)`, at quantile 0.5. Lower is better, and 1.0 matches seasonal-naive.
- *log-sum-exp denominator* — the InfoNCE normaliser at an anchor, that is, the soft-max over all negative similarity scores. Under `--pos-in-denominator` the positive score enters it as well.
- *`--pos-in-denominator`* — training flag: the positive score is included in the log-sum-exp denominator, not only in the numerator.
- *`--subtract-contrastive-floor`* — training flag: the analytic InfoNCE floor at that arm's negative count is subtracted from the reported loss. The logged `loss` of an arm using it is therefore not on the same scale as one that does not.
- *MoCo* — cross-batch keys sourced from an EMA teacher (τ = 0.90) instead of the student.
- *bimoco* — the arm applying MoCo keys to both split terms: `L = L_pred_moco + L_rep_moco`.
- *sibling arms* — the six arms trained in this experiment (1, 3, 4, 5, 6, bimoco), sharing one backbone configuration and differing only in loss shape and key source. arm C is not a sibling arm, since it was trained in an earlier experiment and reused here without retraining.
- *SIGReg* — the spectral isotropy regulariser applied to the encoder (λ_e) and head (λ_h) latents; the champion uses λ_e = λ_h = 1.
- *CPC* — the contrastive predictive-coding auxiliary head, on in every arm.
- *BYOL alignment (`L_align`)* — `2 − 2·cos(f_t, sg(h^T_{t+1}))` (sg = stop-gradient); negative-free, minimum 0.
- *`L_rep_moco`* — normalized InfoNCE over the three h-anchored families with teacher-side keys. The same-batch same-time student ↔ teacher pair sits in both the numerator and the denominator log-sum-exp.
- *`auc` / `top1`* — retrieval quality of the f-anchored positive against the B = 512 cross-batch candidates. `auc` is the ROC area of the positive-score distribution, and `top1` the fraction of anchors whose positive ranks first.
- *B4* — GIFT-Eval evaluation strategy B4 (teacher-forced probe over the full 97-config panel).
- *best / last cell* — the two 12,500-step downstream checkpoints per arm. `best` is the head trained on the best-cell backbone, `last` the head resumed on the step-12,500 backbone.
- *95 %-separated* — the paired-bootstrap 95 % CI on the ratio of two arms' GM-Relative MASE excludes 1.0.
- *28-dataset-clustered bootstrap* — resampling the 28 source datasets rather than the 97 configs. Within-dataset correlation between configs is then not counted as independent evidence.
- *arm C* — the SIGReg-cross champion recipe (`cross_C`: λ_e = 1, λ_h = 1, EMA τ = 0.90), the baseline this sweep is ranked against. The per-task file used here is a same-recipe seed-2 retrain (`experiments/2026-07-10_split_pred_rep/results_armC_seed2/`).
- *Bonferroni family* — 60 contrasts = 15 arm pairs × 4 (head, checkpoint) cells on the full-97 panel, α = 0.05 / 60 = 0.000833.

## Paired-bootstrap 95 % CIs (annex)

![Paired-bootstrap 95 % CIs on GM-Relative MASE ratios](plots/ci_forest.png)

*Paired-bootstrap 95 % CIs on GM-Relative MASE ratios. Circles are the task-level bootstrap, faded squares the 28-dataset-clustered bootstrap, and `*` marks rows confounded by checkpoint selection or by backbone step. Both bootstraps use n_boot = 20,000 and seed 42, except the arm-5/6/bimoco-reference rows and the arm-C rows, which use n_boot = 200,000. The figure shows all 34 sibling-arm-only contrasts (12 arm-1/3/4 pairwise `best` and `last`, 12 arm-1/3/4-vs-bimoco `best` and `last`, and the `last`-cell rows for arm 5 vs arm 1, arms 1/3/4 vs arm 6, and arm 5 vs arm 6) plus 12 sibling-vs-arm-C rows on the `last` cells. Arm-C rows are task-level only — arm C has no 28-dataset-clustered pass — and use the seed-2 retrain at backbone step 12,500 as reference.*

A row counts as separated when the task-level 95 % CI excludes ratio 1.0. A ratio above 1 means the named arm is worse than the reference. The arm-5/6/bimoco references come from `experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_*_nboot200k.csv`, the arm-1/3/4 pairwise from `experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci.csv`, and the arm-C rows from `experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_vs_armC.csv`.

Contrast counts among the sibling arms, all rows on the 12,500-step cells:

| contrast set | rows | separated at 95 % (task-level CI) |
| --- | --: | --: |
| arm 1 / 3 / 4 pairwise | 12 | 2 / 12 (both `best`, checkpoint-confounded) |
| arm 5 vs arm 1 / 3 / 4 | 12 | 12 / 12 (arm 5 worse) |
| arm 6 vs arm 1 / 3 / 4 | 12 | 4 / 12 (6L: best arm 3; last arms 1, 3, 4 lower than arm 6) |
| arm 5 vs arm 6 | 4 | 3 / 4 (arm 6 lower) |
| **arm 1 / 3 / 4 vs bimoco** | **12** | **12 / 12 (bimoco lower)** |
| arm 5 / arm 6 vs bimoco | 8 | 8 / 8 (bimoco lower) |

Within the arm-1/3/4-vs-bimoco family, 10 of the 12 rows also clear the Bonferroni threshold α = 0.05 / 60 = 0.000833. The two that miss are 2L / best arm 4 vs bimoco, at p₂ = 0.00435, and 6L / best arm 3 vs bimoco, at p₂ = 0.00426. Both come from the `two_sided_p` column of `experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_bimoco_nboot200k.csv`.

Contrasts vs arm C (seed-2 retrain at the matching backbone step), from `pairwise_bootstrap_ci_vs_armC.csv`:

| sibling cell | arm 1 | arm 3 | arm 4 | arm 5 | arm 6 | bimoco |
| --- | --: | --: | --: | --: | --: | --: |
| 2L / last (vs step-12,500) | 1.020 [1.004, 1.039] | 1.021 [1.009, 1.034] | 1.009 [0.999, 1.021] | 1.126 [1.092, 1.162] | 1.024 [1.004, 1.046] | **0.977 [0.964, 0.991]** |
| 6L / last (vs step-12,500) | 1.021 [1.006, 1.038] | 1.017 [1.003, 1.032] | 1.008 [0.994, 1.021] | 1.078 [1.052, 1.105] | 1.040 [1.019, 1.065] | **0.980 [0.963, 0.994]** |
| 2L / 25k  (vs step-25,000) | 1.030 [1.010, 1.052] | 1.006 [0.987, 1.024] | 0.993 [0.980, 1.007] | 1.076 [1.055, 1.098] | 1.026 [1.006, 1.048] | 0.993 [0.978, 1.009] |
| 6L / 25k  (vs step-25,000) | 1.016 [0.997, 1.036] | 0.986 [0.969, 1.002] | **0.978 [0.967, 0.988]** | 1.050 [1.032, 1.069] | 1.029 [1.008, 1.052] | 0.999 [0.983, 1.018] |
| 2L / 50k  (vs step-50,000) | 1.048 [1.023, 1.075] | **0.974 [0.959, 0.988]** | **0.970 [0.950, 0.989]** | 1.135 [1.098, 1.173] | 1.012 [0.988, 1.038] | — |
| 6L / 50k  (vs step-50,000) | 1.054 [1.031, 1.078] | **0.975 [0.958, 0.991]** | **0.973 [0.959, 0.987]** | 1.067 [1.039, 1.095] | 1.027 [1.001, 1.057] | — |

Ratio A / arm C: values below 1 mean the sibling arm beats arm C at that cell. Bold marks a CI that excludes 1.0 on the "lower than arm C" side. The seed-2 retrain of arm C absorbs one seed of noise into each ratio.

The periodic subset covers 37 of the 97 configs (`experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_periodic.csv`). There bimoco is again the lowest arm in every cell, and separated at 95 % on 10 of the 12 arm-1/3/4 rows. Both exceptions are arm 4 `best`: the 2L CI is 0.976–1.103 and the 6L CI is 0.991–1.108.
