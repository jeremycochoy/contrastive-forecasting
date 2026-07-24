# Splitting the contrastive loss: MoCo on both terms leads the sibling arms at 12,500 steps; no arm is established as beating the SIGReg champion

**Question.** The champion loss merges five negative tensors under one pooled log-sum-exp denominator. Does splitting it into `L_pred` (f-anchored) and `L_rep` (h-anchored) improve GM-Relative MASE on the GIFT-Eval 97-config panel, and does adding EMA-teacher MoCo keys or replacing `L_pred` with BYOL alignment change the answer? (terms defined in annex)

**Answer.** No arm is established as beating the SIGReg champion (arm C). Among the six *sibling arms* — the six recipes trained here, sharing one backbone configuration and differing only in loss shape and key source — `L_pred_moco + L_rep_moco` (*bimoco*: the split loss with EMA-teacher MoCo keys on both terms) has the lowest GM-Relative MASE at all four 12,500-step (head, checkpoint) cells and is 95 %-separated (task-level 95 % CI excludes ratio 1.0) from arms 1, 3, 4, 5, 6 in every cell (paired task-bootstrap, below); on the two arm-4 `best` rows the compared backbones are 11,800 steps apart (arm 4 step 600 vs bimoco step 12,400), so those two rows mix loss shape with backbone step. bimoco has a lower point estimate than arm C in all four cells, but arm C has no per-task file, so that difference carries no CI. The lead does not hold past 12,500 steps: arm 4 (pooled + MoCo) is the lowest arm at 25,000 steps. Single seed (20260520).

## Result

![Downstream GM-Relative MASE per arm](plots/headline_relmase.png)

*Downstream GM-Relative MASE per arm at each (head, checkpoint) cell (point estimates; N = 1 seed). Dashed line = seasonal-naive (1.0). The hatched bar is arm C ref † — an external aggregate read from `experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`, not reproducible from this experiment's results, and with no CI. Arm separation is the paired task-bootstrap in the CI-forest figure below, not a bar-to-bar comparison here.*

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo on `L_pred`) | 1.1548 | 1.1683 | 1.1338 | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | 1.1546 | 1.1603 | 1.1405 |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm 6 (`L_align` + `L_rep_moco`) | 1.1771 | 1.1712 | 1.1768 | 1.1767 |
| arm bimoco (`L_pred_moco` + `L_rep_moco`) | **1.1225** | **1.1180** | **1.1138** | **1.1087** |
| arm C ref (SIGReg-cross champion) † | 1.1682 | 1.1491 | 1.1561 | 1.1254 |

Sibling-arm cells: the `Aggregate GM-Relative MASE (97 configs)` line of `experiments/2026-07-10_split_pred_rep/results*/gift_eval_full_*/summary.txt`.

† arm C ref: aggregate read from `experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv` (arm `cross_C`); no per-task file exists. These four point estimates come from an external file and cannot be reproduced from this experiment's results.

## GM-Relative MASE across backbone step

![GM-Relative MASE per arm across backbone step](plots/gm_curve_per_arm.png)

*GM-Relative MASE per arm across backbone step (2k / best / 12,500 / 25k / 50k where available), shared y-axis, with the arm C champion † best (dotted) and last (dashed) cells as horizontal references. The 12,500 `best` / `last` cells use a 30k best-loss head (+10k resume for `last`); the 2k / 25k / 50k cells use a fresh 40k head on that snapshot.*

At 25,000 steps arm 4 (pooled + MoCo) is the lowest arm in both panels (6L: arm 4 = 1.1073 vs bimoco = 1.1319; 2L: arm 4 = 1.1332 vs bimoco = 1.1339), so the bimoco lead is established only at the four 12,500-step cells used for the arm ranking. Other 6L / 25k cells: arm 3 = 1.1170, arm 1 = 1.1500 (`experiments/2026-07-10_split_pred_rep/results*/gift_eval_full_*_25k_*L/summary.txt`).

## Paired-bootstrap 95 % CIs on GM-Relative MASE ratios

![Paired-bootstrap 95 % CIs on GM-Relative MASE ratios](plots/ci_forest.png)

*Paired-bootstrap 95 % CIs on GM-Relative MASE ratios. Circle = task-level bootstrap; square (faded) = 28-dataset-clustered bootstrap. `*` marks checkpoint-selection- or step-confounded rows. n_boot = 20,000, seed 42. The figure shows 34 of the 60 contrasts counted in the table below: all 12 arm-1/3/4 pairwise rows and all 12 arm-1/3/4-vs-bimoco rows (`best` and `last`), plus the `last`-cell rows for arm 5 vs arm 1, arms 1/3/4 vs arm 6, and arm 5 vs arm 6. The remaining 26 contrasts are counted in the table but not plotted.*

Rows counted as separated when the task-level 95 % CI excludes ratio 1.0 (`experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_*_nboot200k.csv` for the arm-5/6/bimoco references, `experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci.csv` for the arm-1/3/4 pairwise; direction ratio > 1 → the named arm is worse than the reference):

| contrast set | rows | separated at 95 % (task-level CI) |
| --- | --: | --: |
| arm 1 / 3 / 4 pairwise | 12 | 2 / 12 (both `best`, checkpoint-confounded) |
| arm 5 vs arm 1 / 3 / 4 | 12 | 12 / 12 (arm 5 worse) |
| arm 6 vs arm 1 / 3 / 4 | 12 | 4 / 12 (6L: best arm 3; last arms 1, 3, 4 lower than arm 6) |
| arm 5 vs arm 6 | 4 | 3 / 4 (arm 6 lower) |
| **arm 1 / 3 / 4 vs bimoco** | **12** | **12 / 12 (bimoco lower)** |
| arm 5 / arm 6 vs bimoco | 8 | 8 / 8 (bimoco lower) |

Within the arm-1/3/4-vs-bimoco family, 10 of the 12 rows also clear the Bonferroni threshold α = 0.05 / 60 = 0.000833; the two that miss are 2L / best arm 4 vs bimoco (p₂ = 0.00435) and 6L / best arm 3 vs bimoco (p₂ = 0.00426) (`two_sided_p` column of `experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_bimoco_nboot200k.csv`). On the periodic subset (37 of the 97 configs, `experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_periodic.csv`) bimoco is again the lowest arm in every cell, separated at 95 % on 10 of the 12 arm-1/3/4 rows; both exceptions are arm 4 `best` (2L CI 0.976–1.103, 6L CI 0.991–1.108).

## Denominator share

![Per-family denominator share](plots/gradient_share_stack.png)

*Per-family denominator share at each arm's best-cell backbone snapshot (arm 1: step 12,500; arm 3: step 11,800; arm 4: step 600), on a mixed batch and a periodic-only batch (solar / electricity windows), τ = 0.10, probe B = 64. `share_i = exp(mean(logit_i − log-denominator))` is a per-anchor geometric mean, so the families need not sum to 1; each bar's column sum Σ is printed above it.*

Measured share of the cross-batch `f ↔ h′` family (`log_neg_cross_batch`) in the term carrying the prediction pairs (`experiments/2026-07-10_split_pred_rep/results/gradient_share_measurement.csv`):

| arm | term | mixed batch | periodic batch |
| --- | --- | --: | --: |
| arm 1 (split, step 12,500) | `L_pred` | 0.901 | 0.991 |
| arm 3 (split + MoCo, step 11,800) | `L_pred` | 0.937 | 0.997 |
| arm 4 (pooled, step 600) | pooled | 0.003 | 0.003 |

The three probed snapshots span 11,900 steps, so this split-vs-pooled difference is not separated from backbone step; bimoco was not probed. The share is measured at probe B = 64 (diagnostic), whereas training ran at B = 512; cross-batch share is B-dependent, so the absolute shares are indicative, not identical to the training-time split.

## Method (annex)

All backbones: B = 512, T = 4096, C = 1, τ = 0.10, `lr = 1e-3`, seed 20260520, dataset `gift-pretrain-full-4096 / small_v1`, EMA teacher τ = 0.90, SIGReg λ_e = λ_h = 1, CPC auxiliary, 12,500 steps.

Two backbone snapshots per arm feed the downstream head. The **best-cell backbone** (file `bb_<run>_FINAL.pth`) is a copy of the arm's `_best_loss.pth` save; arm 1 recorded no `best_loss` save, so its best-cell backbone is byte-identical to its step-12,500 backbone (`experiments/2026-07-10_split_pred_rep/results/backbone_step_verification.log`). The **step-12,500 backbone** (file `bb_<run>_final.pth`) is the end-of-training checkpoint. The `best` cell trains a fresh 30k-step quantile head (2L or 6L) on the best-cell backbone; the `last` cell resumes that head +10k steps on the step-12,500 backbone. The 2k / 25k / 50k trajectory cells each train a fresh 40k-step head on the corresponding backbone snapshot.

Eval: GIFT-Eval 97 configs, strategy B4, quantile-median MASE divided by the seasonal-naive reference in `experiments/2026-07-10_split_pred_rep/results/seasonal_naive_all_results.csv`.

f-anchored retrieval is saturated across all arms after step 600 — `auc` ≥ 0.9975, minimum `top1` 0.8348 (arm 1, step 3,343), every other arm above 0.95 (`auc` / `top1` columns of `experiments/2026-07-10_split_pred_rep/runs*/bb_*_losses*.csv`) — so this diagnostic does not separate the arms.

## Arms (annex)

| arm | loss shape | flags | key structure |
| --- | --- | --- | --- |
| arm 1 | `cosine_similarity_batch_split_pred_rep` | — | `L = L_pred + L_rep`, no MoCo |
| arm 3 | `cosine_similarity_batch_split_pred_rep` | `--moco-negatives` | `L_pred` uses EMA-teacher keys on cross-batch f ↔ h |
| arm 4 | `cosine_similarity_batch_full_hh_negs_xshh_allt` | `--moco-negatives --pos-in-denominator --subtract-contrastive-floor` | pooled champion shape with EMA-teacher keys |
| arm 5 | `cosine_similarity_batch_rep_only` | `--align-loss-weight 1.0` | `L = L_align + L_rep`; `L_align` is BYOL to teacher-encoder next-step latent |
| arm 6 | `cosine_similarity_batch_rep_only` | `--align-loss-weight 1.0 --moco-rep-keys` | arm 5's `L_align` + `L_rep_moco`: h-family keys routed through EMA teacher |
| arm bimoco | `cosine_similarity_batch_split_pred_rep` | `--moco-negatives --moco-rep-keys` | `L = L_pred_moco + L_rep_moco`; MoCo + positive-in-denominator on both terms |
| arm C ref | `cosine_similarity_batch_full_hh_negs_xshh_allt` | — | SIGReg-cross champion (λ_e = 1, λ_h = 1, EMA τ = 0.90), reused without retraining |

## Backbone step (annex)

| arm | `best` cell step | `last` cell step | best-cell backbone = |
| --- | --: | --: | --- |
| arm 1 | 12,500 | 12,500 | the step-12,500 checkpoint, not a loss-argmin checkpoint |
| arm 3 | 11,800 | 12,500 | `_best_loss.pth` at step 11,800 |
| arm 4 | 600 | 12,500 | `_best_loss.pth` at step 600 |
| arm 5 | 11,800 | 12,500 | `_best_loss.pth` at step 11,800 |
| arm 6 | 8,700 | 12,500 | `_best_loss.pth` at step 8,700 |
| arm bimoco | 12,400 | 12,500 | `_best_loss.pth` at step 12,400 |
| arm C ref | not exported | 12,500 | — |

Arms 1 and 4's `best`-cell backbone is not the argmin of a comparable curve: arm 1's shipped checkpoint is step 12,500, and arm 4's loss never returns below its step-600 value. Six of the twelve arm-1/3/4 pairwise rows, and six of the twelve arm-1/3/4-vs-bimoco rows, are `best` rows and mix the loss-shape axis with this checkpoint-selection axis.

## Definitions (annex)

- *f* / *h* — *f* is the forecaster's next-step latent (the forecast); *h* is the encoder's original latent. *f-anchored* / *h-anchored* families take *f* resp. *h* as the query.
- *`L_pred`* — normalized InfoNCE, f-anchored: positive `cos(f_t, h'_{t+1})`; denominator = adjacent `f_{t+1} ↔ f_t` negatives plus cross-batch `f_t ↔ h'_{t+1}` negatives.
- *`L_rep`* — pooled log-sum-exp of three h-anchored families (cross-channel `h ↔ h`, within-series all-time `h ↔ h`, cross-series all-time `h ↔ h'`), with no positive term.
- *GM-Relative MASE* — geometric mean over the 97 GIFT-Eval configs of `(model MASE) / (seasonal-naive MASE)`, at quantile 0.5. Lower is better; 1.0 = seasonal-naive.
- *log-sum-exp denominator* — the InfoNCE normaliser at an anchor: the soft-max over all negative similarity scores (and, under `--pos-in-denominator`, the positive score too).
- *MoCo* — cross-batch keys sourced from an EMA teacher (τ = 0.90) instead of the student.
- *bimoco* — the arm applying MoCo keys to both split terms: `L = L_pred_moco + L_rep_moco`.
- *sibling arms* — the six arms trained in this experiment (1, 3, 4, 5, 6, bimoco), sharing one backbone configuration and differing only in loss shape and key source. arm C is not a sibling arm: it was trained in an earlier experiment and reused here without retraining.
- *SIGReg* — the spectral isotropy regulariser applied to the encoder (λ_e) and head (λ_h) latents; the champion uses λ_e = λ_h = 1.
- *CPC* — the contrastive predictive-coding auxiliary head, on in every arm.
- *BYOL alignment (`L_align`)* — `2 − 2·cos(f_t, sg(h^T_{t+1}))` (sg = stop-gradient); negative-free, minimum 0.
- *`L_rep_moco`* — normalized InfoNCE over the three h-anchored families with teacher-side keys; the same-batch same-time student ↔ teacher pair sits in both numerator and denominator log-sum-exp.
- *`auc` / `top1`* — retrieval quality of the f-anchored positive against the B = 512 cross-batch candidates: ROC area of the positive-score distribution, and the fraction of anchors whose positive ranks first.
- *B4* — GIFT-Eval evaluation strategy B4 (teacher-forced probe over the full 97-config panel).
- *best / last cell* — the two downstream checkpoints per arm: `best` = head on the best-cell backbone, `last` = head resumed on the step-12,500 backbone.
- *arm C* — the SIGReg-cross champion recipe (`cross_C`: λ_e = 1, λ_h = 1, EMA τ = 0.90), the baseline this sweep is ranked against.
- *Bonferroni family* — 60 contrasts = 15 arm pairs × 4 (head, checkpoint) cells on the full-97 panel, α = 0.05 / 60 = 0.000833.
</content>
