# Splitting the contrastive loss (L_pred + L_rep) is not established as beating the SIGReg champion; MoCo-on-both (bimoco) point-beats the champion in all four (head, checkpoint) cells and leads the sibling arms with CIs at 12,500 steps

**Question.** The champion loss merges five negative tensors under one pooled log-sum-exp denominator. Does splitting it into `L_pred` (f-anchored) and `L_rep` (h-anchored) improve GM-Relative MASE on the GIFT-Eval 97-config panel, and does adding EMA-teacher MoCo keys or replacing `L_pred` with BYOL alignment change the answer? (terms defined in annex)

**Answer.** No arm is established as beating the SIGReg-champion (arm C). Among the six sibling arms, `L_pred_moco + L_rep_moco` (bimoco) has the lowest GM-Relative MASE at all four 12,500-step (head, checkpoint) cells and is 95 %-separated (task-level 95 % CI excludes ratio 1.0) from arms 1, 3, 4, 5, 6 in every cell (paired task-bootstrap, below). bimoco's point estimates are also below arm C in all four cells (narrowest gap 0.0167 at 6L / last), but arm C has no per-task file on this branch, so that gap is a point-estimate difference with no CI. The lead does not hold past 12,500 steps: arm 4 (pooled + MoCo) matches or beats bimoco at 25k. Single seed (20260520).

## Result

![Downstream GM-Relative MASE per arm at each (head, checkpoint) cell (point estimates; N = 1 seed). Dashed line = seasonal-naive (1.0). The hatched bar is arm C ref † — an external aggregate read from `experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv`, not reproducible from this experiment's `results/`, and with no CI. Arm separation is the paired task-bootstrap in the CI-forest figure below, not a bar-to-bar comparison here.](plots/headline_relmase.png)

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --: | --: | --: | --: |
| arm 1 (split) | 1.1654 | 1.1669 | 1.1575 | 1.1557 |
| arm 3 (split + MoCo on `L_pred`) | 1.1548 | 1.1683 | 1.1338 | 1.1511 |
| arm 4 (pooled + MoCo) | 1.1602 | 1.1546 | 1.1603 | 1.1405 |
| arm 5 (`L_align` + `L_rep`) | 1.3374 | 1.2883 | 1.2554 | 1.2201 |
| arm 6 (`L_align` + `L_rep_moco`) | 1.1771 | 1.1712 | 1.1768 | 1.1767 |
| arm bimoco (`L_pred_moco` + `L_rep_moco`) | **1.1225** | **1.1180** | **1.1138** | **1.1087** |
| arm C ref (SIGReg-cross champion) † | 1.1682 | 1.1491 | 1.1561 | 1.1254 |

† arm C ref: aggregate read from `experiments/2026-06-28_sigreg_lambda_tau_cross/results/gm_table.csv` (arm `cross_C`); no per-task file exists on this branch. These four point estimates are read from an external file and cannot be reproduced from `results/` in this experiment.

## GM-Relative MASE across backbone step

![GM-Relative MASE per arm across backbone step (2k / best / 12,500 / 25k / 50k where available), shared y-axis, with the arm C champion † best (dotted) and last (dashed) cells as horizontal references. The 12,500 `best`/`last` cells use a 30k best-loss head (+10k resume for `last`); the 2k / 25k / 50k cells use a fresh 40k head on that snapshot.](plots/gm_curve_per_arm.png)

Continuing past 12,500 steps lowers no arm below its 12,500-step value, and arm 4 (pooled + MoCo) matches or beats bimoco at 25k (arm 4 6L 25k = 1.1073 vs bimoco 6L 25k = 1.1319), so the bimoco lead is established only at the four 12,500-step cells used for the arm ranking.

## Paired-bootstrap 95 % CIs on GM-Relative MASE ratios

![Paired-bootstrap 95 % CIs on GM-Relative MASE ratios. Circle = task-level bootstrap; square (faded) = 28-dataset-clustered bootstrap. `*` marks checkpoint-selection- or step-confounded rows. n_boot = 20,000, seed 42.](plots/ci_forest.png)

Rows counted as separated when the task-level 95 % CI excludes ratio 1.0 (`experiments/2026-07-10_split_pred_rep/results/pairwise_bootstrap_ci_*_nboot200k.csv` for the arm-5/6/bimoco references, `pairwise_bootstrap_ci.csv` for the arm-1/3/4 pairwise; direction ratio > 1 → the named arm is worse than the reference):

| contrast set | rows | separated at 95 % (task-level CI) |
| --- | --: | --: |
| arm 1 / 3 / 4 pairwise | 12 | 2 / 12 (both `best`, checkpoint-confounded) |
| arm 5 vs arm 1 / 3 / 4 | 12 | 12 / 12 (arm 5 worse) |
| arm 6 vs arm 1 / 3 / 4 | 12 | 4 / 12 (6L: best arm 3; last arms 1, 3, 4 lower than arm 6) |
| arm 5 vs arm 6 | 4 | 3 / 4 (arm 6 lower) |
| **arm 1 / 3 / 4 vs bimoco** | **12** | **12 / 12 (bimoco lower)** |
| arm 5 / arm 6 vs bimoco | 8 | 8 / 8 (bimoco lower) |

Within the arm-1/3/4-vs-bimoco family, 10 of the 12 rows also clear the Bonferroni threshold α = 0.05 / 60 = 0.000833; the two that miss are 2L / best arm 4 vs bimoco (p₂ = 0.00435) and 6L / best arm 3 vs bimoco (p₂ = 0.00426) (`two_sided_p` column of `results/pairwise_bootstrap_ci_bimoco_nboot200k.csv`). On the periodic-cluster subset (`results/pairwise_bootstrap_ci_periodic.csv`) bimoco is again the lowest arm in every cell, separated at 95 % on 10 of the 12 arm-1/3/4 rows (both exceptions are arm 4 `best`).

## Denominator share

![Per-family denominator share at each arm's FINAL.pth backbone snapshot (arm 1: step 12,500; arm 3: step 11,800; arm 4: step 600), on a mixed batch and a periodic-only batch (solar/electricity windows), τ = 0.10, probe B = 64.](plots/gradient_share_stack.png)

Splitting gives the prediction pairs a near-full denominator; pooling collapses their share to 0.003 (both batches).

The share is measured at probe B = 64 (diagnostic), whereas training ran at B = 512; cross-batch share is B-dependent, so the absolute shares are indicative, not identical to the training-time split.

## Method (annex)

All backbones: B = 512, T = 4096, C = 1, τ = 0.10, `lr = 1e-3`, seed 20260520, dataset `gift-pretrain-full-4096 / small_v1`, EMA teacher τ = 0.90, SIGReg λ_e = λ_h = 1, CPC auxiliary, 12,500 steps. Downstream head (quantile, 2L or 6L): the `best` cell trains a fresh 30k-step head on `FINAL.pth` (the best-loss backbone); the `last` cell resumes that head +10k steps on `final.pth` (step 12,500). The 2k / 25k / 50k trajectory cells each train a fresh 40k-step head on the corresponding backbone snapshot. Eval: GIFT-Eval 97 configs, strategy B4, quantile-median MASE divided by the branch-committed seasonal-naive reference (`experiments/2026-07-10_split_pred_rep/results/seasonal_naive_all_results.csv`).

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

| arm | `best` cell step | `last` cell step | `FINAL.pth` = |
| --- | --: | --: | --- |
| arm 1 | 12,500 | 12,500 | `FINAL.pth` is the step-12,500 checkpoint, not a loss-argmin checkpoint |
| arm 3 | 11,800 | 12,500 | `best_loss.pth` at step 11,800 |
| arm 4 | 600 | 12,500 | `best_loss.pth` at step 600 |
| arm 5 | 11,800 | 12,500 | `best_loss.pth` at step 11,800 |
| arm 6 | 8,700 | 12,500 | `best_loss.pth` at step 8,700 |
| arm bimoco | 12,400 | 12,500 | `best_loss.pth` at step 12,400 |
| arm C ref | not exported | 12,500 | — |

Arms 1 and 4's `best`-cell backbone is not the argmin of a comparable curve: arm 1's shipped checkpoint is step 12,500, and arm 4's loss never returns below its step-600 value. Six of the twelve arm-1/3/4 Bonferroni-family rows are `best` rows and mix the loss-shape axis with this checkpoint-selection axis.

## Definitions (annex)

- *f* / *h* — *f* is the forecaster's next-step latent (the forecast); *h* is the encoder's original latent. *f-anchored* / *h-anchored* families take *f* resp. *h* as the query.
- *`L_pred`* — normalized InfoNCE, f-anchored: positive `cos(f_t, h'_{t+1})`; denominator = adjacent `f_{t+1} ↔ f_t` negatives plus cross-batch `f_t ↔ h'_{t+1}` negatives.
- *`L_rep`* — pooled log-sum-exp of three h-anchored families (cross-channel `h ↔ h`, within-series all-time `h ↔ h`, cross-series all-time `h ↔ h'`), with no positive term.
- *GM-Relative MASE* — geometric mean over the 97 GIFT-Eval configs of `(model MASE) / (seasonal-naive MASE)`, at quantile 0.5. Lower is better; 1.0 = seasonal-naive.
- *log-sum-exp denominator* — the InfoNCE normaliser at an anchor: the soft-max over all negative similarity scores (and, under `--pos-in-denominator`, the positive score too).
- *MoCo* — cross-batch keys sourced from an EMA teacher (τ = 0.90) instead of the student.
- *SIGReg* — the spectral isotropy regulariser applied to the encoder (λ_e) and head (λ_h) latents; the champion uses λ_e = λ_h = 1.
- *CPC* — the contrastive predictive-coding auxiliary head, on in every arm.
- *BYOL alignment (`L_align`)* — `2 − 2·cos(f_t, sg(h^T_{t+1}))` (sg = stop-gradient); negative-free, minimum 0.
- *`L_rep_moco`* — normalized InfoNCE over the three h-anchored families with teacher-side keys; the same-batch same-time student ↔ teacher pair sits in both numerator and denominator log-sum-exp.
- *`auc` / `top1`* — retrieval quality of the f-anchored positive against the B = 512 cross-batch candidates: ROC area of the positive-score distribution, and the fraction of anchors whose positive ranks first.
- *B4* — GIFT-Eval evaluation strategy B4 (teacher-forced probe over the full 97-config panel).
- *best / last cell* — the two downstream checkpoints per arm: `best` = head on the best-loss backbone, `last` = head resumed on the step-12,500 backbone.
- *arm C* — the SIGReg-cross champion recipe (`cross_C`: λ_e = 1, λ_h = 1, EMA τ = 0.90), the baseline this sweep is ranked against.
- *Bonferroni family* — 60 contrasts = 15 arm pairs × 4 (head, checkpoint) cells on the full-97 panel, α = 0.05 / 60 = 0.000833.

## f-anchored retrieval saturation (annex)

f-anchored retrieval is saturated across all arms after step 600: `auc` ≥ 0.9975, minimum `top1` 0.8348 (arm 1, step 3,343; `runs/bb_split_pred_rep_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau090_losses_full.csv`, `top1` column), every other arm above 0.95.
