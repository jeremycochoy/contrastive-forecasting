# Splitting the contrastive loss (L_pred + L_rep) and/or MoCo-style teacher latents both improve forecasting

**Question.** The baseline loss, arm C, puts every negative under a single log-sum-exp denominator. Does splitting it into `L_pred` (f-anchored) and `L_rep` (h-anchored) improve GM-Relative MASE, and does adding EMA-teacher MoCo keys or replacing `L_pred` with cosine-distance minimization to the teacher (`L_align`) change the answer?

**Answer.** Two arms beat the arm C baseline with a 95 % paired-bootstrap CI (Annex D). Which one is lowest depends on backbone step: bimoco at 12,500, arm 4 (pooled MoCo) from 25,000 onward (bimoco has no 50k cell).

![GM-Relative MASE per arm across backbone step](plots/gm_curve_per_arm.png)

*Bimoco has no 50,000-step cell.*

## Aligning the training signal with downstream MASE

![Backbone training loss aligned with evaluated GM-Relative MASE snapshots](plots/loss_vs_gm_snapshots.png)

![Retrieval error (1 − ff) aligned with evaluated GM-Relative MASE snapshots, per arm](plots/ff_vs_gm_snapshots.png)

*`ff = cos(f̂_t, f_true_{t+1})`, the positive-pair cosine similarity.*

## Snapshot at 12,500 backbone steps

![Downstream GM-Relative MASE per arm](plots/headline_relmase.png)

## The arms

Notation: `h_t` = encoder latent; `f_t` = forecaster's next-step latent (the forecast); `h′` = cross-batch latent; `ᵀ` = EMA-teacher copy (τ_ema = 0.90); `sg` = stop-gradient. Cosine similarities at τ = 0.10.

```
L_pred        f-anchored NCE:  positive cos(f_t, h′_{t+1});  negatives  adjacent f_{t+1}↔f_t  ∪  cross-batch f_t↔h′_{t+1}
L_pred_moco   = L_pred with the cross-batch keys taken from the teacher
L_rep         h-anchored LSE:  cross-channel h↔h  ∪  within-series h↔h  ∪  cross-series h↔h′;  no positive term
L_rep_moco    = L_rep with teacher-side keys; the same-time student↔teacher pair is the positive and also sits in the denominator
L_align       2 − 2·cos(f_t, sg(hᵀ_{t+1}))   cosine-distance minimization, no negatives
```

| arm | loss | shape (prefix `cosine_similarity_batch_`) + flags |
| --- | --- | --- |
| arm 1 | `L_pred + L_rep` | `split_pred_rep` |
| arm 3 | `L_pred_moco + L_rep` | `split_pred_rep` `--moco-negatives` |
| arm 4 | one pooled denominator over all five negative families, teacher keys | `full_hh_negs_xshh_allt` `--moco-negatives --pos-in-denominator --subtract-contrastive-floor` |
| arm 5 | `L_align + L_rep` | `rep_only` `--align-loss-weight 1.0` |
| arm 6 | `L_align + L_rep_moco` | `rep_only` `--align-loss-weight 1.0 --moco-rep-keys` |
| bimoco | `L_pred_moco + L_rep_moco` | `split_pred_rep` `--moco-negatives --moco-rep-keys` |
| arm C | arm 4's pooled shape with student keys | baseline; seed-2 retrain (Annex C) |

## Annex A — Trajectory cells

One cell holds one arm's GM-Relative MASE at one backbone training step; a row is that arm's trajectory across training.

| arm | 2L 2k | 2L best | 2L last (12,500) | 2L 25k | 2L 50k | 6L 2k | 6L best | 6L last (12,500) | 6L 25k | 6L 50k |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| arm 1 (best @ 12,500) | 1.2005 | 1.1654 | 1.1669 | 1.1761 | 1.2334 | 1.1938 | 1.1575 | 1.1557 | 1.1500 | 1.2129 |
| arm 3 (best @ 11,800) | 1.2075 | 1.1548 | 1.1683 | 1.1484 | 1.1461 | 1.1825 | 1.1338 | 1.1511 | 1.1170 | 1.1221 |
| arm 4 (best @ 600) | 1.1874 | 1.1602 | 1.1546 | **1.1332** | 1.1414 | 1.1564 | 1.1603 | 1.1405 | **1.1073** | 1.1199 |
| arm 5 (best @ 11,800) | 1.2208 | 1.3374 | 1.2883 | 1.2279 | 1.3357 | 1.1829 | 1.2554 | 1.2201 | 1.1889 | 1.2279 |
| arm 6 (best @ 8,700) | 1.1601 | 1.1771 | 1.1712 | 1.1714 | 1.1907 | 1.1604 | 1.1768 | 1.1767 | 1.1655 | 1.1823 |
| arm bimoco (best @ 12,400) | 1.1438 | **1.1225** | **1.1180** | 1.1339 | — | 1.1337 | **1.1138** | **1.1087** | 1.1319 | — |
| arm C ref (seed 2) | — | — | 1.1441 | 1.1415 | 1.1768 | — | — | 1.1318 | 1.1325 | 1.1510 |

Bold = column minimum.

## Annex B — Denominator share

How much of each loss term's denominator does the cross-batch `f ↔ h′` family occupy?

![Per-family denominator share](plots/gradient_share_stack.png)

*Per-family denominator share at every arm's step-12,500 backbone; we probe on a mixed batch and a periodic-only batch at τ = 0.10, B = 64. `share_i = exp(mean(logit_i − log-denominator))` is a per-anchor geometric mean, so families need not sum to 1; each bar shows its column sum Σ above.*

Share of the cross-batch `f ↔ h′` family in the term carrying the prediction pairs (`results/gradient_share_measurement_step12500.csv`):

| arm | term | mixed batch | periodic batch |
| --- | --- | --: | --: |
| arm 1 | `L_pred` | 0.901 | 0.991 |
| arm 3 | `L_pred_moco` | 0.925 | 0.995 |
| bimoco | `L_pred_moco` | 0.515 | 0.893 |
| arm 4 | pooled | 0.004 | 0.004 |

## Annex C — Method

The six arms share a backbone recipe: B = 512, T = 4096, C = 1, τ = 0.10, `lr = 1e-3`, seed 20260520, dataset `gift-pretrain-full-4096 / small_v1`, EMA teacher τ = 0.90, SIGReg λ_e = λ_h = 1, CPC auxiliary. Each ran 12,500 steps; we extended five of them to 25,000 and 50,000. Arm C, the baseline, uses the best recipe of the `2026-06-28_sigreg_lambda_tau_cross` experiment (λ_e = 1, λ_h = 1, τ = 0.90); we retrained it here with a fresh seed and evaluated at steps 12,500 / 25,000 / 50,000.

Eval: GIFT-Eval 97 configs, strategy B4, quantile-median MASE / seasonal-naive. For each `2k` / `25k` / `50k` cell we train a fresh 40k-step quantile head on that snapshot; for `best` we train a 30k-step best-loss head, and for `last` we resume that head a further 10k steps. Per-cell source paths and checkpoint layout: `experiments/2026-07-10_split_pred_rep/README.md`.

## Annex D — Paired-bootstrap 95 % CIs

![Paired-bootstrap 95 % CIs on GM-Relative MASE ratios](plots/ci_forest.png)

| contrast set | rows | separated at 95 % (task-level CI excludes 1.0) |
| --- | --: | --: |
| arm 1 / 3 / 4 pairwise | 12 | 2 / 12 (both `best`, checkpoint-confounded) |
| arm 5 vs arm 1 / 3 / 4 | 12 | 12 / 12 (arm 5 worse) |
| arm 6 vs arm 1 / 3 / 4 | 12 | 4 / 12 (6L: best arm 3; last arms 1, 3, 4 lower than arm 6) |
| arm 5 vs arm 6 | 4 | 3 / 4 (arm 6 lower) |
| **arm 1 / 3 / 4 vs bimoco** | **12** | **12 / 12 (bimoco lower)** |
| arm 5 / arm 6 vs bimoco | 8 | 8 / 8 (bimoco lower) |

10 of the 12 arm-1/3/4-vs-bimoco rows also clear Bonferroni α = 0.05 / 60. On the 37-config periodic subset bimoco is again lowest in every cell.
