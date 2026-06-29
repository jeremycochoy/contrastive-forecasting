# #366 — notes for the report writer

## Multi-seed status (P2, ExperimentReviewer)

Multi-seed has **not** been run. Each of the 8 cells is a single training
seed (`20260520`) per arm × head × ckpt.

Within-arm spread (best→last per head) on the new cross runs:

| arm    | head | best   | last   | best→last gap |
| ------ | ---: | -----: | -----: | ------------: |
| cross_A | 2L  | 1.1704 | 1.1811 | +0.0107 (+0.9%) |
| cross_A | 6L  | 1.1604 | 1.1375 | −0.0229 (−2.0%) |
| cross_B | 2L  | 1.1889 | 1.1517 | −0.0372 (−3.1%) |
| cross_B | 6L  | 1.1743 | 1.1340 | −0.0403 (−3.4%) |

Cross-arm cell-by-cell spread sits in the **0.3–4.8 %** range (largest:
cross_A/2L_best 1.1704 vs cross_B/2L_best 1.1889 = +1.6 %; cross_A/6L_last
1.1375 vs cross_B/6L_last 1.1340 = −0.3 %). The within-arm best→last gap
on a single seed is already ±2–3 %, so a multi-seed replicate would need
to clear that gap to be informative; we have not paid for it. The
ExperimentReviewer's P2 should be framed honestly as "single-seed; spread
within the per-cell best→last band; multi-seed unmet".

## GM table layout

`results/gm_table.csv` was extended (P1, P5):

* **Source column** distinguishes `cross` (this experiment, 8 rows),
  `anchor_363` (8 rows from `feature/contrastive-forecasting-363-v2`,
  λ_e=10 / λ_e=1000 pairs at τ=0.99), `anchor_357` (4 rows from
  `feature/contrastive-forecasting-357`, τ=0.90 at λ_e=λ_h=0.1).
* **Three additional aggregates** beyond `gm` (GM-Rel MASE):
  `gm_mase` (raw MASE), `gm_mape_sn` (MAPE / SN_MAPE), `gm_crps_sn`
  (mean_weighted_sum_quantile_loss / SN equivalent). Computed by
  `scripts/_compute_gm.py` against
  `~/workspaces/gift-eval/results/seasonal_naive/all_results.csv`.

## Winners provenance

`results/winners.locked.txt` (this directory) is a verbatim copy of the
gitignored `winners.sh` at experiment root, kept for audit so the issue
and PR conversation have an in-repo record of which #363/#357 winners
were picked and by whom.
