# GM-table aggregates note (#356 review P3.7)

The GIFT-Eval full-97 wrapper used in this run (`scripts/run_gift_eval_full.sh`) emits only **GM-Relative MASE** in each `summary.txt`. Inspected files:

- `results/gift_eval_full_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_2L/summary.txt` → `Aggregate GM-Relative MASE (97 configs): 1.1610`
- `..._last_2L/summary.txt` → `1.1758`
- `..._6L/summary.txt` → `1.1543`
- `..._last_6L/summary.txt` → `1.1556`

No `GM-MASE`, `GM-MAPE_SN`, or `GM-CRPS_SN` aggregates are emitted by this wrapper. The per-config `all_results.csv` carries raw `MASE[0.5]`, `MAPE[0.5]`, and `mean_weighted_sum_quantile_loss` (CRPS-like) for our model, but **not** the seasonal-naive denominators needed to form the SN-relative versions — those would require re-running the GIFT-Eval wrapper with a flag to emit additional aggregates, out of scope for this artefact set.

`gm_table.csv` therefore carries one `gm` column (GM-Relative MASE), uniformly to 4 decimal places. The #344 / #353 reference values are embedded as constants in `experiments/2026-06-20_lejepa_sigreg/scripts/build_report.py` (`REF_GM`, `EMA_GM`) at their reported published precision; #353's came from the PR #354 head-matched table, #344's from the #344 PR. No fresh head-matched re-eval of those references was run at this code revision.
