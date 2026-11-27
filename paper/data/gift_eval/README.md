# GIFT-Eval tables of the paper

Each CSV has the 97 GIFT-Eval configurations of one model, one row per configuration. `paper/scripts/plot_gift_eval_radar.py` reads them and draws `paper/figures/gift_eval_radar.png` (Section 6.1 of the paper). The score of a model is the geometric mean, over the 97 rows, of `eval_metrics/MASE[0.5]` divided by the same column of `seasonal_naive.csv`. Join the tables on the `dataset` column: the row orders differ.

| File | Model | Source of the copy |
|---|---|---|
| `green_w64_cd135_166k.csv` | Our model at width 64, run green (`g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_cd135`, exponential cooldown from 135k), backbone at 166k, head `h30k_control`. GM-MASE 1.0057. | elisa: `/home/jupyter/cf_runs/hp64/ff/heads/eval/<run>_bb166k_h30k_control/gift/all_results.csv` |
| `green_w384_cd125_166k.csv` | Our model at width 384, run green-384 (`g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix_cd125`, cooldown from 125k), at 166k. GM-MASE 0.9401. | same folder on elisa |
| `grey_w64_166k.csv` | Not in the paper any more: run grey (`g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10harm_rollfix`), backbone at 166k, head `h30k_control`. GM-MASE 1.0164. | elisa: `/home/jupyter/cf_runs/hp64/ff/heads/eval/<run>_bb166k_h30k_control/gift/all_results.csv` |
| `value_space_w64_166k.csv` | Moirai-style value-space model, width 64 (`cf462_mpe_w64`), at 166k. GM-MASE 1.0090. | `reports/2026-11-14_moirai_width64/results/per_config/cf462_mpe_w64_native166k.csv` |
| `value_space_w384_166k.csv` | Moirai-style value-space model, width 384 (`cf421ew_moirai_k3`), at 166k. GM-MASE 0.9362. | `reports/2026-09-06_moirai_small_size/results/per_config/radar/moirai421ew_native166k.csv` |
| `seasonal_naive.csv` | Seasonal naive. | `reports/2026-09-06_moirai_small_size/results/per_config/seasonal_naive.csv` |

To draw the figure again:

```bash
python3 paper/scripts/plot_gift_eval_radar.py
```
