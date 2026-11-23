# GM-MASE by training step

`gm_mase_by_step.csv`: the GM-MASE over the 97 GIFT-Eval configurations at each scored stop (thousands of steps).

| width | series | run | source |
|---|---|---|---|
| 64 | ours | green, `g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix` | elisa `cf_runs/hp64/ff/results/score_<run>_bb<k>k_h30k_control.txt` |
| 64 | ours_cooldown | green, exponential cooldown from its 135k save (`..._cd135`). The 1-sqrt cooldown (`_cdsqrt135`) scores 1.0084 and 1.0036 | same folder |
| 64 | moirai_style | `cf462_mpe_w64` | `reports/2026-11-14_moirai_width64/results/score_cf462_mpe_w64_bb<k>k_native.txt` |
| 384 | ours | green-384, `g166b_w384e1f5_..._align10_rollfix` | elisa `cf_runs/hp64/ff/results` |
| 384 | ours_cooldown | green-384, exponential cooldown from its 125k save (`..._cd125`) | same folder |
| 384 | moirai_style | `cf421ew_moirai_native` (paper: `cf421ew_moirai_k3`) | `reports/2026-09-06_moirai_small_size/results/gm_trajectories.tsv` |

The 250-step stop of green (1.2572) is left out.
