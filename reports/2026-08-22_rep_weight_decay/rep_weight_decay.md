# A decay of the L_rep weight to zero never beats the no-decay reference at k = 32

The linear decay of the L_rep weight from 1.0 to 0.0 does not improve the GM-Relative MASE of the k = 32 cell, which is the configuration `arm6_v2_combab_alignT` at rollout depth 32. Its best setting loses to the no-decay reference by 3.9 times the seed range.

![scores](plots/scores.png)

GM-Relative MASE of every scored arm, with the decay (filled) and the same EMA schedule without it (open).

Both axes have their best value inside the tested range, and the 10,000-step ramp is the best of the four ramps.

![axes](plots/axes.png)

Score against the EMA momentum at the stop (left) and against the decay ramp (right), with the reference and the seed range.

One run lost the contrastive task, momentum 0.500 at step 10,162.

![auc](plots/auc.png)

Contrastive AUC per run to the 40,000-step stop, trailing mean, against the 0.55 gate.

Every scored arm still reduces its total loss over steps 20,000 to 40,000, and the slope over the last 10,000 steps ranks no arm.

![loss terms](plots/loss_terms.png)

Loss by term to the 40,000-step stop: total, weight on L_rep, L_rep, L_align reduced, mean cos_err.

`L_align, reduced` is the residual `(loss - rep_w * l_rep - sigreg_e - sigreg_h) / align_w`, `cos_err` is the mean of the forecast cosine error over the 33 rollout depths d0 to d32, and the trainer computes no L_rep past the ramp, so the `l_rep` lines end there.

## Tables

Score of each arm at the 40,000-step stop. An arm is one backbone run, named by its EMA schedule, decay ramp and seed (`scripts/arms.tsv`). The last column is the same EMA schedule with no decay, from the reference study.

| arm | EMA schedule | momentum at the stop | decay ramp, steps | seed | score | same schedule, no decay |
|---|---|---|---|---|---|---|
| dec_m050_fix | 0.5 fixed | 0.500 | 10,000 | 20260520 | lost the task | not run |
| dec_m070_fix | 0.7 fixed | 0.700 | 10,000 | 20260520 | 1.3534 | not run |
| dec_ramp5k_m080 | 0.8 to 1.0 at 200k | 0.840 | 5,000 | 20260520 | 1.2727 | 1.1782 |
| dec_m080_r200 | 0.8 to 1.0 at 200k | 0.840 | 10,000 | 20260520 | **1.2352** | 1.1782 |
| dec_m080_r200_s24 | 0.8 to 1.0 at 200k | 0.840 | 10,000 | 20260524 | pending | 1.1782 |
| dec_ramp20k_m080 | 0.8 to 1.0 at 200k | 0.840 | 20,000 | 20260520 | 1.3178 | 1.1782 |
| dec_ramp30k_m080 | 0.8 to 1.0 at 200k | 0.840 | 30,000 | 20260520 | 1.3623 | 1.1782 |
| dec_s20 | 0.9 to 1.0 at 100k | 0.940 | 10,000 | 20260520 | 1.2670 | 1.1507 |
| dec_s22 | 0.9 to 1.0 at 100k | 0.940 | 10,000 | 20260522 | 1.2593 | 1.1507 |
| dec_s24 | 0.9 to 1.0 at 100k | 0.940 | 10,000 | 20260524 | 1.2812 | 1.1507 |
| dec_m099_fix | 0.99 fixed | 0.990 | 10,000 | 20260520 | 1.2849 | not run |
| reference | 0.9 to 1.0 at 100k | 0.940 | no decay | 20260520, 20260522 | | 1.1491, 1.1507 |

Gaps against the seed range of 0.0219, the range of `dec_s20`, `dec_s22`, `dec_s24` (`results/rank_gate.tsv`). A gap over the seed range is a rank. A gap under it is not.

| comparison | gap | gap over seed range | verdict |
|---|---|---|---|
| each scored arm against the reference 1.1491 | +0.0861 to +0.2132 | 3.9 to 9.7 | rank |
| each scored arm against the same schedule with no decay | +0.0570 to +0.1841 | 2.6 to 8.4 | rank |
| best arm 1.2352 against the second 1.2593 | +0.0241 | 1.1 | not a rank |

Contrastive AUC per arm (`results/auc_verdicts.tsv`). The floor is the lowest trailing mean over every leg of the arm.

| arm | momentum at the stop | AUC floor | floor step | last AUC | last step | verdict |
|---|---|---|---|---|---|---|
| dec_m050_fix | 0.500 | 0.5221 | 10,348 | 0.5233 | 10,600 | lost at 10,162 |
| dec_m070_fix | 0.700 | 0.5732 | 11,452 | 0.9166 | 40,000 | held |
| dec_ramp5k_m080 | 0.840 | 0.7157 | 6,630 | 0.9928 | 40,000 | held |
| dec_m080_r200 | 0.840 | 0.7685 | 11,213 | 0.9924 | 40,000 | held |
| dec_ramp20k_m080 | 0.840 | 0.8353 | 27,830 | 0.9659 | 40,000 | held |
| dec_ramp30k_m080 | 0.840 | 0.5428 | 34,826 | 0.9310 | 40,000 | held |
| dec_s20 | 0.940 | 0.8638 | 13,123 | 0.9842 | 40,000 | held |
| dec_s22 | 0.940 | 0.8944 | 11,688 | 0.9841 | 40,000 | held |
| dec_s24 | 0.940 | 0.8680 | 12,323 | 0.9587 | 40,000 | held |
| dec_m099_fix | 0.990 | 0.9049 | 19,755 | 0.9695 | 40,000 | held |

Total loss and its slope per 10,000 steps, fitted on 1,000-step blocks (`results/loss_slope.csv`, `results/loss_terms_at_stop.csv`).

| arm | total loss at 40,000 | slope, steps 20,000 to 40,000 | slope, steps 30,000 to 40,000 | mean cos_err at 40,000 |
|---|---|---|---|---|
| dec_m070_fix | 0.367 | -0.194 | -0.258 | 0.236 |
| dec_ramp5k_m080 | 0.318 | -0.113 | -0.201 | 0.210 |
| dec_m080_r200 | 0.208 | -0.179 | -0.088 | 0.151 |
| dec_ramp20k_m080 | 0.555 | -0.163 | -0.457 | 0.387 |
| dec_ramp30k_m080 | 0.486 | -2.261 | -0.389 | 0.307 |
| dec_s20 | 0.553 | -0.082 | +0.106 | 0.330 |
| dec_s22 | 0.471 | -0.118 | +0.009 | 0.301 |
| dec_s24 | 0.721 | -0.079 | +0.071 | 0.482 |
| dec_m099_fix | 0.460 | -0.235 | -0.256 | 0.310 |

## Protocol

- Cell: `arm6_v2_combab_alignT`, rollout depth k = 32, reduction `mean`, align target the EMA teacher. `scripts/study.sh` holds every value.
- Decay: `--rep-loss-weight 1.0 --rep-loss-weight-end 0.0 --rep-loss-weight-ramp-steps <ramp>`, linear. The ramp of each arm is column 5 of `scripts/arms.tsv`.
- EMA: `--ema-tau <tau>`, with `--ema-tau-end 1.0 --ema-tau-ramp-steps <ramp>` on the ramped schedules. The momentum at the stop is the value the schedule holds at step 40,000.
- Backbone: 40,000 steps, seed 20260520 unless the arm name carries `s22`, `s24`. Checkpoints every 5,000 steps.
- AUC gate: `scripts/auc_guard.sh`, trailing window 500 steps, threshold 0.55, warm-up 1,000 steps. A run under the gate stops and gets no head.
- Head: 30,000 steps on the student encoder, head seed 20260722, runner `reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh`.
- Eval: GIFT-Eval, 97 configs, batch 4, forecast length 16. Score: GM-Relative MASE, lower is better.
- Reference: the EMA momentum study at `reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md`, the same cell with no decay. Its cell, stop, head, head seed, encoder and eval match this study on 11 of 11 items (`results/reference_match.tsv`). Its best score is 1.1491, and its own range is 0.0016 over two seeds.
- Seed range: the three seeds of `0.9 to 1.0 at 100k` under the decay, `results/rank_gate.tsv`.
- Pending: `dec_m080_r200_s24`, the repeat seed of the best arm, which gives its error bar. `scripts/make_plots.sh` rebuilds every figure and table from it.

## Annex

Runs without a score and why (`results/RUN_STATE.md`, `notes/execution_log.md`):

| arm | reached | why no score |
|---|---|---|
| dec_m050_fix | 10,600 | the AUC gate stopped it |
| dec_s23, dec_s25 | 22,900, 22,700 | not restarted, the schedule already had three seeds |
| dec_m090_r60, dec_m095_fix | 100, 0 | not restarted |
| dec_m090_fix, dec_m090_r200, dec_m095_r100 | never started | every tested momentum lost to the reference by 3.9 times the seed range or more, so no further momentum ran |
