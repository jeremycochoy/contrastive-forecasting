# A decay of the L_rep weight to zero costs 0.086 GM-Relative MASE at k = 32

The linear decay of the L_rep weight from 1.0 to 0.0 does not improve the GM-Relative MASE of the k = 32 cell. Its best setting scores 1.2352 against the reference of 1.1491, a cost of 0.0861, and no EMA momentum or ramp length recovers it.

![The GM-Relative MASE of every scored arm, with the decay (filled) and the same EMA schedule without it (open)](plots/scores.png)

Score of each arm at the 40,000-step stop, with the decay, against the same schedule with no decay.

Both axes have their best value inside the tested range, and the card's own ramp of 10,000 steps is the best of the four ramps.

| EMA momentum at the stop | schedule | ramp | score | same schedule, no decay |
|---|---|---|---|---|
| 0.500 | 0.5 fixed | 10,000 | lost the task at step 10,162 | not run |
| 0.700 | 0.7 fixed | 10,000 | 1.3534 | not run |
| 0.840 | 0.8 to 1.0 at 200k | 5,000 | 1.2727 | 1.1782 |
| 0.840 | 0.8 to 1.0 at 200k | 10,000 | **1.2352** | 1.1782 |
| 0.840 | 0.8 to 1.0 at 200k | 10,000, seed 20260524 | pending, at step 6,600 of 40,000 | 1.1782 |
| 0.840 | 0.8 to 1.0 at 200k | 20,000 | 1.3178 | 1.1782 |
| 0.840 | 0.8 to 1.0 at 200k | 30,000 | 1.3623 | 1.1782 |
| 0.940 | 0.9 to 1.0 at 100k | 10,000 | 1.2670, 1.2593, 1.2812 | 1.1507, 1.1491 |
| 0.990 | 0.99 fixed | 10,000 | 1.2849 | not run |

Only the 0.9-to-1.0 schedule has replicate seeds, with a range of 0.0219 over three seeds, so this report ranks the decay against the reference and does not rank the decay schedules against each other.

| comparison | gap | verdict |
|---|---|---|
| each scored arm against the reference 1.1491 | +0.0861 to +0.2132 | rank, 3.9 to 9.7 times the seed range |
| each scored arm against the same schedule with no decay | +0.0570 to +0.1841 | rank, 2.6 to 8.4 times the seed range |
| best arm 1.2352 against the second 1.2593 | +0.0241 | 0.0022 over the seed range, not a rank |

The reference 1.1491 is a cross-study number from the EMA momentum sweep, and its cell, 40,000-step backbone stop, 30,000-step head, head seed 20260722, student encoder and 97-config GIFT-Eval match this card's exactly (`results/reference_match.tsv`, 11 items, 0 mismatches).

![The contrastive AUC of every run to 40,000 steps, with the decay ramp shaded](plots/auc.png)

Contrastive AUC per run, trailing mean, against the 0.55 gate.

One run lost the contrastive task, momentum 0.500 at step 10,162, and the arm that held the AUC best, momentum 0.990, scored worst of the schedules that survived.

| arm | momentum at the stop | AUC floor | at step | AUC at 40,000 | verdict |
|---|---|---|---|---|---|
| dec_m050_fix | 0.500 | 0.5221 | 10,348 | stopped at 10,600 | lost at 10,162 |
| dec_m070_fix | 0.700 | 0.5732 | 11,452 | 0.9166 | held |
| dec_ramp5k_m080 | 0.840 | 0.7157 | 6,630 | 0.9928 | held |
| dec_m080_r200 | 0.840 | 0.7685 | 11,213 | 0.9924 | held |
| dec_ramp20k_m080 | 0.840 | 0.8353 | 27,830 | 0.9659 | held |
| dec_ramp30k_m080 | 0.840 | 0.5428 | 34,826 | 0.9310 | held |
| dec_s20 | 0.940 | 0.8638 | 13,123 | 0.9842 | held |
| dec_s22 | 0.940 | 0.8944 | 11,688 | 0.9841 | held |
| dec_s24 | 0.940 | 0.8680 | 12,323 | 0.9587 | held |
| dec_m099_fix | 0.990 | 0.9049 | 19,755 | 0.9695 | held |

![The training loss by term to the 40,000-step stop: total, L_rep weight, L_rep, reduced L_align, mean cos_err](plots/loss_terms.png)

Loss by term, one line per arm, log axis on the total, the align term and cos_err.

Every scored arm still reduces its total loss over steps 20,000 to 40,000, and the slope over the last 10,000 steps ranks no arm, because the three seeds of one schedule span from +0.009 to +0.106 while the falling arms sit between -0.088 and -0.457.

| arm | total loss at 40,000 | slope, steps 30,000 to 40,000, per 10,000 | mean cos_err at 40,000 | score |
|---|---|---|---|---|
| dec_m080_r200 | 0.208 | -0.088 | 0.151 | 1.2352 |
| dec_s22 | 0.471 | +0.009 | 0.301 | 1.2593 |
| dec_s20 | 0.553 | +0.106 | 0.330 | 1.2670 |
| dec_ramp5k_m080 | 0.318 | -0.201 | 0.210 | 1.2727 |
| dec_s24 | 0.721 | +0.071 | 0.482 | 1.2812 |
| dec_m099_fix | 0.460 | -0.256 | 0.310 | 1.2849 |
| dec_ramp20k_m080 | 0.555 | -0.457 | 0.387 | 1.3178 |
| dec_m070_fix | 0.367 | -0.258 | 0.236 | 1.3534 |
| dec_ramp30k_m080 | 0.486 | -0.389 | 0.307 | 1.3623 |

Five EMA schedules of the catalogue (`scripts/arms.tsv`) never trained, because every tested momentum loses to the reference by 0.0861 or more, four times the seed range, so one more schedule on that axis could not close the gap.

## Protocol

- Cell: `arm6_v2_combab_alignT`, rollout depth k = 32, reduction `mean`, align target the EMA teacher. `scripts/study.sh` holds every value.
- Decay: `--rep-loss-weight 1.0 --rep-loss-weight-end 0.0 --rep-loss-weight-ramp-steps <ramp>`, linear. Ramp 10,000 on every arm except the three `dec_ramp*` arms (5,000, 20,000, 30,000).
- EMA: `--ema-tau <tau>`, with `--ema-tau-end 1.0 --ema-tau-ramp-steps <ramp>` on the ramped schedules. The momentum at the stop is what names an arm.
- Backbone: 40,000 steps, seed 20260520 unless the arm name carries `s22`, `s24`. Checkpoints every 5,000 steps.
- AUC gate: `scripts/auc_guard.sh`, trailing window 500 steps, threshold 0.55, warm-up 1,000 steps. A run under the gate stops and gets no head.
- Head: 30,000 steps on the student encoder, head seed 20260722, runner `reports/2026-08-08_rollout_depth/scripts/head_eval_bb.sh`.
- Eval: GIFT-Eval, 97 configs, batch 4, forecast length 16. Score: GM-Relative MASE, lower is better.
- Reference: `reports/2026-08-19_ema_momentum_k32/ema_momentum_k32.md`, the same cell with no decay. Open dots and the "no decay" column come from it.
- Seed range: the three seeds of `0.9 to 1.0 at 100k` under the decay, `results/rank_gate.tsv`. The reference's own range is 0.0016 over two seeds.
- Pending: `dec_m080_r200_s24`, the best arm at seed 20260524, is at step 6,600 of 40,000 on elisa GPU 0. Its score goes in the table above, and `scripts/make_plots.sh` rebuilds every figure and table from it.
- Rebuild: `bash scripts/make_plots.sh` from this directory, on elisa, where `/home/jupyter/checkpoints_backup/cf-409` holds the losses CSVs.

## Annex

Runs without a score and why (`results/RUN_STATE.md`):

| arm | reached | why no score |
|---|---|---|
| dec_m050_fix | 10,600 | the AUC gate stopped it |
| dec_s23, dec_s25 | 22,900, 22,700 | an outside kill at 14:18 on 08-23, not restarted: the schedule had three seeds |
| dec_m090_r60, dec_m095_fix | 100, 0 | the Hub outage of 08-23 18:48, not restarted |
| dec_m090_fix, dec_m090_r200, dec_m095_r100 | never started | the momentum axis was closed, see above |

The align term in the loss figure is the residual `(loss - rep_w * l_rep - sigreg_e - sigreg_h) / align_w`, per `notes/loss_decomposition.md`. Past step 10,000 the trainer computes no L_rep, so the `l_rep` lines end at the ramp.
