# #401 — every phase-1 checkpoint, measured

Diagnosis note, updated 2026-08-17. Supporting artefacts for PR #402.
This is not the study report. The study report comes when phase 1 ends.

## Verdict

The head and the GIFT-Eval path are correct. Every phase-1 backbone is
collapsed. That includes `k = 32` at bb100k, which scores 1.7939 and is the
best cell of the study.

Effective rank near 1 does not mean the score is bad. It means the encoder
keeps one direction. That one direction still moves with the input, and the
head reads it.

![Collapse against score](../../plots/collapse_vs_score.png)

Left: rank separates `k = 0` from every `k > 0` cell, and orders nothing
inside the collapsed set. Right: inside the collapsed set the score follows
`readout_r`, Spearman -0.76 over 8 cells.

## What was measured

`scripts/diag_collapse.py --all`, the same probe as before, same 21 real
GIFT-Eval windows, same loader, same columns. Only the subject list is
wider. It now covers all 31 backbones on disk: `k = 0`, 8, 16 and 32, at
every periodic step. The five earlier rows reproduce exactly.

Two measurements are new, in their own scripts. Neither changes a number in
`diag_collapse.py`:

| script | what it measures |
|---|---|
| `diag_time_rank.py` | rank and cosine of `h` ALONG TIME, inside one series. The first probe measures across series at one instant. |
| `diag_scalar_readout.py` | `readout_r`, the mean absolute correlation between the input series and the projection of `h` on its top direction. |
| `diag_curve_state.py` | the step at which a 500-step median of the trainer's `auc` crosses 0.55. |

## The table

`collapse_vs_score.md` holds all 31 rows. The 8 scored cells:

| k | stop | GM-Relative MASE | eff. rank (series) | mean cos (series) | eff. rank (time) | mean cos (time) | readout r | train AUC |
|---|---|---|---|---|---|---|---|---|
| 0 | bb40k | 1.1600 | 7.221 | 0.06531 | 7.248 | 0.17476 | 0.110 | 0.9574 |
| 8 | bb40k | 2.0357 | 1.000 | 1.00000 | 1.045 | 1.00000 | 0.542 | 0.4998 |
| 8 | bb100k | 7.9344 | 1.000 | 1.00000 | 1.000 | 1.00000 | 0.427 | 0.5000 |
| 8 | bb200k | 2.4755 | 1.000 | 1.00000 | 1.032 | 1.00000 | 0.685 | 0.4997 |
| 16 | bb40k | 4.5297 | 1.015 | 1.00000 | 1.045 | 1.00000 | 0.290 | 0.5002 |
| 16 | bb100k | 12.4827 | 1.905 | 1.00000 | 1.646 | 1.00000 | 0.423 | 0.4999 |
| 16 | bb200k | 2.9331 | 1.000 | 1.00000 | 1.008 | 1.00000 | 0.605 | 0.5000 |
| 32 | bb40k | 7.9575 | 1.194 | 1.00000 | 1.127 | 1.00000 | 0.454 | 0.5000 |
| 32 | bb100k | 1.7939 | 1.570 | 1.00000 | 1.442 | 1.00000 | 0.705 | 0.5002 |

The `k = 0` row is control c2, the parent this study branched from,
measured by #401's own path. `k = 32` bb200k is not in the table. That leg
runs now.

## Which checkpoints are collapsed

All 25 of the `k > 0` checkpoints. Not one of the 6 `k = 0` checkpoints.

| set | mean cos between two series | effective rank |
|---|---|---|
| `k = 0`, 6 checkpoints | 0.0597 to 0.1057 | 6.46 to 10.76 |
| `k > 0`, 25 checkpoints | 0.99996 to 1.00000 | 1.000 to 1.905 |

The gap is a factor of 9 on cosine and a factor of 3 on rank, with no
checkpoint between the two sets. The same split holds along time: `k = 0`
reads cosine 0.163 to 0.471 and rank 6.6 to 7.3, and every `k > 0`
checkpoint reads cosine 0.99992 or above and rank 1.00 to 1.65.

## What the rank does not explain

Inside the collapsed set the score spans 1.7939 to 12.4827, a factor of 7.
Rank does not order it.

| measurement | Spearman with GM-Relative MASE, n = 8 |
|---|---|
| effective rank, across series | +0.41 |
| effective rank, along time | +0.10 |
| dimension std | -0.38 |
| top direction variance share | -0.17 |
| `readout_r` | **-0.76** |

`readout_r` is the correlation between the input and the one direction the
encoder keeps. It is the only one of the five that orders the scores. Eight
points support it. That is a correlation, not a tested mechanism.

`readout_r` summarises a rank-1 encoder. It does not summarise the rank-7
parent, whose variance sits in seven directions, so the `k = 0` row is off
the right panel of the plot.

## When each arm enters the collapsed state

From the trainer's own `auc` column, every step, 500-step median, threshold
0.55. `scripts/diag_curve_state.py`.

| k | first drop below chance | later crossings | last drop, final | AUC after |
|---|---|---|---|---|
| 0 | never | 0 | none | 0.943 to 0.962 |
| 8 | step 4,404 | up at 5,072 | step 7,845 | 0.4997 to 0.5001 |
| 16 | step 347 | none | step 347 | 0.4999 to 0.5002 |
| 32 | step 1,343 | up at 3,137, down at 3,391, up at 4,276 | step 4,968 | 0.4996 to 0.5003 |

No arm leaves the collapsed state after its last drop. The `k = 8`, `k = 16`
and `k = 32` curves stay at chance for the rest of the 40k leg and for the
whole 100k and 200k legs, 200,000 steps of training in total for `k = 8`
and `k = 16`.

**The onset is not monotone in `k`.** `k = 16` collapses first, at step 347.
`k = 32` follows at 1,343. `k = 8` holds longest, to 4,404. An earlier
version of this note predicted a monotone dose-response and stated that
`k = 32` would collapse faster than `k = 16`. The curves refute that.

## What this rules out

**The path is clean.** Two controls ran #401's path, at #401's 30,000 head
steps, on known-good backbones. c1 pathbound, #373's own G1 subject, scores
1.2910 against the 1.2751 that #373 published at 15,000 head steps. c2, the
`k = 0` parent of this cell, scores 1.1600. A path that broke backbones
would not return those.

**The `ARCH` list is not the cause.** `eval_gift_eval_official.py` reads the
architecture off the checkpoint, then calls `load_state_dict` STRICTLY
(`load_models`, line 442). The shard logs show the detection:
`num_encoder_layers=3`, `qk_norm=True`, `attn_out_norm=True`,
`freq_emb_dim=3`, `seasonality_emb_dim=3`. A list that did not match the
checkpoint would raise, not score badly.

**The two checkouts run the same code.** `git diff c7e8af9d 8a9b567a`
outside `reports/` is one unrelated probe script and one test file. #373's
worktree is clean.

**The trainer command is #373's.** #401 adds no flag. `run_arm_k.sh` calls
#373's own `run_leg_k.sh`, which holds the whole command line. The one
input that differs is `K`.

**The damage is not horizon-shaped, it is global.** `k = 8` at bb40k is
worse than #373's A4 on 95 of 97 configs, `k = 16` on 97 of 97, `k = 8` at
bb100k on 97 of 97. See `per_config_vs_373.csv`.

## What the loss code says, and what it does not

`--train-rollout-depth k` sums `k + 1` copies of every term that ties the
forecaster output `f` to the encoder latent `h`. This cell runs
`--loss-shape cosine_similarity_batch_rep_only`, whose depth copies return
zeros, so the only term that repeats is `L_align`:

```
L_align = weight * (2 - 2*cos(f_t, sg(h_{t+1})))          src/loss.py:686
```

`L_align` has no negatives. Its minimum is `cos = 1`, which one direction
reaches for every depth at once. The three terms that resist collapse enter
once at every `k`: `L_rep` with MoCo keys, SIGReg on the embedding, SIGReg
on the encoding. This cell sets `--cpc-infonce-weight 0.0`, so there is no
InfoNCE.

The step-1 loss rises with `k`: 18.39 at `k = 3`, 27.37 at `k = 8`, 42.63 at
`k = 16`. That is a reading of the loss, not a test of the collapse.

The measurements do NOT support a `k + 1` dose-response on the outcome. The
onset step is not ordered by `k`, and the score inside the collapsed set is
not ordered by `k` either. `k = 32` at bb100k beats `k = 8` at bb100k by a
factor of 4.4. Whether an experiment that holds the f-side weight fixed
(`--align-loss-weight 1/(k+1)`) separates depth from alignment weight is
untested. That is outside this card.

## Effect on the card

The card asks whether more iterative rollout improves GM-Relative MASE. Every
`k = 8`, `k = 16` and `k = 32` backbone collapses inside the first 5,000
steps and never recovers, so the ladder measures collapsed encoders, not
deeper rollouts. The best cell, 1.7939, is still worse than the `k = 0`
anchor of 1.1600.

## Files

| file | what |
|---|---|
| `collapse_all.csv` | 31 checkpoints, latent spread across series, from `scripts/diag_collapse.py --all` |
| `time_rank.csv` | the same 31, latent spread along time |
| `scalar_readout.csv` | the same 31, `readout_r` and the top direction's variance share |
| `curve_state.csv`, `curve_state.out` | AUC crossings and per-checkpoint AUC |
| `collapse_vs_score.csv`, `collapse_vs_score.md` | the join: score beside every measurement |
| `collapse.csv` | the first, narrow run of the probe, 5 checkpoints |
| `per_config_vs_373.csv` | 97 configs, #401 against #373's A4 k = 3 bb40k |
| `diag_path.log`, `score_c1_*.txt`, `score_c2_*.txt` | the two path controls |
| `../../plots/collapse_vs_score.png` | score against rank, and against `readout_r` |
| `../../plots/collapse_onset.png` | AUC, dimension usage, cos_err_d0 against step |
| `../../plots/latent_rank.png` | the first five checkpoints, through the eval's loader |
