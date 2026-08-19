# #401 — every checkpoint of both arms, measured

Diagnosis note, updated 2026-08-19. Supporting artefacts for PR #402.
This is not the study report.

The card ran two arms. Both train the same cell at the same depths. They
differ in one flag, `--train-rollout-reduce`, which says how a k-depth run
combines its `k + 1` depth copies.

| arm | flag | checkpoints measured |
|---|---|---:|
| summed | `sum`, the trainer's default | 27 |
| mean | `mean` | 20 |
| `k = 0` parent, neither arm | one copy, so the flag has no effect | 6 |

## Verdict

The head and the GIFT-Eval path are correct.

**The reduction decides the collapse.** Every one of the 27 summed
checkpoints keeps one latent direction. Not one of the 20 mean checkpoints
does. The mean arm reads the same as the `k = 0` parent on every probe.

**The collapse is not what holds the mean arm back.** Its six scored cells
run 1.1637 to 1.2898, and the `k = 0` anchor on this same path is 1.1600.
So a healthy encoder at depth still does not beat no rollout at all.

![Collapse against score](../../plots/collapse_vs_score.png)

Left: rank separates the two reductions by a factor of 4. It puts no mean
cell under the `k = 0` anchor. Right: inside the collapsed set the score
follows `readout_r`, Spearman -0.76 over 8 cells.

## What was measured

`scripts/diag_collapse.py --all`, the same probe as before, same 21 real
GIFT-Eval windows, same loader, same columns. Only the subject list is
wider. It now covers all 53 backbones on disk, over both arms: `k = 0`, 8,
16 and 32, at every periodic step. The five earliest rows reproduce exactly.

Every table carries a `reduce` column. The two arms share their depths and
their stops, so without it one arm's rank would join the other arm's score.
`scripts/diag_latent.sh` runs the whole set on the CPU, beside the head
queue.

Two measurements are new, in their own scripts. Neither changes a number in
`diag_collapse.py`:

| script | what it measures |
|---|---|
| `diag_time_rank.py` | rank and cosine of `h` ALONG TIME, inside one series. The first probe measures across series at one instant. |
| `diag_scalar_readout.py` | `readout_r`, the mean absolute correlation between the input series and the projection of `h` on its top direction. |
| `diag_curve_state.py` | the step at which a 500-step median of the trainer's `auc` crosses 0.55. |

`diag_curve_state.py` reads the fuller of a losses CSV and its one-deep
`.prev`. The sync loop rotates the old copy aside before the new one lands,
so a fetch that dropped mid-transfer leaves a shorter current file. The
`k = 32` mean 200k leg is that case: the current file stops at step 181,225
and the `.prev` reaches 199,800.

## The table

`collapse_vs_score.md` holds all 53 rows. Every cell either arm scored, at
the study's fixed 30,000-step head:

| arm | k | stop | GM-Relative MASE | eff. rank (series) | mean cos (series) | eff. rank (time) | readout r | train AUC |
|---|---|---|---|---|---|---|---|---|
| — | 0 | bb40k | 1.1600 | 7.221 | 0.06531 | 7.248 | 0.110 | 0.9574 |
| mean | 8 | bb40k | 1.2433 | 4.196 | 0.08699 | 2.136 | 0.429 | 0.9982 |
| mean | 8 | bb100k | 1.2857 | 5.652 | 0.12860 | 2.279 | 0.195 | 0.9976 |
| mean | 8 | bb200k | 1.2898 | 5.841 | 0.11964 | 2.138 | 0.175 | 0.9980 |
| mean | 32 | bb40k | 1.2082 | 7.782 | 0.08120 | 6.174 | 0.220 | 0.9641 |
| mean | 32 | bb100k | 1.1803 | 7.364 | 0.10554 | 6.425 | 0.498 | 0.9649 |
| mean | 32 | bb200k | 1.1637 | 7.404 | 0.10895 | 6.433 | 0.491 | 0.9618 |
| sum | 8 | bb40k | 2.0357 | 1.000 | 1.00000 | 1.045 | 0.542 | 0.4998 |
| sum | 8 | bb100k | 7.9344 | 1.000 | 1.00000 | 1.000 | 0.427 | 0.5000 |
| sum | 8 | bb200k | 2.4755 | 1.000 | 1.00000 | 1.032 | 0.685 | 0.4997 |
| sum | 16 | bb40k | 4.5297 | 1.015 | 1.00000 | 1.045 | 0.290 | 0.5002 |
| sum | 16 | bb100k | 12.4827 | 1.905 | 1.00000 | 1.646 | 0.423 | 0.4999 |
| sum | 16 | bb200k | 2.9331 | 1.000 | 1.00000 | 1.008 | 0.605 | 0.5000 |
| sum | 32 | bb40k | 7.9575 | 1.194 | 1.00000 | 1.127 | 0.454 | 0.5000 |
| sum | 32 | bb100k | 1.7939 | 1.570 | 1.00000 | 1.442 | 0.705 | 0.5002 |

The `k = 0` row is control c2, the parent this study branched from, measured
by #401's own path. The summed arm stopped before its `k = 32` bb200k leg.

## Which checkpoints are collapsed

All 27 of the summed arm. Not one of the 20 mean ones, and not one of the 6
`k = 0` ones.

| set | mean cos between two series | effective rank |
|---|---|---|
| `k = 0`, 6 checkpoints | 0.0597 to 0.1057 | 6.46 to 10.76 |
| mean arm, 20 checkpoints | 0.0527 to 0.1305 | 4.13 to 8.01 |
| summed arm, 27 checkpoints | 0.99996 to 1.00000 | 1.000 to 1.905 |

The gap is a factor of 8 on cosine and a factor of 4 on rank, with no
checkpoint between the summed set and the other two. The same split holds
along time: the summed arm reads cosine 0.99992 or above and rank 1.00 to
1.65, and the mean arm reads cosine 0.167 to 0.932 and rank 2.09 to 6.86.

The trainer's own AUC says the same, from the first step to the last. Every
mean leg starts above 0.55 and never crosses down, over all eight legs of
the two arms. Every summed leg crosses down and stays down.

So the reduction is what collapses the encoder, not the depth. The mean arm
runs the same two depths.

## What the rank does not explain

**Inside the summed arm.** The score spans 1.7939 to 12.4827, a factor of 7,
at rank 1.00 to 1.91. Rank does not order it.

| measurement | Spearman with GM-Relative MASE, summed arm, n = 8 |
|---|---|
| effective rank, across series | +0.41 |
| effective rank, along time | +0.10 |
| dimension std | -0.38 |
| top direction variance share | -0.17 |
| `readout_r` | **-0.76** |

`readout_r` is the correlation between the input and the one direction the
encoder keeps. It is the only one of the five that orders the scores. Eight
points support it. That is a correlation, not a tested mechanism.

**Across the two arms.** Rank separates them cleanly and explains the factor
of 7 between them. It stops there. The mean arm reaches the parent's rank
and does not reach the parent's score.

`readout_r` reads ONE direction. It summarises a rank-1 encoder. It does not
summarise an encoder whose variance sits in seven directions. The right
panel of the plot therefore holds only cells whose top direction carries at
least half of the variance. That rule keeps the parent and all six mean
cells off it, on the measurement and not on the arm's name: the summed arm
reads a top-direction share of 0.55 to 1.00, the mean arm 0.13 to 0.39, and
the parent 0.11 to 0.15.

## When each arm enters the collapsed state

From the trainer's own `auc` column, every step, 500-step median, threshold
0.55. `scripts/diag_curve_state.py`.

| arm | k | first drop below chance | later crossings | last drop, final | AUC after |
|---|---|---|---|---|---|
| — | 0 | never | 0 | none | 0.943 to 0.962 |
| mean | 8 | never | 0 | none | 0.988 to 0.998 |
| mean | 32 | never | 0 | none | 0.948 to 0.966 |
| sum | 8 | step 4,404 | up at 5,072 | step 7,845 | 0.4997 to 0.5001 |
| sum | 16 | step 347 | none | step 347 | 0.4999 to 0.5002 |
| sum | 32 | step 1,343 | up at 3,137, down at 3,391, up at 4,276 | step 4,968 | 0.4996 to 0.5003 |

No summed arm leaves the collapsed state after its last drop. Its `k = 8`,
`k = 16` and `k = 32` curves stay at chance for the rest of the 40k leg and
for the whole 100k and 200k legs, 200,000 steps of training in total for
`k = 8` and `k = 16`.

No mean arm ever enters it. Both mean arms hold above the threshold for all
200,000 steps, at the same two depths.

**Inside the summed arm, the onset is not monotone in `k`.** `k = 16`
collapses first, at step 347.
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

The card asks whether more iterative rollout improves GM-Relative MASE.

**The summed arm cannot answer it.** Every one of its backbones collapses
inside the first 5,000 steps and never recovers, so its ladder measures
collapsed encoders, not deeper rollouts.

**The mean arm can answer it, and the answer is no.** It holds a healthy
encoder for all 200,000 steps at both depths. Its best cell, `k = 32` at
bb200k, scores 1.1637 against the `k = 0` anchor of 1.1600 on this same
path. The two sit 0.0037 apart, well inside the published head-seed band of
+/-0.0384. So more rollout depth buys nothing this study can measure, and it
is not because the encoder broke.

## Files

| file | what |
|---|---|
| `collapse_all.csv` | 53 checkpoints of both arms, latent spread across series, from `scripts/diag_collapse.py --all` |
| `time_rank.csv` | the same 53, latent spread along time |
| `scalar_readout.csv` | the same 53, `readout_r` and the top direction's variance share |
| `diag_latent.log`, `*.out` | the run record of the three probes |
| `curve_state.csv`, `curve_state.out` | AUC crossings and per-checkpoint AUC |
| `collapse_vs_score.csv`, `collapse_vs_score.md` | the join: score beside every measurement |
| `collapse.csv` | the narrow run of the probe, 7 checkpoints, which `latent_rank.png` draws |
| `per_config_vs_373.csv` | 97 configs, #401 against #373's A4 k = 3 bb40k |
| `diag_path.log`, `score_c1_*.txt`, `score_c2_*.txt` | the two path controls |
| `../../plots/collapse_vs_score.png` | score against rank, and against `readout_r` |
| `../../plots/collapse_onset.png` | AUC, dimension usage, cos_err_d0 against step |
| `../../plots/latent_rank.png` | seven checkpoints of the three sets, through the eval's loader |
