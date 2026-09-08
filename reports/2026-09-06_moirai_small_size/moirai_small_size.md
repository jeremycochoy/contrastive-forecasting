# At Moirai-2-Small size, the learning rate moves the k = 3 score more than the ten-fold capacity increase

At 11.4M parameters, k = 3 and 40,000 steps, a rate change from 1e-3 to
5.6e-4 gains 2.6 seed bands. The score falls from 1.3495 to 1.1820
GM-Relative MASE, both at seed 20260520. The second 1e-3 seed scores 1.2927,
still 1.7 bands behind 5.6e-4. The ten-fold capacity increase moves the k = 3
cell at most 0.6 bands. It makes the k = 32 cell 4.8 bands worse. So the rate
claim holds on k = 3 only. No arm of this card beats the 1.0651 project best,
and more steps make the best arm worse.

## Definitions

- **GM-Relative MASE**: the geometric mean, over the 97 GIFT-Eval configs, of
  MASE divided by seasonal-naive MASE. Lower is better.
- **The band**: 0.0649, the spread of this card's two seeds at 5.6e-4, the
  rate it ranks at. Two numbers closer than the band are not ranked.
- **A cell**: one combination of depth, reduction, rate and decay.

The other definitions open the tables section.

## The rate has an interior minimum at 5.6e-4

![the rate bracket](plots/rates.png)

GM-Relative MASE against the rate, k = 3 at 11.4M, one seed per point, with
the two project references.

## The scores at 40,000 steps

![the scores](plots/scores.png)

Every pass-1 score against its 1.1M twin, and each label gives the arm's
settings.

## More steps make the best arm worse, and no stop reaches 1.0651

![the climb](plots/climb.png)

`k3_r100_09` at 1e-3 and `k3_r100_09_lr56` at 5.6e-4, both at 11.4M, each
against its 1.1M twin over three stops. A hollow marker marks a stop that is
not head-matched.

**This is the size answer.** At 5.6e-4 more steps cost score: 1.1820 at
40,000 steps, 1.3170 at 100,000 and 1.3189 at 200,000. The first climb costs
2.1 bands and the second changes nothing. The 200,000-step number sits 3.9
bands behind the 1.0651 project best. The 40,000-step number sits 1.8 bands
behind it. So 11.4M at 5.6e-4 does not beat 1.0651 at any stop.

At 1e-3 the three scores span 0.0515, inside the band, so no stop is ranked
above another. Both climbs run seed 20260520.

## Three of the six mean arms lost the task, and none of the seven sum arms

![the contrastive AUC](plots/auc.png)

The guard's rolling AUC of every run, with the three stopped arms in red and the
live legs dotted.

All three lost arms run `mean`, and every `sum` arm held. The lost steps,
18,634, 26,413 and 28,152, are the live guard's verdicts
(`results/auc_verdicts.tsv`). The guard stops a run whose rolling AUC median
falls under 0.55. It ranks nothing.

A k = 3 against k = 32 comparison moves three settings together: the depth,
the reduction, and the weight of the rollout term. The sum reduction
multiplies that weight by k + 1, so 4 copies at k = 3 and 33 at k = 32.
This card cannot assign the split to the depth or the rate alone,
and the inheritance table below shows the confound.

**The rate does not rescue the k = 32 mean cell.** `k32_r100_09_lr56` scores
1.4404 at 5.6e-4 against 1.4629 at 1e-3, a gap of 0.35 bands. The two are not
ranked. The rate gains 2.6 bands on k = 3 and nothing here.

## One lost arm falls to near-zero dimension usage, the other rises

![the dimension usage of the two lost arms](plots/lost_uniformity.png)

`u_temporal` and `u_batch` of the two lost arms, 200-row rolling mean, with
the lost step dashed. The two statistics measure dimension usage of the
latents across time and across the batch (`src/metrics.py:179` and
`src/metrics.py:184`).

`L_rep` carries the contrastive negatives. `k32_r200_08` carries it at weight
1.0 for all 28,500 logged steps, so the decay is not required to lose the
task. Dropping `L_rep` does not cause a loss either: `k3_r100_09_dec` runs
38,000 steps at weight 0.0, holds an AUC floor of 0.9797 and scores 1.3236,
inside its cell's seed range.

## The AUC says that a run is alive, and nothing more

![the score against the AUC](plots/auc_score.png)

GM-Relative MASE against the AUC at the stop, one point per scored leg, with
the stopped arms on the top axis line.

**A higher AUC does not mean a better score.** Inside each reduction group the
two move the wrong way. The Pearson correlation of the AUC against the score
is +0.533 over the seven sum arms. It is +0.954 over the three mean arms.
Higher is worse. The best arm of the card, `k3_r100_09_lr56` at 1.1820, holds
the LOWEST AUC of the seven sum arms, 0.9973. The best mean arm,
`k32_r100_09_lr56` at 1.4404, holds the lowest AUC of its three, 0.7098. The
statistic is the `auc` column of `results/loss_terms.csv`, the guard's rolling
median at the stop.

So this report uses the AUC for one thing. The guard stops a run that reached
chance, because a dead run gives no score. GM-Relative MASE decides every
rank. `k32_r100_09` is 4.8 bands behind its 1.1M twin at 1.1507, a gap this
card cannot split between the width and the rate misfit.

## The loss by term

![the loss by term](plots/loss_terms.png)

`L_rep` and `L_align`, every run of the card. The live `L_rep` weight is 1.0
on every arm, and 0.0 by step 2,000 on the two decay arms.

## What this design can say

**One setting changes: the width.** Every arm trains the parent configuration
`arm6_v2_combab_alignT` at `d_model` 384 (11,431,548 parameters, against
Moirai-2-Small's 11.4M). Each pass-1 arm takes its depth, reduction, momentum,
seed, decay and rate from a published 1.1M parent (`scripts/arms.tsv`).

**The rate sweep is coarse.** The width-scaled rate 1.67e-4 (1e-3 times 64
over 384) over-corrects by 1.4 bands against 5.6e-4. Only the k = 3 cell ran
the sweep, and 5.6e-4 is the best of four coarse points, one seed each. No
5.6e-4 arm has reached 200,000 steps, the stop of the 1.0651 project best.

**This report applies the band outside the cell that measured it.** 0.0649 is
a two-seed spread at k = 3, sum, 5.6e-4, 40,000 steps. No replicate exists at
other stops or under the mean reduction, so those rankings assume the band
transfers.

**The head budget comes from the 64-wide parents.** A 30,000-step head that
under-trains a 384-wide encoder biases every 11.4M score the same way.
Comparisons inside this card survive, and the size headline carries the
caveat.

**A decay verdict at 40,000 steps is a 40,000-step verdict.** This card does
not project its decay pair past the stop it measured.

## The tables

- **A leg**: one continuous training segment of an arm toward one stop.
- **Head-matched**: both sizes score under a 30,000-step head.
- **AUC** (area under the curve): a diagnostic probe, not a loss term
  (`src/metrics.py:361`). It counts how often the forecast beats a lagged
  latent by cosine similarity, and 0.5 is chance. This report prints the
  guard's statistic, a rolling median over 500 training rows, except where a
  value says raw.
- **Lost**: the guard (median under 0.55 after a 1,000-step warm-up) ended
  the run early. **Held**: the guard did not end the run.
- **EMA** (exponential moving average): the teacher is an EMA of the student,
  and `L_align` pulls the forecast toward its latent.

Every 1.1M reference number comes from `scripts/plot_style.py` and
`scripts/arms.tsv`. The k = 3 twin is cell A3 of the rollout-depth study, at
seed 20260520. That seed matches every k = 3 row except `k3_r100_09b`, which
runs 20260525, so its twin is not seed-matched. The twin's head is 15,000
steps at the 40,000-step stop and 30,000 steps at the two later stops. The
k = 32 twins come from the EMA-momentum study (seeds 20260520 and 20260524)
and the L_rep-decay study, under a 30,000-step head. The AUC anchors 0.978,
0.957 and 0.983 come from the same two studies. The project best, 1.0651, is
the 1.1M align-student run at 200,000 steps (align student: `L_align` targets
the student latent, not the EMA teacher).

### The references

| number | GM-Relative MASE |
|---|---|
| the best arm of this card (k = 3, 5.6e-4, 40,000 steps) | 1.1820 |
| the project best (1.1M, align student, 200,000 steps) | 1.0651 |
| Moirai-2-Small, the same 97 GIFT-Eval configs | 0.728 |

### The scores

Fourteen scores over ten arms, because one arm scores three stops
(`results/scores.csv`). The band column reads each row against 1.1820, the
best of the card, in units of 0.0649. The `twin minus 11.4M` column subtracts
the row's 11.4M score from its 1.1M twin, and positive favours the 11.4M
arm.

| arm | k | reduce | seed | lr | decay | stop | 11.4M | bands behind 1.1820 | 1.1M twin | twin minus 11.4M | head-matched |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09_lr56 | 3 | sum | 20260520 | 5.6e-4 | no | 40,000 | 1.1820 | 0.0 | never run | — | — |
| k3_r100_09b_lr56 | 3 | sum | 20260525 | 5.6e-4 | no | 40,000 | 1.2469 | 1.0 | never run | — | — |
| k3_r100_09_lr33 | 3 | sum | 20260520 | 3.3e-4 | no | 40,000 | 1.2483 | 1.0 | never run | — | — |
| k3_r100_09_lr17 | 3 | sum | 20260520 | 1.67e-4 | no | 40,000 | 1.2612 | 1.2 | never run | — | — |
| k3_r100_09b | 3 | sum | 20260525 | 1e-3 | no | 40,000 | 1.2927 | 1.7 | 1.3618 (not seed-matched) | +0.0691 | no, 15,000-step head |
| k3_r100_09_lr56 | 3 | sum | 20260520 | 5.6e-4 | no | 100,000 | 1.3170 | 2.1 | never run | — | — |
| k3_r100_09_lr56 | 3 | sum | 20260520 | 5.6e-4 | no | 200,000 | 1.3189 | 2.1 | never run | — | — |
| k3_r100_09_dec | 3 | sum | 20260520 | 1e-3 | yes | 40,000 | 1.3236 | 2.2 | never run | — | — |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 100,000 | 1.3395 | 2.4 | 1.3010 | -0.0385 | yes |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 40,000 | 1.3495 | 2.6 | 1.3618 | +0.0123 | no, 15,000-step head |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 200,000 | 1.3910 | 3.2 | 1.3998 | +0.0088 | yes |
| k32_r100_09_lr56 | 32 | mean | 20260520 | 5.6e-4 | no | 40,000 | 1.4404 | 4.0 | never run | — | — |
| k8_r100_09 | 8 | mean | 20260520 | 1e-3 | no | 40,000 | 1.4537 | 4.2 | never run | — | — |
| k32_r100_09 | 32 | mean | 20260520 | 1e-3 | no | 40,000 | 1.4629 | 4.3 | 1.1507 (1.1491 to 1.1507) | -0.3122 | yes |

### The reduction and the depth are confounded

The confound comes by inheritance: the k = 3 arms take `sum` from their
published parents and the deeper arms take `mean`.

| | sum | mean |
|---|---|---|
| k = 3 | 7 arms | 1 arm, and the guard stopped it |
| k = 8, k = 32 | 1 arm, running | 5 arms |

Pass 2 added `k3_r100_09_mean` to break the confound, and the guard stopped it
at 26,413 steps, so it holds no score. Pass 3 runs the other diagonal,
`k32_r100_09_sum`.

### The k = 32 treatments

Compare this table by row and by column only, because the two lost arms
differ in both the momentum and the decay. Neither ramp completes by 40,000
steps: the momentum reached 0.8285 (`k32_r200_08`, last log at step 28,500),
0.9191 (`k32_r100_09_dec`, last log at step 19,100) and 0.9400
(`k32_r100_09` at 40,000), per `results/loss_terms.csv`.

| EMA momentum | no decay | `L_rep` decay to 0.0 by 2,000 |
|---|---|---|
| 0.9 to 1.0 at 100k | held, ends 0.773 | lost at 18,634 |
| 0.8 to 1.0 at 200k | lost at 28,152 | not run |

### The contrastive AUC, step by step

Lower is worse, and a run at 0.5 has lost the task. The last two columns give
what the same cell, seed and stop reached at 1.1M parameters, and where.

| arm | k | EMA momentum | decay | 2,000 | 5,000 | 8,000 | 10,000 | 12,000 | 15,000 | 18,600 | 25,000 | 28,000 | 40,000 | verdict | 1.1M twin at 40,000 | parent study |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09 | 3 | 0.9 to 1.0 at 100k | no | 0.993 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.999 | 0.999 | 0.999 | held | — | — |
| k3_r100_09b | 3 | 0.9 to 1.0 at 100k | no | 0.993 | 0.995 | 0.998 | 0.999 | 0.999 | 0.997 | 0.998 | 0.999 | 0.999 | 0.998 | held | — | — |
| k3_r100_09_lr33 | 3 | 0.9 to 1.0 at 100k | no | 0.997 | 0.992 | 0.997 | 0.998 | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 | 0.998 | held | — | — |
| k3_r100_09_lr17 | 3 | 0.9 to 1.0 at 100k | no | 0.998 | 0.996 | 0.991 | 0.996 | 0.997 | 0.998 | 0.998 | 0.998 | 0.999 | 0.999 | held | — | — |
| k3_r100_09_lr56 | 3 | 0.9 to 1.0 at 100k | no | 0.996 | 0.998 | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 | 0.997 | 0.998 | 0.996 | held | — | — |
| k3_r100_09_dec | 3 | 0.9 to 1.0 at 100k | yes | 0.996 | 0.992 | 0.986 | 0.980 | 0.984 | 0.992 | 0.992 | 0.997 | 0.997 | 0.997 | held | — | — |
| k8_r100_09 | 8 | 0.9 to 1.0 at 100k | no | 0.971 | 0.961 | 0.920 | 0.881 | 0.812 | 0.717 | 0.769 | 0.769 | 0.772 | 0.730 | held | — | — |
| k32_r100_09 | 32 | 0.9 to 1.0 at 100k | no | 0.968 | 0.937 | 0.889 | 0.892 | 0.840 | 0.793 | 0.763 | 0.843 | 0.789 | 0.773 | held | 0.978 | the EMA-momentum study |
| k32_r200_08 | 32 | 0.8 to 1.0 at 200k | no | 0.877 | 0.814 | 0.788 | 0.749 | 0.739 | 0.751 | 0.796 | 0.568 | 0.555 | — | lost at 28,152 | 0.957 | the EMA-momentum study |
| k32_r100_09_dec | 32 | 0.9 to 1.0 at 100k | yes | 0.978 | 0.746 | 0.758 | 0.742 | 0.642 | 0.747 | 0.718 | — | — | — | lost at 18,634 | 0.983 | the L_rep-decay study |

Compare two rows only when they differ in one column. These are the pairs,
and there are no others:

- `k32_r100_09` against `k32_r100_09_dec`, which moves the L_rep decay
- `k32_r100_09` against `k32_r200_08`, which moves the EMA momentum
- `k32_r100_09` against `k8_r100_09`, which moves the rollout depth
- `k3_r100_09_lr17` against `k3_r100_09_lr33`, which moves the learning rate
- `k3_r100_09_lr17` against `k3_r100_09_lr56`, which moves the learning rate
- `k3_r100_09_lr33` against `k3_r100_09_lr56`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09_dec`, which moves the L_rep decay
- `k3_r100_09` against `k3_r100_09_lr17`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09_lr33`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09_lr56`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09b`, which moves the seed

### The loss by term

| arm | stop | last step | total loss | L_rep | L_align | L_rep weight | EMA momentum |
|---|---|---|---|---|---|---|---|
| k32_r100_09 | 40,000 | 40,000 | 13.4947 | 11.6270 | 1.7583 | 1.00 | 0.9400 |
| k32_r100_09_dec | 40,000 | 19,100 | 1.8013 | — | 1.8219 | 0.00 | 0.9191 |
| k32_r200_08 | 40,000 | 28,500 | 13.6105 | 11.5968 | 2.0018 | 1.00 | 0.8285 |
| k3_r100_09 | 100,000 | 100,000 | 12.3238 | 11.7174 | 0.1472 | 1.00 | 1.0000 |
| k3_r100_09 | 200,000 | 200,000 | 12.2424 | 11.7181 | 0.1319 | 1.00 | 1.0000 |
| k3_r100_09 | 40,000 | 40,000 | 12.5323 | 11.7822 | 0.1852 | 1.00 | 0.9400 |
| k3_r100_09_dec | 40,000 | 40,000 | 1.0579 | — | 0.2487 | 0.00 | 0.9400 |
| k3_r100_09_lr17 | 40,000 | 40,000 | 12.8115 | 11.7678 | 0.2532 | 1.00 | 0.9400 |
| k3_r100_09_lr33 | 40,000 | 40,000 | 12.5972 | 11.7772 | 0.2020 | 1.00 | 0.9400 |
| k3_r100_09_lr56 | 40,000 | 40,000 | 12.6520 | 11.6891 | 0.2118 | 1.00 | 0.9400 |
| k3_r100_09b | 40,000 | 40,000 | 12.9777 | 11.7567 | 0.2832 | 1.00 | 0.9400 |
| k8_r100_09 | 40,000 | 40,000 | 13.3835 | 11.6129 | 1.7117 | 1.00 | 0.9400 |

### The arms

Sixteen arms run over three passes, and `scripts/arms.tsv` carries the
reasoning per arm. Every arm ramps the EMA momentum from 0.9 to 1.0 over
100,000 steps, except `k32_r200_08` (0.8 to 1.0 over 200,000). The scores
table above holds each arm's k, reduction, seed, decay and rate.

### The pass-2 arms, and what each settled

| run | arm | legs | result |
|---|---|---|---|
| R1 | `k3_r100_09_lr56`, resumed from 40,000 | 100,000 then 200,000 | 1.3170 and 1.3189. More steps cost 2.1 bands, and neither stop reaches 1.0651. |
| R2 | `k32_r100_09_lr56` | 40,000 | 1.4404 against 1.4629 at 1e-3, a gap of 0.35 bands. The rate does not move this cell. |
| R3 | `k3_r100_09_mean` | 40,000 | The guard stopped it at 26,413 steps. No score, so the reduction against depth confound stands. |
| R4 | `k3_r100_09b_lr56` | 40,000 | 1.2469. It gives the band this report ranks with, 0.0649. |

### The pass-3 arms

Three arms train now, and each one takes the 40,000-step stop and a
30,000-step head. `scripts/gate_pass3.sh` holds the rule: a score under
1.1171 is progress, a score over 1.2469 is worse, and the numbers between
are not ranked.

| run | arm | what it settles |
|---|---|---|
| R5 | `k32_r100_09_sum` | The reduction against `k32_r100_09_lr56`, and the depth against `k3_r100_09_lr56`. It is the first k = 32 arm under `sum` at any size. |
| R6 | `k3_r100_09_lr45` | The rate bracket under 5.6e-4. |
| R7 | `k3_r100_09_lr70` | The rate bracket over 5.6e-4. |

### The cost

The totals sum the 28 logged pass-1 stage rows in `results/`.

| stage | logged runs | total hours |
|---|---|---|
| backbone | 8 legs | 51.2 |
| head, 30,000 steps each | 10 heads | 17.0 |
| GIFT-Eval, 97 configs each | 10 evals | 35.9 |

## How to repeat it

The evaluation protocol is the parents'. `scripts/head_eval.sh` calls the
rollout-depth study's `head_eval_bb.sh` unchanged, and adds only
`CF_BB_SHAPE`, which gives the head trainer and the evaluation the new width.
The head is a 2-layer transformer quantile head: forecast length 16, batch
256, rate 1e-3, seed 20260722. The score comes from the 97 GIFT-Eval configs
under strategy B4, GIFT-Eval's official evaluation strategy.

```bash
cd reports/2026-09-06_moirai_small_size
BB_GPU=0 bash run.sh size smoke trial     # the shape, the cost, one arm end to end

# The heads. One sweep per card, so start it before the backbones.
BB_GPU=1 bash scripts/head_sweep.sh &

# Pass 1: two backbones of one card at a time, largest memory need first.
BB_GPU=0 CF412_QUEUE="k32_r100_09 k32_r100_09_dec" bash scripts/queue_backbones.sh
BB_GPU=0 CF412_QUEUE="k32_r200_08 k8_r100_09"      bash scripts/queue_backbones.sh
BB_GPU=1 CF412_QUEUE="k3_r100_09 k3_r100_09b k3_r100_09_dec" \
  bash scripts/queue_backbones.sh
BB_GPU=1 CF412_QUEUE="k3_r100_09_lr56 k3_r100_09_lr33 k3_r100_09_lr17" \
  bash scripts/queue_backbones.sh

# The climb. `phase1.sh` resumes the furthest checkpoint, so the stops are one
# continuous run and a stop it never asks for costs nothing later.
BB_GPU=0 STOPS="100000 200000" ARMS="k3_r100_09" bash run.sh phase1

# Pass 2, one ordered lane per GPU, no head in the lane.
BB_GPU=0 CF412_LEGS="k3_r100_09_lr56:100000 k3_r100_09_lr56:200000" \
  bash scripts/pass2_lane.sh
BB_GPU=1 CF412_LEGS="k32_r100_09_lr56:40000 k3_r100_09b_lr56:40000 k3_r100_09_mean:40000" \
  bash scripts/pass2_lane.sh

# Pass 3, the same lane, one per GPU.
BB_GPU=1 CF412_LEGS="k32_r100_09_sum:40000" bash scripts/pass2_lane.sh
BB_GPU=0 CF412_LEGS="k3_r100_09_lr45:40000 k3_r100_09_lr70:40000" \
  bash scripts/pass2_lane.sh

bash run.sh collect && bash scripts/make_plots.sh && bash scripts/gate.sh
bash scripts/gate_pass3.sh          # the pass-3 verdict against 1.1820
python3 scripts/lost_arm_terms.py   # the loss terms of the stopped arms
```
