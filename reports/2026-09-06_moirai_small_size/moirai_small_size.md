# At Moirai-2-Small size, the learning rate moves the score more than the capacity

Ten times the capacity buys nothing at the published rate of 1e-3, and the
same configuration at 5.6e-4 gains 0.1675 GM-Relative MASE. No arm at that
rate has reached
200,000 steps, so the size question stays open until pass 2 lands. Next: #414.

## The answer

![the scores](plots/scores.png)

Every 40,000-step score of pass 1, against the parent study at 1.1M parameters.
The band on every figure and table of this report is **0.0568**: the spread of
this card's two seeds on one cell (k = 3, sum reduction, 1e-3, 40,000 steps).
It is a two-seed spread, not a standard deviation.

| number | GM-Relative MASE |
|---|---|
| the best arm of this card (k = 3, 5.6e-4, 40,000 steps) | 1.1820 |
| the project best (1.1M, align student, 200,000 steps) | 1.0651 |
| Moirai-2-Small, the same 97 GIFT-Eval configs | 0.728 |

The best 11.4M arm sits 2.1 bands behind the project best, so at the stops this
card has scored, the capacity has not paid.

## The rate draws a U, and 1e-3 sits outside it

![the rate bracket](plots/rates.png)

The four rates draw a U with an interior minimum at 5.6e-4, which beats 1e-3
by 0.1675 on the same seed (2.9 bands), and all three lower rates beat 1e-3.

| rate | score | Δ against 1.3495, same seed | Δ against 1.2927, other seed |
|---|---|---|---|
| 1e-3, seed 20260520 | 1.3495 | — | — |
| 1e-3, seed 20260525 | 1.2927 | 1.0 band | — |
| 5.6e-4 | 1.1820 | 2.9 bands | 1.9 bands |
| 3.3e-4 | 1.2483 | 1.8 bands | 0.8 bands |
| 1.67e-4 | 1.2612 | 1.6 bands | 0.6 bands |

The rule, fixed before the sweep scored, voids every 1e-3 number of this card
as a capacity statement: the bar was 1.2359, the better 1e-3 seed minus one
band, and 1.1820 clears it by 0.0539. The width-scaled rate over-corrects:
1.67e-4, which is 1e-3 times 64 over 384, lands 1.4 bands worse than 5.6e-4.
Two limits: only the k = 3 cell carries this sweep (each mean arm has one
rate), and 5.6e-4 is the best of four points on a coarse grid, one seed each,
not a tuned optimum.

**So no comparison exists at the stop that decides.** The 1.0651 reference is a
200,000-step number, and the only rate that fits this width holds one arm, one
seed, one 40,000-step stop. Pass 2 (R1, R4 below) closes that gap, and the
card's question stays open until it lands.

## A longer stop does not help at 1e-3

![the climb](plots/climb.png)

The two sizes tie inside the band at all three stops, and the 200,000-step
score is worse than the 40,000-step one.

| backbone steps | 11.4M | 1.1M twin | gap | head-matched |
|---|---|---|---|---|
| 40,000 | 1.3495 | 1.3618 | -0.0123 | no, a 15,000-step 1.1M head |
| 100,000 | 1.3395 | 1.3010 | +0.0385 | yes |
| 200,000 | 1.3910 | 1.3998 | -0.0088 | yes |

The 100,000 and 200,000-step gaps are ranked with the band, which was measured
at 40,000 steps only (see the design section).

**The pre-registered gate was overruled.** `results/gate_40k.txt` reads "CLIMB
k3_r100_09b 1.2927", and that arm never climbed: the orchestrator had fixed
the climb on `k3_r100_09` before the seed scores existed, because its 1.0651
reference is a 200,000-step number on that lineage. So the 200,000-step row
above rides seed 20260520, at 1.3495 the worse of the two seeds.

## Every mean arm erodes the contrastive task, and every sum arm holds it

![the contrastive AUC](plots/auc.png)

The AUC is a diagnostic probe, not a loss term: `src/metrics.py:361` counts how
often the forecast beats a lagged latent by cosine similarity, and 0.5 is
chance. Every AUC in this report is the guard's statistic, a rolling median
over 500 training rows. **Held** has one meaning here: the guard (median under
0.55 after a 1,000-step warm-up) never fired on the arm. The guard stopped two
of the ten pass-1 arms, so eight arms held, and the ten scores of pass 1 come
from those eight (one arm scores three stops).

Both stopped arms run the mean reduction, and all six sum arms ended above
0.99. But a k = 3 against k = 32 comparison moves the rollout depth, the
reduction, and the effective weight of the rollout term together (the factor
is k + 1: 4 copies at k = 3, 33 at k = 32), so this card cannot attribute the
split to the depth alone. The reduction and the depth are confounded over all
ten arms, by inheritance from the published parents:

| | sum | mean |
|---|---|---|
| k = 3 | 6 arms | none (R3 runs it now) |
| k = 8, k = 32 | none | 4 arms |

The rate is not ruled out either: both stopped arms ran at 1e-3, and no mean
arm had run below it. R2 and R3 below separate these.

At k = 32, each treatment finishes the job by itself:

| EMA momentum | no decay | `L_rep` decay to 0.0 by 2,000 |
|---|---|---|
| 0.9 to 1.0 at 100k | held, ends 0.773 | guard fired at 18,634 |
| 0.8 to 1.0 at 200k | guard fired at 28,152 | not run |

Read that table by row and by column, never on the diagonal: the two stopped
arms differ in both the momentum and the decay, so the gap between 18,634 and
28,152 measures nothing.

**The two stopped arms do not share one mechanism.** From their losses CSVs,
as a 200-row rolling mean at step 2,000 against the last logged step
(`results/lost_arm_terms.txt`):

| term | k32_r200_08 | k32_r100_09_dec |
|---|---|---|
| u_temporal | 0.046 to 0.097, rises | 0.012 to 0.009, falls |
| u_batch | 0.056 to 0.101, rises | 0.027 to 0.010, falls |
| l_align | 1.46 to 1.97, climbs | 0.78 to 1.78, climbs |

Neither arm breaks at one step. `k32_r200_08` peaks at a raw AUC of 0.980 at
step 411 and declines from there: its raw AUC first crosses 0.55 at step
23,901, and the guard's median verdict fired at 28,152. On the decay arm the
raw crossing is at step 5,131 and the guard fired at 18,634. `k32_r200_08`
carries `L_rep` at weight 1.0 for all of its 28,500 logged steps, so the decay
is not required to lose the task.

Dropping `L_rep` does not cause it either: `k3_r100_09_dec` runs 38,000 steps
at weight 0.0, holds an AUC floor of 0.9797 (step 9,690), ends at 0.9974, and
scores 1.3236, inside the 1.2927-to-1.3495 seed range of its cell. That floor
beats 0.904, the best the plain k = 32 arm reaches after step 10,000.

## A healthy AUC says nothing about the score

| group | AUC at the stop | score |
|---|---|---|
| eight scores from the six sum arms | 0.996 to 1.000 | 1.1820 to 1.3910 |
| `k8_r100_09` | 0.730 | 1.4537 |
| `k32_r100_09` | 0.773 | 1.4629 |
| the two stopped arms | chance | no score |

The AUC is a floor, not a ranking: the two eroded arms hold the two worst
scores, while inside the healthy group the ordering runs backwards, and the
lowest healthy AUC, 0.996, holds the best score. The two eroded arms sit 0.16
of a band apart, so they are not ranked against each other. `k32_r100_09` is
5.5 bands behind its own 1.1M twin at 1.1507, a gap this card cannot assign
between the width and the rate misfit.

## The loss by term

![the loss by term](plots/loss_terms.png)

`L_rep` carries the contrastive negatives and `L_align` pulls the forecast
toward the EMA teacher's latent. The empty `l_pred` column of the losses CSVs
is not a logging fault, because the loss shape
`cosine_similarity_batch_rep_only` has no prediction term.

## Pass 2 runs now

| run | arm | legs it holds | it settles | at writing |
|---|---|---|---|---|
| R1 | `k3_r100_09_lr56`, resumed from 40,000 | 100,000 then 200,000 | the size question at the stop that decides | step 54,400, AUC 0.999 |
| R2 | `k32_r100_09_lr56` | 40,000 | whether the mean cells erode at 5.6e-4 too | step 4,500, AUC 0.970 |
| R3 | `k3_r100_09_mean` | 40,000 | the depth-against-reduction confound | queued |
| R4 | `k3_r100_09b_lr56` | 40,000 | the seed band at 5.6e-4 | queued |

Their rows land in the tables below when they score (#414).

## What this design can say

**One thing changes: the width.** Every arm trains #373's cell
`arm6_v2_combab_alignT` at `d_model` 384, `n_heads` 8, 3 encoder and 3 decoder
layers (11,431,548 parameters against Moirai-2-Small's 11.4M) and takes its
depth, reduction, EMA momentum, seed, decay and rate from its parent.
`scripts/arms.tsv` carries the reasoning per row.

**The band travels further than it was measured.** 0.0568 is a two-seed spread
on one cell, under the sum reduction, at 40,000 steps. The 100,000 and
200,000-step gaps and every mean-arm gap are ranked with it, and no replicate
exists at those stops or under that reduction, so those rankings assume the
band transfers. R4 measures the band at 5.6e-4.

**The head budget comes from the 64-wide parents.** A 30,000-step student head
that under-trains a 384-wide encoder biases every 11.4M score the same way:
comparisons inside this card survive, the size headline carries the caveat.
#373's 40,000-step references use a 15,000-step head, so that stop is not
head-matched, and the tables name the budget of every reference.

**The EMA labels name schedules the 40,000-step runs do not traverse.**
"0.8 to 1.0 at 200k" reached a momentum of 0.8285, and "0.9 to 1.0 at 100k"
reached 0.940 at that stop.

**A decay verdict at 40,000 steps is a 40,000-step verdict.** #409 carried its
best decay arm to 200,000 steps and the gap moved, so this card does not
project its decay pair past the stop it measured.

## The arms

| arm | pass | k | reduce | EMA start | EMA end | ramp | seed | L_rep decay | lr |
|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09 | 1 | 3 | sum | 0.9 | 1.0 | 100000 | 20260520 | - | 1e-3 |
| k32_r100_09 | 1 | 32 | mean | 0.9 | 1.0 | 100000 | 20260520 | - | 1e-3 |
| k32_r200_08 | 1 | 32 | mean | 0.8 | 1.0 | 200000 | 20260520 | - | 1e-3 |
| k3_r100_09_dec | 1 | 3 | sum | 0.9 | 1.0 | 100000 | 20260520 | 2000 | 1e-3 |
| k32_r100_09_dec | 1 | 32 | mean | 0.9 | 1.0 | 100000 | 20260520 | 2000 | 1e-3 |
| k8_r100_09 | 1 | 8 | mean | 0.9 | 1.0 | 100000 | 20260520 | - | 1e-3 |
| k3_r100_09b | 1 | 3 | sum | 0.9 | 1.0 | 100000 | 20260525 | - | 1e-3 |
| k3_r100_09_lr56 | 1 | 3 | sum | 0.9 | 1.0 | 100000 | 20260520 | - | 5.6e-4 |
| k3_r100_09_lr33 | 1 | 3 | sum | 0.9 | 1.0 | 100000 | 20260520 | - | 3.3e-4 |
| k3_r100_09_lr17 | 1 | 3 | sum | 0.9 | 1.0 | 100000 | 20260520 | - | 1.67e-4 |
| k32_r100_09_lr56 | 2 | 32 | mean | 0.9 | 1.0 | 100000 | 20260520 | - | 5.6e-4 |
| k3_r100_09_mean | 2 | 3 | mean | 0.9 | 1.0 | 100000 | 20260520 | - | 1e-3 |
| k3_r100_09b_lr56 | 2 | 3 | sum | 0.9 | 1.0 | 100000 | 20260525 | - | 5.6e-4 |

## The tables

### The scores

Ten scores from eight arms. Two numbers closer than the 0.0568 band are not
ranked.

| arm | k | reduce | seed | lr | L_rep decay | stop | 11.4M | 1.1M twin | gap | head-matched |
|---|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09_lr56 | 3 | sum | 20260520 | 5.6e-4 | no | 40,000 | 1.1820 | never run | — | — |
| k3_r100_09_lr33 | 3 | sum | 20260520 | 3.3e-4 | no | 40,000 | 1.2483 | never run | — | — |
| k3_r100_09_lr17 | 3 | sum | 20260520 | 1.67e-4 | no | 40,000 | 1.2612 | never run | — | — |
| k3_r100_09b | 3 | sum | 20260525 | 1e-3 | no | 40,000 | 1.2927 | 1.3618 | -0.0691 | no, 15,000-step head |
| k3_r100_09_dec | 3 | sum | 20260520 | 1e-3 | yes | 40,000 | 1.3236 | never run | — | — |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 100,000 | 1.3395 | 1.3010 | +0.0385 | yes |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 40,000 | 1.3495 | 1.3618 | -0.0123 | no, 15,000-step head |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 200,000 | 1.3910 | 1.3998 | -0.0088 | yes |
| k8_r100_09 | 8 | mean | 20260520 | 1e-3 | no | 40,000 | 1.4537 | never run | — | — |
| k32_r100_09 | 32 | mean | 20260520 | 1e-3 | no | 40,000 | 1.4629 | 1.1507 (seeds 1.1491 to 1.1507) | +0.3122 | yes |

### The contrastive AUC, per leg the guard watched

| arm | leg to | verdict | AUC floor | at step | AUC last | at step |
|---|---|---|---|---|---|---|
| k3_r100_09 | 40,000 | held | 0.9931 | 1,942 | 0.9988 | 40,000 |
| k3_r100_09 | 100,000 | held | 0.9974 | 40,001 | 0.9994 | 100,000 |
| k3_r100_09 | 200,000 | held | 0.9992 | 144,399 | 0.9995 | 200,000 |
| k3_r100_09b | 40,000 | held | 0.9922 | 1,869 | 0.9982 | 40,000 |
| k3_r100_09_dec | 40,000 | held | 0.9797 | 9,690 | 0.9974 | 40,000 |
| k3_r100_09_lr56 | 40,000 | held | 0.9936 | 34,553 | 0.9956 | 40,000 |
| k3_r100_09_lr56 | 100,000 (R1, runs) | held | 0.9956 | 40,001 | 0.9987 | 54,500 |
| k3_r100_09_lr33 | 40,000 | held | 0.9913 | 3,337 | 0.9980 | 40,000 |
| k3_r100_09_lr17 | 40,000 | held | 0.9852 | 5,991 | 0.9985 | 40,000 |
| k8_r100_09 | 40,000 | held | 0.6847 | 14,106 | 0.7302 | 40,000 |
| k32_r100_09 | 40,000 | held | 0.6827 | 20,872 | 0.7732 | 40,000 |
| k32_r100_09_lr56 | 40,000 (R2, runs) | held | 0.9667 | 1,788 | 0.9700 | 4,500 |
| k32_r100_09_dec | 40,000 | guard fired at 18,634 | 0.5014 | 19,100 | 0.5014 | 19,100 |
| k32_r200_08 | 40,000 | guard fired at 28,152 | 0.5347 | 28,500 | 0.5347 | 28,500 |

### The contrastive AUC, step by step

Lower is worse, 0.5 is chance. The last column is what the same cell, seed and
stop reached at 1.1M parameters.

| arm | k | EMA momentum | L_rep decay | 2,000 | 5,000 | 8,000 | 10,000 | 12,000 | 15,000 | 18,600 | 25,000 | 28,000 | 40,000 | verdict | 1.1M twin at 40,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09 | 3 | 0.9 to 1.0 at 100k | no | 0.993 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.999 | 0.999 | 0.999 | held | — |
| k3_r100_09b | 3 | 0.9 to 1.0 at 100k | no | 0.993 | 0.995 | 0.998 | 0.999 | 0.999 | 0.997 | 0.998 | 0.999 | 0.999 | 0.998 | held | — |
| k3_r100_09_lr33 | 3 | 0.9 to 1.0 at 100k | no | 0.997 | 0.992 | 0.997 | 0.998 | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 | 0.998 | held | — |
| k3_r100_09_lr17 | 3 | 0.9 to 1.0 at 100k | no | 0.998 | 0.996 | 0.991 | 0.996 | 0.997 | 0.998 | 0.998 | 0.998 | 0.999 | 0.999 | held | — |
| k3_r100_09_lr56 | 3 | 0.9 to 1.0 at 100k | no | 0.996 | 0.998 | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 | 0.997 | 0.998 | 0.996 | held | — |
| k3_r100_09_dec | 3 | 0.9 to 1.0 at 100k | yes | 0.996 | 0.992 | 0.986 | 0.980 | 0.984 | 0.992 | 0.992 | 0.997 | 0.997 | 0.997 | held | — |
| k8_r100_09 | 8 | 0.9 to 1.0 at 100k | no | 0.971 | 0.961 | 0.920 | 0.881 | 0.812 | 0.717 | 0.769 | 0.769 | 0.772 | 0.730 | held | — |
| k32_r100_09 | 32 | 0.9 to 1.0 at 100k | no | 0.968 | 0.937 | 0.889 | 0.892 | 0.840 | 0.793 | 0.763 | 0.843 | 0.789 | 0.773 | held | 0.978 (#404) |
| k32_r200_08 | 32 | 0.8 to 1.0 at 200k | no | 0.877 | 0.814 | 0.788 | 0.749 | 0.739 | 0.751 | 0.796 | 0.568 | 0.555 | — | guard fired at 28,152 | 0.957 (#404) |
| k32_r100_09_lr56 | 32 | 0.9 to 1.0 at 100k | no | 0.973 | — | — | — | — | — | — | — | — | — | held, runs | — |
| k32_r100_09_dec | 32 | 0.9 to 1.0 at 100k | yes | 0.978 | 0.746 | 0.758 | 0.742 | 0.642 | 0.747 | 0.718 | — | — | — | guard fired at 18,634 | 0.983 (#409) |

**Read this table by row and by column, never on the diagonal.** Two rows are
comparable only when they differ in one column. These are the pairs, and there
are no others:

- `k32_r100_09` against `k32_r100_09_dec`, which moves the L_rep decay
- `k32_r100_09` against `k32_r100_09_lr56`, which moves the learning rate
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

The depth pair holds one more reading: `k8_r100_09` goes under `k32_r100_09`
at step 12,000 (0.812 against 0.840) and stays under to the stop (0.730
against 0.773), so the erosion does not scale with k.

### The loss by term, last logged row per leg

| arm | leg to | last step | total loss | L_rep | L_align | L_rep weight | EMA momentum | AUC |
|---|---|---|---|---|---|---|---|---|
| k3_r100_09 | 40,000 | 40,000 | 12.5323 | 11.7822 | 0.1852 | 1.00 | 0.9400 | 0.9993 |
| k3_r100_09 | 100,000 | 100,000 | 12.3238 | 11.7174 | 0.1472 | 1.00 | 1.0000 | 0.9997 |
| k3_r100_09 | 200,000 | 200,000 | 12.2424 | 11.7181 | 0.1319 | 1.00 | 1.0000 | 0.9938 |
| k3_r100_09b | 40,000 | 40,000 | 12.9777 | 11.7567 | 0.2832 | 1.00 | 0.9400 | 0.9984 |
| k3_r100_09_dec | 40,000 | 40,000 | 1.0579 | — | 0.2487 | 0.00 | 0.9400 | 0.9980 |
| k3_r100_09_lr56 | 40,000 | 40,000 | 12.6520 | 11.6891 | 0.2118 | 1.00 | 0.9400 | 0.9973 |
| k3_r100_09_lr56 | 100,000 | 54,400 | 13.4350 | 11.7984 | 0.3706 | 1.00 | 0.9544 | 0.9990 |
| k3_r100_09_lr33 | 40,000 | 40,000 | 12.5972 | 11.7772 | 0.2020 | 1.00 | 0.9400 | 0.9983 |
| k3_r100_09_lr17 | 40,000 | 40,000 | 12.8115 | 11.7678 | 0.2532 | 1.00 | 0.9400 | 0.9992 |
| k8_r100_09 | 40,000 | 40,000 | 13.3835 | 11.6129 | 1.7117 | 1.00 | 0.9400 | 0.7563 |
| k32_r100_09 | 40,000 | 40,000 | 13.4947 | 11.6270 | 1.7583 | 1.00 | 0.9400 | 0.7629 |
| k32_r100_09_lr56 | 40,000 | 4,500 | 12.8616 | 11.6032 | 1.0379 | 1.00 | 0.9045 | 0.9649 |
| k32_r100_09_dec | 40,000 | 19,100 | 1.8013 | — | 1.8219 | 0.00 | 0.9191 | 0.5010 |
| k32_r200_08 | 40,000 | 28,500 | 13.6105 | 11.5968 | 2.0018 | 1.00 | 0.8285 | 0.5254 |

### The cost

| run | stage | steps | hours |
|---|---|---|---|
| k3_r100_09 | backbone | 40,000 | 4.6 |
| k3_r100_09 | backbone, to 200,000 | 160,000 | 13.4 |
| k3_r100_09b | head, 40k backbone | 30,000 | 1.4 |
| k3_r100_09_bb40k_h30k_student | head | 30,000 | 1.9 |
| k3_r100_09_bb100k_h30k_student | head | 30,000 | 1.5 |
| k3_r100_09_bb200k_h30k_student | head | 30,000 | 1.8 |
| k3_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 2.8 |
| k3_r100_09_bb100k_h30k_student | GIFT-Eval, 97 configs | — | 2.9 |
| k3_r100_09_bb200k_h30k_student | GIFT-Eval, 97 configs | — | 4.0 |
| k3_r100_09b_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.1 |
| k3_r100_09_dec | backbone | 40,000 | 4.2 |
| k3_r100_09_dec_bb40k_h30k_student | head | 30,000 | 1.8 |
| k3_r100_09_dec_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 4.2 |
| k3_r100_09_lr56 | backbone | 40,000 | 4.8 |
| k3_r100_09_lr56_bb40k_h30k_student | head | 30,000 | 1.7 |
| k3_r100_09_lr56_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 4.2 |
| k3_r100_09_lr33 | backbone | 40,000 | 4.2 |
| k3_r100_09_lr33_bb40k_h30k_student | head | 30,000 | 1.7 |
| k3_r100_09_lr33_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.9 |
| k3_r100_09_lr17 | backbone | 40,000 | 5.1 |
| k3_r100_09_lr17_bb40k_h30k_student | head | 30,000 | 1.8 |
| k3_r100_09_lr17_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.9 |
| k8_r100_09 | backbone | 40,000 | 5.3 |
| k8_r100_09_bb40k_h30k_student | head | 30,000 | 1.6 |
| k8_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 4.0 |
| k32_r100_09 | backbone | 40,000 | 9.6 |
| k32_r100_09_bb40k_h30k_student | head | 30,000 | 1.8 |
| k32_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 2.9 |

## How to repeat it

The evaluation protocol is the parents': `scripts/head_eval.sh` calls #373's
`head_eval_bb.sh` unchanged (a 2-layer transformer quantile head, forecast
length 16, batch 256, head rate 1e-3, head seed 20260722, then the 97
GIFT-Eval configs under strategy B4) and adds only `CF_BB_SHAPE`, which gives
the head trainer and the evaluation the new width. Backbones train on one GPU
and heads on the other: a backbone queue trains no head, and one sweep starts
every head that a checkpoint lacks.

```bash
cd reports/2026-09-06_moirai_small_size
BB_GPU=0 bash run.sh size smoke trial     # the shape, the cost, one arm end to end

# The heads. One per card, so start it before the backbones.
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

# Pass 2 (#414), one ordered lane per GPU, no head in the lane.
BB_GPU=0 CF412_LEGS="k3_r100_09_lr56:100000 k3_r100_09_lr56:200000" \
  bash scripts/pass2_lane.sh
BB_GPU=1 CF412_LEGS="k32_r100_09_lr56:40000 k3_r100_09b_lr56:40000 k3_r100_09_mean:40000" \
  bash scripts/pass2_lane.sh

bash run.sh collect && bash scripts/make_plots.sh && bash scripts/gate.sh
python3 scripts/lost_arm_terms.py   # the loss terms of the two stopped arms
```

`scripts/arm_busy.sh <arm>` and `scripts/head_busy.sh <arm> <stop>` answer
whether an arm or a checkpoint is already training. Ask them before you start
anything by hand, because nothing under this card stops two trainers on one
arm.
