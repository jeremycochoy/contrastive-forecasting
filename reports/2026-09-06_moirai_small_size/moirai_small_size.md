# The same objective at Moirai-2-Small size

**DRAFT — the card still runs. The verdict below covers the arms that landed.
`results/tables.md` and `plots/` refresh every 15 minutes.**

Every score of this project comes from a backbone of 1.1M parameters.
Moirai-2-Small holds 11.4M and scores 0.728 on the same 97 GIFT-Eval configs,
where our best is 1.0651. This card retrains the published teacher-align
configurations at 11.4M parameters, so we learn whether the objective is weak
or whether the model is small.

## The answer

Ten times the capacity buys nothing at the stops that landed.

![the scores](plots/scores.png)

At 100,000 backbone steps, head-matched, the 11.4M model scores 1.3395 and its
1.1M twin scores 1.3010. The 11.4M model is 0.0385 worse, which is inside this
size's 0.0568 seed band, so the two are not ranked. On the k = 32 cell at
40,000 steps the 11.4M model scores 1.4629 against 1.1507, worse by 5.5 bands.

The 200,000-step stop answers the card, because the 1.0651 reference is a
200,000-step number. That leg runs.

## A longer stop does not rescue this size

![the climb](plots/climb.png)

`k3_r100_09` scores 1.3495 at 40,000 steps and 1.3395 at 100,000. The gain is
0.0100 against a seed band of 0.0568. So a later stop does not move this
size's rank, and the 40,000-step order of this card stands. The 200,000-step
leg runs, because 1.0651 is a 200,000-step number.

## The learning rate does not clear the bar, yet

![the rate sweep](plots/rates.png)

`k3_r100_09_lr33` trains configuration 1 at 3.3e-4 and scores 1.2483, the best
11.4M number on this card. Whether that is a rate effect depends on which 1e-3
run you subtract, and both are valid runs.

| comparison | D | against the 0.0568 band |
|---|---|---|
| the same seed 20260520, 1.3495 | +0.1012 | 1.8 bands |
| the other seed 20260525, 1.2927 | +0.0444 | 0.8 bands |

The seed-matched pair is the controlled one: `lr33` differs from the 1.3495 run
in the rate column alone. But 1e-3 is a RANGE here, 1.2927 to 1.3495, and
1.2483 sits 0.0444 below its nearest edge. One seed of one rate is not
separated from that range.

SO NO 1e-3 NUMBER IS VOID. The rule needed 1.2359 and this arm reached 1.2483,
short by 0.0124. What would settle it is a SECOND SEED of `lr33`, not a third
rate: the question is whether 1.2483 is a rate or a draw, and only a repeat
seed answers that. This card did not run one.

Two readings do not depend on it. The size comparison is a within-rate
comparison, 1e-3 on the 11.4M side and 1e-3 on every 1.1M parent, so a rate
that suits width 384 better would move both sides. And no rate changes the
contrastive AUC: `lr33` holds 0.998 at 40,000 steps where the k = 32 arms read
0.773 and lower.

## The width erodes the contrastive task at every depth above 3

![the contrastive AUC](plots/auc.png)

Every AUC in this report is a rolling median over 500 training rows, which is
the statistic the guard reads against its 0.55 threshold. A single row is
noisy, and one row under the threshold is not a lost run. An AUC goes to three
places here and a GM-Relative MASE to four, because the 0.0568 band turns on
the fourth place and no AUC reading does.

Three k = 32 arms of this card have an exact 1.1M twin, and every twin held the
task. #404's plain twin ended at AUC 0.978 at 40,000 steps. #409's decay twin,
`dec_m090r100_ramp2k`, ended at 0.983 and scored 1.2295. #404's `s08` twin, on
the other momentum schedule, ended at 0.957. Each twin matches its arm on the
cell, the momentum, the decay ramp and the seed 20260520.

At 11.4M all three fall, and TWO of them lose the task outright. `k32_r100_09`
goes from 0.968 at step 2,000 to 0.773 at 40,000 and holds. `k32_r100_09_dec`
reaches chance at step 18,634. `k32_r200_08` reaches it at step 28,152. The
guard stopped both, so neither has a head or a score.

The k = 3 arm is the POSITIVE CONTROL at the same width, and it does not merely
survive. Its AUC floor rises with every leg: 0.993 to 40,000 steps, 0.997 to
100,000 and 0.999 to 200,000. Same width, same cell, same align target, same
seed 20260520. The worst reading of its last leg beats the worst reading of its
first.

So it takes BOTH the width and a depth above 3. At 1.1M all three k = 32 twins
held, at 0.978, 0.957 and 0.983. At 11.4M this cell at k = 3 improves for
200,000 steps. Neither the width nor the depth loses the task alone.

At k = 32 and 11.4M, two treatments each finish the job by themselves:

| EMA momentum | no decay | `L_rep` decay to 0.0 by 2,000 |
|---|---|---|
| 0.9 to 1.0 at 100k | held, ends 0.773 | LOST at step 18,634 |
| 0.8 to 1.0 at 200k | LOST at step 28,152 | not run |

THE DECAY IS NOT REQUIRED. `k32_r200_08` carries `L_rep` at weight 1.0 for
every one of its 28,152 steps and loses the task anyway. A ramp that starts at
0.8 does it with the full objective in place.

Read the table by column and by row, never on the diagonal. The decay column is
the pair at 0.9 to 1.0, where the no-decay arm HELD at 0.773 and its decay twin
lost at 18,634. The momentum column is the no-decay pair, where 0.9 held and
0.8 lost at 28,152. The two lost arms differ from each other in BOTH the
momentum and the decay, so the gap between 18,634 and 28,152 measures nothing.

NOT EVERY k = 32 ARM LOSES THE TASK. Two of the three did. `k32_r100_09` ran
degraded and held, with a floor of 0.683 at step 20,872 that never crossed.

The worst AUC is also the worst score: 0.773 and 1.4629, against 0.999 and
1.2927 to 1.3495 on the k = 3 arms.

### Every depth above 3 erodes, and they are not ordered by depth

`k8_r100_09` puts a third depth beside the two cells, and it erodes as well. So
losing the task is not a property of k = 32.

| step | k = 3 | k = 8 | k = 32, EMA 0.9 | k = 32, EMA 0.8 |
|---|---|---|---|---|
| 2,000 | 0.993 | 0.971 | 0.968 | 0.877 |
| 5,000 | 0.998 | 0.961 | 0.937 | 0.814 |
| 8,000 | 0.998 | 0.920 | 0.889 | 0.788 |
| 10,000 | 0.998 | 0.881 | 0.892 | 0.741 |

The k = 3 column holds 0.998 and does not move. Every deeper column falls.

THE DEEPER ARMS ARE NOT ORDERED BY DEPTH. At step 10,000 the k = 8 arm reads
0.881 against the k = 32 arm's 0.892, so the DEEPER of the two is the healthier
one, and at 10,500 they read 0.863 and 0.881. The two run within 0.011 of each
other from step 8,000 and both sit far under k = 3. So the erosion does not
scale with k on this evidence, and this report does not say that it does.

`k8_r100_09` has run 10,700 steps of its 40,000, so its column is partial.

The EMA momentum moves the whole curve as well. `k32_r200_08` starts its ramp
at 0.8 and sits under `k32_r100_09` at every step, by 0.091 at step 2,000 and
by 0.101 at step 8,000. The two arms share the cell, the depth, the reduction
and the seed, so the momentum is the one thing between them.

The k = 3 arms have no 1.1M AUC anchor, because #373 published no AUC column
for cells A3 and A4. That limits a SIZE comparison at k = 3. It does not limit
the reading above, which compares four depths at one width inside this card.

The decay damages the two cells differently, and it shows early. At step 5,000
the k = 32 decay arm reads 0.746 against 0.937 for its no-decay twin, a gap of
0.191. The k = 3 decay arm reads 0.992 against 0.998, a gap of 0.006. Both
carry their `L_rep` weight at 0.0 from step 2,000.

`k3_r100_09_dec` then held the task to 40,000 steps. Its floor is 0.980 and it
ends at 0.997, against its twin's 0.999. That floor is higher than 0.904, which
is the BEST the k = 32 plain arm reaches after step 10,000. So the decay costs
nothing measurable at k = 3 over 38,000 steps at weight 0.0, and the same
treatment reaches chance at k = 32.

Two limits hold that reading. It is the AUC axis, and `k3_r100_09_dec` has no
1.1M twin, because every decay run of #409 used k = 32. It is also a
40,000-step reading: #409 carried its best decay arm to 200,000 steps and the
gap moved, and this card carries neither decay arm past 40,000.

## The loss by term

![the loss by term](plots/loss_terms.png)

This cell has two loss terms. `L_rep` carries the contrastive negatives, and
`L_align` pulls the forecast toward the latent. The losses CSV also writes an
`l_pred` column and leaves every row of it empty. The loss shape is
`cosine_similarity_batch_rep_only`, which has no prediction term, so the empty
column is not a logging fault.

## What this design can say

**One thing changes: the width.** Every arm trains #373's cell
`arm6_v2_combab_alignT` at `d_model` 384, `n_heads` 8, 3 encoder layers and 3
decoder layers: 11,431,548 trainable parameters against Moirai-2-Small's
11.4M. The depth `k`, the reduction, the EMA momentum, the seed and the
`L_rep` decay of each arm are its parent's values.

**The protocol is the parents', by construction.** `scripts/head_eval.sh`
calls #373's `head_eval_bb.sh` unchanged: a quantile head, a 2-layer
transformer, forecast length 16, batch 256, learning rate 1e-3, head seed
20260722, then the 97 GIFT-Eval configs under strategy B4. The one thing this
card adds is `CF_BB_SHAPE`, which gives the head trainer and the evaluation
the new width.

**The head budget is not matched at every stop.** This card trains a
30,000-step head at every stop. #404 and #409 did the same at 40,000 steps, so
their numbers compare directly. #373 trained a 15,000-step head at 40,000
steps and 30,000 at 100,000 and 200,000, so its 40,000-step numbers do not.
The tables name the budget of every reference.

**The head was tuned on a 64-wide backbone.** A 30,000-step head that
under-trains a 384-wide encoder biases every 11.4M score in this report the
same way. Comparisons inside one size survive that bias. The size headline
does not, and it carries this caveat.

**The learning rate is not proven at this width.** Every arm above trains at
1e-3, which is the Moirai recipe. This project does not use muP, and the
trainer builds one AdamW group over all parameters. So a rate that fits
`d_model` 64 need not fit 384. Three arms sweep the rate on configuration 1,
at 5.6e-4, 3.3e-4 and 1.67e-4. Each one moves the rate column alone.

D is the best rate arm against BOTH seeds of configuration 1, 1.3495 and
1.2927, and not against 1.3495 alone. 1.3495 is the worse of the two, so an
arm that beats it by less than 0.0568 shows a seed draw and not a rate
effect. A D above 0.0568 on the better seed voids every 1e-3 number of this
card, and phase 1 repeats at the winning rate. Every arm inside the band, or
worse, ends the learning-rate explanation at width 384.

**The band is 0.0568, and this card measured it.** `k3_r100_09b` repeats
`k3_r100_09` at seed 20260525 alone, and it scores 1.2927 against 1.3495 at
the 40,000-step stop. That spread is this size's own seed band. It is wider
than the 0.0471 #409 measured at 1.1M, so this report ranks on 0.0568. Two
numbers closer than that are not ranked.

**A decay verdict at 40,000 steps is a 40,000-step verdict.** The two decay
arms pair with their plain twins at that stop alone. #409 carried its best
decay arm to 200,000 steps and the gap moved, so this card does not project
its pair past the stop it measured.

## The arms

_(from `scripts/arms.tsv`)_

## The tables

_(from `results/tables.md`)_

## How to repeat it

Backbones on one card, heads on the other. A backbone queue trains no head,
and one sweep starts every head that a checkpoint lacks.

```bash
cd reports/2026-09-06_moirai_small_size
BB_GPU=0 bash run.sh size smoke trial     # the shape, the cost, one arm end to end

# The heads. One per card, so start it before the backbones.
BB_GPU=1 bash scripts/head_sweep.sh &

# The backbones, two of one card at a time, largest memory need first.
# A small arm ahead of a large one takes the window the large one needs.
BB_GPU=0 CF412_QUEUE="k32_r100_09 k32_r100_09_dec" bash scripts/queue_backbones.sh
BB_GPU=0 CF412_QUEUE="k32_r200_08 k8_r100_09"      bash scripts/queue_backbones.sh
BB_GPU=1 CF412_QUEUE="k3_r100_09 k3_r100_09b k3_r100_09_dec" \
  bash scripts/queue_backbones.sh
BB_GPU=1 CF412_QUEUE="k3_r100_09_lr56 k3_r100_09_lr33 k3_r100_09_lr17" \
  bash scripts/queue_backbones.sh

# The climb. `phase1.sh` resumes the furthest checkpoint, so the stops are one
# continuous run and a stop it never asks for costs nothing later.
BB_GPU=0 STOPS="100000 200000" ARMS="k3_r100_09" bash run.sh phase1

bash run.sh collect && bash scripts/make_plots.sh && bash scripts/gate.sh
```

`scripts/arm_busy.sh <arm>` and `scripts/head_busy.sh <arm> <stop>` answer
whether an arm or a checkpoint is already running. Ask them before you start
anything by hand. Nothing under this card stops two trainers on one arm.
