# The same objective at Moirai-2-Small size

**DRAFT — the card still runs. The verdict below covers the arms that landed.
`results/tables.md` and `plots/` refresh every 15 minutes.**

Every score of this project comes from a backbone of 1.1M parameters.
Moirai-2-Small holds 11.4M and scores 0.728 on the same 97 GIFT-Eval configs,
where our best is 1.0651. This card retrains the published teacher-align
configurations at 11.4M parameters, so we learn whether the objective is weak
or whether the model is small.

## The answer

_(filled when the head-matched stops land)_

![the scores](plots/scores.png)

## The climb

![the climb](plots/climb.png)

## One arm lost the contrastive task

![the contrastive AUC](plots/auc.png)

`k32_r100_09_dec` is configuration 2 with the `L_rep` decay. Its rolling AUC
median reached chance at step 18,634 and stayed there, so the guard stopped
the leg. That arm has no head and no score.

#409 ran the same decay on the same cell at 1.1M parameters over 22 runs. One
run lost the task there, and it is the run at the fixed EMA momentum 0.500.
This arm carries the ramping momentum, which held at 1.1M. The width is the
one column that changed.

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
at 5.6e-4, 3.3e-4 and 1.67e-4. Each one moves the rate column alone. D is
1.3495 minus the best of the three. A D above the 0.0471 band voids every
1e-3 number of this card, and phase 1 repeats at the winning rate. Every arm
inside the band, or worse, ends the learning-rate explanation at width 384.

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

```bash
cd reports/2026-09-06_moirai_small_size
BB_GPU=0 bash run.sh size smoke trial       # the shape, the cost, one arm end to end
BB_GPU=0 STOPS=40000 ARMS="k32_r100_09 k32_r100_09_dec" bash run.sh phase1
BB_GPU=1 STOPS=40000 ARMS="k3_r100_09 k3_r100_09b k3_r100_09_dec" bash run.sh phase1
BB_GPU=0 STOPS="100000 200000" ARMS="k3_r100_09" bash run.sh phase1
BB_GPU=1 STOPS=40000 ARMS="k3_r100_09_lr33 k3_r100_09_lr56" bash run.sh phase1
BB_GPU=1 STOPS=40000 ARMS="k3_r100_09_lr17" bash run.sh phase1
bash run.sh collect && bash scripts/make_plots.sh
```
