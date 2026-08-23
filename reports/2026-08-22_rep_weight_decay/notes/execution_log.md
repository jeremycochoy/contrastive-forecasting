# Execution log

Operational events of the run. The report keeps the science, this file keeps
the events (`reports/REPORT_STANDARD.md`).

## The machine

elisa runs every arm. It holds two RTX 4090 cards. Other agents already train
on both cards, so this card shares them and stops no other run.

## The first attempt, and why it was stopped

The first implementation moved the study to #373's k = 3 cell, under the `sum`
reduction, with `--align-target student`, on the Fable opinion in
`scripts/fable_opinion.md`. It ran two control arms for about 2.5 hours.

The user stopped it. That cell answers a different question, and the card gives
the cell: k = 32 under `mean`, against the teacher, at the sweep's best EMA
momentum. See the issue comment of 2026-08-23.

Everything that attempt wrote is deleted: the checkpoints under
`/home/jupyter/checkpoints_backup/cf-409` and `cf-409-trial`, and every file
under `results/` and `plots/`. No number of that attempt enters this card.

## The latent-drift probe

The probe is a diagnostic. It draws a fixed ARMA batch once, then does one
no-grad forward of it at every save step. At the trainer's own batch of 64 that
forward allocates a 4.32 GB block, and the allocator keeps it for the run.

elisa's cards carry other agents' work, so every arm of this card runs at
`--latent-drift-probe-batch-size 16`. That is `CF409_PROBE_BS` in `study.sh`,
and every arm takes the same value.

That flag cannot move the training. `generate_arma_batch` draws the probe batch
from `np.random.default_rng(seed)`, a LOCAL generator, and `probe()` runs under
`torch.no_grad()`. It changes the drift CSV, which this card does not read.

## The checkout

`cf409_check_checkout` refused `~/workspaces/contrastive-forecasting`. That
checkout carries none of the things this card needs, and `origin/experiments`
does not carry them either: `--rep-loss-weight-end` is on this card's branch
and no merge has happened.

The study therefore runs from the branch worktree,
`/tmp/contrastive-forecasting-409`, which `study.sh` takes as `CF409_WT` by
default. `experiments/hf_token.txt` is gitignored, so a fresh worktree needs
one copy of the token before the gate passes.

## Events

| time (UTC) | event |
|---|---|
| 2026-08-22 22:44 | first attempt, at the k = 3 cell |
| 2026-08-23 01:48 | the user stopped it. Its two legs killed, its GPUs cleared |
| 2026-08-23 02:00 | its checkpoints and results deleted |

## The cards, as this run found them

Card 0 carried another agent's run at the launch, 17.3 GB and about 50
percent. This card never touched it.

Card 1 was free. One leg of this cell holds 2.1 GB and takes the card to 26
percent, so card 1 holds SEVERAL legs. This run put three lanes on it, two
arms each. `scripts/launch.sh` with `GPUS="1 1 1"` deals them.

Two lanes on one card each deleted the other's head queue, because the queue
name carried the card index alone. The name now carries the lane's PID.

## The trial

`CF409_TRIAL=400` ran the whole pipeline on card 1 before the arms:

| stage | result |
|---|---|
| backbone, 400 steps | rc=0, 3.4 minutes, 1.96 steps/s |
| the five flags | decay `1.0 0.0 100`, seed 20260520, teacher, mean, ema `0.9 1.0 100000` |
| `rep_w` at step 200 | 0.0, and `l_rep` blank, which is the decay |
| head, 200 steps | rc=0, 31 seconds |
| GIFT-Eval, 97 configs | rc=0, 72 minutes, 4 shards on the CPU |
| `results/trial/scores.csv` | one row, `rep_w_at_stop` 0.000, score 1.5216 |
| `results/trial/auc_verdicts.tsv` | one row, verdict `held` |

The trial score is a wiring check at 400 backbone steps, not a measurement.
`results/trial/` holds it, apart from the study's own results.

## How many lanes card 1 holds

Measured over 900 s, with the trial's GIFT-Eval on the CPU beside them:

| lanes on card 1 | steps/s per lane | steps/s total | card |
|---|---|---|---|
| 1 | 1.96 | 1.96 | 26 % |
| 3 | 1.33 to 1.67 | 4.56 | 98 % |

Three lanes take 2.3 times the work of one. The card is full at three, so this
run adds no fourth lane. Six arms at 40,000 steps is 240,000 steps, which is
about 14.6 hours of backbone.

## Card 0 came free for four minutes, and was taken again

At 03:06 card 0 showed 21 GB free. The plan was to move the three arms that had
not started onto it, which needs the lane managers to release their second arm.
At 03:10, before any process was touched, the same agent's run was back on card
0 with 17.3 GB. The move was dropped and no process of this run was stopped.

The card-0 watch now asks for 20 minutes free, not one reading.

## A third agent took 11.5 GB of card 1 at 04:10

`rnd-472` started a run on card 1. This card shares it and stops nothing.

The risk it makes is the HEAD, not the backbone. `head_eval_bb.sh` asks for
7000 MiB free before it trains a head, and card 1 then held 5573 MiB free. A
lane frees only its own 2.4 GB when a backbone ends, and it starts the next
backbone at once, so a head could wait out its four-hour ceiling and abort with
no score.

`head_eval.sh` is idempotent, so a head that aborts is re-run. The watch reads
`results/stops.log` for `waiting for VRAM`, `TIMEOUT` and `ABORT`.

## The head needs the card, and card 1 could not give it

At 08:59 card 1 held 4805 MiB free: `rnd-472` had 11.5 GB and this card's three
backbones 7.1 GB. A head trainer started there dies at once —

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.32 GiB.
```

so `head_eval_bb.sh`'s 7000 MiB gate is a real need, not a margin. The head
batch is 256 and the references come from it, so the batch does not move.

A lane frees only its own 2.4 GB when a backbone ends, and it starts the next
backbone at once. So the three heads of the first three arms would have waited
out their four-hour ceiling and aborted.

`results/HOLD_ABOVE` holds 1000. `run_leg_k.sh` reads it fresh at every leg,
so the three arms that had not started refuse with exit 9, which is a code the
lane does not re-fire. The lanes then drain their head queues, and the heads
get the 7.1 GB back.

The three arms that refuse start again after that, at a lane count this run
takes from the head's own measured footprint.

## What one head holds, measured

The first head of this card, at 09:48 on card 1:

| process | card 1 |
|---|---|
| head trainer, 30,000 steps, batch 256 | 5464 MiB |
| one backbone leg | 2628 MiB |
| `rnd-472`, another agent | 11504 MiB |

Card 1 holds 24564 MiB. So while a head runs, this card takes TWO backbone
lanes and not three: three lanes leave 5176 MiB, and the next head's gate asks
for 7000. Two lanes leave 7804 MiB, which passes.

The three held arms therefore start again as two lanes. The third starts when
the heads of the first three arms are done.

## The two waves

| time | event |
|---|---|
| 02:28 | arms dec_s20, dec_s24, dec_s22 start, three lanes on card 1 |
| 09:47 to 09:50 | all three reach 40,000 steps |
| 09:47 to 09:51 | dec_s23, dec_s26, dec_s25 refuse on `HOLD_ABOVE`, exit 9 |
| 09:47 | the head of dec_s20 takes the card. Two heads queue behind it |
| 09:51 | `HOLD_ABOVE` removed |
| 09:52 | dec_s23 and dec_s25 start as two lanes |

dec_s26 starts when the three heads release the card.
