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

## 14:18 — the two running arms were killed from outside

`dec_s23` and `dec_s25` both exited rc=137 (SIGKILL) in the same second, at
steps 22,800 and 22,600. Their lane managers went with them. No trace in any
log of this card, no host OOM, and the head that was training on card 1
survived.

At 14:19 the pipeline orchestrator started a new Implementer stage for #409 in
THIS worktree, `/tmp/contrastive-forecasting-409`. It had already changed
`tests/test_409_launcher_shape.py`.

Two things follow, and this run did both.

**The code is frozen.** An Implementer edits `src/loss.py` and the trainer,
which is what the arms train. Arms four to six would then train other code than
arms one to three. `git archive HEAD` at `614788a5` gives
`/home/jupyter/cf409_frozen`, and the three arms restart from there through
`WT`. The trainer files were identical to the worktree at that moment, and no
commit had touched `src/`, `experiments/` or #373's runner since 02:00. So all
six arms train one code state.

`CF409_RESULTS` and `CF409_PLOTS` still point at the worktree, so every
artefact of this card stays in one place.

**The legs save every 5000 steps.** `SAVE_EVERY` was 20,000, so this kill cost
2,800 and 2,600 steps. The three restarted legs resume from their 20,000-step
checkpoints with their optimizer state. A save cadence does not touch the
weights a step produces.

## 14:26 — card 0 came free, and this time it held

`rnd-434` ended and left 21 GB. The three restarted arms run there. That takes
this card off the contended card 1, which `rnd-472` shares, and gives the heads
of those arms the 7000 MiB their gate asks for.

## 15:01 — the seven new schedules start, and the three seed arms stop

A new session picked the study up. `dec_s23`, `dec_s25` and `dec_s26` are seed
repeats of the schedule `0.9 to 1.0 at 100k`. That schedule is measured twice
already, at 1.2670 (`dec_s20`) and 1.2593 (`dec_s22`), so a third and fourth
seed of it buys no answer this card asks for. They do not start again. Their
20,000-step checkpoints stay on disk.

The card carries SEVEN schedules that no run of #409 has trained.

| lane | card | arms |
|---|---|---|
| A | 1 | dec_m099_fix, dec_m095_fix, dec_m080_r200 |
| B | 1 | dec_m090_r60, dec_m090_fix |
| C | 0, when it frees | dec_m095_r100, dec_m090_r200 |

Card 1 holds `rnd-472` at 11.5 GB, so it has 13.0 GB free. Two legs (2×2628)
and one head (5464) take 10.7 GB of that. Three legs would leave 5.2 GB and the
head gate asks for 7.0, so card 1 takes TWO lanes.

Card 0 holds 22.4 GB of two other agents' runs. Lane C waits on
`lane_when_free.sh`, which now asks for 9000 MiB over FOUR readings, 20 minutes
apart in total. One reading is not enough: card 0 showed 21 GB free at 03:06
and was full again at 03:10.

`WT=/home/jupyter/cf409_frozen` on every lane. The trainer, #373's runner, the
head scripts and `src/` of that archive are identical to the worktree, so all
ten arms train one code state even if an Implementer edits the worktree again.

### The legs now save every 5000 steps

`SAVE_EVERY` defaults to 20,000 in #373's runner, and nothing set it. Two
outside kills already cost this card 2,800 and 2,600 steps at that cadence. One
checkpoint pair is 11 MB, so eight pairs an arm cost 88 MB and buy back up to
15,000 steps of a 7.4-hour leg. Both lanes restarted at 15:04 with
`SAVE_EVERY=5000`, five minutes into the first leg.

## 08-23 18:48 — the box lost DNS, and the lane read it as three dead arms

`Failed to resolve 'huggingface.co'`. The data streams from the Hub, so both
live legs died with rc=1 in about 3 seconds.

`phase1.sh` then spent an arm's whole retry ladder in two minutes, because each
re-fire hit the same dead network and died in 3 seconds. It declared the arm
dead, moved to the next one and did the same. Three arms went in seven minutes.
Nothing ran for the next 27 hours.

`dec_m080_r200` was at 19,900 steps at `SAVE_EVERY=20000`, so it had saved no
step checkpoint and lost all of it.

### The fix

`scripts/hub_gate.sh` is new and shared. It reads a dead leg's tail, probes the
Hub and holds the growing wait.

| Change | Where |
|---|---|
| A Hub failure exits 20, not 1 | `run_leg_k.sh` |
| Code 20 costs no try, and waits over hours | `phase1.sh` |
| A lane probes the Hub before any arm | `phase1.sh` |
| A fresh start needs a cell with no step checkpoint | `run_leg_k.sh` |
| `SAVE_EVERY` defaults to 5000 | `study.sh` |

One leg gets at most `CF409_NET_DEADLINE` (6 hours) of outage. Past that the
lane stops, and every arm keeps its checkpoints for a later lane.

## 08-25 16:07 — a rebuild ran `launch.sh` for real, and it started `dec_s23`

The rebuild of the loss-by-term artefacts moved the `RUN_STATE.md` writer out
of `launch.sh` into `scripts/run_state.sh`. The check of that edit ran
`bash scripts/launch.sh` with no `CF409_DRY_RUN=1`, so the launcher started.

It ran for two minutes and was killed by process group at 16:09.

What it did:

| what | outcome |
|---|---|
| resumed `dec_s23` from its 20,000-step checkpoint | ~100 steps, killed |
| appended 100 rows to `dec_s23_r2_losses.csv` | steps 20,001 to 20,100 |
| started the head of `dec_s20` | `SKIP — already scored 1.2670` |
| appended to `arms.log`, `phase1.log`, `stops.log`, `heads.log` | |

What it did NOT do. No `.pth` moved: the save cadence is 5,000 steps and the
leg ran 100. No score file, no `scores.csv` row and no qhead checkpoint
changed. `dec_s23` still reaches 22,900 steps, from its base CSV.

### It exposed a real bug in the reader

`dec_s23` writes TWO CSVs: a base that reached 22,900 steps, and an `_r2` that
gave up at 20,300. `arm_style.read_run` took the files in name order, so the
short `_r2` overwrote 300 steps in the middle of the long base run.

`read_run` now orders the files by the step each one REACHES and lets the
furthest win. That is right on both shapes: `dec_m099_fix` ran on in `_r2` to
40,000, so `_r2` wins, and `dec_s23` ran on in its base file, so the base wins.
`tests/test_409_score_pipeline.py::TestTheReaderStitchesARefiredLeg` pins it.

### The rule

`launch.sh` starts backbones. Use `bash scripts/run_state.sh "<note>"` to
refresh the state file, and `CF409_DRY_RUN=1` to check the launcher's plan.

## 08-25 16:14 — the eighth backbone started, and it left the EMA axis

`dec_ramp30k_m080` started on card 0. A second session started it, detached,
while this session rebuilt the loss-by-term artefacts.

It is `dec_m080_r200` again — the same cell, the same schedule 0.8 to 1.0 at
200k, the same seed 20260520 — with ONE change: the decay ramp runs to 30,000
steps, not 10,000. `CF409_REP_W_RAMP=30000` on its lane sets it, so the arms
table cannot tell the pair apart and neither can any figure keyed on the
schedule.

Two files carry that:

- `scripts/arms.tsv` names the arm and its ramp in the "Why each row earns a
  backbone" section, and states that the ramp is not a column.
- `arm_style.read_arms` now marks such a row `ambiguous`, so `arm_label`
  falls back to the arm name. Without it, `plots/scores.png` would draw two
  rows called "0.8 to 1.0 at 200k, seed 20260520".

It also fixed a second thing. `plot_scores.py` reads `repeat` to draw the seed
spread bar, and `repeat` used to mean "another row shares this schedule". The
new arm shares one and it is NOT a seed repeat, so the bar read 4 seeds and a
range of 0.0460 where the truth is 3 seeds and 0.0219. `repeat` now means
"another row shares this schedule at a DIFFERENT seed".

Why the eighth arm left the EMA axis: five schedules from 0.500 to 0.990 all
lose to the card's target of 1.1491 by 0.0861 or more, which is nearly four
times the measured seed spread of 0.0219. A sixth schedule cannot close that.

The figures and the tables skip the arm until it passes 1,000 steps, and each
one names it on stdout. Re-run `bash scripts/make_plots.sh` when it stops.

## 08-25 17:15 — a watch that rebuilds the artefacts when the last leg stops

`scripts/refresh_when_done.sh` runs detached. It polls every 10 minutes for a
leg of this study under `$CF409_ROOT`, and when none is left it runs
`make_plots.sh` and `run_state.sh`. It gives up after 14 hours and rebuilds
anyway. `results/refresh_when_done.log` holds its ticks.

Why: gap 4 of the run review was stale artefacts, and the cause was a launcher
that died and stopped refreshing them. `dec_ramp30k_m080` lands in about seven
hours plus a head and an eval, so the same staleness would return.

It starts no leg, kills nothing and commits nothing. Four tests pin that. A
session must commit the refreshed artefacts: a background `git` beside a live
session is how two sessions lose each other's work.

## 08-27 02:20 — the decay ramp became a column of the arms table

Four rows, `dec_m080_r200`, `dec_ramp5k_m080`, `dec_ramp20k_m080` and
`dec_ramp30k_m080`, held identical values. They differed only in
`CF409_REP_W_RAMP` on the lane that ran each one, so a reader could not
reproduce an arm from its row. The 08-25 entry above records that state.

`scripts/arms.tsv` now carries a `rep_ramp` column, filled with the ramp each
arm actually ran. Every value matches the arm's leg log and the `ramp` column
of `results/scores.csv`, and `collect.sh` rewrites `scores.csv` byte for byte.

`cf409_decay_ramp_of` reads that column, and no environment value moves it: a
stray `CF409_REP_W_RAMP` in a lane environment would otherwise rewrite the ramp
of every arm in the tables. `cf409_ramp <arm>` is what one LEG runs, and
`CF409_REP_W_RAMP` overrides it for a dry run of a ramp that has no row yet.
`cf409_decay_args`, `cf409_decay_sig` and `cf409_rep_w_at` now take the arm.

`arm_style.DECAY_RAMP`, a second copy of the same three arm names, is gone.
`plot_auc.py` reads the ramp of each arm it draws and shades the longest one,
because one band at one arm's ramp read as every arm's.

## 08-31 08:10 — the stop became part of the key of `scores.csv`

Two arms ran past the 40,000-step stop, to 80,000. `collect.sh` built one row
for each arm, so each of those two arms lost a measured score. It also read the
score files alone, and `results/` is under git: a checkout took the 40,000-step
score files of `dec_m090r100_ramp5k`, `dec_m090r100_ramp2k` and
`dec_m090r100_ramp1k`, and the table then showed the 80,000-step numbers in
their place.

The key is now (arm, stop), and a second source reads the aggregate line of the
eval's own log when this results directory no longer holds the score file. The
table carries 15 rows over the two stops. The three lost score files are back,
from their own eval logs, so no head runs a second time.

`plot_scores.py`, `plot_axes.py` and `rank_gate.py` held three copies of one
`read_scores`, each keyed by the arm alone. One copy is left, in
`arm_style.py`, and it takes one stop. Every figure and the gate read the
40,000-step rows, which is the stop each of them names.

`plots/axes.png` still draws one ramp family, the one at momentum 0.840. The
three new arms move the ramp on the 0.940 schedule, so no panel holds them.

The AUC gate's 1,000-step warmup was inside every ramp before these three
arms. The shortest ramp is now 1,000 steps, so the gate turns on at the step
that arm's weight reaches 0.0. All three held the task, at floors 0.9075,
0.9092 and 0.8688 against the 0.55 threshold.
