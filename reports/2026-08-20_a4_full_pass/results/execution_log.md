# Execution log — #407, A4 to one full pass

Operational record. The report holds the science. This file holds the events.

## Time zone

Every time in this file, in `throughput.txt` and in `schedule.txt` is UTC.

elisa runs `Europe/London`, which is `BST`, `UTC+1`, in August. The parent
study's scripts stamp their logs with `date '+%m-%d %H:%M:%S'`, so
`leg_<cell>.log`, `run_<run>.log`, `run_pass.log` and `stops.log` carry
LOCAL time and are one hour ahead of this file. Subtract one hour to read
them as UTC. This card's own scripts stamp `date -u`, so
`replicate_*.log`, `watchdog.log` and the mirror lines are already UTC.

## 2026-08-20, launch

Machine: elisa. Card 0 of two RTX 4090 cards. Other projects hold both cards.

### Preflight

| check | result |
|---|---|
| `full_pass.py --check-resume /home/jupyter/cf373_r3/sync` | pass |
| md5 `..._r2_200k.pth` | `f477c03525bf5e169704715511f1c6d7`, equal to the card |
| md5 `..._r2_200k_optimizer.pth` | `740891276637ff7bce744b1d9109d57a`, equal to the card |
| `full_pass.py --check-leg 300000` | pass, resumes `..._r2_200k.pth` |
| `experiments/hf_token.txt` | present, 37 bytes |
| `cell_claims.txt`, `HOLD_ABOVE` | absent, so no refusal |
| GIFT-Eval data at `~/workspaces/gift-eval-data` | present |
| GPU 0 free memory | 7522 MiB, above the 6500 MiB gate |
| disk free | 124 GB |

No artefact from an earlier attempt exists. No `run_pass.sh` process runs.
The `leg_300k`, `leg_450k` and `leg_665k` directories do not exist yet.

### Command

```
WT=/tmp/contrastive-forecasting-407 BB_GPU=0 nohup setsid bash \
  reports/2026-08-20_a4_full_pass/scripts/run_pass.sh \
  > reports/2026-08-20_a4_full_pass/results/run_pass.out 2>&1 &
```

Start 2026-08-20 18:21:31 UTC. `run_pass.log` stamps the same moment
`[08-20 19:21:31]`, in local time.

### Continuity, confirmed

`train.py` printed these three lines:

```
[checkpoint] Restored optimizer from .../cf393_..._r2_200k_optimizer.pth (step=200000, best_ff=0.2490)
Resumed from .../cf393_..._r2_200k.pth at step 200000
  [dataloader] 4274 shards, 42740000 total rows, target skip 12800000
  [dataloader] Fast-skip: starting at shard 1280/4274, then skipping 0 rows within it
```

The weights, the optimizer state, the step counter and the data pointer all
continue. The run does not start at step 0.

The leg writes into a new directory, `leg_300k`, so it does not overwrite the
200k checkpoint set. Checkpoint safety rule 2 holds.

## Notes

The driver runs three legs in series, and each leg trains two heads and runs
two GIFT-Eval passes. One leg plus its two stops must finish before the next
leg starts. A dead head costs one point, not the stops behind it, because the
driver records the gap and continues.

## Known, and not a fault

`collect.sh` prints `downsample failed` while a leg still runs. The losses CSV
logs one row per step, and `downsample_curve.py` refuses a file that gives it
fewer than a few points at `--stride 200`. A leg of 100,000 steps gives 500
points, so the guard passes at the stop. The driver calls `collect.sh` after
each stop, which is when the CSV is complete.

## Throughput, measured

A 900-second wall-clock window on the running leg: step 200,200 to 203,400,
so 3.556 steps per second. `train.py` prints 3.6 sps in the same window. The
leg holds 5,388 MiB on card 0.

The parent study gives the cost of one stop. A4 at 200k took 35 minutes for
the student head and 72 minutes for its GIFT-Eval. The driver runs the two
heads and the two evals in series, so one stop costs about 3.6 hours after
the leg.

`results/schedule.txt` holds the expected times. The whole card ends about
2026-08-22 17:22 UTC.

## 2026-08-20, review gaps closed

The ExperimentReviewer gap list is PR comment 5360229413. What ran, and when.

### 18:50 UTC — the head-seed band, on card 1 (gap 1)

`replicate_heads.sh 200000` on GPU 1, beside the leg on GPU 0. Two seeds,
20260723 and 20260724, two heads each, 30,000 steps each, then the 97
GIFT-Eval configs. The backbone is the card's own, md5
`f477c03525bf5e169704715511f1c6d7`, so the band is measured at the exact
point the card compares against.

Card 1 had 6,933 MiB free and another session's job on it, so the head VRAM
gate went from its 7,000 MiB default to 6,400 MiB. Measured: one head holds
5,468 MiB. The first head reached step 500 at loss 0.2097, against the
protocol seed's 0.2095 at the same step.

The heads run at 7.4 steps per second here against the protocol draw's 14.3,
because card 1 carries other work. The leg on card 0 stayed at 3.5 steps per
second through the launch, so the band costs the card no backbone time.

### 18:51 UTC — the numbers leave /tmp (gap 5)

`mirror_durable.sh` copies the scores, this study's `results/` and the two
logs the continuity gates read to `/home/jupyter/cf407_durable`. The
watchdog runs it every tick.

### 18:58 UTC — the watchdog (gap 6)

`watchdog.sh`, 1,800-second period. It re-fires the driver only when the
process is gone AND neither the train log nor the driver log moved for two
ticks. `pgrep -f run_pass.sh` alone is not enough: the shell that launched
the driver and the tail that watches it both carry that name in their own
command lines, so the test reads `argv[1]` out of `/proc`.

### 18:58 UTC — the teacher is frozen (gap 4)

`teacher_move.py`, on checkpoints already on disk. 100k against 200k moves
0 of 52 teacher tensors, bit for bit, while the student moves 106 of 110 at
relative L2 0.599. The 40k against 100k control moves all 52. So the answer
did not wait for the 300k stop.

`teacher_check.sh` repeats the test on every later pair, from the watchdog.

### 19:01 UTC — the shard order (gap 3)

`shard_order.py` read the `meta` and `source_id` columns of 12 shards from
`small_v1`, spanning shard 0 to shard 4273 and including the 1279/1280
boundary the 200,000-step mark falls on. It read no `series` column, so the
check moved a few MB and did not compete with the training stream.

### 18:55 to 19:02 UTC — the interval, the metrics, the figure (gaps 2, 9)

`stop_bootstrap.sh`, `metrics_table.py` and the ribbon in
`plot_full_pass.py`. All three read CSVs that already exist, so none costs
GPU time.

### 19:1x UTC — how selected the target is (gap 8)

`selection_context.py` reads #373's 99 score files. 1.0660 is rank 1 of 99,
and the runner-up is 1.0801, 0.0141 above it.

## 2026-08-20, round 3 review gaps

Nine items. Seven are closed on this page. Two need card 1 and are armed.

### 19:20 UTC — the two draws card 1 still owes (items 2, 7)

`band_queue.sh`, detached, 300-second period, card 1. It waits on card 1
rather than on a clock, because `head_vram_gate` serialises every head on
one flock.

Stage 1 is item 2: head seed 20260722 drawn AGAIN at 200,000 steps, here and
on this code. #373 drew that seed on another round and another box, so the
published 1.0660 carries head seed, machine and code version together. The
re-draw holds the seed still. Its tag carries `_s20260722`, so it cannot
overwrite the card's own number. It fires when the 20260723 and 20260724
chains drain, near 01:00 UTC.

Stage 2 is item 7: the band at 450,000 steps, two more head seeds, fired on
the CHECKPOINT rather than on the score. The watchdog already holds the band
at 665k. So 200k, 450k and 665k carry error bars, and 300k carries one draw.

### 19:2x UTC — which tensors the teacher head reads (item 3)

`teacher_head_inputs.py`. It does not argue about the assumption. It builds
the state dict the head trainer loads, for both checkpoints, and it compares
them. Then it runs both backbones over one fixed batch.

| pair | loaded | from teacher | from student | teacher moved | student moved |
|---|---:|---:|---:|---:|---:|
| 100k to 200k | 110 | 74 | 36 | 0 | 32 |
| 40k to 100k | 110 | 74 | 36 | 74 | 32 |

The 40k row is the control and it moves everything, as the EMA ramp says it
must. The 100k row is the one that matters. The teacher covers
`teacher_input_to_latent.*` and `teacher_encoder_layers.*` only. The
frequency table, the seasonality table and the three forecaster layers stay
the student's, and they keep training. The latents the head reads move with
them: relative L2 0.0136 on the encoder latents and 0.1013 on the forecaster
latents. The card's head reads both, under `--head-train-input e_then_f`.

`teacher_check.sh` now runs this on every later pair, from the watchdog.

### 19:3x UTC — the teacher pool (item 4)

`teacher_pool.py`. It gives the pool the review asked for and it labels it.
The teacher points share one encoder stack. They do not share one head
input, so the pool is not a null. It is how far the teacher head travels
while its encoder stack stands still. At n = 2 the range is 0.0046. It grows
to n = 5 as the stops land, and the watchdog refreshes it every tick.

**Corrected 2026-08-21, 09:xx UTC.** The label above was still too kind. A
number that pools five DIFFERENT models has no meaning, whatever label sits
beside it, and a reader who sees `mean`, `std` and `range` in a CSV reads a
draw statistic. `teacher_pool.py` is now `teacher_frozen_track.py`. It prints
each stop as its own model, it prints the change from one stop to the next,
and it computes no mean, no standard deviation and no pooled range over the
stops. Outputs: `teacher_frozen_track.csv`, `teacher_frozen_track.txt`.

### 19:0x UTC — 40 shards, not 12 (item 9)

`shard_order.py` reads 40 of the 4,274 shards and the verdict quotes that
count. It also pools each half of the run into ONE mix, which is the number
the card's question asks for: a per-shard distance mixes a real mix change
with the sampling noise of one shard, and `small_v1` holds short shards.

| half | shards | rows | total variation |
|---|---:|---:|---:|
| below shard 1280, which #373 read | 15 | 132,085 | reference |
| shard 1280 and up, which this card reads | 25 | 221,818 | 0.0008 |

The widest single shard sits 0.0326 from shard 0's, and that shard holds 424
rows against 10,000 in a full shard. So its distance is sampling noise.

### 19:4x UTC — the band comparison and the figure (items 5, 6, 8)

`head_band.py` prints the card's own band beside #393's published one, and
beside the gap that made 1.0660 the best. #393's pooled 0.0384 is the
largest range over EVERY cell. This cell's own rows are the closer
comparison, and `noise_band.py` gives them: 0.0118, 0.0080, 0.0049, 0.0047.
The script prints a verdict against the 0.0141 selection gap once the band
lands.

`plot_full_pass.py` writes `results/figure_caption.txt` and draws the same
text under the axes. It states that the ribbon is one number pooled over
both heads, which stops carry draws, and that the ribbon is an extrapolation
everywhere else. It gives the measured range and the draw count beside the
standard deviation. It also states that 40k comes from `cf373_r2` while 100k
and 200k come from `cf373_r3`, and that the 200k file carries an `_r2_`
infix.

### 19:5x UTC — claim 3, corrected (item 1)

The round-2 page called the `medium_long` interval a false positive. That
was a subset picked after the numbers were seen. The aggregate row of
`results/teacher_delta_bb100k_bb200k.csv` (named
`null_frozen_teacher.csv` at the time) reads delta -0.0046, interval
[-0.0199, 0.0123], p_improved 0.711, so the aggregate bootstrap PASSED.
`results/pr_comment_20260820_gaps.md` now shows all three rows and reads the
aggregate. `stop_bootstrap.sh` is sound and its docstring is accurate.

The same page also called 0.0046 a pure repeatability difference. Item 3
shows it is not. Both corrections are marked on that page.

### 19:42 UTC — the firing path, run rather than read (items 2, 7)

Items 2 and 7 wait on card 1, so the code that fires them is the risk. A
sandbox runs `band_queue.sh` against a stub, and eight tests cover it: stage
1 fires seed 20260722 at 200k, stage 1 waits while a band holds the card,
stage 2 waits for stage 1, stage 2 fires on the checkpoint, stage 2 waits
without one, a half-scored re-draw does not read as done, the fire cap stops
a runaway, and a drained band ends the queue.

The sandbox found one defect. `replicate_alive` matched
`*/replicate_heads.sh` from ANY checkout. The band running now was launched
by a relative path, so `replicate_alive` now resolves `argv[1]` against the
process's own working directory and demands this checkout's copy. Two
worktrees no longer read as one band.

It also gained a fire cap, `QUEUE_MAX_FIRES`, default 4. A draw that dies at
once would otherwise re-launch every 300 seconds for the whole card.

The queue restarted on the verified script at 19:42 UTC. It had fired
nothing, so the restart cost no work.

`watchdog.sh` carried the same loose test, so it took the same fix. It
restarted at 19:50 UTC and its first tick read driver=yes, step 218,200.
A restart costs the quiet counter and the re-fire counter only.

### 19:56 UTC — item 2 starts, ahead of the queue

The band's four draws run one at a time: `head_vram_gate` holds one flock per
card. Draw one of four was 65 minutes into its 68-minute head at 19:55 UTC.
So the band's GPU work ends near 00:20 UTC, and the band PROCESS ends near
01:30 UTC, after its last 72-minute eval on the CPU.

`band_queue.sh` waits for the process, so it would have started the re-draw
at 01:30 UTC and left card 1's GPU idle for about 70 minutes.

So the re-draw started now instead, by hand, with the same command the queue
would use. It holds no GPU. It sits in the same flock behind the band, and it
takes the card the moment a band head releases it.

    [2026-08-20T19:56:47Z] [rep200k] seeds 20260722  heads student teacher
    [2026-08-20T19:56:47Z] [rep200k] DRAW A4_k3_bb200k_student_s20260722 start

`fuser` on `/tmp/cf373_head_gpu1.lock` shows the band's head holding it and
the re-draw waiting. Both GPUs read compute mode `Default`, so `gpu_gate`
returns at once and the flock is the only queue.

The queue did not double-fire. A `QUEUE_ONCE` probe against the live machine
reads the re-draw as up and launches nothing. The queue now marks stage 1
done when both re-draw heads score, and it keeps stage 2 for the 450k band.

### 20:00 UTC — the read-back waits in a background task

`await_redraw.sh` blocks on the two re-draw score files, then runs
`collect_replicates.sh`, `head_band.py`, `teacher_frozen_track.py`,
`plot_full_pass.py` and `mirror_durable.sh`. So the numbers land in the
checkout the moment they exist, and no agent sits in a poll loop for hours.

It tells the re-draw's chain from the band's by `argv[3]`, the seed. It exits
2 on a timeout of 7 hours and 3 when the chain dies unscored. On 3 it does
nothing, because `band_queue.sh` owns that retry inside its own cap.

## What the report must state, and where each number comes from

The gap list asks for these claims in the report itself. Each one has an
artefact behind it, so the report cites rather than asserts.

| claim | artefact |
|---|---|
| The teacher tensors are frozen from step 100,000 on. | `results/teacher_move_100k_200k.json`, `results/teacher_move_40k_100k.json` |
| The teacher HEAD still reads 36 student-owned tensors, and 32 of them move over that span. So the teacher stops are five models, not five draws of one. `src/checkpoint.py:266` is the reason. | `results/teacher_head_inputs_100k_200k.json`, `results/teacher_frozen_track.txt` |
| The 100k-to-200k teacher pair is a change between two models, not a null. The aggregate config bootstrap reads delta -0.0046, [-0.0199, 0.0123], p_improved 0.711. | `results/teacher_delta_bb100k_bb200k.csv`, `results/teacher_delta_bb100k_bb200k.txt` |
| The noise band the report reads: three head seeds on ONE backbone at 200,000 steps. Student range 0.0018, teacher range 0.0064. | `results/head_band.csv` |
| The shard order carries no data-mix confound. Two halves, pooled, sit 0.0008 apart in total variation over 40 shards. | `results/shard_order.json` |
| The head-seed band, measured here, against #393's published one and against the 0.0141 selection gap. | `head_band.py` review-gap-6 block, `results/head_band.csv` |
| The protocol seed drawn again here at 200k, against #373's published anchor. Machine and code drift at one head seed. | `results/replicate_200k.log`, `results/head_band.csv` |
| One backbone seed. The card answers "did THIS run keep improving", not "does A4 improve with more data". | `run_leg_k.sh` line 113 pins `SEED=20260520`. #373's own gap table already records that backbone-seed variance stays unmeasured. A second backbone seed costs 40 GPU-hours and this card does not buy one. |
| 1.0660 is a selected number, so a point near it is not a plateau. | `results/selection_context.json`: rank 1 of 99, runner-up +0.0141 |

A move smaller than the head-seed band decides nothing, whichever direction
it points. `band_queue.sh` now covers 200k, 300k and 450k, and the watchdog
covers 665k, so every stop of the card gets its own band. Beside a stop
whose band has not drained yet, the report must say that the stop carries
one draw per head.


## Round 4 — the read-back, and the idle card

### 02:47 UTC, 21 Aug — the 200k band reads back

Six draws, all on disk, all across the 97-config gate. `collect_replicates.sh`
reports 6 pairs and 0 skips.

| head | s20260722 | s20260723 | s20260724 | mean | std | range |
|---|---|---|---|---|---|---|
| student | 1.0660 | 1.0652 | 1.0642 | 1.0651 | 0.0009 | 0.0018 |
| teacher | 1.0828 | 1.0809 | 1.0764 | 1.0800 | 0.0033 | 0.0064 |

The protocol re-draw reproduces #373 EXACTLY on both heads, delta +0.0000.
Machine drift and code drift are zero at one head seed. So the whole band is
head-seed spread, and no part of it comes from the move between boxes.

The largest range, 0.0064, stays under the 0.0141 gap that made 1.0660 the
project's best. A move larger than the band is therefore readable.

`head_band.py`, `teacher_frozen_track.py`, `plot_full_pass.py` and
`mirror_durable.sh`
all ran. The mirror holds 82 files.

### 02:53 UTC — the 300k band takes the idle card

The 300k checkpoint landed at 02:52 UTC and the 300k leg closed at 8.5 h.
Card 1 then stands idle until the 450k checkpoint, about 15 hours. Two more
head seeds at 300k cost that idle time and no more.

`band_queue.sh` now holds a stage TABLE rather than two hard-coded stages:

| stop | seeds | gate | state at 02:53 UTC |
|---|---|---|---|
| 200,000 | 20260722 | now | done |
| 300,000 | 20260723, 20260724 | ckpt | FIRED at 02:53:52Z |
| 450,000 | 20260723, 20260724 | ckpt | pending, armed |

A `ckpt` stage fires on the CHECKPOINT and not on the score, so the extra
seeds train while the driver still scores the protocol seed at the same stop.
The queue runs ONE band at a time, because `head_vram_gate` serialises card 1
on one flock and a second band would only queue behind the first.

The gate now demands the `_optimizer.pth` sidecar beside the checkpoint.
`save_snapshot` in `train.py` writes the backbone first and the sidecar
second, so a sidecar on disk proves the backbone write finished. Without that
test the queue could glob a checkpoint the driver is still writing.

Five decision paths were tested against the live machine under `QUEUE_DRY`:

- gate `now`, no score, card free: fires.
- gate `ckpt`, backbone on disk: fires.
- gate `ckpt`, no backbone: waits.
- a chain up at any stop: no stage fires, on either stop.
- every seed scored: reports DONE and the queue exits.

The queue reads its state off the disk at every start, so this restart did
not repeat the 200k re-draw and did not lose the 450k band.

### 03:05 UTC — the read-back stops depending on an agent

The 200k read-back only happened because an agent ran it by hand. Round 3
put it behind `await_redraw.sh`, a harness background task, and that task
died with its session. So the draws sat scored on disk while the checkout
kept the previous numbers and the figure went stale.

The lesson is not "do not use a background task". The lesson is that no
ARTEFACT may depend on one.

`read_back.sh` now holds the five steps in one place: `collect_replicates.sh`,
`head_band.py`, `teacher_frozen_track.py`, `plot_full_pass.py`,
`mirror_durable.sh`.
Two things that outlive an agent call it.

| caller | when |
|---|---|
| `watchdog.sh` | every hourly tick, for whatever drained since |
| `replicate_heads.sh` | the moment its own band drains |

The watchdog's bare `mirror_durable.sh` call moved inside `read_back.sh`, so
the mirror still runs every hour. The first tick of the restarted watchdog
read back clean at 03:05:06Z and mirrored 87 files. The same tick extended
`teacher_check.sh` to the new stop and wrote `teacher_move_200k_300k.json`
and `teacher_head_inputs_200k_300k.json`.

`await_redraw.sh` is deleted. `await_band.sh` replaces it and carries NO
work. It blocks until one stop's band scores and then exits, so an agent
wakes on the event rather than on a clock. It exits 0 when every draw
scored, 2 on its deadline, and 3 when no chain for that stop is alive. When
it dies with its session, nothing is lost.

Tests: 196 in `test_407_full_pass.py`.

## 2026-08-22

### 00:52 UTC — the 450k stop drained and the driver hit a wall

The driver scored the 450k stop and then blocked at its VRAM gate:

```
[08-22 01:52:37] [cf407] waiting for VRAM on GPU 0: 3883 MiB free, need 6500
```

Card 0 carried another project, 17,270 MiB, and three Jupyter kernels. The
665k leg is 215,000 steps, about 17.6 hours, so each hour in the gate is an
hour of the card lost.

### 00:57 UTC — the last leg moved to card 1

Card 1 held 24,207 MiB free and 0 percent. The 450k band's last draw
(teacher, seed 20260724) was already past its head training and inside
GIFT-Eval, which runs `--device cpu`. So card 1 was free of GPU work.

The blocked driver (pid 3369703) waited in `wait_vram`. Its only child was
`sleep 60`, so `kill` lost no training. The relaunch:

```
WT=/tmp/contrastive-forecasting-407 RUNS=/home/jupyter/cf373_r3/sync \
  BB_GPU=1 HEAD_GPU=1 nohup setsid bash scripts/run_pass.sh 665000 &
```

One stop, not three. That is what `open_stops` returns and what the
watchdog's own re-fire would pass.

The continuity gates passed and the leg resumed:

```
[08-22 01:57:55] [cf407] start stops=665000 ... bb_gpu=1 head_gpu=1
[08-22 01:57:55] [arm6_v2_combab_alignS] RESUME from ..._450k.pth (step 450k)
[08-22 01:57:55] [arm6_v2_combab_alignS] START target=665000 gpu=1
```

Card 1 then read 5,396 MiB and 37 percent, and the step counter moved.

### 00:58 UTC — the watchdog moved with it

The watchdog carried `BB_GPU=0` from its own launch. Two things follow from
that variable, and both were wrong after the move:

- a re-fire would put the driver back on the blocked card;
- `band_at_last_stop` takes `BAND_GPU` as `1 - BB_GPU`, so the 665k band
  would fire on card 0.

So the watchdog restarted with `BB_GPU=1 HEAD_GPU=1 BAND_GPU=1`. It was
asleep between ticks, so the restart interrupted no work. Its first tick
read `driver=yes step=450200 quiet=0 open='665000'`.

The 665k band stays armed: `replicate_665k.log` does not exist, so
`band_at_last_stop` still fires when the 665k checkpoint lands. Card 1 holds
24 GB, the leg holds 5.4 GB, and `head_vram_gate` holds one flock per card,
so the driver's own heads and the band draws take the card in turn.

`band_queue.sh` needed no change. It already ran with `BAND_GPU=1`.

### the numbers read back at this point

| stop | student | teacher |
|---|---|---|
| 200k | 1.0660 | 1.0828 |
| 300k | 1.0867 | 1.1030 |
| 450k | 1.0691 | 1.0986 |

Band draws at 450k: student 1.0761 and 1.0778, teacher 1.0924. The second
teacher draw (seed 20260724) is still in GIFT-Eval.

## 2026-08-22, the compute audit

### 12:52 UTC — the 665k band is conditional, not armed

The orchestrator audited the compute after the 450k band drained. The band
at the last stop was ARMED: `watchdog.sh` fired it when the 665,000-step
checkpoint landed, whatever the stop then scored. Two head seeds, two heads
each, cost about 8 GPU-hours.

The audit disarmed it. The band now fires on the SCORE, and only inside a
window.

**The rule.** Fire the band when the 665,000-step STUDENT score lands
inside

    |score - 1.0651| <= 0.01

**The center, 1.0651.** The mean of the 200,000-step student band, over
head seeds 20260722, 20260723 and 20260724. It is row `200000,student` of
`results/head_band.csv`. That number is the comparison the card makes.

**The radius, 0.01.** This card's measured pooled standard deviation is
0.0029, over both heads and the three banded stops. The radius is 3.4 of
those standard deviations.

**Why.** Inside the window, one draw cannot decide the comparison, so the
card buys the band. Outside it, a clearly high or a clearly low point reads
on its own, and the band buys no information. The card then keeps about 8
GPU-hours.

**Where the rule lives.** `scripts/band_decision.py`, not a comment. It
returns 0 for FIRE, 10 for SKIP and 20 for WAIT. `band_at_last_stop` in
`watchdog.sh` branches on those codes. `--explain` prints the two constants,
the window and the pooled standard deviation:

```
center     1.0651  (mean of the 200k student band)
radius     0.0100
window     [1.0551, 1.0751]
pooled sd  0.0029  (both heads, every banded stop)
radius     3.4 pooled sd
```

**The tests.** Section 14 of `tests/test_407_full_pass.py`, 28 tests. They
cover the rule and the firing path:

- 1.0651 tracks `head_band.csv` rather than a number typed twice.
- The pooled standard deviation reproduces the audit's 0.0029.
- FIRE inside the window, SKIP outside, WAIT with no score.
- The edge belongs to FIRE. Binary floats put `1.0651 + 0.01` at
  1.0751000000000002, so the rule carries a 1e-9 tolerance.
- A watchdog tick with no 665k score fires nothing and records nothing.
- A tick at 1.0660 fires the band. A tick at 1.0450 does not.
- A SKIP decides once. The second tick reads the latch file.
- A fired band does not fire twice.
- The last tick decides BEFORE the watchdog exits on an empty `open_stops`.

The suite is 224 tests and all pass.

**The latch.** `results/band_665k_decision.txt` holds the verdict. A SKIP
latches on that file. A FIRE latches on `replicate_665k.log`, exactly as
before. A WAIT writes nothing, so an early tick cannot freeze the decision.

**The restart.** `watchdog.sh` changed, so the watchdog restarted at
12:52:12 UTC on the new code. The old process (pid 1742859) was asleep
between ticks and its only child was `sleep 1800`, so the restart
interrupted no work. A restart costs the quiet counter and the re-fire
counter only. The first tick read:

```
[2026-08-22T12:52:16Z] tick driver=yes step=631200 quiet=0 open='665000'
WAIT   stop 665000 student: no score yet
```

The 665k leg on card 1 was not touched. `run_pass.sh` (pid 1741671) kept
its step counter through the restart.

`watchdog.sh` also gained `WATCHDOG_ONCE`, the test seam. It is the same
seam `band_queue.sh` carries as `QUEUE_ONCE`. The band decision runs inside
the tick loop, so no test could reach it without one.

**What did not change.** Both heads stay at the 665,000-step stop, as the
card asks. `band_queue.sh` owns no 665k stage and needed no change.

## 2026-08-22, round 7 review gaps

The reviewer returned eleven items. This section records what changed on
disk and what the report must carry. Every claim below names its artefact.

### 18:41 UTC — the wait for the last head

`await_stop.sh 665 teacher` started again at 18:41 UTC. The round-6 copy of
that waiter died with its session at about 13:00 UTC. The waiter carries no
work, so nothing was lost. `read_back.sh` runs from the watchdog every
thirty minutes and brings each number into the checkout on its own.

### 18:44 UTC — the 665,000-step student stop crossed (item 1)

`collect.sh` moved the student pair into this study: the score, the 97
per-config rows, the eval log and the leg's losses CSV.

| artefact | path |
|---|---|
| score | `results/score_A4_k3_bb665k_student.txt` |
| 97 rows | `results/eval/A4_k3_bb665k_student/all_results.csv` |
| summary | `results/eval/A4_k3_bb665k_student/summary.txt` |
| losses | `results/curves/leg_665k__cf393_arm6_v2_combab_alignS_cf373k3_losses.csv` |

### 18:50 UTC — the figure, redrawn (items 2, 3, 9, 10)

The pooled ribbon is gone. Four changes:

- The two lines join the PER-STOP MEANS. At 450,000 steps the old line ran
  through the protocol draw, 1.0691, while the tables carried the band
  mean, 1.0743. The picture and the text now agree.
- Every head-seed draw is a small dot in its head colour. The dots are the
  measured spread, so no pooled number stands between the reader and the
  data. The old ribbon was one number, plus and minus 0.0029, over both
  heads and every stop. The measured ranges run from 0.0018 to 0.0087.
- A hollow ring marks the rollout-depth study's published point at 40k,
  100k and 200k. At 200,000 steps that point is one of three draws, so the
  ring shows which draw the parent study published.
- The rule label reads "prior best, 1.0660" and the legend reads
  "rollout-depth study point". No issue number reaches the axes.

The 665,000-step point carries one draw, so it carries no spread.

`results/figure_caption.txt` holds the caption, word for word.
`results/figure_provenance.txt` holds the two checkpoint trees behind the
three hollow points. The provenance stays off the axes.

### 18:52 UTC — the paired bootstrap and the goal metrics (item 6)

Both analyses read `all_results.csv` files that already exist, so both cost
CPU seconds and no GPU time.

`stop_bootstrap.sh` ran for every stop and head on disk. The resampling
unit is the dataset, not the config. Output: `results/stop_bootstrap.csv`
and `results/stop_bootstrap.txt`.

| comparison | delta | 95% CI | improved |
|---|---:|---|---:|
| 200k to 300k, student | +0.0207 | [+0.0051, +0.0373] | 0.4% |
| 200k to 450k, student | +0.0031 | [-0.0057, +0.0114] | 24.1% |
| 200k to 665k, student | +0.0123 | [+0.0009, +0.0240] | 1.7% |
| 200k to 300k, teacher | +0.0202 | [+0.0085, +0.0330] | 0.0% |
| 200k to 450k, teacher | +0.0158 | [+0.0010, +0.0311] | 1.7% |

A positive delta is worse. This bootstrap covers the protocol seed only,
and it measures the spread over the 97 configs. It does not measure head
noise or backbone noise.

`metrics_table.py` ran for every stop, head and head seed. It writes the
per-draw table and the per-stop means:

| file | what |
|---|---|
| `results/metrics_table.csv`, `.md` | one row per (stop, head, seed) |
| `results/metrics_table_means.csv`, `.md` | one row per (stop, head) |

The recomputed GM-Relative MASE agrees with every published score to
better than 5e-4, so one seasonal-naive denominator is in play.

The three goal metrics do not all move the same way. Student head, band
mean at 200,000 steps against the one draw at 665,000 steps:

| metric | 200k mean | 200k range | 665k, one draw | reads |
|---|---:|---:|---:|---|
| GM-MASE | 1.4889 | 0.0024 | 1.5073 | worse, outside the range |
| GM-MAPE_SN | 1.0481 | 0.0191 | 1.0846 | worse, outside the range |
| GM-CRPS_SN | 0.7792 | 0.0063 | 0.7779 | inside the 200k range |

The 200,000-step CRPS draws are 0.7755, 0.7802 and 0.7818. The 665,000-step
draw, 0.7779, sits between them. So GM-CRPS_SN shows no readable move.

`head_band.csv` gives the 200k teacher mean as 1.0800 and
`metrics_table_means.csv` gives 1.0801. The first averages the 4-decimal
score files. The second recomputes each draw from its 97 rows. The
difference is 1.8e-5, and it is rounding.

### 18:56 UTC — the band rule, and what the skip costs (item 4)

The rule went in at 12:52 UTC on 2026-08-22. The leg was at step 631,200
and no 665,000-step score was on disk. `band_decision.py` fires a band when
the 665,000-step STUDENT score lands inside [1.0551, 1.0751], which is the
200,000-step student band mean plus and minus 0.0100. The student scored
1.0783 at 17:52 UTC, 0.0132 from the center, so the rule skipped the band.
The order of those two events is what makes the skip legitimate.

| fact | value | artefact |
|---|---|---|
| rule in place | 12:52 UTC, 2026-08-22, step 631,200 | `results/watchdog.log` |
| window | [1.0551, 1.0751] | `band_decision.py --explain` |
| score | 1.0783 at 17:52 UTC | `results/score_A4_k3_bb665k_student.txt` |
| verdict | SKIP | `results/band_665k_decision.txt` |

The skip does not move the verdict. `band_decision.py --offsets` measures
how far the protocol draw sits from its own band mean, at every banded stop
of this card, and applies the same offsets to the one draw at 665,000
steps. Output: `results/band_665k_offsets.txt`.

```
   stop  band mean   protocol    offset
 200000     1.0651     1.0660   +0.0009
 300000     1.0864     1.0867   +0.0003
 450000     1.0743     1.0691   -0.0052
measured offsets: -0.0052 to +0.0009
stop 665000 has one draw, 1.0783. Its band mean lands between 1.0774 and 1.0835.
against the 1.0651 band mean at 200000 steps: +0.0123 to +0.0184
pooled head-seed sd 0.0029, so the rise is 4.2 to 6.3 of it
```

The band would have cost about 8 GPU-hours and it would not have changed
the answer.

### what the report may say about the last three points (item 7)

The student means move -0.0121 from 300,000 to 450,000 steps, which is 4.2
pooled standard deviations. They then move +0.0040 from 450,000 to 665,000
steps, which is 1.4 pooled standard deviations and inside the widest
measured head-seed range, 0.0087.

The head-seed band bounds head noise only. Nothing in this card measures
how much the score moves between two nearby checkpoints of one run.

- The report may say that every stop past 200,000 steps is worse than
  200,000 steps.
- The report may not say that 665,000 steps is worse than 450,000 steps.
- The report may not draw a trend through the last three points, and the
  figure draws none.

### what one run supports (item 8)

Every point comes from one backbone seed and one continuous trajectory.
`run_leg_k.sh` line 113 pins `SEED=20260520`.

The recipe holds constant across the card's span. The launcher passes
`--lr 1e-3` and no schedule flag, and `train.py` prints `lr=0.001` at the
start of all four legs. The EMA momentum ramp is anchored to a fixed step
count, `--ema-tau-ramp-steps 100000`, and it holds at 1.0 past that step.
So no schedule change sits between 200,000 and 665,000 steps.

- The report must write "this run did not improve past 200,000 steps".
- The report must not write "A4 does not improve with more data".
- The report must state the constant learning rate.
- The report must give no mechanism for the rise at 300,000 steps. The run
  measured that it happened, not why.

### 1.0660 is a selected minimum (item 9)

`results/selection_context.json`: rank 1 of 99 published scores, runner-up
1.0801, gap 0.0141. A point near 1.0660 is not a plateau.

The like-for-like comparison is band mean against band mean: 1.0651 at
200,000 steps against 1.0783 at 665,000 steps, a rise of 0.0132.

### one full pass, to the digit (item 11)

The card claims one full pass. 665,000 steps is not exactly one pass, and
the three row counts disagree:

| source | rows | steps for one pass | 665,000 is |
|---|---:|---:|---:|
| `small_v1/manifest.json` | 42,571,692 | 665,182 | 99.97% |
| shard arithmetic | 42,740,000 | 667,812 | 99.58% |
| the card's own number | - | 665,156 | 99.98% |

`full_pass.ROW_COUNTS` holds the first two. The report must name the count
it uses and the percentage, next to the claim.

### the smaller items (item 11)

- `results/figure_caption.txt` was stale. `plot_full_pass.py` writes it now,
  so the file cannot drift from the figure again.
- The retracted "free null of 0.0046" stays out of the report. It appears in
  four PR comments and in the `pr_comment_*.md` files under `results/`.
  Those files are the round-by-round record and they stay where they are.
  The corrected fact is the one to state: the teacher head loads 36 student
  tensors, and 32 of them move at every stop.
  Artefacts: `results/teacher_head_inputs_450k_665k.json`,
  `results/teacher_frozen_track.txt`.
- This card reuses the parent study's launcher, so the launcher writes the
  scores and the training log into
  `reports/2026-08-08_rollout_depth/results/`. `collect.sh` mirrors every
  one of them into this study, and `read_back.sh` runs it from the watchdog
  every thirty minutes. Nothing this card measured lives in one place only.
- The report is one file at the study root, `a4_full_pass.md`. The eight
  `pr_comment_*.md` files and this execution log stay under `results/`.

### tests

`tests/test_407_full_pass.py` is 240 tests and all pass. Round 7 replaced
the ribbon caption tests with tests on the new figure contract, and it added
tests on the offsets mode of the band rule.
