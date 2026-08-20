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
`results/null_frozen_teacher.csv` reads delta -0.0046, interval
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

## What the report must state, and where each number comes from

The gap list asks for these claims in the report itself. Each one has an
artefact behind it, so the report cites rather than asserts.

| claim | artefact |
|---|---|
| The teacher tensors are frozen from step 100,000 on. | `results/teacher_move_100k_200k.json`, `results/teacher_move_40k_100k.json` |
| The teacher HEAD still reads 36 student-owned tensors, and 32 of them move over that span. So the teacher stops are not draws of one encoder. | `results/teacher_head_inputs_100k_200k.json`, `results/teacher_pool.txt` |
| The aggregate config bootstrap on the 100k-to-200k teacher pair straddles zero: delta -0.0046, [-0.0199, 0.0123], p_improved 0.711. Its half-width, about 0.016, is wider than the 0.0141 selection gap. | `results/null_frozen_teacher.csv` |
| The shard order carries no data-mix confound. Two halves, pooled, sit 0.0008 apart in total variation over 40 shards. | `results/shard_order.json` |
| The head-seed band, measured here, against #393's published one and against the 0.0141 selection gap. | `head_band.py` review-gap-6 block, `results/head_band.csv` |
| The protocol seed drawn again here at 200k, against #373's published anchor. Machine and code drift at one head seed. | `results/replicate_200k.log`, `results/head_band.csv` |
| One backbone seed. The card answers "did THIS run keep improving", not "does A4 improve with more data". | `run_leg_k.sh` line 113 pins `SEED=20260520`. #373's own gap table already records that backbone-seed variance stays unmeasured. A second backbone seed costs 40 GPU-hours and this card does not buy one. |
| 1.0660 is a selected number, so a point near it is not a plateau. | `results/selection_context.json`: rank 1 of 99, runner-up +0.0141 |

A move smaller than the head-seed band decides nothing, whichever direction
it points. 300k carries one draw per head and no band, so the report must
say so beside that point.
