«Agent ExperimentRunner claude-opus-5 writing»

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

## The 200k head-seed band is complete, and read back

Six draws, all on disk, all across the 97-config gate. `collect_replicates.sh`
reports 6 pairs and 0 skips.

| head | s20260722 | s20260723 | s20260724 | mean | std | range |
|---|---|---|---|---|---|---|
| student | 1.0660 | 1.0652 | 1.0642 | 1.0651 | 0.0009 | 0.0018 |
| teacher | 1.0828 | 1.0809 | 1.0764 | 1.0800 | 0.0033 | 0.0064 |

The protocol seed, drawn again on this machine and on this code, reproduces
#373 exactly.

| stop | head | #373 | here | delta |
|---|---|---|---|---|
| 200,000 | student | 1.0660 | 1.0660 | +0.0000 |
| 200,000 | teacher | 1.0828 | 1.0828 | +0.0000 |

Machine drift and code drift are zero at one head seed. So the whole band is
head-seed spread, and no part of it comes from the move between boxes.

**Headline.** The largest head-seed range is 0.0064. The gap that made 1.0660
the project's best is 0.0141. The band is narrower than the gap, so a move
larger than the band is readable.

The read-back ran in full: `collect_replicates.sh`, `head_band.py`,
`teacher_pool.py`, `plot_full_pass.py`, `mirror_durable.sh`. The figure is
`plots/full_pass.png` and the table is `results/head_band.csv`. The durable
mirror holds 82 files.

## The 300k band takes the idle card

The 300k checkpoint landed at 02:52 UTC and that leg closed in 8.5 h. Card 1
then stands idle until the 450k checkpoint, about 15 hours. Two more head
seeds at 300k cost that idle time and no more.

`band_queue.sh` now holds a stage TABLE, not two hard-coded stages.

| stop | seeds | gate | state |
|---|---|---|---|
| 200,000 | 20260722 | now | done |
| 300,000 | 20260723, 20260724 | ckpt | **fired 02:53:52Z** |
| 450,000 | 20260723, 20260724 | ckpt | **pending, armed** |

```
[2026-08-21T02:53:52Z] [cf407-queue] FIRE band at 300k, seeds 20260723 20260724, on GPU 1
[2026-08-21T02:53:53Z] [rep300k] backbone .../leg_300k/cf393_arm6_v2_combab_alignS_cf373k3_300k.pth
[2026-08-21T02:53:53Z] [rep300k] DRAW A4_k3_bb300k_student_s20260723 start
[2026-08-21T02:53:53Z] [rep300k] DRAW A4_k3_bb300k_student_s20260724 start
```

So the card reads a band at 200k, 300k, 450k and 665k, and not at two stops
only. The watchdog still owns 665k.

Two chains are up on card 1. The backbone md5 is `618e433e`, recorded in
`results/replicate_300k_backbone_md5.txt`. Card 0 runs the driver's own 300k
heads. Nothing above touched card 0.

## The 450k stage is armed

Row 3 of the table above: gate `ckpt`, state `pending`. The queue is pid
4048853, launched under `nohup setsid`, so it survives this session. It reads
its state off the disk at every start, so the restart did not repeat the 200k
re-draw and did not lose the 450k band.

## What changed in the queue

A `ckpt` stage fires on the CHECKPOINT and not on the score, so the extra
seeds train while the driver still scores the protocol seed at the same stop.
The queue runs ONE band at a time, because `head_vram_gate` serialises card 1
on one flock and a second band would only queue behind the first.

The gate now demands the `_optimizer.pth` sidecar. `save_snapshot` in
`train.py` writes the backbone FIRST and the sidecar second, so a sidecar on
disk proves the backbone write finished. Without that test the queue could
glob a checkpoint the driver is still writing.

Five decision paths ran against the live machine under `QUEUE_DRY`:

- gate `now`, no score, card free: fires.
- gate `ckpt`, backbone on disk: fires.
- gate `ckpt`, no backbone: waits.
- a chain up at any stop: no stage fires, on either stop.
- every seed scored: reports DONE and the queue exits.

## Runs completed

| run | state |
|---|---|
| driver leg to 300,000 steps, card 0 | done in 8.5 h |
| 200k band, seeds 20260723 and 20260724, both heads | scored |
| 200k re-draw, seed 20260722, both heads | scored |
| 300k band, seeds 20260723 and 20260724, both heads | running since 02:53 UTC |
| 450k band | armed on its checkpoint |
| 665k band | armed in the watchdog |

The bootstrap carries no new claim. The report reads the all-97 aggregate row
of `results/null_frozen_teacher.csv`.

Tests: 186 in `test_407_full_pass.py`, 2,083 in the suite.
