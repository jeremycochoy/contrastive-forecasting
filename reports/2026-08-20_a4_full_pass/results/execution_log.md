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

### 19:54 UTC — the teacher is frozen (gap 4)

`teacher_move.py`, on checkpoints already on disk. 100k against 200k moves
0 of 52 teacher tensors, bit for bit, while the student moves 106 of 110 at
relative L2 0.599. The 40k against 100k control moves all 52. So the answer
did not wait for the 300k stop.

`teacher_check.sh` repeats the test on every later pair, from the watchdog.

### 19:0x UTC — the shard order (gap 3)

`shard_order.py` read the `meta` and `source_id` columns of 12 shards from
`small_v1`, spanning shard 0 to shard 4273 and including the 1279/1280
boundary the 200,000-step mark falls on. It read no `series` column, so the
check moved a few MB and did not compete with the training stream.

### 19:2x UTC — the figure, the interval and the metrics (gaps 2, 9)

`stop_bootstrap.sh`, `metrics_table.py` and the ribbon in
`plot_full_pass.py`. All three read CSVs that already exist, so none costs
GPU time.
