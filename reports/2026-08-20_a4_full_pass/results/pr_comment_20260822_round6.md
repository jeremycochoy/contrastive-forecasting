«Agent ExperimentRunner claude-opus-5 writing»

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

## The 450,000-step stop is complete

| head | seed | GM-Relative MASE |
|---|---|---|
| student | 20260722 | **1.0691** |
| teacher | 20260722 | **1.0986** |
| student | 20260723 | 1.0761 |
| student | 20260724 | 1.0778 |
| teacher | 20260723 | 1.0924 |
| teacher | 20260724 | in GIFT-Eval, 80 of 97 configs |

Every draw reads one backbone, md5 `f505688b3168e32b72eb45dad0a897e0`. Every
eval holds the 97 GIFT-Eval configs.

## Headline numbers

| stop | student | teacher |
|---|---|---|
| 40,000 | 1.0862 | 1.0855 |
| 100,000 | 1.0801 | 1.0874 |
| 200,000 | **1.0660** | 1.0828 |
| 300,000 | 1.0867 | 1.1030 |
| 450,000 | 1.0691 | 1.0986 |
| 665,000 | on card 1 now | |

## The band at each stop

`results/head_band.csv`. Three head seeds on one backbone.

| stop | head | draws | mean | range |
|---|---|---:|---:|---:|
| 200,000 | student | 3 | 1.0651 | 0.0018 |
| 200,000 | teacher | 3 | 1.0800 | 0.0064 |
| 300,000 | student | 3 | 1.0864 | 0.0042 |
| 300,000 | teacher | 3 | 1.1009 | 0.0038 |
| 450,000 | student | 3 | 1.0743 | 0.0087 |
| 450,000 | teacher | 2 | 1.0955 | 0.0062 |

Pooled head-seed standard deviation: 0.0032. Largest range: 0.0087. Both
stay under the 0.0141 gap that made 1.0660 the best, so a move larger than
the band is readable.

## The card's question, at 450,000 steps: still no

The curve turned back down between 300k and 450k. The student band mean
falls by 0.0121 and the teacher band mean falls by 0.0054. The student move
is 3.8 pooled standard deviations, so it is readable.

The curve did not reach the 200k level. The student band mean at 450k sits
0.0092 above the 200k band mean, which is 2.9 pooled standard deviations.
The protocol-seed draw, 1.0691, sits 0.0031 above 1.0660. That single number
falls inside the 450k student range of 0.0087, so the band means carry the
comparison and the single draws do not.

A4 at 450,000 steps is worse than A4 at 200,000 steps, on both heads. The
665,000-step stop is the last point, and it decides the card.

## The last leg moved to card 1

Card 0 filled with another project, 17,270 MiB, plus three Jupyter kernels.
The driver blocked in its VRAM gate at 00:52 UTC with the 665k leg still to
run:

```
[08-22 01:52:37] [cf407] waiting for VRAM on GPU 0: 3883 MiB free, need 6500
```

Card 1 was free. The 450k band's last draw had already left the GPU for a
GIFT-Eval, which runs `--device cpu`.

The blocked driver held no GPU work. Its only child was `sleep 60`, so the
kill lost nothing. The relaunch, on card 1:

```
WT=/tmp/contrastive-forecasting-407 RUNS=/home/jupyter/cf373_r3/sync \
  BB_GPU=1 HEAD_GPU=1 nohup setsid bash scripts/run_pass.sh 665000 &
```

The continuity gates passed and the leg resumed:

```
[08-22 01:57:55] [cf407] start stops=665000 ... bb_gpu=1 head_gpu=1
[08-22 01:57:55] [arm6_v2_combab_alignS] RESUME from ..._450k.pth (step 450k)
[08-22 01:57:55] [arm6_v2_combab_alignS] START target=665000 gpu=1
[ 450800] loss=15.2514  ema_loss=15.0976  4.1 sps  ETA 14.6h
```

The watchdog moved with it. It carried `BB_GPU=0`, and two things follow
from that variable: a re-fire would put the driver back on the blocked card,
and `band_at_last_stop` takes `BAND_GPU` as `1 - BB_GPU`. So it restarted
with `BB_GPU=1 HEAD_GPU=1 BAND_GPU=1`. It was asleep between ticks, so the
restart interrupted no work. Its first tick read
`driver=yes step=450200 quiet=0 open='665000'`.

The 665k band stays armed. `replicate_665k.log` does not exist, so
`band_at_last_stop` fires when the 665k checkpoint lands. Card 1 holds 24 GB,
the leg holds 5.4 GB, and `head_vram_gate` holds one flock per card, so the
driver's own heads and the band draws take the card in turn.

Nothing above touched card 0.

## Runs completed

| run | state |
|---|---|
| driver leg to 300,000 steps | done in 8.5 h |
| driver leg to 450,000 steps | done in 11.7 h |
| driver leg to 665,000 steps | on card 1, ETA 14.6 h |
| 200k re-draw, seed 20260722, both heads | scored, delta +0.0000 against #373 |
| 200k band, seeds 20260723 and 20260724, both heads | scored |
| 300k stop, seed 20260722, both heads | scored |
| 300k band, seeds 20260723 and 20260724, both heads | scored |
| 450k stop, seed 20260722, both heads | scored |
| 450k band, student seeds 20260723 and 20260724 | scored |
| 450k band, teacher seed 20260723 | scored |
| 450k band, teacher seed 20260724 | in GIFT-Eval on CPU |
| 665k band | armed in `watchdog.sh`, on card 1 |

Tests: 199 in `test_407_full_pass.py`.
