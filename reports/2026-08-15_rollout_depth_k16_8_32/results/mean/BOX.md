# The box, and what it costs

The primary compute of this study. It trains both backbone arms, one arm per
card. elisa is the extra compute: it trains every head and runs every
97-config GIFT-Eval, because the eval data and the `gift_eval` package are
there.

## The machine

| field | value |
|---|---|
| contract id | 47976049 |
| label | `cf401-mean-box` |
| GPU | 2 x NVIDIA GeForce RTX 5090, 32,607 MiB each |
| CPU | AMD EPYC 9J14 96-Core, 384 threads |
| host | ssh2.vast.ai:16048, CZ, reliability 0.998, datacenter |
| rate | $2.2340 / h |
| driver / CUDA | 590.48.01 / 13.1, torch 2.8.0+cu128 |
| provisioned | 2026-08-17 22:52 |

`vastrun-provision` returned this contract id. One earlier instance,
47975987, never reached `running` and the provisioner destroyed it in the
same minute.

## What one step costs on this box

`scripts/smoke_depth.sh 400`, GPU 0, the MEAN objective, in
`smoke_depth_box_5090.csv`. The median over the `timing:` windows after the
warm-up window.

| k | step time | peak memory | `cos_err_dj` columns |
|---:|---:|---:|---:|
| 0 | 155.8 ms | 5,530 MiB | 0 |
| 8 | 240.5 ms | 5,572 MiB | 9 |
| 32 | 482.6 ms | 5,690 MiB | 33 |

The column count is `k + 1` at both depths, which proves the depth flag
reached the trainer. `--train-rollout-reduce mean` is on every command line
the box runs, and `run_arm_k.sh` reads it back out of the trainer's own line.

The same objective on elisa's RTX 4090 costs 299.1 ms at k = 8 and 598.3 ms
at k = 32, so this box is about 1.24 times faster per step.

## What the two arms cost

Each arm has its own card, so the two run at once and the wall clock is the
k = 32 arm.

| arm | steps left after the move | time on this box |
|---|---:|---:|
| k = 8 | 180,000 | 12.0 h |
| k = 32 | 180,000 | 24.1 h |

## The budget

The card has about $80. At $2.2340/h the k = 32 arm sets the bill. Destroy
the box when its last leg ends, not before.
