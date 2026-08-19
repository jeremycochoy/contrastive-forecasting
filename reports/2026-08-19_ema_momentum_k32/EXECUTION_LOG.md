# #404 — execution log

Operational events only. The science goes in the report
(REPORT_STANDARD: "Science, not journey").

Times are BST, 2026-08-19 unless another date is given.

## The machines

| role | machine | what it does |
|---|---|---|
| box | vast.ai `cf404-box-a`, instance 48109999, 4 x RTX 5090 | the four backbone arms, one per card |
| elisa | GPU 0 or GPU 1 | one student head per arm, then that head's 97 GIFT-Eval configs |

The box carries no gift-eval data and no `gift_eval` package, so it trains
backbones only (`launch_box.sh`, `CF404_HEADS=0`). elisa reads the box's tree
through the sync loop.

## Events

- 13:25 `vastrun-search` gave 3 offers under the CPU filter. #373 measured this
  cell's step rate against the CPU, not the card: 5.6 to 6.7 steps/s on a Zen 4
  desktop part against 1.1 steps/s on an EPYC 7452. The filter takes Zen 4 and
  Sapphire Rapids parts only.
- 13:27 `provision_box.sh` returned instance 48109999 at ssh1.vast.ai:29998,
  4 x RTX 5090 (32607 MiB each), AMD Ryzen Threadripper PRO 7965WX, 48 threads,
  driver 580.159.03, $1.7180/h. The label `cf404-box-a` is this session's own.
- 13:28 bootstrap started (`scripts/bootstrap_box.sh`).
- 13:31 bootstrap OK. `GPU-GATE: device_count 4`, torch 2.8.0+cu128 on the
  5090s, the trainer and the head trainer both parse, and the box's own
  `launch_box.sh` dry run prints four arms.
- 13:31 smoke started on GPU 0, 300 steps per arm.
- 13:35 sync loop up, pid 1268406, local root `/home/jupyter/cf404_sync/box_a`,
  15-minute ticks. The first tick landed a file, confirmed by `ls`.
- 13:39 smoke done, 0 failures. Every arm's momentum reached its trainer, and
  every arm wrote 33 `cos_err_dj` columns, which is k + 1 at k = 32.
  Step time 0.44 s, so 40,000 steps is about 4.9 h on one card.
- 13:40 the four arms started, one per card, `GPUS="0 1 2 3"`. The lanes stagger
  180 s so four cold HuggingFace readers do not open together.

## The head stage, and why it moves to the box

At 13:44 elisa's cards hold two runs of another session (a GRPO batch-size
study, one of them up 1 day 14 h) and one #373 head. Free VRAM is 2.4 GB on
GPU 0 and 1.1 GB on GPU 1. A head of this study needs 7 GB.

So the four heads train on the box, one per card, in parallel, and only the
97-config GIFT-Evals run on elisa. #373 made the same move in its round 2, for
the same reason, with the same head script and the same budget. The protocol
does not change: `head_eval_bb.sh`, 30,000 steps, head seed 20260722,
`--encoder-source student`.

The box carries no `gift_eval` package and no gift-eval data, so
`head_eval_bb.sh` trains and saves the head, then stops at the eval with
`ABORT: no eval script`. It writes no score file, so elisa's `collect.sh`
cannot read a half-made result. elisa then re-fires the same script per arm,
which skips the head that is already on disk and runs its eval.

## The four arms

- 13:49 all four lanes hold a live trainer, one per card. Each arm's guard read
  its own momentum and its own reduction back off the trainer's command line
  before the leg was left to climb:

      a08  0.8 - -           mean
      a09  0.9 - -           mean
      s08  0.8 1.0 200000    mean
      s09  0.9 1.0 200000    mean

## The heads, the box teardown, and the evals

- 18:24 to 18:30 the four backbones reached 40,000 steps, one after the other,
  and the sync loop landed each one on elisa. Every file is 5,195,883 B.
- 19:29 to 19:31 the four heads reached 30,000 steps on the box. `heads_box.sh`
  then ended at `ABORT: no eval script`, as its header says it must: the box
  carries no `gift_eval` package.
- 19:29 to 19:30 `drive.sh` pulled each head and its bb40k backbone with
  `safe_pull.sh`. Every head is 449,977 B.
- 19:30 the four 97-config GIFT-Evals started on elisa's CPUs.
- 19:34 `vastrun-destroy 48109999 cf404-box-a` returned. Uptime 6 h 9 m,
  $10.59. The box trained nothing after 19:31, and the evals need no GPU.
- 19:35 a `pkill` pattern for the sync loop also matched the eval shards,
  because an eval command line carries the sync root. The four evals died at
  three configs each. `evals_elisa.sh` re-fired at 19:36 and
  `eval_gift_eval_official.py --resume` took those three configs back, so the
  cost was one minute of CPU. The sync loop is down on purpose: its box is
  gone.
- 19:38 `finish.sh` armed, detached. It waits for the eval driver, then runs
  `report_assets.sh` and `make_plots.sh`. No stage now waits on a session.
- 21:28 the four evals returned rc=0, each on 97 configs. `finish.sh` wrote the
  scores, copied the artefacts and drew the figures, then wrote `FINISH DONE`
  at 21:30.
- 21:38 a read of the three PNGs found four things a reader could not read, and
  the figures were redrawn from the same tables:
  - `momentum.png` drew the inner repeat band and the 1.0862 line with no
    legend entry. Both now carry one.
  - `momentum.png` carried no warning that the two dotted lines come from
    200,000-step runs. It now prints that inside the frame, so the PNG travels
    with it.
  - `loss_curves.png` printed every y tick as `1.25 x 10^1`, and the fixed arm
    of each momentum hid under the ramped arm below step 500. The ticks now
    read as numbers, and the fixed arms take a dash and go on top.
  - `domain_radar.png` printed `0.772` against `0.8` at the centre, 4% of the
    radius apart. A crowded end tick now keeps its ring and drops its text.
  `results/scores.csv` and `results/splits.csv` are byte-identical after the
  redraw. No number moved.
