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

## Round 2 — the three arms the review of PR #405 asks for

The review asks for a fixed arm above 0.9, a fixed arm at 0.85, and a repeat
of the best arm at a second backbone seed. `scripts/arms.tsv` takes three rows,
`a095`, `a085` and `s08b`, and a fifth column: the backbone seed. Nothing else
in the study changes.

| role | machine | what it does |
|---|---|---|
| box_b | vast.ai `cf404-box-b`, 48149747, 1 x RTX 5090 | a085 |
| box_c | vast.ai `cf404-box-c`, 48149866, 1 x RTX 5090 | a095 |
| box_d | vast.ai `cf404-box-d`, 48150020, 1 x RTX 5090 | s08b |
| elisa | CPU | the three 97-config GIFT-Evals |

Round 1 rented one box with four cards at $1.7180/h. The offers that carry the
CPU this cell needs are single-card boxes at $0.3611/h, so three of them cost
$1.0833/h for the same three lanes. `scripts/round2.sh` holds a watchdog at
10.5 hours of box life, which is $11.37 against the $12 this round may spend.

### Events

- 22:03 `round2.sh` started, detached. Credit $18.59.
- 22:05 to 22:09 the three boxes came up: 48149747, 48149866, 48150020, all
  RTX 5090 at $0.3611/h.
- 22:08 box_b passed the bootstrap. Its sync loop landed a first tick at
  22:08:37, confirmed by `ls`.

## Round 3 — the same three arms, on one box, after round 2 started no trainer

### Why round 2 gave nothing

`round2_box.sh` asked each box whether a trainer was already up, over SSH:

    ssh box "pgrep -f 'run_leg_k.sh arm6_v2_combab_alignT' >/dev/null"

sshd runs that string through a shell, so the box then holds a process whose
command line IS the pattern. `pgrep -f` reads full command lines and drops only
its own pid, so it matched that shell. The check read true on a bare box, the
driver started no trainer, and the three boxes billed at 0% GPU. Line 181
carried the same defect for the head trainer. Round 1 used `launch_box.sh` and
never took this path.

`cf404_pgrep_pattern` in `scripts/study.sh` now puts a bracket class on the
first character. `scripts/test_trainer_check.sh` proves both directions through
`bash -c`, which is what sshd does with a remote command: 10 assertions, 0
failures. Three more remote calls carried the same shape and take the fix too
(`drive.sh`, `heads_box_await.sh`, `finish.sh`).

The test carries a per-run nonce. A bare `train_forecasting_head` on elisa
matches two runs of the #401 session AND the agent shell, whose prompt merely
names the file, so a test on the bare name would report a neighbour as a
regression.

### The box, and why it carries one card

The round-2 boxes were non-datacenter. Round 3 refuses that: `vastrun-search`
returns a non-datacenter host only when it is given `--prosumer`, which
`round3.sh` never passes, and `--min-reliability 0.99` carries the rest. The
pool is real: 29 offers without `--prosumer` against 60 with it.

The card asks for one machine with enough cards for the three arms. The
datacenter pool held NO 3-card or 4-card offer under $20/h at 22:47, and its
one 2-card offer (RTX 5090, EPYC 9654) costs $1.8687/h and carries a server
CPU. #373 measured this cell at 5.6 to 6.7 steps/s on a Zen 4 desktop part
against 1.1 steps/s on an EPYC 7452, so the CPU and not the card sets the step
rate.

So the three arms share ONE RTX 5090 on a 1-card datacenter box with a Ryzen 7
7800X3D at $0.3356/h. `gpu_gate` returns at once on a `Default`-mode card, so
the three legs do not serialise. `round3.sh` refuses a card in
`Exclusive_Process` mode, where the second CUDA context would die.

The three heads DO serialise: `head_vram_gate` holds an exclusive lock per card
for the whole of one head training. That is its purpose, and one card is enough
VRAM for one head at a time.
