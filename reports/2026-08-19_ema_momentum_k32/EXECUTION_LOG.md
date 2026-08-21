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

## Round 3b — `a095` first, then `s08b`, on the box round 3 rented

### Why the plan changed

The user read the momentum figure. With alpha held constant, 0.8 scores 1.2309
and 0.9 scores 1.1819. That segment falls steeply and it does not turn, so the
next value belongs ABOVE 0.9. The user asks for `a095` first and picks the
third value from its result.

`a085` is dropped. Alpha 0.85 interpolates inside a segment that already falls
in one direction. `scripts/arms.tsv` loses its row, so no table, figure or
guard of this study names an arm the study does not run.

`s08b` still runs, second. It repeats `s08` at backbone seed 20260521, and that
pair is the only thing that measures this cell's own repeat spread. Without it
a reader cannot say whether the 0.95 number is a result.

### The card index

Round 3's plan print read `arm a095 gpu=1` while the box carries ONE card, at
index 0. The print came from a dry run that passed no GPUS, so `launch_box.sh`
took its own default, `0 1`. The launch itself passed `GPUS='0 0 0'`, so the
lane would have landed on card 0 — but the plan an operator reads named a card
that is not there, and `heads_box.sh` carried the same default at `0 1 2 3`.

Two changes in `scripts/study.sh` close both halves:

- `cf404_require_gpus` reads the card count off the driver and refuses every
  index at or above it. Both launchers call it BEFORE the plan print.
- `cf404_default_gpus` makes the launcher default the card list the box
  carries.

`scripts/test_gpu_guard.sh` proves both: 20 assertions, 0 failures, on elisa
and on the box. On the box, `GPUS=1` returns rc=2 and prints what it refused,
`GPUS=0` returns rc=0, and a plan with no GPUS names `gpu=0` for both arms.

### One arm at a time

`scripts/round3b.sh` runs each arm end to end — backbone, head, pull, eval,
score — before the next one starts. Two arms on one card would both land at the
end. The user needs the 0.95 number first.

The driver reuses instance 48152799 and provisions nothing. A box that does not
answer is an ABORT.

### Events

- 22:47 to 22:52 `round3.sh` provisioned and bootstrapped instance 48152799,
  label `cf404-box-r3`, 1 x RTX 5090, datacenter, AMD Ryzen 7 7800X3D (8c),
  driver 580.95.05, $0.3611/h. The user stopped that driver by pid before it
  started an arm.
- 23:02 `round3b.sh` started, detached. Credit $17.90. The watchdog holds two
  ceilings: 22 hours of box life, and $11 of spend.
- 23:03 the box printed its own plan at `GPUS='0'`: both arms on `gpu=0`.
- 23:03 `a095` started on card 0.
- 23:06 the launch is VERIFIED, off the box: 5,684 MiB of GPU memory in use,
  one compute app, 33 depth columns in the losses CSV, which is k + 1 at
  k = 32, 501 CSV rows, and the guard line
  `arm a095 ema='0.95 - -' reduce=mean seed=20260520 OK`.
  Step rate 2.8 sps, ETA 4.0 h. Round 1 measured 2.4 to 2.7 sps for the same
  cell with four arms on one 4-card box.
- The sync loop of round 3 stays up for the whole round, 15-minute ticks, local
  root `/home/jupyter/cf404_sync/box_r3`.

## Round 3c — the two machines run together

### Why the order changed

`round3b.sh` runs one arm end to end before the next one starts. The eval of an
arm is a 97-config GIFT-Eval on elisa's CPUs, and round 1 measured 1.9 hours for
it. The box holds ONE card. Under round 3b's order that card sits at 0% for
every hour of the `a095` eval, at $0.3611/h, and `s08b` starts 2 hours late.

The two stages use two machines, so they overlap. `scripts/round3c.sh` runs the
same two arms in this order:

1. the `a095` head on the box, 30,000 steps, head seed 20260722.
2. the `s08b` backbone on the same card, the minute the head process leaves it.
3. the `a095` 97-config GIFT-Eval on elisa, at the same time as (2).

The card gets no idle hour, and the 0.95 number arrives at the same time as
before.

### What round 3c inherits

`round3b.sh` trained the `a095` backbone to 40,000 steps and started its head.
`round3c.sh` picks that head up where it stands. Every stage is idempotent, so
nothing is retrained.

### The comment carries the number

The user waits on the `a095` score to pick the third momentum. `round3c.sh`
posts that score to PR #405 itself, from `results/scores.csv`, the minute the
score file appears. A session that ends does not hold the number back.

### Events

- 02:55 the `a095` backbone reached 40,000 steps. `round3b.sh` started the head
  on card 0, seed 20260722, at 30,000 steps.
- 03:00 the head is VERIFIED, off the box: 5,616 MiB of GPU memory in use, one
  compute app, 81% GPU, and the head losses CSV at step 1,600 of 30,000.
  Measured rate 7.6 steps/s, which is 66 minutes for the head.
- 03:02 the round 3b driver stopped, by pid (1959256) with its watchdog
  (1959292). No pattern, and the box sync loop (1948422) was not touched. The
  head runs under `nohup setsid` on the box, so it did not notice.
- 03:03 `round3c.sh` started, detached, under `nohup setsid`. Credit $16.42, box
  spend $1.51. Its watchdog holds two ceilings: 16 hours of box life, and $9 of
  spend. The card check read one card at index 0 again.
- 03:38 the `a095` artefacts landed on elisa. Its 97-config GIFT-Eval started on
  the elisa CPUs, detached, with 6 shard processes. The box took the same card
  back for the `s08b` backbone.
- 04:59 `a095` scored 1.1907. The driver posted that number to PR #405 by
  itself, built from `results/scores.csv`.
- 07:26 the `s08b` backbone reached 40,000 steps. It took 3.8 hours at 2.9
  steps/s. The driver started the head on card 0, seed 20260722, at 30,000
  steps.
- 07:31 the head is VERIFIED, off the box: 5,616 MiB of GPU memory in use on
  card 0, and the head losses CSV at steps 1 to 5, loss 0.4729 down to 0.4281.
  Measured rate 12.5 steps/s, which is 40 minutes for the head.
- 07:35 `finish_round3c.sh` started, detached, under `nohup setsid`.

### What the finisher holds, and why it is a second process

The driver ends at its own stage 7. It never touches git, and an in-session
waiter dies with its session. So the last steps live in their own process:

1. Wait for the driver BY PID. A pattern here would also match the waiter
   itself, and on 2026-08-19 a pattern for the sync loop matched four running
   eval shards.
2. Put the six arms' artefacts into the study directory, under ONE curves
   directory. The driver labels its own run `box_r3`, and that splits six arms
   across two directories that hold one tree.
3. Rebuild `scores.csv`, `splits.csv`, the three figures and the table.
4. Write `results/verify_round3c.txt`: every artefact of every arm, by name and
   by size, with a `missing:` count at the end.
5. Destroy the box, post the comment and commit, each only if the driver did
   not. Every one of the three is idempotent.

### The comment answers the card's question

The comment carried the repeat spread but never made the call the card asks
for: does that spread separate the constant 0.90 arm from the constant 0.95
arm? `repeat_spread.separation` reads both arms out of `scores.csv` and
compares their gap against the spread. The test is strict, so a gap equal to
the spread does not separate.

## Round 4 — four backbone seeds of one arm

### Why the round exists

`s08b` was meant to measure this cell's run-to-run spread. It measured a
COLLAPSE instead. Its contrastive AUC went 0.91 at 10,000 steps to 0.84, 0.67
and 0.57 at 40,000. Its top1 went 0.273 to 0.009 and its gap_ratio went 0.772
to 1.007. Its `ema_tau` trajectory matches `s08` exactly, so the schedule
reached the trainer, and its eval covered all 97 configs. The arm trained the
right cell and the backbone died inside it.

Every other arm is stable, AUC 0.93 to 0.98 at 40,000 steps, and all five carry
backbone seed 20260520. One seed collapsed. Two readings fit that:

  - 20260521 was unlucky, and a collapse here is rare, or
  - 20260520 was lucky, and this cell is unstable.

Under the second reading every ranking of this card rests on one seed. Two more
seeds of the SAME arm tell the two apart: `s08c` at 20260522 and `s08d` at
20260523. Every other flag is `s08`'s.

### The machine, and why it carries one card

The card asks for ONE box with TWO cards, a datacenter host at reliability 0.99
or better, and a DESKTOP-class CPU. The step rate of this cell is set by the
CPU and not by the card: #373 measured 5.6 to 6.7 steps/s on a Zen 4 desktop
part against 1.1 steps/s on an EPYC 7452.

On 2026-08-20 the whole datacenter multi-GPU pool carries SERVER CPUs.
`results/round4_pool_evidence.txt` holds the search, with no price ceiling:

| GPUs | offers | CPU class | cheapest |
|---|---|---|---|
| 2 | 23 rows, every one a server part | EPYC 7763, 7V13, 7452, 9274F, Xeon Gold 6133, Xeon 6767P | $1.0437/h |
| 1 | 4 of 5 rows are desktop parts | Ryzen 7 7800X3D | $0.3356/h |

At six times the step time the cheapest two-card box needs about 22 hours per
arm and about $23, against a limit of $7. A two-card server box is SLOWER and
DEARER than one desktop box, and it breaks the budget.

So `round4.sh` asks the pool for the card's own shape FIRST and falls back to
one card of the same class. The two arms then share that card. They fit: one
leg holds 5.7 GB of a 32 GB RTX 5090 and leaves the card at 27 to 34 %
utilization, so the second leg takes idle silicon. `gpu_gate` returns at once
on a `Default`-mode card, and the driver refuses an `Exclusive_Process` card,
where the second lane would die inside `.to(device)`.

### What every reader had to learn

Before this round the tables read `s08` against `s08b`, called their distance
of 0.3677 "the repeat spread this card measures", and then reported that every
arm sits within one repeat of the winner. That is not a spread. It is the
distance between a healthy run and a dead one, and under it the card ranked
nothing.

`scripts/seed_report.py` now holds the study's ONE definition of a collapse:
the contrastive AUC at the stop, against a line at 0.80. The five stable arms
hold 0.93 to 0.98 and the collapsed one holds 0.57, so any line inside that
band classifies the same arms.

Five readers share that definition and cannot disagree:

  - `plot_backbone_health.py` paints red by the data, not by an arm name.
  - `plot_momentum.py` keeps a collapsed arm out of every mean and every bar,
    and gives it a red X off the line.
  - `make_table.py` gains a seed column and an AUC column.
  - `pr_comment.py --sync-root` answers the card's four questions.
  - `make_plots.sh` draws both new outputs, so "redraw every figure" covers
    them.

### Events

- 09:39 `round4.sh` started, detached, under `nohup setsid`. Credit $13.95. The
  watchdog holds two ceilings: 14 hours of box life, and $6 of spend.
- 09:39 the two-card search returned 0 offers under the ceiling. The driver
  fell back to one card and said so.
- 09:42 instance 48192413, label `cf404-box-r4`, at ssh9.vast.ai:32412. One RTX
  5090 in `Default` compute mode, AMD Ryzen 7 7800X3D, 8 threads, 30 GB RAM,
  datacenter, reliability 0.99 or better.
- 09:42 the card check read one card at index 0, so both arms take lanes
  `0 0`. The box printed its own plan and named `s08c` and `s08d` on `gpu=0`.
- 09:45 bootstrap OK. The box carries this round's arms table.
- 09:45 the sync loop is up, pid 2588325, local root
  `/home/jupyter/cf404_sync/box_r4`, 15-minute ticks.
- 09:42 the round 4 heartbeat loop replaced round 3c's. It follows the round
  through `ROUND` and reads EVERY live losses CSV, because two lanes run at a
  time and a probe that reads one file calls a dead lane live.
- 09:55 the sync loop's first tick landed nothing but its own log, because the
  box had no run directory yet, so `launch_sync.sh` waited out its whole
  10-minute window. The card was idle for it, about $0.07. A later round should
  start the lanes first and the loop straight after.
- 09:56 both backbones started, lanes `0 0`.
- 09:59 the guard lines are in, off each trainer's OWN command line:

      arm s08c ema='0.8 1.0 200000' reduce=mean seed=20260522 OK
      arm s08d ema='0.8 1.0 200000' reduce=mean seed=20260523 OK

  The seed is the value this round turns on, so it is checked like the
  momentum. Two arms that differ in the seed alone write the same file names.
- 10:00 the launch is VERIFIED, off the box: 11,363 MiB of 32,607 MiB in use,
  two compute apps at 5,674 MiB each, 89 % GPU, 33 depth columns in both losses
  CSVs, which is k + 1 at k = 32, and 701 and 201 CSV rows.
- 10:00 step rates: `s08c` 2.8 sps, ETA 3.9 h. `s08d` 2.2 sps, ETA 4.9 h. One
  lane of round 3 held 2.8 sps on the same class of box at 27 to 34 % GPU. Two
  lanes on one card cost the second lane 0.6 sps and take the card to 89 %, so
  both seeds land in about the time one seed took.
- 10:01 `finish_round4.sh` started, detached, under `nohup setsid`, pid 2601993.
  It waits for the driver BY PID.

## Round 5 — the ramp length, at the two momenta the ladder brackets

### Why the round exists

The card wants a LOWER GM-Relative MASE. Round 4 measured the collapse rate,
and that work does not move the score. The card still has no arm that beats its
own parent: the k = 0 parent scores 1.1600 at 40,000 steps, the best arm here
scores 1.1782, and k = 3 scores 1.0862 at the same stop.

The RAMP LENGTH is the one axis this card never moved. Every ramp arm of rounds
1 to 4 runs 200,000 steps. The EMA schedule ladder,
`reports/2026-08-04_ema_sched_ladder/`, trained ten runs on that axis. A
momentum that reaches 1.0 at step 100,000 scored 0.0259 BELOW the fixed 0.9
reference at the 40,000-step stop, where the momentum held 0.94. The same run
scored 0.0251 ABOVE at the 100,000-step stop, where the momentum held 1.0. The
ladder's latent table gives the reason: past step 100,000 the teacher latent
moves 0.019 or less per 20,000 steps, so a momentum at 1.0 freezes the teacher.
A 100,000-step ramp read at 40,000 steps is the good half of that curve.

Two arms, both at a 100,000-step ramp, both at backbone seed 20260520:

    r100_09  --ema-tau 0.9 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000
    r100_08  --ema-tau 0.8 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000

`r100_09` lands on 0.940 at the stop, the value the ladder measured as its
best. `r100_08` lands on 0.880 and brackets it from below. Neither reaches 1.0
at the stop, so neither freezes its teacher there.

### What the round dropped, and in which order

`s08c` and `s08d` reached 40,000 backbone steps at 14:18 and their checkpoints
are on disk. Their heads and their evals are DROPPED. Their scores measure the
collapse rate and this round spends the card on the score.

THE ORDER OF THE THREE KILLS MATTERS. `finish_round4.sh` destroys the box the
moment the round 4 driver leaves the process table. So it went FIRST, by pid,
and the driver went after it. A driver killed first would have taken the box
with it, and this round would have had to rent another.

  - 15:20 `finish_round4.sh` (2601993) stopped, then the round 4 watchdog
    (2581862), then the driver (2581821), then the round 4 heartbeat loop
    (2583886). Every one by pid, never by pattern.
  - 15:21 the box-side head of `s08d` stopped, with its two parents, by pid.
    The card then read 2 MiB in use and 0 compute apps, so it was free.
  - 15:21 `vastrun-status` still showed 48192413 running. The box survived
    every kill, which is what the order was for.

### The machine

ONE box, and the SAME box: 48192413, `cf404-box-r4`, one RTX 5090 in `Default`
compute mode, AMD Ryzen 7 7800X3D. `round5.sh` NEVER provisions. When the box
does not answer it stops, because a second box is not allowed.

Spend at the handover was $2.40 of a $6 limit. The watchdog holds MAX_SPEND at
$5.50 of TOTAL box spend, which is what `vastrun-status` reports.

THE TEARDOWN COMES BEFORE THE EVALS, and this is the one place round 5 differs
from round 4. The 97-config GIFT-Eval runs on elisa CPUs, not on the box, so
the box does no work for the 2.5 hours it takes. Round 4 held the box through
it. That costs about $1.07 and does not fit under the limit. So stage 8 pulls
every artefact, LOADS each one with `torch.load` to prove it is readable, and
only then destroys the box. A size floor does not prove a checkpoint reads: a
half-written file is large. When a file does not load, the box STAYS UP so it
can be pulled again.

### Events

- 15:22 `arms.tsv` gained the two rows. Nothing else defines an arm: every
  script, guard and figure reads that file.
- 15:23 `round5.sh` started, detached, under `nohup setsid`, pid 2906374.
- 15:24 the box carried round 4's scripts, so its arms table had no row for
  either arm. Stage 3 ships `scripts/` again and then reads the table back off
  the box: `a08 a09 s08 s09 a095 s08b s08c s08d r100_09 r100_08`.
- 15:24 both backbones started, lanes `0 0`.
- 15:27 the guard lines are in, off each trainer's OWN command line:

      arm r100_09 ema='0.9 1.0 100000' reduce=mean seed=20260520 OK
      arm r100_08 ema='0.8 1.0 100000' reduce=mean seed=20260520 OK

- 15:29 the launch is VERIFIED, off the box: 11,363 MiB of 32,607 MiB in use,
  2 compute apps, 33 depth columns in both losses CSVs, which is k + 1 at
  k = 32, and 701 and 201 CSV rows.
- 15:29 step rates: `r100_09` 2.7 sps, ETA 4.0 h. `r100_08` 2.4 sps, ETA 4.7 h.
- 15:25 the round 5 heartbeat loop replaced round 4's, pid 2909402.

### What the readers had to learn

**A ramp arm does not hold the momentum it names.** This is new to round 5. Up
to round 4 every ramp ran 200,000 steps, so the momentum at step 0 named the
arm without ambiguity. It does not now: `s08` and `r100_08` both start at 0.8
and hold 0.840 and 0.880 at the stop, and `s09` and `r100_09` both start at 0.9
and hold 0.920 and 0.940.

`plot_momentum.py` grouped its points by the momentum at step 0 within a
schedule. With two ramp lengths it would have averaged `s08` and `r100_08` into
ONE marker and drawn the distance between them as a vertical bar. A vertical
bar means a repeat spread everywhere else on that figure, so the figure would
have reported two different momenta as one arm trained twice. The series key is
now the schedule TOGETHER WITH the ramp length, and each ramp length takes its
own colour and marker. `seed_report.family` already keyed on all three.

Four readers gained the reached value:

  - `cf404_momentum_at` in `study.sh` prints it without a Python interpreter.
    `scripts/test_momentum_at.sh` holds it against `src.models.ema_tau_at_step`
    over every arm and ten steps: 100 pairs agree.
  - `plots/momentum_at_stop.png` is a new figure. It puts the reached momentum
    on the x axis, so every arm takes its own tick and the card's question
    reads straight off it.
  - `pr_comment.py` gained a `holds at 40k` column.
  - `pr_comment.py` also gained a section for backbones that trained and carry
    no score, so `s08c` and `s08d` are not hidden.

**The heartbeat could not read an arm name with an underscore.** `r100_09` is
the first. The backbone pattern printed the whole file name and the head
pattern printed `head_r100`. Both patterns now anchor on a fixed neighbour,
`_cf373k<N>_` on the left and `_bb<N>k_` on the right.

### A result this round did not pay for

`s08c` holds contrastive AUC 0.9776 at 40,000 steps and `s08d` holds 0.9746.
Both are healthy, against `s08b` at 0.5745. So three of four seeds of the `s08`
arm lived and one died. The collapse was one unlucky seed, and the ranking of
this card does not rest on one lucky one. Their heads never ran, so neither arm
has a GM-Relative MASE.

## Round 6 — the L_align weight, as a third lane on the round 4 box

### Why the round exists

The user asked for it. Take the best config of the card and move ONE flag.

For this loss shape the rollout depth touches the align term alone. A depth
copy of `cosine_similarity_batch_rep_only` "has nothing to substitute and adds
exactly zero", because that shape is h-anchored, and the align add-on "IS
duplicated" (`src/loss.py`). The reduction is a mean, so 33 copies of L_align
average back to about one copy's magnitude. The loss then holds ONE copy of the
h-anchored repel term against the MEAN of 33 copies of the f-anchored pull
term. `--align-loss-weight` is the only flag that sets that balance, and no arm
of rounds 1 to 5 moved it.

    w3_s08   s08 (1.1782, the card's best) at --align-loss-weight 3.0

Everything else is round 1's, down to both seeds.

### What the round dropped

`s08c` and `s08d` keep their heads and their evals DROPPED. Their backbones
stay on disk. The user asked for that: their scores measure the collapse rate,
and this round spends the card on the score.

### The order of the two kills

`finish_round5.sh` destroys the box the moment the round 5 driver leaves the
process table. So it went FIRST, by pid, and the driver went after it. Then the
round 5 heartbeat loop.

  - 15:47 `finish_round5.sh` (2947274) stopped, then the round 5 watchdog
    subshell (2906449), then the driver (2906374), then the heartbeat loop
    (2909402). Every one by pid, never by pattern.
  - 15:47 `vastrun-status` still showed 48192413 running at $2.61. The box
    survived every kill, which is what the order was for.

### The two things round 5 did per round, and round 6 does per arm

**The launch.** Round 5 asked "does a trainer run on the box?" and skipped the
launch when one did. Three lanes make that answer useless: two trainers already
ran, so the third would never have started. `round6.sh` asks the question of
EACH ARM, off that arm's own run name, and starts only the arms that neither
finished nor run. It read `r100_09: a trainer already runs` and
`r100_08: a trainer already runs`, then started `w3_s08` alone.

**The heads.** Round 5 waited for every backbone and then started every head. A
head reports 0 % GPU utilization on this card, so it costs the trainers beside
it almost nothing. Round 6 starts each arm's head THE MOMENT that arm's
backbone lands, so the two 100,000-step ramps are scored while `w3_s08` still
trains. It takes about an hour of box time off the tail, which is what makes
the round fit under the limit.

### Events

- 15:48 `arms.tsv` gained a SIXTH COLUMN, the align weight, and one row. A `-`
  or an absent column takes the cell's own 1.0, so every row above `w3_s08`
  keeps the command line it ran.
- 15:49 `run_arm.sh` appends `--align-loss-weight` to `GAP_ARGS`, which is the
  LAST thing on the trainer command line. The cell states the flag earlier, so
  a repeat is what moves it: argparse keeps the last value.
- 15:49 the guard gained the weight, and it needed a NEW READER.
  `cf404_arg_of_cmdline` stops at the first hit, so it reports the cell's 1.0
  on every arm. `cf404_align_of_cmdline` reads the LAST hit.
  `scripts/test_align_guard.sh` holds all of it: 24 checks pass.
- 15:50 `round6.sh` started, detached, under `nohup setsid`, pid 2959282.
- 15:51 the box's arms table now holds `w3_s08`, and the box's OWN copy of
  `run_arm.sh` builds it at `align_w=3.0`. Stage 3 reads that back over SSH
  before it starts the lane, so a table that shipped without column 6 stops
  the round instead of training a duplicate of `s08` for five hours.
- 15:51 the third lane started. Lanes `0 0 0`.
- 15:51 the guard line is in, off the trainer's OWN command line:

      arm w3_s08 ema='0.8 1.0 200000' reduce=mean seed=20260520 align_w=3.0 OK

- 15:55 the launch is VERIFIED, off the box: 17,042 MiB of 32,607 MiB in use,
  THREE compute apps at 5,674 MiB each, 83 % GPU, 33 depth columns in all three
  losses CSVs, which is k + 1 at k = 32. The verify file also carries the
  weight read back off each lane's command line: 1.0, 1.0, 3.0.
- 15:55 step rates: `r100_09` 2.6 sps ETA 3.8 h. `r100_08` 2.6 sps ETA 3.9 h.
  `w3_s08` 2.2 sps ETA 5.1 h. The third lane cost the other two 0.1 sps each
  and took the card from 88 % to 83 %, so the card was NOT saturated at two
  lanes.
- 15:55 the round 6 heartbeat loop replaced round 5's. It reads the newest SIX
  losses CSVs, not four: three lanes plus three heads write at once.
- 15:57 `finish_round6.sh` started, detached, under `nohup setsid`, pid
  2967588. It waits for the driver BY PID.

### A number the launch already gives

`w3_s08` writes loss 17.99 at step 1 where the two round 5 lanes write 14.28,
off the same data and the same seed. The align term is the only term that
moved, and 3.0 times one copy of it against 1.0 times the same copy is the
difference. So the weight reached the objective and not only the command line.

### The budget

The limit is $6 of TOTAL box spend. The box had spent $2.61 at the handover, at
$0.4278/h. `MAX_SPEND` is $5.60 and the rest is margin for the teardown itself.

THE WATCHDOG PULLS BEFORE IT DESTROYS, which round 5's did not. The cap is a
budget event and not a data event: whatever the box holds at that moment is
still worth a head on elisa, and `head_eval.sh` trains a head that is not on
disk before it evals. So a cap that fires mid-round still ends with a score.

## Round 6, part 2 — the head the round 6 driver did not wait for

### What the round 6 driver did

`finish_round6.sh` pulled the three backbones and the two heads that existed,
then logged this pair of lines:

    MISSING qhead_w3_s08_bb40k_h30k_student_s20260722_final.pth
    w3_s08: WARNING — an artefact did not land

Then it destroyed instance 48192413. The `w3_s08` head was at about 26,000
steps of 30,000. That head is gone, and no eval ran for any of the three new
arms.

A missing final checkpoint is a STOP, not a warning.

### What survived, by name and by size

    r100_09  backbone _40k.pth 5196347 B   head _final.pth 450113 B
    r100_08  backbone _40k.pth 5196347 B   head _final.pth 450113 B
    w3_s08   backbone _40k.pth 5196231 B   NO head

So one head had to train again, and nothing else.

### The order, and why the two CPU evals went first

`r100_09` and `r100_08` hold a head each. Their evals read `gift-eval-data`
and the `gift_eval` package, both on elisa, and they run on the CPU. They need
no GPU and no box, so they started FIRST, at 23:05, four shards each. They cost
nothing to start early and they finish while the third arm trains.

### Why the head rented a box

Both elisa cards were full: 22,297 MiB and 22,746 MiB of 24,564 MiB. A head
needs about 5.6 GB and the VRAM gate asks for 7,000 MiB free. Card 407 also
holds the head lock on this machine and has two more heads to run, so a wait
here has no known end.

One RTX 5090, $0.3611/h, was the cheaper answer. The head took 31 minutes and
the box cost $0.30, against a $2 limit.

### The rule the new driver carries

`scripts/recover_w3_head.sh` asks ONE question before it destroys anything: is
the head on elisa's disk, by name and above 400,000 bytes? `destroy_box`
returns without acting when the answer is no. Every other exit path calls
`leave_box_alive`, which names the instance and prints the command that
destroys it. A box that stays alive costs $0.36 an hour and a person can see
it. A box destroyed early costs the run.

### Events

- 23:05 the two CPU evals started, detached, under `nohup setsid`. Both
  skipped head training: `head_eval_bb.sh` reads `head-train SKIP (final
  exists)` off the head already on disk.
- 23:11 the recovery driver started, detached, pid 3775388.
- 23:13 instance 48246956, RTX 5090, 32,607 MiB, Default compute mode.
- 23:13 THREE ROUND 6 LEFTOVERS STOPPED BY PID, the finisher FIRST. It acts
  when the driver leaves the process table, and it would have posted a second
  PR comment over this one. Then the driver (2959282), then the heartbeat loop
  (2963911). Never a pattern: card 407 runs its own evals on this machine.
- 23:16 bootstrap OK, 3.5 minutes.
- 23:17 the backbone went UP this time, 5,196,231 B on both sides.
- 23:17 the sync loop started for `box_a`, 15 minute ticks, for the whole run.
- 23:17 the head started. Its command line carries `--encoder-source student
  --quantile-head --forecast-len 16 --batch-size 256 --lr 1e-3 --total-steps
  30000 --seed 20260722 --head-arch transformer`, which is round 1's protocol.
- 23:22 the launch is VERIFIED off the box: trainer pid 2661, 5,616 MiB, 73 %.
- 23:48 the head landed on the box at 450,079 B, 31 minutes for 30,000 steps.
  The sync loop stopped first, by pid, so two writers could not land on one
  `.tmp` name. Then the pull.
- 23:48 the head is on elisa. ONLY THEN the box was destroyed. $0.22 billed,
  credit $6.91 to $6.61.
- 23:48 the third eval started on elisa's CPUs.

### The head's own insurance

The sync loop pulled the periodic head checkpoints as they were written: 5k,
10k, 15k and `best`, with their optimizers and the losses CSV. A box lost at
minute 25 would have left a head to fine-tune, not an empty directory.

## Round 7 — the winning direction, extended past 0.940

### Why this round exists

Nine arms have scored and ONE goes below the k = 0 parent of this cell.
`r100_09` raises the momentum from 0.9 to 1.0 over 100,000 steps, holds 0.940
at the 40,000-step stop, and scores 1.1507 against the parent's 1.1600. No arm
of rounds 1 to 6 holds more than 0.950 at that stop.

Two readings of the nine fit that result:

  - the RAMP LENGTH sets the score. Two arms start at 0.9 and the faster ramp
    wins by 0.0277. But two arms start at 0.8 and the faster ramp LOSES by
    0.0453, so a fast ramp helps from a HIGH start only.
  - the momentum AT THE STOP sets the score, and 0.940 is not the top of that
    curve.

### The two arms

    r60_09    --ema-tau 0.9  --ema-tau-end 1.0 --ema-tau-ramp-steps 60000
    r100_095  --ema-tau 0.95 --ema-tau-end 1.0 --ema-tau-ramp-steps 100000

`r60_09` holds 0.967 at the stop and `r100_095` holds 0.970. Both go past
0.940 and neither reaches 1.0, so neither freezes its teacher there. The pair
differs in the START value at one similar END value, so it says whether the
start or the end sets the score.

Everything else is round 1's, flag for flag: k = 32, the mean reduction, the
align target teacher, align weight 1.0, 40,000 backbone steps, 30,000 head
steps, head seed 20260722, backbone seed 20260520.

### The rule this round carries forward

`round7.sh` takes `recover_w3_head.sh`'s rule over two arms. `destroy_box`
acts only when EVERY head of the round is on elisa's disk, by name and above
400,000 bytes. Every other exit path calls `leave_box_alive`, which names the
instance and prints the command that destroys it.

THE CARD GATE IS THE ONE EXCEPTION. A box this round rented seconds ago, whose
card is not in Default compute mode or is too small for two lanes, holds no
artefact. `discard_box` sends that box back and the round stops. It refuses to
act on a box read out of the `.env` file, which can hold a lane from an
earlier invocation.

### Events

- 01:26 the momentum guard reads 130 (arm, step) pairs over 13 arms and every
  one agrees with `src.models.ema_tau_at_step`. The align guard passes 26
  checks and the trainer-pattern guard passes 10.
- 01:31 `round7.sh` started, detached, under `nohup setsid`, pid 3917976. The
  heartbeat loop followed at pid 3918321, hourly.
- 01:31 the search asked for ONE card, a datacenter host, reliability at or
  above 0.99, a desktop-class CPU and a bid at or under $0.45/h. Three offers
  went away on the first attempt.
- 01:33 instance 48255116, one RTX 5090, 32,607 MiB, `Default` compute mode,
  $0.3611/h. The card gate passed on all three counts, so `discard_box` never
  fired.
- 01:36 bootstrap OK, 2 minutes 43 seconds. The box's arms table holds all 13
  arms, `r60_09` and `r100_095` among them.
- 01:36 the box builds each arm's command line with the RAMP on it, read back
  off the box's OWN copy of `run_arm.sh`. The two arms of this round differ
  from the nine that scored in the ramp length alone, and `r100_095` differs
  from `r100_09` in the start value alone, so a stale table would train a
  duplicate of an arm that already has a number.
- 01:46 the sync loop started for `box_r7`, 15 minute ticks, for the whole
  run.
- 01:47 and 01:53 the two lanes started on card 0, six minutes apart. Two cold
  HuggingFace readers on one connection need the stagger.
- 01:52 the first tick landed, verified by `ls` and not by the sync log: the
  `r60_09` losses CSV at 866,188 B, its attention-amplitude CSV at 3,490 B and
  its latent-drift CSV at 78 B.
- 01:59 the launch is VERIFIED off the box: 11,363 MiB of 32,607 MiB in use,
  TWO compute apps at 5,674 MiB each, 88 % GPU, 33 depth columns in both
  losses CSVs, which is k + 1 at k = 32.
- 01:59 the guard lines, off each trainer's OWN command line:

      arm r60_09 ema='0.9 1.0 60000' reduce=mean seed=20260520 align_w=1.0 OK
      arm r100_095 ema='0.95 1.0 100000' reduce=mean seed=20260520 align_w=1.0 OK

- 01:59 step rates: `r60_09` 2.7 sps ETA 3.9 h. `r100_095` 2.6 sps ETA 4.2 h.
  Two lanes cost each other 0.1 sps against round 6's three lanes, which is
  what the card predicted.

### A number the launch already gives

Both lanes write loss 14.2752 at step 1 and part at step 2: 13.8292 against
13.8323. The teacher is identical at step 1 and the two momentum schedules
differ from step 2 on, so the schedule reached the objective and not only the
command line.

### The budget

The credit is $6.58 and the limit for this round is $4. `MAX_SPEND` is $3.20,
which is 8.9 h of runway at $0.3611/h, and the whole round needs about 5.5 h.
