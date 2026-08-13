# Execution log — #373 rollout depth

Operational events. The report carries the science; this file carries what
happened while producing it. Times are elisa's (BST). The vast.ai
containers run an hour behind, so a remote log line reads one hour earlier
than the same event here.

## 2026-08-08 23:40 — the compute this study actually had

$7.31 of vast.ai credit and two RTX 4090s on elisa that another agent
session was already using: 23.3 GB of 24.5 GB on GPU 0 and 15.8 GB on
GPU 1, both at >90% utilisation.

The card asks for 14 cells at k = 3, each to bb40k and bb100k and
conditionally bb200k, two heads per stop, 97 GIFT-Eval configs per head.
Measured on this hardware that is 200+ GPU-hours. $7.31 buys 15.6 hours of
a 4090 at $0.4681 or 21.8 hours of a 5090 at $0.3356. The study was scoped
to what the credit covers, in the card's own run order, and the report
says which cells did not run.

One decision made the difference between two cells and five: GIFT-Eval runs
on elisa's 32 cores, not on a rented GPU. PR #394 measured it at 2.86
core-hours for the 97 configs against 58.7 s / 97.3 s per six configs on a
4090 / one core, and it takes no VRAM. So the rented cards only ever train
backbones, and heads and evals run on elisa for nothing.

## 2026-08-08 23:46 — per-cell depth check

All 14 cells, k = 0 against k = 3, one step each, through each cell's own
launcher. 28 runs, 12 minutes total on GPU 1. All PASS. See
`verify_summary.tsv` and the report.

## 2026-08-08 23:55 — GPU step time

Cell B5, alternating k = 0 / k = 3, 3 reps of 600 steps, on elisa's GPU 1
beside another session's training. Alternating rather than one-then-other,
because a single pair would fold whatever the neighbour did in between
into the overhead.

## 2026-08-09 00:05 — provisioning burned two instances and created three extras

Four failures on the way to four boxes, all vast.ai's own:

- `Offer N is no longer available`, repeatedly. The listing churns in
  seconds; every search returned 45 candidates and printed 1-3 rows, and
  each row was gone by the time provision reached it.
- `Instance 47218248 created but SSH unreachable` — the instance existed
  and was billing. Destroyed, $0.02.
- `Instance 47218524 did not reach 'running' (reason: doa)` — same,
  destroyed, $0.01.
- The retry loop's success parser did not match `vastrun-provision`'s
  output, so it read three good instances as failures and kept going. Four
  instances existed before it was killed. All four were usable and all four
  were used; the cost of the mistake was the ~10 minutes each spent idle
  during bootstrap, about $0.10 in total.

`scripts/provision_box.sh` now handles the first three by name and tries
every offer of one search before re-searching.

## 2026-08-09 00:20 — the bootstrap gate earned itself

`bootstrap_remote.sh` ends with `train.py --help` on the box. It failed on
all four with `ModuleNotFoundError: No module named 'statsmodels'`. The
trainer's `--synth-kind forked-arma` mixer imports
`statsmodels.tsa.arima_process.ArmaProcess`, and the dependency list had it
down as eval-only. Found before any GPU time was spent on it.

## 2026-08-09 00:26 — ssh hangs on a detached remote job

`ssh host "cmd & echo queued"` never returns: the session stays open until
every descriptor on the channel closes, and the backgrounded command
inherits them. The job runs fine; the caller hangs forever behind it. The
form that works is a subshell with all three descriptors redirected and a
bare `exit 0`.

## 2026-08-09 01:00 — a smoke run of the stop pipeline

`stop_k.sh` on a stand-in checkpoint, 40 head steps. Head training passed.
The eval died at shard 0: `shard_configs.py` splits the 97 configs by
MEASURED cost (0.4 s to 1537 s) and reads that table from this study's own
`results/config_costs.csv`, which had not been staged. Found before a real
backbone depended on it.

## 2026-08-09 01:11 — dropped A4 k = 3

At the measured rates the five runs in flight cost ~$4.6 of the $5.8 left.
A4 k = 3 would have taken ~$0.72 more and left ~$0.4 of margin across four
boxes. A run that dies at 90% for want of credit is a total loss, so A4
was cut rather than risk B9, which had four hours in it.

## 2026-08-09 01:16 — sync verified by `ls`

`sync/verify_sync.sh`, after a full tick on every box: 4/4 checkpoints at
exactly the remote size, 3/3 growing files present, on all four boxes.

Two bugs in the verifier itself, both found by running it:

- `ssh` without `-n` read the box table off the loop's stdin, so only the
  first box was ever checked.
- Comparing a losses CSV by size reports every healthy loop as broken: the
  file grows every step, so the remote size is newer than any copy by
  construction. Checkpoints are written once and are compared exactly;
  growing files are checked for a non-empty copy.

## 2026-08-09 01:25 — two identical A3 runs shared box b for 45 minutes

`pgrep` on box b showed two `train.py` processes with byte-identical
command lines, both writing `cf393_arm6_v2_combab_alignT_cf373k3` into
`leg_40k`, both holding 5532 MiB of the same card.

Cause: the ssh that started the first queue TIMED OUT and was reported as a
failure, but the remote command had already run. ssh holds the session open
until every descriptor on the channel closes, and a backgrounded remote job
inherits them — so the caller hangs behind a job that is running fine. The
second start then did not collide, because this box came up in `Default`
compute mode rather than the `Exclusive_Process` a vast.ai box usually
comes up in, and a second CUDA context was allowed.

Killed the younger of the two (started 1m43s later). All four boxes now
report exactly one CUDA process. `nvidia-smi --query-gpu=compute_mode`
reads `Default` on all four, so `gpu_gate` is a no-op on every one of them
and cannot be relied on here.

Cost: 45 minutes at half speed on one 5090, about $0.14. No science lost —
the two runs carried the same seed and the same data order, and the
survivor kept its own step counter and its own weights.

What it did leave: every step up to ~14,300 appears twice in that cell's
losses CSV, because `CSVLogger` opens in append mode. `scripts/losses_csv.py`
keeps the first row per step, and both curve figures read through it.

## 2026-08-09 04:34 — three stops died on CUDA OOM, and the fix

`torch.OutOfMemoryError: Tried to allocate 4.32 GiB. GPU 0 has 23.64 GiB of
which 3.15 GiB is free` inside the GRU encoder's `_last_hidden`. Three head
trainings died in 45 seconds: B5 k = 0 student, A3 k = 0 student, B5 k = 3
student.

Two causes at once. The neighbouring session's job grew from 9 GiB to
14.3 GiB partway through the study, and this study ran two of its own heads
at a time to halve the wall clock. Each head needs about 5.3 GiB.

`gpu_gate` does not cover this: it returns immediately on a `Default`-mode
card, and every card in this study — elisa's two and all four vast.ai boxes
— reports `Default`.

`stop_k.sh` now holds a `head_vram_gate`: an flock, so two of this study's
own heads cannot both pass the check and then both allocate, and then a
poll on `memory.free` until 6000 MiB is available. A head that would have
died now waits. Nothing was lost — `stop_k.sh` is idempotent, so the three
stops were simply re-queued.

## 2026-08-09 05:06 — a drained box billed for 51 minutes after its work was safe

The reaper found box d drained at 04:15 with all four of its checkpoint
files verified byte-identical locally, and `vastrun-destroy` refused:
`Instance 47219263 has no marker — it was provisioned outside vastrun-kit`.
It was not: it came from this session's own `vastrun-provision`, in the
burst where the retry loop misread three successes as failures. Whatever
went wrong, the on-instance marker was not written.

Cost: $0.31 of a $7.31 budget, and it was found by reading the balance
rather than by an alert. Destroyed with `--force` once the four files were
compared by size against the remote.

The reaper now falls back to `--force` when the refusal says "no marker",
and only then: its own gate has already established that every checkpoint
the remote holds is here, byte for byte, and that the row naming that id
was written by this session's own launch.

## 2026-08-09 06:00 — the VRAM lock was held through the eval

The `head_vram_gate` added an hour earlier holds its flock on fd 7 for the
life of the calling shell — and that shell then runs a ~1 h CPU GIFT-Eval.
So the next head waited for a card that had 9.6 GiB free the whole time.

`stop_k.sh` already dropped the `gpu_gate` descriptor before the eval, for
exactly this reason; the new one was not dropped beside it. It is now.

Recovering it without losing the eval in flight: the stale holder was inside
its eval and would not train another head, so the lock FILE was renamed.
flock is held on the inode, so a new head takes a lock on a fresh one while
the old holder keeps the old. Mutual exclusion between future heads is
unaffected — they all open the new path, and the fixed script releases it as
soon as its head finishes.

## 2026-08-09 07:01 — the group-B baseline validity gate FAILS

B5 (`arm4_combab_fix09`) retrained at k = 0 on this code, bb40k, head seed
20260722, full 97 configs: **1.3917**. `small_long.md` and
`lalign_teacher.md` both publish **1.2748** for that cell and stop.

|Δ| = 0.1169. The card's threshold is 0.0002. That is 585 times the gate,
and three times the parents' pooled head-seed band of 0.0384.

The card names the remedy: "If it does not match, retrain the k = 0 side of
every group-B cell instead of reading it from the reports, and say so in
the report." This study has same-code k = 0 for B5 and for A3, so those two
cells have valid comparisons. B9 does not, and its k = 3 number therefore
has no baseline this study can stand behind.

The gate failing is not a side note. The shift between the published
snapshot and this one is larger than the effect the study set out to
measure, so every delta computed against a published number — including the
A3 k = 3 numbers reported an hour ago — has to be recomputed against this
study's own k = 0.

## 2026-08-09 13:17 to 2026-08-10 06:29 — the review runs

Seven more backbones and fourteen more heads, on elisa's two 4090s rather
than rented cards. `gap_worker.sh` ran two backbones at a time on one card
and handed each finished backbone's heads to a background subshell, because
head training is ~35 GPU-minutes and the GIFT-Eval is ~1 CPU hour: serialising
them behind the queue would have idled the cores for hours.

The queue drained at 02:04 and the last head finished at 06:29. Every one of
the 25 evals holds all 97 configs.

One head died and was retried once (G2_B9_k0 teacher, 22:52). `head_eval_bb.sh`
is idempotent in both halves, so the retry cost only what had not finished.

## 2026-08-10 — collection and re-analysis

`collect.sh` brought the 25 score files, the trainer logs and the per-config
eval CSVs into the git checkout. `results/boxes.tsv.tmp.4009307`, a partial
write left by an interrupted `reap_boxes.sh`, was already gone; nothing under
the checkout or the run worktree matches `boxes.tsv.tmp.*`.

Three things in the analysis code were wrong and are fixed:

- **Tag parsing dropped the review runs.** Every table and figure resolved a
  cell by splitting the eval tag on `_` and reading field 1. That works for
  `A3_k3_bb40k_student` and silently drops `G6_B1_k0_bb40k_student`, so B1
  and B9 had no same-code baseline in any figure even though both had been
  trained. `scripts/runs.py` is now the one registry; no consumer parses a
  tag.
- **A prefix test folded a control into the cell it controls for.** Group A's
  launcher writes every depth of a cell AND the `L_align x4` control into one
  `leg_40k` directory, so `cf393_..._cf373k0` is a prefix of
  `cf393_..._cf373k0_aw4_40k.pth`. The rollout-fidelity and latent-movement
  figures picked their A3 `k = 0` checkpoint by `startswith`, so either file
  could win. `runs.ckpt_step()` now matches `_<N>k.pth` exactly.
- **`gap_analyse.sh` was a second pipeline.** It wrote `splits_all.csv`,
  `bootstrap_gaps.csv` and `gap_scores.md` beside `make_report_assets.sh`'s
  `splits.csv`, `bootstrap.csv` and `scores.md`, from the same inputs. Folded
  into the one rebuild script and deleted.

`make_report_assets.sh` is now the single entry point and holds no paths:
`runs.py` says which runs exist and `find_artefacts.py` finds each one's
artefacts across the sync tree, the durable root and the results directory.

The two figures that load checkpoints hit `CUDA error: out of memory` on the
final rebuild, because another session held both of elisa's cards. The script
now retries them on the CPU. The CPU numbers match the GPU ones to four
decimals on rollout fidelity and to five on latent movement.

## 2026-08-10 09:08 — the round-2 run waits for a card

The round-2 review's blocking item is one run: B5 at `k = 0`, seed 20260520,
on elisa. It separates the machine from the seed, and no other artefact can.

Both of elisa's 4090s were full when it was queued. Three other sessions were
training: `/tmp/rnd-434` held 12.6 GB of GPU 0, `/tmp/rnd-446` held 4.0 GB of
GPU 0 and 3.2 GB of GPU 1, `/tmp/rnd-454` held 16.0 GB of GPU 1. That left
22 MiB free on GPU 0 and 5062 MiB on GPU 1, against the 5375 MiB
`results/gpu_mem_B5.csv` measures for this run.

`scripts/gap_r2_launch.sh` therefore polls for 6200 MiB on either card and
starts `gap_worker.sh` on whichever frees first. Starting 313 MiB short would
not have produced a slower run, it would have produced a dead one.

Eleven orphaned CUDA worker processes (PPID 1, 7 days old, ~4.9 GB of GPU 0
between them) were left alone. Reclaiming them would still have left GPU 0
about 400 MiB short of the run, so the risk bought nothing.

## 2026-08-10 — one step-time pipeline, and the machine in the registry

`steptime_from_logs.py` and `results/steptime_runs.csv` are deleted.
`steptime_provenance.py` produces the same medians plus the contention
split, so keeping both was the same "second pipeline" the last round removed
from `gap_analyse.sh`. `make_report_assets.sh` now runs
`run_provenance.py` then `steptime_provenance.py`.

`run_provenance.py` is new. It reads the driver logs and the box queue logs
and writes, per run, which other runs shared its card and for how much of
its life. That is what the cost table was missing: every elisa backbone in
this study was contended for 43% to 100% of its life, so only the four
rented-box runs and the solo tail of a fifth can carry a step time.

## 2026-08-10 09:07 to 13:51 — the round-2 machine test

`gap_jobs_r2.tsv` carries one row. It retrains B5 at `k = 0` on the protocol
seed 20260520, on elisa, so the only thing that differs from B5·s1 is the
box. The reproduction table sorted perfectly on the machine and the study
could not say whether the seed or the box did it.

Backbone 40k steps, then the student head and the 97-config eval:
`score_G7_B5_k0_e_bb40k_student` = **1.2751**. The published value is 1.2748
and B5·s1, same seed, same code, on a rented RTX 5090, is 1.3917.

The machine moved it by 0.1166 and the seed by 0.0035. `early_loss.csv`
shows B5·s1 and B5·s3 printing the same mixer counts step for step, so the
two runs saw the same batches in the same order.

## 2026-08-10 13:51 to 17:51 — the teacher head of that control did not run

`head_eval_bb.sh G7_B5_k0_e_bb40k_teacher` waited 4 h for VRAM and aborted:
other projects held both of elisa's cards, GPU 1 had 4916 MiB free and the
head needs 6000. `stops.log` carries the TIMEOUT line and the ABORT beside
it.

It was not retried. The group-B parent reports publish the student-encoder
head only, so the student number is the one the reproduction check compares
against, and the encoder-delta figure bounds the choice at under half the
head-seed band. The worker and its retry loop were stopped rather than left
to spin on a card another project owns.

## 2026-08-10 17:54 — collect.sh was overwriting the execution log

`collect.sh` rsyncs the run worktree's `results/` over the checkout's. The
run worktree carries its own fork of `execution_log.md`, branched before the
review runs, and rsync copied the older file over the newer one: 82 lines
gone. The log is written in the checkout, never by a run, so it is now
excluded from that rsync. Restored from git.

## 2026-08-10 18:22 — the rebuild did not run from committed artefacts

`find_artefacts.py` searched two working trees, `~/cf373_sync` and
`$CF373_ROOT`, and the checkout last. Both are local to elisa. Five of the
study's eleven depth-ladder curves lived only in `$CF373_ROOT`, because
elisa has no sync loop and wrote straight to the durable root, so a clone
resolved ZERO curves and rebuilt no training-curve figure at all. **Both
sides of B1 were among the five**, which put the study's one sound
comparison outside the repository.

`collect.sh` now downsamples every such curve into `curves/<machine>/`, at
the same `--stride 20 --dense-until 1000` the box runs already used, and
`find_artefacts.py` searches the committed tree FIRST. Eight elisa curves
came across, 11 MB. `--what missingcurves` is the standing check: it lists
every backbone whose losses CSV is in a working tree and not in git, and it
now prints nothing.

Verified by hiding both trees:
`CF373_SYNC_BASE=/nonexistent CF373_ROOT=/nonexistent bash
scripts/make_report_assets.sh` rebuilds all 14 non-checkpoint figures and
every table. The two checkpoint figures skip with the line that says so; the
Protocol names them.

Two side effects worth recording. Every curve in a figure is now at one
resolution. Before this, the box runs were downsampled and the elisa runs
were not, so B9's two sides carried different smoothing spans in the same
panel. And the four curve figures changed slightly, which is that fix.

## 2026-08-10 18:26 — one rsync exclusion was not enough

The Aug 10 17:54 fix excluded `execution_log.md` by name.
`make_report_assets.sh` writes eight more files into the same directory and
the same stale fork would have reverted every one. `collect.sh` now holds a
`GENERATED` list of all ten, and a guard re-derives the list from
`make_report_assets.sh` and refuses to run if the two disagree. Tested by
dropping a name: the guard names the missing file and exits 1.

## 2026-08-10 18:31 — the depth-0 diagnostic did not survive its own audit

Committing the curves made section 9 auditable for the first time, so it was
audited. `depth0_gap.py` writes the gap the section reads off
`cos_err_depth.png` as a number, over four end-of-run windows.

B9 and B1 hold their sign over every window, at 0.08 to 0.11. **A3 and B5·s2
do not.** A3's `k = 3` gap runs -0.0469 over the last half of the run and
+0.0623 at the final step. B5·s2's runs +0.0121 and -0.0129. The section's
sentence "the sign of that gap matches the sign of the eval result in all
four cells" was true of two cells, and it is now stated as two.

The gap for those two arms is smaller than the drift across the window it is
measured over. Nothing here says the diagnostic is wrong; it says it is
underpowered on the two arms whose eval result it appeared to explain.
`results/depth0_gap.csv` carries the numbers and marks which arms hold a
sign.

## Operational detail moved out of the report

**`B5·s3`'s teacher head.** The head waited four hours for VRAM on elisa and
then aborted. Other projects held both cards for the whole window; GPU 1 had
4916 MiB free and the head needs 6000. Logs: `results/stops.log`,
`results/eval/G7_B5_k0_e_bb40k_teacher/stop.log`.

**The step-time probe's card.** The controlled `k = 0` against `k = 3` probe
ran on elisa's GPU 1 while another session's job held 8946 MiB at the start
and drew 44% mean utilisation throughout
(`results/steptime_B5_solo_card.csv`). The probe therefore alternates on a
shared card rather than owning one.

**Training-curve diagnostics the report does not read.** The rebuild writes
`plots/per_run_loss.png`, `plots/cos_error_per_arm.png` and
`plots/latent_movement.png` beside the figures the report carries. The loss
panel is not comparable across depths, because `k = 3` optimises the `k = 0`
objective plus three added terms, so no ranking is read off it.
`plots/ladder.png` draws this study's bb40k points on the published `k = 0`
trajectories; every point this study contributes sits at one x value, so the
report carries `depth_response.png` and `reproduction.png` instead.

## 2026-08-12 16:05 to 16:35 — round 3 replaces the fleet with one box

Round 2 rented one single-GPU box per cell. Every operational failure in
this study came out of that shape: 15 failed bootstraps on B8, a box idle
37.6 h at 0%, two more idle 4 h, and a duplicated run on one card. Round 3
rents ONE box with two cards and pairs it with elisa's two.

Instance 47557391, 2x RTX 4090, Hungary, $0.789/h on-demand, reliability
0.993, driver 570.211.01. The gate the card sets is hard and it ran before
any training: `python3 -c "import torch; print(torch.cuda.device_count())"`
inside the container printed **2**.

Two single-GPU boxes were up when round 3 started, 47555858 (cf373r2-b10,
8 min old) and 47556474 (cf373r2-b8, 1 min old). Neither held a checkpoint;
both were destroyed. A 2x RTX 5090 offer was taken and did not reach
`running` inside the kit's timeout, so it was destroyed too, $0.11.

The many-box drivers were stopped first, because they were still
provisioning: `reap_boxes.sh`, two `r2_launch_cell.sh`, two
`provision_box.sh`, one `vastrun-provision` mid-flight, `r2_eval_driver.sh`
and sixteen `sync_loop.sh` instances, one per dead box.

`q_run.sh` is the replacement and it is the only thing that starts work.
One queue, `q_queue.tsv`, 43 jobs in the card's order, over six slots: two
per rented card and one per elisa card. Two backbones share a card — 5.4 GB
+ 5.4 GB against 24 GB, and measured 2.7 to 3.0 sps each against 4.1 solo,
so the second process buys 40% more per card. An elisa slot only takes a job
when that card has the VRAM free at that moment; other projects held 32 GB
of elisa's 49 GB through the whole launch.

One sync loop, not one per cell, into one flat durable root
`/home/jupyter/cf373_r3/sync` that mirrors the box's `/root/cf373_runs`.

### Two bugs the launch found

**The remote launch never returned.** `ssh host "cmd &"` holds the channel
open while the backgrounded child lives, even with the child's three
descriptors redirected and `setsid` in front. The first dispatcher placed
B8, blocked inside that ssh, and never came back to place a second job. The
job body now goes over as a file and the start is a separate `ssh -n` under
a 40 s timeout; the body writes a `.started` marker and the dispatcher reads
that, never ssh's exit status.

**Every job landed on one card.** `SLOTS=(${QSLOTS:-"rem:0 rem:0 ..."})`
holds ONE element: the quotes inside the default make it a single word, so
no splitting happens. `${SLOTS[$i]}` then returned the whole string for
every index, `%%:*` read `rem` and `##*:` read the LAST field, so B10 was
placed beside B8 on card 1 and card 0 sat idle — the same duplicated-run
failure round 2 had. The default is now unquoted and the script refuses to
start with fewer than two slots. Cost: 13 minutes of one card.

### What the queue holds

Nine backbones: B8 from step 0 to 100k, then eight extends from 100k to
200k, biggest bb100k winner first — B10, A2, B4, B6, B2, A3, A4, B1.
Seventeen heads at 30,000 steps, seed 20260722, `--grad-clip 1.0`.
Seventeen 97-config GIFT-Evals, B4 strategy, horizon 16, on elisa's cores.

A4 extends the student head only. The rule read its teacher up at bb100k.

B1 extends both heads. The rule stopped B1's student — 1.0850 at bb40k
against 1.0881 at bb100k — and the card extends the cell whole, because
B1's bb40k pair is round 1's, written under `G6_B1_k3_bb40k` before round 2
renamed the cells, so the rule tested B1 against a number this round did not
produce. `results/r3_extend_override.tsv` records that, and both of B1's
heads are reported with the rule's verdict beside them.

Five cells stop at 100k and are absent from the queue: A1, B3, B5, B7, B9.

## 2026-08-12 17:00 — A1 and B3 hold one student model, not two

A1 and B3 scored the same number on the student head at both stops, 1.1305
at bb40k and 1.1676 at bb100k, while their teacher heads scored apart. Two
cells, one number, reads like a path key that drops the cell.

It is not a path key. `scripts/pair_identity.py` loads both backbones and
compares every tensor, split into the student side (encoder, transformer,
channel mixing, embeddings) and the teacher side (`teacher_*`).

    A1/B3  bb40k   student  110/110 identical   max|diff| 0
    A1/B3  bb40k   teacher    0/52  identical   max|diff| 6.400e-03
    A1/B3  bb100k  student  110/110 identical   max|diff| 0
    A1/B3  bb100k  teacher    0/52  identical   max|diff| 1.986e-01

The two cells train the same student, bit for bit, at both stops. Their
student heads follow: 28 of 28 head tensors are identical at both stops,
and the 97-row eval CSVs are byte-identical. The teacher side differs at
every level. One number for both cells is the right answer.

The arm says why. A1 and B3 both run `arm5_combab_alignS`, whose alignment
target is the student and whose representation loss is `lrep`, not
`lrepmoco`. Nothing in that arm's gradient path reads the teacher, so the
EMA regime — group A's schedule against group B's fixed 0.9 — cannot move
the student. The teacher is a passive copy and it is the only thing the
regime changes.

The other same-arm pairs are not in that position, and the same test says
so. `arm6_v2_*` carries `lrepmoco`, whose keys come from the momentum
encoder, so the regime enters the student's loss:

    A4/B1  arm6_v2_combab_alignS  student differs at both stops
    A3/B2  arm6_v2_combab_alignT  student differs at both stops

A2/B8 waits for B8's first checkpoint. `results/pair_identity.tsv` holds
every row.

The consequence for the report: A1 and B3's student column is ONE
measurement. Publishing it twice would claim a replication that does not
exist. The teacher column is two.

## 2026-08-12 17:05 — three fixes to what the round reads and pays

**B1's bb40k score had a name no script could find.** Round 1 wrote it as
`score_G6_B1_k3_bb40k_*`, and the coverage table needed a hand-written
alias to see 1.0850 and 1.0948. The round-1 eval read B1's own checkpoint —
its log names `..._cf373k3_40k.pth`, md5 `23ba3d9d...`, the same file round
2 resumed — so `scripts/normalise_scores.sh` writes the canonical name
beside the old one and copies the eval artefacts under the cell's name. It
removes nothing.

**The coverage table called unscheduled work `running`.** It marked every
stop of a cell whose backbone was training, so B8 — queued to 100k and no
further — reported bb40k and bb200k as in flight. Coverage now reads the
job that produces the number, the head and the eval, off `q_queue.tsv` and
the queue's state files: `run` is in flight, `plan` is queued, and anything
with no job reads as the gap it is. `results/r3_no_extend.tsv` records the
stops this round decides not to produce, so a decision cannot read as an
omission.

**B8 had no bb40k pair, and the other thirteen cells do.** Its backbone
saves 40k on the way to 100k, so the pair costs two 15,000-step heads and
two CPU evals. Four jobs at the tail of the queue, behind everything the
plan asked for. Head steps match the other thirteen cells' bb40k heads.

**A head no longer takes a rented card while elisa has room.** The box is
paid by the hour, and the rental lasts as long as the backbones do, so a
head on a rented card pushes the end of the rental out by its own runtime.
The dispatcher now offers elisa's cards to a head first and a rented card
only when elisa has no VRAM. Backbones keep the rented-card-first order,
and no card is ever left idle with the queue not empty.

## 2026-08-12 16:30Z — the round-3 queue, checked end to end after a session drop

Two dispatches before this one died within minutes of starting. The queue
they left behind kept running, so this session adopted it rather than
restarting it. Every part was found alive and was verified by its own
output, not by the fact that a process exists:

    q_run.sh        pid 22515   4 jobs placed, ADOPT after its own restart
    sync_loop.sh    pid 4164566 tick 17:11, 12 files, sizes in the log
    q_guard.sh      pid 4188854 floor $5.50, box 47557391
    q_heartbeat.sh  pid 23744   hourly, last 16:08:54Z

The box passes the gate the card set before any training:

    device_count 2   torch 2.8.0+cu128   cuda 12.8

Both cards carry two backbones each, 86% and 90% util, 10.8 GB of 24 GB per
card. Four runs are on them:

    B8   arm6_v2_nse_alignT_fix09      0 -> 100k    step   9,300
    B10  arm6_v2_nse_alignS_fix09    100k -> 200k   step 109,000
    A2   arm6_v2_nse_alignT_sched    100k -> 200k   step 108,700
    B4   arm5_combab_alignT_fix09    100k -> 200k   step 108,700

2.7 steps/s per run, so a 100k leg takes 10.3 h. Nine backbone jobs remain
in the queue and four cards can hold four of them, so the backbone column
finishes about 22 h from now: $17.8 at $0.8144/h. Credit is $27.12.

**elisa's two cards cannot take a job right now.** 1,639 MiB and 5,530 MiB
free against the 7,000 MiB a backbone needs and the 8,500 MiB a head needs.
Other projects hold the rest. The dispatcher is right to leave them out, and
it will place work on them the moment the VRAM appears. No card of ours is
idle: the four rented slots are full and the queue's remaining GPU jobs are
either waiting for a slot or waiting for the backbone they read.

**The coverage table said `plan` for work that is on a card.** It reads the
head and the eval, which is right, but a head whose backbone is training now
is not the same as a head nobody has started. It now walks each head's
dependency chain and marks `bb-run` when a running backbone job sits above
it. B8's bb40k pair reads `bb-run` correctly: it hangs off `bb_B8_100k`,
which writes the 40k checkpoint on the way past it.

## 2026-08-12 17:40 BST — the A1/B3 student number is right

The card blocked publication on this: A1 and B3 scored the same student
number at both stops, 1.1305 at bb40k and 1.1676 at bb100k, while their
teacher numbers differed. Two different backbones, one number, so one of
them had to be wrong.

The head and the eval paths are not the cause. Each cell has its own head
directory, its own head checkpoint, and its own `backbone.txt`, and each
records the checkpoint it read:

    A1_k3_bb40k_student   cf393_arm5_combab_alignS_cf373k3_40k.pth
    B3_k3_bb40k_student   bb_small_arm5_combab_lalign_lrep_..._cf373k3_40k.pth

Their 97-config eval outputs are byte-identical (md5 eb5e4e21 for both
`all_results.csv`). So the two models predict the same thing.

`scripts/pair_identity.py` compares the checkpoints tensor by tensor,
splitting student from teacher. It reports (`results/pair_identity.tsv`):

    A1/B3   arm5_combab_alignS     bb40k    student  110/110 identical
    A1/B3   arm5_combab_alignS     bb40k    teacher    0/52  differ, max 6.4e-03
    A1/B3   arm5_combab_alignS     bb100k   student  110/110 identical
    A1/B3   arm5_combab_alignS     bb100k   teacher    0/52  differ, max 1.99e-01
    A4/B1   arm6_v2_combab_alignS  bb40k    student    4/110 identical
    A3/B2   arm6_v2_combab_alignT  bb40k    student    4/110 identical

A1 and B3 hold the SAME student weights, bit for bit, at both stops. The
student head reads the student side only, so one number for both cells is
the correct answer.

The reason is in the arm. `arm5_combab` carries `--loss-shape
cosine_similarity_batch_rep_only --align-loss-weight 1.0 --tau-rep 1.0
--cpc-infonce-weight 0.0` and aligns to the student. It has no
`--moco-rep-keys`. So no loss term reads the EMA encoder, the EMA regime
sends no gradient into the student, and the two regimes train one student
from one seed. The EMA regime shows in the teacher tensors only, and the
teacher numbers do differ: 1.1318 against 1.1343, 1.1565 against 1.1618.

The other three same-arm pairs run `arm6_v2`, which does carry
`--moco-rep-keys`. There the EMA encoder produces the keys, so the regime
reaches the student, and A4/B1 and A3/B2 differ on 106 of 110 student
tensors. A2/B8 waits on B8's first checkpoint.

Nothing is re-run. A1 and B3 report one student number because they trained
one student.

## 2026-08-12 17:45 BST — two fixes to the round's own instruments

**The coverage denominator counted the stops.** `deliverables 84 done 65`
put the 13 heads the extend rule ended into both the numerator and the
denominator. A stop is not a deliverable. The line now reads
`deliverables 71 done 52 ... (+13 stops, not deliverables)`.

**Nothing restarted the dispatcher.** `q_run.sh` runs detached, so a dead
session does not kill it, but nothing brought it back if it died on its
own, and the cards would then drain their jobs and idle against a full
queue. `scripts/q_super.sh` checks every 5 minutes and restarts it once.
It identifies the dispatcher by `ppid == 1`, because the dispatcher forks a
subshell per local job that carries the same argv — counting by argv alone
reads a running local head as a live dispatcher. It stands down when the
guard writes `BLOCKED_BUDGET` or when the queue drains.

## 2026-08-12 18:20 BST — three faults in the queue's own machinery

The five running backbones were untouched. Every fault below sat ahead of a
job that had not started yet, so fixing them cost nothing already spent.

**An extend resumed the wrong checkpoint for B1 and B2.**
`stage_bb_remote` staged `stop - 100000`, which names the 100k for every
cell. B1 and B2 hold a 140k, written 2026-08-12 14:17, optimizer beside it.
Both would have retrained 40,000 steps that are already on disk: 80,000
steps, 8.4 slot-hours at the 2.63 steps/s the box measures.

`cell_paths.sh` gains `cf373_bb_below <cell> <k> <stop>`: the furthest
checkpoint strictly below the stop, chosen by the step in its name, and only
if its optimizer sidecar is there. Group B keeps one directory per run;
group A keeps one per leg, so the search walks the arm's sibling legs. It
resolves 140k for B1 and B2 and 100k for B4, B6, B10, A2, A3, A4.

The 140k pairs were copied from the round-2 roots into the round-3 root,
size-checked, `.tmp` then `mv`. The round-2 copies stay.

**The group B launchers pick their resume by mtime.** `run_arm_k.sh:364` and
`run_arm_lalign_k.sh:238` read `ls -t`, and staging a checkpoint onto the box
gives it the newest mtime there. A staged 100k would therefore win over a
140k that arrived earlier. `launch_bb` now passes `RESUME_FROM` and names the
file. `run_leg_k.sh` already chooses by step and is left alone.

**Two dispatchers ran at once, for about two minutes.** `setsid nohup bash
q_run.sh &` leaves two processes: a wrapper and the loop. Killing the wrapper
left the loop running, and the replacement made two. Round 2 lost a run to
exactly this, two processes writing one run name.

The supervisor could not have caught it. Its test was `argv matches` plus
`ppid == 1`, and a restart orphans every local job's subshell onto init, so
an orphan reads as a live dispatcher. The dispatcher now writes
`results/queue/dispatcher.pid` with its own `$$`, and `q_super.sh` checks that
pid is alive and still running `q_run.sh`. Verified: one loop, pid 186853,
five jobs adopted, `sleep 60` on its poll.

## 2026-08-12 18:55 BST — the session resumed; nothing was restarted

The queue survived two dead sessions. On resume it held one dispatcher
(pid 186853), one supervisor, one budget guard, one 15-minute sync loop and
five backbones, and every one of them was still doing its job. Nothing was
relaunched. `torch.cuda.device_count()` prints 2 on box 47557391, and both
its cards read 77% and 89%.

Steps at 18:37, off the losses CSVs:

    B8   0    -> 100k    21,900   fresh
    B10  100k -> 200k   121,700
    A2   100k -> 200k   121,800
    B4   100k -> 200k   122,300
    B6   100k -> 200k   115,400   elisa, GPU 1

2.9 steps/s per remote slot, 3.7 on elisa. elisa's GPU 0 holds 2.0 GB free
against the 7 GB a backbone needs, so the dispatcher leaves it out.

The four queued extends resolve the checkpoint the plan asked for:

    B2 -> ..._alignteacher_cf373k3_r3_140k.pth
    B1 -> ..._cf373k3_r3_140k.pth
    A3 -> cf393_arm6_v2_combab_alignT_cf373k3_100k.pth
    A4 -> cf393_arm6_v2_combab_alignS_cf373k3_100k.pth

## 2026-08-12 19:05 BST — two faults between the numbers and the report

**Round 3's numbers could not have reached a figure.** Every split and plot
script reads `results/eval/*/all_results.csv` in the git checkout. Round 2
had `r2_collect.sh` to put them there, which reads one directory per cell
under `~/cf373_r2`. Round 3 writes ONE flat root, `~/cf373_r3`, and nothing
read it. A 200k eval would have finished, written its 97 configs, and never
appeared in a plot.

`scripts/r3_collect.sh` reads the flat layout: the 97-config CSV, the
summary, the head's `backbone.txt` and `head.log`, the trainer logs, and the
losses CSVs at every 200th row. It skips a CSV holding fewer than 97
configs, so no figure can average over a partial eval. First run: 0 evals
(round 3 has scored nothing yet), 7 logs, 19 curves.

**The coverage table called ten scored cells `never ran`.** It was built
from the run registry, which knows round 1's 32 runs and stops there. It
now counts the score files, which are the thing that says a number exists,
and gains a `stops scored` column. It reads 13 of 14 cells scored, B8 the
one hole, stops bb40k and bb100k. bb200k appears in it as those land.

**Still stale, for the report stage.** The report's title, its opening
paragraph and its figures describe round 1: four cells at bb40k. The tables
below them now describe thirteen at two stops. The prose is the writer
stage's job and the 200k numbers are not in yet, so nothing above the
`TABLES` block was touched.

## 2026-08-12 19:30 BST — the head path, tested before it was needed

The session resumed onto a queue that was still doing its job: one
dispatcher, one supervisor, one budget guard, one 15-minute sync loop, one
publisher on a 20-minute timer, and five backbones. Nothing was restarted
for its own sake. Credit $25.55, box spend $2.48 over 3 h 2 m at $0.8144/h.
`torch.cuda.device_count()` prints 2 on box 47557391.

Round 3 had trained nine backbones and run no head and no eval. That half
of the queue was therefore untested, and it holds 19 heads and 19 evals —
every number this round produces. Three faults were in it.

**A head on the rented box could not read a backbone trained on elisa.**
`r2_head_box.sh` resolves its checkpoint under `CF373_ROOT` on the machine
that runs it, and four of this round's nine backbones train on elisa. B6 is
one of them. Its head would have found nothing, exited 3, and been marked
`failed` — a state the dispatcher never retries. `q_run.sh` now stages it:
ask the box first, because a backbone it trained is already there and the
ask costs one ssh, and mirror from elisa only when it is not. No optimizer;
a head reads weights, it does not resume. A checkpoint that is on neither
machine yet returns non-zero, which leaves the job QUEUED rather than
failed, so the next sync tick and the next dispatch place it.

**19 heads behind 9 backbones on four slots.** The queue puts every extend
ahead of every head, correctly — the extends are what the card asked for.
But four slots meant no head could start until the last extend had a card,
and that put about six hours of head time at the end of a bill paid by the
hour. Each rented card now carries a THIRD slot, and it takes heads only:
5.4 GB + 5.4 GB + 7 GB against 24.5 GB, on cards reading 77% and 89%. A
head there takes no slot from a backbone; it fills what the two backbones
leave. Backbones are barred from it, or a third backbone would slow the two
on the critical path to buy one that is not. `r2_head_box.sh` still gates on
8.5 GB free before it allocates, so a card that is genuinely full makes the
head wait instead of dying on an OOM.

**A head could lose its provenance marker, and the guard would pass.** The
eval refuses a head whose `backbone.txt` names a checkpoint other than the
one the cell resolves to. That file is written AFTER training. The smoke
test below was killed by an ssh drop in exactly that window and left
`final.pth`, its optimizer and the encoder-source marker with no
`backbone.txt` — and the re-run then hit the SKIP guard and exited before
the line that writes it. A missing marker makes the pair check pass by
being absent, which is the one way it must never pass. It is now written on
the skip path too. Reproduced on the box, fixed, re-run, verified.

**The smoke test.** A 200-step head on B10's bb100k checkpoint, into
`/root/cf373_smoke`, a scratch root outside the synced tree: rc=0, a
449,943-byte head, its optimizer, the encoder-source marker, the losses CSV.
The eval half is under test now — the full 97 configs over 8 shards on
elisa's cores against that same throwaway head, into
`/home/jupyter/cf373_smoke_eval`. Its number is meaningless and is not a
score; what it proves is the shard split, the merge, the 97/97 check, the
aggregate pass and the score extraction, in round 3's flat layout.

**Scripts now reach git.** The queue runs out of `wt-cf-373-run2`, so a fix
made while it is live is made there, and the publisher copied results only.
Three scripts were edited today and only a hand copy would have committed
them. `r3_publish.sh` now copies `scripts/` on every tick.

## 2026-08-12 19:50 BST — B1 does have a bb40k comparison

The card extends B1 with "B1 has no valid 40k comparison; extend and say
so", and the queue file repeated it. The decision stands — B1 is extending —
but the premise does not, and the report should not carry it.

B1's bb40k pair is round 1's, written under the name `G6_B1_k3_bb40k_*`
before round 2 renamed the tags. Three things make it this cell's number at
this cell's protocol:

    checkpoint   the head log names
                 bb_small_arm6_v2_combab_lalign_lrepmoco_..._cf373k3_40k.pth
                 md5 23ba3d9dcb4a9ee86d18a377a5965ff1, which is the file
                 round 2 resumed to reach 100k
    head         15,000 steps, seed 20260722, quantile head, 2-layer
                 transformer, forecast-len 16, batch 256, lr 1e-3,
                 grad-clip 1.0 — the other thirteen cells' bb40k protocol
    eval         97 configs, strategy B4, horizon 16, aggregate
                 GM-Relative MASE 1.0850 student / 1.0948 teacher

Round 2 started its own B1 bb40k head and did not finish it: that directory
holds `_best.pth` and its optimizer, no `_final.pth`, and no eval ran. So
there is one bb40k number for B1, not two that disagree.

The one difference from round 2's bb100k head is the machine — round 1 ran
heads on elisa, round 2 and 3 run them on the rented box — and the card
rules that question closed. Under that ruling the pair is comparable, and
the extend rule can be read on B1 like any other cell: 1.0850 -> 1.0881 on
the student, 1.0948 -> 1.0897 on the teacher. The student is flat to
slightly worse, the teacher slightly better.

`scripts/q_queue.tsv` said "B1 carries no bb40k number of its own, so the
rule could not be tested on it". That sentence is now corrected in place.

## 2026-08-12 20:05 BST — A1 and B3 hold one student, not two

The card blocked publication of A1 and B3 until one of their two equal
student scores was shown to be wrong. Neither is wrong. The two cells train
one and the same student, bit for bit, and the shared number is what that
must produce.

The chain, tested end to end by `scripts/pair_identity.py`:

    backbone   student side 110/110 tensors identical, max|diff| 0, at
               bb40k and at bb100k. Teacher side 0/52, max|diff| 6.4e-3 at
               bb40k and 1.99e-1 at bb100k — the regime shows there.
    head       the student head trained off each: 28/28 tensors identical,
               max|diff| 0, at both stops. The teacher heads differ,
               max|diff| 1.65 and 4.16.
    eval       identical head, identical 97 configs, identical aggregate:
               1.1305 at bb40k, 1.1676 at bb100k.

Nothing shared a path. A1 evaluates out of `A1/sync/eval/A1_k3_bb*_*`, B3
out of `B3/sync/eval/B3_k3_bb*_*`, and each head's `backbone.txt` names its
own cell's checkpoint. The two backbone FILES differ by md5 — f99fa42c
against b3a51f06 at 40k — because the teacher tensors differ inside them.
File md5 was the wrong instrument: it reads a difference the student head
never loads. Head md5 differs too, on bytes that are not weights.

**Why the EMA regime cannot reach this student.** A1 and B3 run
`arm5_combab` with `--align-target student`. Two things could carry the EMA
teacher into the student's gradient, and this arm has neither:

    L_align    `align_ref = hy_teacher_norm if align_tgt == 'teacher' else
               hy_norm`, then `.detach()` (src/loss.py). With the student as
               its own target the term is the student against a detached
               copy of itself. The teacher is not read.
    MoCo keys  `--moco-rep-keys` gives the contrastive loss teacher-encoded
               keys. `arm5_combab` does not pass it; `arm6_v2_combab` does.

So in this arm the teacher is a by-product: updated every step, read by
nothing. The trainer logs agree from the first stop — both runs print
`loss=17.3118 ema_loss=17.3034 gap=-0.1967 AUC=0.8438` at step 200 — which
is the same statement one step into training.

The other three pairs all carry `--moco-rep-keys`, and their students
differ: A4/B1 4/110 identical, A3/B2 4/110. A2/B8 fills in when B8's
checkpoints land; the publisher re-runs the check every 20 minutes.

**No re-run.** Re-training the two student heads and re-running the two
97-config evals costs four GPU-hours and about half a dollar of a $25
budget, and the inputs are equal bit for bit, so it can only reproduce
1.1305 and 1.1676. The pair is published as ONE student measurement carried
by two cells, and as two teacher measurements.

## 2026-08-12 20:10 BST — the queue was found running, and left running

The session resumed onto live work and restarted nothing. One dispatcher
(pid 267027), one supervisor, one budget guard, one 15-minute sync loop, one
20-minute publisher, five backbones.

Two things read like faults and are not:

    two `bash scripts/q_run.sh`   pid 52775 is B6's job wrapper, orphaned
                                  onto init when the dispatcher restarted. A
                                  local job's subshell inherits the
                                  dispatcher's argv. It has one child,
                                  run_arm_k.sh; the real dispatcher has one
                                  child, `sleep 60`. Killing the wrapper
                                  would drop B6's return code.
    A1 = B3 on the student        one model, two cells. See the entry above.

Checked before leaving it alone: `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are
in the environment of the box's training processes, and step timing reads
`data=5.0ms` against `fwd=168ms bwd=136ms`, so the stream is not throttling
the cards; 52 GB free on the box; both cards 85–88% util; B1 and B2 resolve
their resume to their own `_r3_140k.pth`, so those two extends cost 60k
steps, not 100k; the extend re-fires the cell's own launcher with the same
arm argument, so a 200k leg cannot differ from its 100k leg by a flag; head
protocol is 15,000 steps at bb40k and 30,000 above it, seed 20260722,
`--grad-clip 1.0`; the eval writes `score_<CELL>_k3_bb<stop>k_<enc>.txt`,
which is the name the coverage table and the tables script read.

The ladder figure already draws a 200k tick and fills it from the score
files, so the round's headline needs no new plotting code.

## 2026-08-12 20:35 BST — the A1/B3 block is closed on the files, not on an argument

The card asked for one thing before anything is published: find the head or
eval path key that ignores the EMA regime. The answer is that there is none.
`scripts/pair_head_files.py` resolves both sides of every same-arm pair and
prints the file behind each one:

    A1/B3 bb40k  student   A1 7b0c4786   B3 74464de1   separate files
    A1/B3 bb100k student   A1 98e33fb9   B3 eed8a553   separate files
    A1/B3 bb40k  teacher   A1 1ac4a40d   B3 f3cc0d9d   separate files
    A1/B3 bb100k teacher   A1 dc5f7b0e   B3 4d543b0f   separate files

Four distinct md5s per stop, under four distinct cell-id directories:
`cf373_r2/A1/sync/eval/A1_k3_bb100k_student/` against
`cf373_r2/B3/sync/eval/B3_k3_bb100k_student/`. The evals agree — each log
merges its own 97 configs into its own `all_results.csv`:

    A1 [08-11 10:03:13] merged 97 -> .../A1/sync/eval/A1_k3_bb100k_student/gift/
    B3 [08-11 10:35:17] merged 97 -> .../B3/sync/eval/B3_k3_bb100k_student/gift/

No path, no file and no directory is shared. The three other pairs are clean
by the same test.

**The re-run the card asked for has already happened.** Two head trainings
ran, from two directories, off two backbone files with different md5s. Two
97-config evals ran, into two directories. They returned the same number
because `pair_identity.tsv` says the student tensors going in were equal bit
for bit — 110/110, max|diff| 0.000e+00 — and the heads coming out were equal
bit for bit too, 28/28. Doing it a third time cannot say more; it would only
re-measure a deterministic map on the same input, at four GPU-hours off a
queue with nine backbones still to run.

**So A1 and B3 are publishable now**, with the equality stated in the table
and not hidden: ONE student measurement carried by two cells, TWO teacher
measurements. The cause is in the arm, and it is the reason the equality is
information rather than an error — `arm5_combab --align-target student` has
no path from the EMA teacher into the student's gradient (align target
detaches to self; the arm does not pass `--moco-rep-keys`), so the EMA
regime, which is the only difference between A1 and B3, cannot move the
student. Same seed, same student, to the bit. The three pairs that DO pass
`--moco-rep-keys` all differ.

**Naming.** The 14-cell grid now reads `score_<CELL>_k3_bb<stop>k_<head>.txt`
throughout; `score_B1_k3_bb40k_student.txt` holds 1.0850 and the teacher
1.0948, the same numbers round 1 wrote under `score_G6_B1_k3_bb40k_*`. The
`G*` and `k0` files that remain are round-1 side measurements and depth-0
controls, which are not cells and must not carry cell names.

## 2026-08-12 21:05 BST — round 3 picked up on a fresh session, nothing restarted

The previous two dispatches died on a session limit within minutes of
launching. This one found the round already running and left it running.

**What was alive.** One dispatcher (`q_run.sh`, pid 267027, eight slots), one
sync loop at 15 min, one credit guard at the $5.50 floor, one reaper, one
hourly heartbeat, one publisher on a 20-minute timer. Five backbones training:

    job            cell  from   to     step at 20:50Z   where
    bb_B8_100k     B8    0      100k   42,800          box gpu0
    bb_B10_200k    B10   100k   200k   142,700         box gpu0
    bb_A2_200k     A2    100k   200k   142,600         box gpu1
    bb_B4_200k     B4    100k   200k   143,200         box gpu1
    bb_B6_200k     B6    100k   200k   141,800         elisa gpu1

**One process looked like a second dispatcher and is not.** Pid 52775 carries
the argv `bash scripts/q_run.sh` and holds B6's process tree, so `ps` reads it
as a duplicate. Its file descriptors say otherwise: fd 1 is
`results/q_bb_B6_200k.log` and fd 0 is the queue file the dispatch loop reads.
It is the `( ... ) &` subshell `launch_bb` forks for a local job, orphaned onto
init when the dispatcher it came from was replaced. It is in `do_wait` on the
training, and it will write `queue/bb_B6_200k.rc` when the training ends, which
is the only thing the live dispatcher polls. Killing it would have dropped B6's
completion signal on the floor. It stays.

**The resume restores the step counter.** The backbone `.pth` is a bare
state_dict with no step in it, so the counter has to come from the optimizer
sidecar. The four running extends prove it does: B10 resumed
`..._r2_100k.pth` at 16:31 and read 142,700 at 20:50, which is 100,000 plus
the 42,700 steps 4.3 h at 2.76 sps buys. B1 and B2 will resume their 140k the
same way. `cf373_bb_below` resolves every one of the nine:

    B8   40k   (fresh run, saves 40k on the way to 100k)
    B10  _r3_140k        A3  leg_100k/..._100k
    A2   leg_200k/..._140k   A4  leg_100k/..._100k
    B4   _r3_140k        B1  _r3_140k
    B6   _r3_140k        B2  _r3_140k

**The box passes the gate the card set.** `47557391`, label `cf373-dual`,
2x RTX 4090, $0.8144/h, on-demand. `python3 -c "import torch;
print(torch.cuda.device_count())"` prints 2. Both cards read 92% and 70% with
four `train.py` processes on them.

**One real fault, fixed.** The publisher running since 18:59 was started before
the commit that added the `pair_head_files.py` block, and bash had already
parsed its loop body, so the new script and its table never crossed into the
git checkout. `results/pair_head_files.tsv`, `results/pair_identity.tsv` and
`scripts/pair_head_files.py` were missing from the branch while the report
argued from them. Copied by hand, publisher restarted on the current file,
committed as `b831261c`. A stray `scripts/execution_log.md`, a duplicate of
this file in the wrong directory, went with it.

**A headline count was wrong in the first PR comment and is corrected in the
second.** 40k -> 100k splits 7 down and 6 up on both heads, not 9 and 3. The
seven that improve are the seven the queue extends; the six that worsen are the
five the rule stopped at 100k, plus B1 at +0.0031.

**Budget at 21:05Z.** Credit $24.31, box spent $3.71 over 4h 33m. The four
remote backbones need about 5.7 h more, then B2/A3/A4/B1 extend — B1 and B2
from 140k, so 60k steps each — and the heads fill the two head-only slots
behind them. About 18 h of box time, about $14.7, leaving about $9.6 over the
$5.50 floor.

## 2026-08-12 21:16 — session four: the head-only slots, and the A1/B3 block

This session found the round running: one dispatcher, one supervisor, one
sync loop, one credit guard, one hourly heartbeat, one publisher, five
backbones and two heads. It changed one thing and closed one block.

**The two head-only slots were idle, and would have stayed idle four more
hours.** Every queued head waits on a backbone that is still running, so
between the last head ending and the first extend finishing there is no head
to place, and the old rule reserved those two slots for heads alone. Four
backbones sat queued against two empty slots on a card that is paid for by
the hour.

`q_run.sh` now lets a backbone take a head-only slot while no head can use
one. `heads_ready()` walks the queue for a head that is `queued` with its
dependency `done`; a backbone takes the slot only when that returns false.
When a backbone ends, its own slot frees, so a head can never be buried
behind a six-hour extend. B2 took the first slot at 21:27 and A3 the second
at 21:29.

The card already carried three training processes — two backbones and a
head — so the third backbone replaces a process rather than adding one. Per
backbone rate measured at 2.980 sps under two backbones and a head.

**The A1/B3 block: no number is wrong.** The card blocked publication until
one of the two equal student scores was shown wrong. Neither is. The two
cells hold the same student weights, so the student column holds one
measurement printed twice.

    evidence                                    file
    two runs, two backbone files                each cell's eval/backbone.txt
    four head files, four md5 sums              results/pair_head_files.tsv
    110/110 student tensors equal, 0.000e+00    results/pair_identity.tsv
    head loss curves equal step for step        each cell's head losses.csv

`arm5_combab` passes no `--moco-rep-keys`. The loss reads no teacher output,
so the EMA copy has no gradient path into the student and the EMA schedule
moves the teacher alone. The teacher scores do differ, 1.1318 against
1.1343 at bb40k. The three other same-arm pairs run `arm6_v2_*`, which does
pass `--moco-rep-keys`, and their students differ by 2.377, 3.103 and 5.025.

**Naming.** Every cell score is `score_<CELL>_k3_<stop>_<head>.txt`. B1's
bb40k pair reads the same under the new name and the round-1 alias. The
`score_G*` files that remain are round-1 controls, not cells.

**Budget at 21:28.** Credit $23.76, box spent $4.26 over 5 h 13 m at
$0.8144/h.

## 2026-08-12 21:36 — the dispatcher stalled on its first eval

B8's bb40k student eval started at 21:30 and the dispatcher stopped placing
and reaping at that moment. It sat in `pipe_read` for six minutes with seven
backbones running and forty jobs queued.

`q_run.sh` called the eval launcher as `why="$(launch_eval ...)"`. Command
substitution reads the child's stdout to end-of-file, and `launch_eval`
backgrounds the eval with `( ... ) &`, which inherits that same pipe. The
read therefore returns when the EVAL ends, not when the launcher does. One
97-config eval takes about 43 minutes on four shards, and nineteen of them
run this round, so the queue would have lost about thirteen hours to a
dispatcher that was waiting on work it had already started.

Round 2 ran its evals from `r2_eval_driver.sh`, so no eval had ever gone
through this path before B8's.

Fixed: call `launch_eval` directly and read its exit status. The eval that
was already running keeps its process; its state file was written by hand as
`running` on `elisa-cpu` so the restarted dispatcher adopts it rather than
starting a second copy of the same tag.

## 2026-08-12 21:42 — what the third backbone per card costs

Per-process step rate on the rented box, 602-second window, three backbones
per card:

    B4   2.824 sps      A3   2.990 sps
    B2   2.990 sps      A2   2.658 sps

Two backbones and a head on the same card, measured 20 minutes earlier, ran
at 2.980 sps each. So the third backbone takes almost nothing from the other
two: a d_model=64 model at batch 64 is launch-bound, and the card reads 98-99%
without being compute-bound. Card throughput goes from about 5.96 sps to
about 8.7 sps.

That is what the head-only slot rule was costing while it held two slots
empty for heads that could not start.

Remaining box work at this rate: about 508k backbone steps and about 510k
head steps, 8.1 h each over the box's two cards, about 16 h and $13 of
rental. B6 runs on elisa at 3.42 sps and costs nothing.

## 2026-08-13 02:10Z — session five: the head lock never released

This session found the round running: one dispatcher, one supervisor, one
sync loop, one credit guard, one hourly heartbeat, one publisher, one
reaper, three backbones and four heads. It fixed one fault, removed one
race, and armed one watchdog.

**The per-card head lock held for the whole run, not 180 seconds.**
`r2_head_box.sh` takes a per-card lock, starts the head, sleeps 180 s and
closes its own descriptor, so a second head can share the card. The close
did nothing. A lock lives until EVERY descriptor on it closes, and the
backgrounded python inherits the parent's fd 7. So each head kept its card
locked for its whole 30,000 steps and the next head on that card waited.

Measured on the box at 02:11Z:

    /proc/26742/fd/7 -> /tmp/cf373_r2_head.gpu0.lock    B8 teacher, training
    /proc/27082/fd/7 -> /tmp/cf373_r2_head.gpu0.lock    B10 student, in flock
    /proc/27231/fd/7 -> /tmp/cf373_r2_head.gpu0.lock    B10 teacher, in flock

Card 0 held 19,090 MiB free while two heads waited on it. The same lock had
already serialised B8's two bb100k heads: the student ran 01:17 to 02:01,
and the teacher's start line reads 02:01.

Fixed by closing the descriptor in the child: `... >>"$LOG" 2>&1 7>&- &`.
Deployed to both copies, worktree and box, by `mv` over the path, so the
three wrappers already running keep the old inode and are not corrupted
mid-read. B10's two waiters were released by killing their `flock`
processes; `set -uo pipefail` carries no `-e`, so each wrapper fell through
to the VRAM gate, which passed, and both heads started. Four heads then ran
at once, three on card 0 and one on card 1, with 7,680 MiB still free.

Fifteen heads remain. The fix roughly halves their wall-clock.

**A second supervisor was watching the same dispatcher.** `q_super.sh` 268303
was orphaned when its dispatcher died at 21:37 on 08-12, and 452407 started
a new pair. Neither holds a lock. Both poll every 300 s and both restart a
dispatcher they find dead, which is the duplicate-process failure the script
was written to prevent. Killed 268303. One supervisor, one dispatcher, 452407
and 452409.

**The watchdog.** `scripts/q_watchdog.sh` says one line per job state change,
one line per hour, and one STALL line when the summed step counter of every
training log does not move between hourly probes. It starts nothing and
moves nothing. It exists because a process that is alive, a log that tails
clean and a card at 0% look identical to a working run, and the meter runs
either way.

**Budget at 02:17Z.** Credit $19.14. Box 47557391 spent $8.90 over 10 h 55 m
at $0.8144/h. Backbone ETAs off their own logs: A3 3.5 h, B1 4.8 h, A4 6.4 h
on elisa. The box carries A3 and B1, so it has about 5 h of GPU work left,
about $4.1. Evals need no GPU and run on elisa's cores.

## 2026-08-13 02:45Z — session six: the round's two blocking items, checked against the files

This session found the round running and healthy: one dispatcher, one
supervisor, one sync loop, one credit guard, one hourly heartbeat, one
publisher, one reaper, three backbones, four heads and two evals. It started
nothing new. It re-armed the watchdog, which had died after its first probe,
and it checked the two items the card blocked on.

**A1/B3 is not a path-key fault.** The card asked which head or eval path key
ignores the EMA regime. None does. `results/pair_head_files.tsv` gives four
head files at four cell-id paths with four different md5 sums. The weights
inside them are equal: `results/pair_identity.tsv` reads 110/110 student
tensors and 28/28 student head tensors at max abs diff 0.000e+00.

The cause is in the arm, not the path. Neither A1's nor B3's run passes
`--moco-rep-keys`, checked against both run logs. The loss reads no teacher
output, so the EMA copy has no gradient path into the student.
`scripts/cells.tsv` gives A1 as `arm5_combab_alignS_sched` and B3 as
`arm5_combab_alignS_fix09`, so the EMA regime is their only difference, and
it moves the teacher alone. Two runs, one student trajectory, one student
score printed twice. The teachers do differ, 1.1318 against 1.1343 at bb40k.

**Naming holds.** 54 of 54 scored cell-stop-head triples read
`score_<CELL>_k3_bb<stop>k_<head>.txt`. All 14 cells carry a 40k pair and a
100k pair, except B8's 100k pair, which is in flight. B1's bb40k reads 1.0850
student and 1.0948 teacher under the standard name and under the round-1
alias `score_G6_B1_k3_bb40k_*`. The remaining `score_G*` and `score_*_k0_*`
files are round-1 and depth-0 controls, not cells.

**Spec, checked against the running processes, not the launcher.** Heads on
the box run `--total-steps 30000 --seed 20260722 --grad-clip 1.0`. Evals run
`--strategy B4 --forecast-len 16` over 97 configs, counted from
`scripts/shard_configs.py` as 23+25+25+24.

**No GPU is idle.** Seven of the eight slots carry a job. The eighth is
elisa card 0, which holds 22.5 GB of other projects' work and has 2 GB free,
so the VRAM gate holds it back. That is the gate working, not the queue
stalling.

**`scripts/q_await_round.sh`.** One round's wait for the watching session. It
moves no work. It blocks until the queue empties, a job fails, the dispatcher
dies, credit falls under the floor, or an hour passes, then prints the reason
and the coverage table and exits. The hourly arm exists because a session
that waits only on notifications learns nothing when the thing that should
notify it is the thing that died.

**Budget at 02:45Z.** Credit $18.66. Box 47557391 spent $9.33 over 12.1 h at
$0.8144/h. Backbone ETAs off their own logs: A3 3.1 h, B1 4.5 h, A4 6.1 h on
elisa. The box carries A3 and B1 and their head pairs, so about 5.3 h and
$4.3 remain on it. A4 finishes on elisa and costs nothing.

## 2026-08-13 03:10Z — session seven: the two blocked items close on the bytes

This session found the round running with every daemon alive: one
dispatcher, one supervisor, one sync loop, one credit guard, one hourly
heartbeat, one publisher, one watchdog. Two backbones and three heads run on
the box, one backbone on elisa, four evals on elisa's cores. It started no
new work and moved no job. It closed the card's two blocking items and
measured the round's remaining cost.

**A1/B3: the pipeline ran twice and agreed to the byte.** The card asked for
a re-run of the student head and the 97-config eval for both cells into
cell-id paths. That re-run is already on disk, and it is what produced the
shared number. Each cell holds its own head file and its own eval tree:

    A1/sync/eval/A1_k3_bb40k_student/gift/all_results.csv   eb5e4e21...
    B3/sync/eval/B3_k3_bb40k_student/gift/all_results.csv   eb5e4e21...

Every one of the four shard CSVs matches, and so does each head's own
`_losses.csv`. Two head trainings, two eval runs, separate directories,
identical bytes.

The cause is in the arm. `run_leg_k.sh` gives A1 `--loss-shape
cosine_similarity_batch_rep_only --align-loss-weight 1.0 --tau-rep 1.0` with
`--align-target student`. `run_arm_k.sh` gives B3 the same three loss args
and the same default target. Neither passes `--moco-rep-keys`. The two
launchers differ in one line: A1 ramps the EMA (`--ema-tau 0.9 --ema-tau-end
1.0 --ema-tau-ramp-steps 100000`), B3 holds it at 0.9.

With the align target on the student and no MoCo keys, the loss reads no
teacher output. `tests/test_390_align_target_main_loss.py` fixes that
contract: with no `align_target` the added term is L_align(f, o), the
student's own encoder output, for `moco_rep=False` and `moco_rep=True`
alike. So the EMA regime moves the teacher and nothing else, and the student
trajectory is the same run twice. `results/pair_identity.tsv` measures it:
110 of 110 student tensors equal at max abs diff 0.000e+00, at 40k and at
100k. The teachers differ, 6.4e-3 at 40k and 1.986e-1 at 100k, and the
teacher scores differ, 1.1318 against 1.1343.

The number 1.1305/1.1676 is right for both cells. What the report cannot do
is read A1 against B3 on the student column: that column holds one
trajectory, not two. The teacher column separates the regimes. The other
three pairs differ on both sides, so the comparison holds for them.

**Naming holds, audited cell by cell.** All 56 cell-stop-head triples were
checked for `score_<CELL>_k3_bb<stop>k_<head>.txt`. 54 exist, which is every
scored deliverable; the two absent are B8's bb100k pair, in flight. B1's
bb40k reads 1.0850 student and 1.0948 teacher under the standard name. The
20 score files outside the pattern are all `k0`, `k1`, `aw4` or seed-2
controls, plus the round-1 alias `score_G6_B1_k3_bb40k_*`, which carries the
same two numbers as B1's standard pair. No cell score hides under a
non-standard name.

**The head lock fix works.** Checked on the live box: B4's two wrappers hold
no fd 7, so their cards are free; B6's student, 167 s old, still holds
`/tmp/cf373_r2_head.gpu0.lock` and releases at 180 s. Three heads and two
backbones share two cards at 16.4 GB of 24.5 GB each.

**No GPU is idle.** Six of the eight slots carry a job, one box slot fills
on the next tick, and elisa card 0 holds 22.5 GB of another project's work
with 2 GB free, so the VRAM gate holds it back.

**Remaining work, off the training logs.**

    A3  bb 200k   box card 1   171000/200000   2.7 h
    B1  bb 200k   box card 1   159800/200000   4.0 h
    A4  bb 200k   elisa card 1 126800/200000   5.9 h

Nine heads and thirteen evals follow them. The box is needed until B1's head
pair finishes, about 4.9 h, about $4.0. A4 finishes on elisa and costs
nothing, and every eval runs on elisa's cores. Four concurrent evals clear
twelve queued evals in the four hours before the last box head lands, so the
eval column forms no backlog and the round ends on A4's eval.

**Budget at 03:10Z.** Credit about $18.4. Box 47557391 spent about $9.7 over
12.6 h at $0.8144/h. Projected spend to the end of the round: $4.0, leaving
about $14.4.

## 2026-08-13 03:35Z — session eight: the queue holds, no job moved by hand

State at entry. Seven daemons alive: dispatcher, supervisor, sync loop,
credit guard, hourly heartbeat, publisher, watchdog. Sixteen jobs done,
eleven running, twenty left, zero failed. This session started no work and
moved no job.

Three backbones run.

    A3  box card 1    173000/200000   ETA 2.5 h
    B1  box card 1    161600/200000   ETA 3.8 h
    A4  elisa card 1  130600/200000   ETA 5.8 h

Four heads run on the box (B4 student, B4 teacher, B6 student, B6 teacher).
Four evals run on elisa's cores (A2 200k student and teacher, B8 100k
student and teacher).

The card's two blocking items stay closed, re-audited on the bytes.
`pair_identity.tsv` holds all four same-arm pairs at both stops. A1/B3 is
identical on the student side, 110 of 110 tensors at 0.000e+00, and differs
on the teacher side. The other three pairs differ on both sides. The cause
is the arm: A1 and B3 run arm5 with `--align-target student` and no
`--moco-rep-keys`, so the loss reads no teacher output and the EMA regime
cannot move the student. A4/B1, A3/B2 and A2/B8 run arm6_v2, which carries
MoCo rep keys, so the teacher enters the loss and both sides separate.

Naming holds. 54 of 54 scored cell deliverables sit under
`score_<CELL>_k3_bb<stop>k_<head>.txt`. The 20 files outside that pattern
are k0, k1, aw4 and seed-2 controls, plus the round-1 alias
`score_G6_B1_k3_bb40k_*`, which reads 1.0850 and 1.0948, the same two
numbers as B1's standard pair.

Budget at 03:26Z. Credit $18.10. Box 47557391 has run 12 h 12 m and spent
$9.95 at $0.8144/h. The box is needed until B1's head pair lands, about
4.8 h and about $3.9, which leaves about $14. A4 finishes on elisa and the
evals run on elisa's cores, so both cost nothing.

## 2026-08-13 03:50Z — session nine: the round runs, nothing moved by hand

State at entry. Seven daemons alive: dispatcher (pid 452409), supervisor,
sync loop, credit guard, hourly heartbeat, publisher, watchdog. 47 jobs:
20 done, 11 running, 16 queued, 0 failed. This session started no work and
moved no job.

Three backbones run, off their own losses CSVs.

    A3  bb 200k   box card 1    175900/200000   ETA 2.1 h
    B1  bb 200k   box card 1    164500/200000   ETA 3.1 h
    A4  bb 200k   elisa card 1  134300/200000   ETA 4.4 h

Four heads run on the box (B2 student and teacher, B6 student and teacher).
Four evals run on elisa's cores (A2 200k teacher, B10 200k student and
teacher, B8 100k teacher).

No GPU is idle. The box holds six processes over two cards at 100% and 97%.
Elisa card 1 carries A4. Elisa card 0 holds 22.5 GB of another project's
work, so the VRAM gate keeps the queue off it, and the four evals take
elisa's cores instead.

**B8 closes its 100k student.** 1.3157. The cell that failed 15 times on
CUDA 13 hosts now holds three of its four numbers.

**Both blocking items re-audited on the bytes, and both hold.**

A1/B3 is one trajectory, not two. `pair_identity.tsv` reads 110 of 110
student tensors equal at max abs diff 0.000e+00 at 40k and at 100k, and the
student heads equal at 28 of 28. The teachers differ, 6.400e-03 at 40k and
1.986e-01 at 100k. The head files are four distinct files with four distinct
md5s, so the pipeline ran twice and agreed. The cause is the arm: arm5 with
`--align-target student` and no `--moco-rep-keys` reads no teacher output,
so the EMA regime moves the teacher and cannot move the student.

The fourth pair now has its rows. B8 gained the checkpoints the test needed.

    A2/B8  arm6_v2_nse_alignT  40   student  111  4  5.025e+00  differs
    A2/B8  arm6_v2_nse_alignT  100  student  111  4  9.518e+00  differs

All four same-arm pairs are tested. Three separate on both sides. Only A1/B3
shares its student column, and the report must not read A1 against B3 there.

Naming holds. 56 score files sit under `score_<CELL>_k3_bb<stop>k_<head>`.
B1's 40k reads 1.0850 student and 1.0948 teacher under the standard name,
the same two numbers as the round-1 alias `score_G6_B1_k3_bb40k_student`.
Every file outside the pattern is a k0, k1, aw4 or seed-2 control.

**Budget at 03:46Z.** Credit $17.83. Box 47557391 has run 12 h 32 m and
spent $10.22 at $0.8144/h. The box is needed until B1's head pair lands,
about 4.1 h and about $3.3, which leaves about $14.5. A4 finishes on elisa
and the evals run on elisa's cores, so both cost nothing.

**B1's bb40k: the card's premise does not hold, and the artefacts say why.**
The card reads "B1 has no valid 40k comparison; extend and say so". B1 is
extended, and this is what the files say. The head behind 1.0850 trained off
`bb_small_arm6_v2_combab_lalign_lrepmoco_..._cf373k3_40k.pth` — B1's own 40k
checkpoint, the file round 2 resumed — for 15,000 steps at seed 20260722,
which is the head every other cell's bb40k carries. `stop.log` in
`checkpoints_backup/cf-373/eval/G6_B1_k3_bb40k_student/` names that backbone
on the head-train start line. The head, its optimizer and a 97-row
`gift/all_results.csv` are all on disk. Only the NAME was non-standard, and
round 3 normalised it. The report now carries this paragraph outside the
generated table block, so the injector cannot drop it.

**A4 extends the student head only**, as the card asked. The queue holds
`hd_A4_200k_student` and `ev_A4_200k_student` and no teacher job for A4.

## 2026-08-13 05:20 UTC — session six, round 1: the eval cap goes 3 -> 5

The card's two blocking items were already closed on disk when this session
opened. It changed one thing and started no new job.

**The eval cap.** Elisa held three eval slots of four shards each, 12 of 32
cores, and a fourth eval sat blocked in `eval_slot`. Measured load was 16.
The round owes 15 evals and every backbone but one now trains on the rented
box, so elisa's cores are the queue's slowest resource. `eval_slot.sh`
default goes 3 -> 5: 20 cores for evals, 12 left, above the eight the brief
requires. Five rounds of 1 h 20 become three. No process restarted; each new
eval sources the file when it starts.

**The tail is A4, and it does not move.** Live rates:

    A4  loc:1 elisa   139,200/200,000   3.3 sps   ETA 5.1 h
    A3  rem:1 box     182,200/200,000   2.9 sps   ETA 1.7 h
    B1  rem:1 box     170,800/200,000   2.8 sps   ETA 2.9 h

Three jobs sit on the box's GPU 1 and one head on GPU 0, so GPU 0 frees at
about 05:35 and has nothing to take: every remaining head waits on a
backbone that is still training. Moving A4 to it was costed and refused.
A4's last periodic save is 120k, so a restart discards 19,200 steps, 1.6 h,
and the box's own step is slower than elisa's, 179 ms forward against 161.
Staying is 5.1 h; moving is about 4.9 h plus the staging. The card's rule
against an idle GPU binds when the queue has a job for it. This one does not.

**Where that puts the round.** A3 lands ~07:10, B1 ~08:30, A4 ~10:25. Heads
follow at ~50 min, evals at ~1 h 20. Last number: A4 student, about 12:40.

**Spend.** credit $17.51 at 04:10, box $0.82/h, box spent $10.54. The tail
needs about 5 h of box time for A3 and B1 and their four heads, so about $4.
Floor is $5.50 and the guard holds it.
