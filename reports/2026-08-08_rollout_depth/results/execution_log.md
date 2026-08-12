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
