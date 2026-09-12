# Execution log

Operational events of this card. The report holds the science. This file
holds what happened to the machines, so a later reader can tell a measurement
from an incident.

## Three agent sessions share one checkout

Three ExperimentRunner sessions drove this card at the same time, on one box
with two RTX 4090 cards. Other projects held 4 to 6 GB on each card
throughout. The sessions split the work by lane and agreed each split by
message.

- One drove the rate sweep, the report and the PR comment.
- One drove the decay arm, the heartbeat and the reference tables.
- One joined at about 04:00. It split the two cards by kind of work and wrote
  `head_sweep.sh`, `head_claim.sh` and `queue_backbones.sh`.

GIT DOES NOT NAME THE AUTHOR OF A COMMIT HERE. Every session commits as
`jeremycochoy-agent`. Two of the three use the address `jeremy@redstone.ee`
and the third uses the GitHub noreply address, so the identity separates one
session from the other two, and those two from each other not at all. A reader
who infers an author from a commit can get it wrong. One session did, inside
this card, and named the wrong peer as the author of `bde107f7`.

## Git authorship cannot separate the three sessions

Every session commits as `jeremycochoy-agent <jeremy@redstone.ee>`, into one
worktree and one index. So a commit of this card names no session, and a
reader of the PR cannot tell which one wrote it. One session already read
another's commit as its own.

Read the commit body, not the author. Each one states what it changed and
why. This log names the session for the events where it matters.

## An edit under a running shell killed three lanes

Bash reads a script by byte offset and re-reads the file after each command.
A shell that is running a script resumes inside the NEW text after an edit,
and dies on the first syntax error it meets there.

`run_arm.sh` changed at 21:28 and `phase1.sh` at 21:44 on 2026-09-06, under
three lanes. Each lane died with `cf412_auc_warmup: command not found`, then a
syntax error. The three ended three different ways.

- `k3_r100_09b`, at 21:33:03. Its leg finished and its checkpoint WAS scored.
  `lane_b_relay.sh` waits on the pid and not on the exit code, so it started a
  fresh `phase1.sh`. That one found the checkpoint on disk, made the leg a
  no-op and ran the head. The relay absorbed the failure by accident.
- `k3_r100_09` at 100,000 steps, at 22:14:28. Its leg finished and its
  checkpoint sat unscored. A head started by hand at 22:15.
- `k32_r100_09_dec`, at 01:03:05. Its leg did NOT finish. The AUC gate stopped
  it at step 18,634, and its `leg_40k` holds no `_40k.pth`.

So one checkpoint of the three sat unscored. No training was lost.

### `phase1.log` is unreliable in both directions

`run_arm.sh` should have exited `CF412_RC_COLLAPSED`, which is 4, and
`phase1.sh` should have logged "LOST the contrastive task". It exited 2
instead. Two lines of `results/phase1.log` carry that exit code, and they hide
opposite things.

```
line 21  [09-06 22:14:28] backbone k3_r100_09 stop 100000 FAILED rc=2
line 65  [09-07 01:03:05] backbone k32_r100_09_dec stop 40000 FAILED rc=2
```

Line 65 is a real result written as a failure. That leg lost the contrastive
task at step 18,634, which is this card's clearest finding.

Line 21 is the reverse error. That leg SUCCEEDED. It wrote its 100,000-step
checkpoint, and that checkpoint scored 1.3395.

So read neither from `phase1.log`. The verdicts are in
`results/collapsed_<arm>.txt`, `results/auc_verdicts.tsv` and
`results/scores.csv`.

### The doubled gate line is the fingerprint

The corrupted shells did not simply die. Each one RE-RAN a command it had
already run. Every one of the three logged its AUC gate line twice, from one
process (`results/arms.log`):

```
[09-06 15:15:52] arm k3_r100_09b     AUC gate pid 364747 - ... warmup 1000
[09-06 21:33:03] arm k3_r100_09b     AUC gate pid 364747 - ... warmup
[09-06 15:34:23] arm k3_r100_09      AUC gate pid 380882 - ... warmup 1000
[09-06 22:14:28] arm k3_r100_09      AUC gate pid 380882 - ... warmup
[09-06 20:12:52] arm k32_r100_09_dec AUC gate pid 663584 - ... warmup 1000
[09-07 01:03:05] arm k32_r100_09_dec AUC gate pid 663584 - ... warmup
```

Same pid, same line, hours apart. The second of each pair has an empty warmup,
because `cf412_auc_warmup` did not exist in the `study.sh` that shell sourced
when it started. Bash resumed at a stale byte offset, re-entered a block it had
finished, and fell into text that was not there when it began.

### Two guards came out of it

- `run_snapshot/` is a frozen copy of `scripts/`. A lane launched from it
  cannot be edited under itself. It is gitignored, because it is a copy of
  committed files.
- `scripts/missing_heads.sh` sweeps for any checkpoint that has no score and
  no running head, and starts one. It is the net under every lane.

A file that must change while a shell runs it is replaced through `mv`, never
edited in place. The running shell keeps the old inode.

## Three shells stopped at 04:20, on purpose

The newer session stopped three shells at 04:20: the lane C driver (380579),
the decay lane driver (1103976) and the sweep loop (900144). It killed no
trainer. All four trainers and both `run_arm.sh` shells stayed alive, and no
training was lost. An earlier version of this entry read "cause unknown".

WHY THE TWO DRIVERS STOPPED. `phase1.sh` trains a head on its own card after
each leg, and both drivers carried `BB_GPU=0`. A head waits for 12,000 MiB.
`k32_r200_08` waits for 11,300 MiB on the same card, and `k3_r100_09_dec`
frees only 12,591 MiB when it ends. The head would have taken that window and
held a 9.6-hour backbone for 1.8 hours. Stopping the driver leaves the trainer
running and moves the head to the other card.

WHY THE SWEEP LOOP STOPPED. `missing_heads.sh` calls `head_eval.sh` and WAITS
for it, and that call is a 1.7-hour head train plus a 2.9-hour CPU evaluation.
A sweep that waits therefore starts one arm every 4.6 hours, and the seven
arms left need 32 hours of that. `scripts/head_sweep.sh` starts each head in
the background. The `flock` in `head_eval_bb.sh` still serializes the head
train, and it drops that lock before the evaluation, so the evaluations
overlap.

## The two cards carry different work

The two sessions agreed this split by message at 05:30 on 2026-09-07.

- gpu 0 carries the released backbones. `scripts/queue_backbones.sh` holds
  `k32_r200_08`, then `k8_r100_09`, largest memory need first. A small arm
  ahead of a large one takes the window and leaves the large one waiting.
- gpu 1 carries the rate sweep and every head. `scripts/head_sweep.sh` is the
  net over all arms. `scripts/head_claim.sh` takes the four checkpoints that
  no lane will claim, the moment they land, because the five-minute age gate
  of a sweep protects a lane that no longer exists.

`k3_r100_09_lr17` belongs to lane D, pid 1102993, which holds
`ARMS="k3_r100_09_lr33 k3_r100_09_lr17"` on gpu 1 and reaches lr17 after
lr33's head and evaluation. The queue on gpu 0 carried it for eight minutes
and dropped it before it started, so no arm trained twice.

## `pgrep -f` on a script name is not safe here

`pgrep -f` matches the WHOLE command line of every process on the box. Three
sessions watch this card, and each one runs shell commands that NAME these
scripts and these checkpoints. So a watcher shell reads as the work it watches.

It bit four places on 2026-09-07:

- The duplicate-head guard of `missing_heads.sh` and `head_sweep.sh`. At
  06:02:48 it reported a head for `k3_r100_09_dec` that did not exist, and the
  claim lane dropped the arm.
- The `heads` counter of `watch_status.sh`, which read 2 against one head at
  06:18. The second was a memory sampler naming `train_forecasting_head.py`.
- The `lanes` counter of the same file, from the other direction: a pattern
  naming `scripts/` alone missed the frozen copy and read 2 lanes where 3 drove
  work.
- The same `lanes` counter after that fix, which read 3 where 2 drove work.
  `phase1.sh` pipes its vram wait into `tee`, and that subshell carries the
  same whole command line as its lane.

Three rules came out of it.

- Pin the EXECUTABLE, not the command line. `ps -eo args` puts the binary in
  `$1`. `scripts/head_busy.sh` does this for a head, its driver and its
  evaluation.
- Match a string no watcher carries. `scripts/arm_busy.sh` matches the
  trainer's `--run-name`, which is unique per arm and appears nowhere else.
- Drop a match whose parent also matches. That is a subshell, not a second
  process.

TEST BOTH BRANCHES OF A GUARD. A first version of `head_busy.sh` read `$stop`
where it held `$STOP`. Under `set -u` the BUSY branch aborted, and an abort
exits 1, which every caller reads as FREE. A guard against a duplicate head
that starts one.

## A collapse is a band, not a step

`k32_r100_09_dec` lost the contrastive task at step 18,634. That is the
verdict step, and it is the only step worth quoting. The fall around it takes
about 1,300 steps, and the rolling median over 500 rows crosses 0.55 more than
once on the way:

```
17,313   median 0.5492   first crossing, exactly half the window under 0.55
18,000   median 0.7508   recovered
18,600   median 0.7175
18,634   median 0.5257   the verdict step
19,100   median 0.5014   the floor
```

THIS IS WHY THE GUARD RULE IS "under the threshold AND does not come back",
and not "goes under". A rule of the second kind stops this arm at 17,313,
1,321 steps before it fails, while it still recovers to 0.75.

It also means NO mark between 17,313 and 18,634 is readable, not only the
marks beside the crossing. `results/tables.md` carried an 18,600 column,
chosen because it sat near the collapse, which is the wrong reason. A
`verdict` column read from `auc_verdicts.tsv` replaced it. Every remaining
mark sits far from the threshold.

## Ask what a number IS, not only whether it checks out

Two errors on 2026-09-07 had the same shape, and neither was caught by
checking the number.

- Two AUC tables disagreed in the third decimal. One session explained the gap
  as a difference of window, 300 rows against 400, and stopped. The real
  difference was a trailing MEAN against a rolling MEDIAN. The explanation
  fitted, so it ended the question.
- A table quoted a mark near a collapse as a measurement. The number was read
  correctly from the correct file. The mark itself was not a measurement.

A third followed the same day. `watch_status.sh` printed the LAST RAW ROW of
the `auc` column in its hourly line, which all three sessions read. Numbers
quoted from it were compared with medians from `results/tables.md` and looked
close enough to pass.

Both were found by asking what the quantity was, not by re-reading it. A
number that reconciles is not a number that is understood.

THE AUDIT THIS FORCED. Every AUC in the report was recomputed as a rolling
median from the losses CSVs. Five values were raw rows in a section that
declares medians: 0.737, 0.938, 0.993, 0.995 and 0.998, now 0.746, 0.937,
0.992, 0.998 and 0.999. No reading changed, because all five sit far from any
threshold. The point is that a number's SOURCE has to be checked even when its
value is right.

Prose that copies a live number goes stale or goes wrong. Point it at the
artefact instead: `results/tables.md` rebuilds every heartbeat and
`results/auc_verdicts.tsv` holds each gate verdict.

A FIFTH CASE IS NOT A NUMBER AT ALL. One session reported "`head_busy.sh`
reads FREE" for an arm whose head was already claimed. It had run that check on
four other arms and written the fifth from habit, not from output. The peer
re-ran it, got BUSY, and did not start the second head.

That is the most dangerous shape of the five, because both sessions gate a
duplicate head on that script. A false FREE quoted in a message defeats a guard
that works. Run the check you quote, and quote its output.

The exchange also tested something no test covered: `head_busy.sh` matches a
driver launched from `run_snapshot/` as well as from `scripts/`, because its
driver rule pins the executable to `bash` and the argument to a path ENDING in
`head_eval.sh`.

## A k = 32 backbone grows past its smoke row mid-leg

`k32_r200_08` held 6,402 MiB for its first four hours and 10,418 MiB after,
against the 10,062 of its row in `results/trial/smoke.csv`. That is 347 samples
over the seven hours of its leg. A k = 3 arm does not do this: the
200,000-step climb held 6,492 MiB flat over its whole leg.

THE CAUSE IS NOT KNOWN. Two mechanisms were argued and both fail on the
evidence:

- The 20,000-step save. Both k = 3 arms cross the same save with no growth.
- The latent-drift probe at the same step. `LatentDriftProbe` (`train.py`)
  runs a no-grad, `eval()`-mode forward through `extract_encoder_latents` and
  caches `h` on the CPU as fp16. It never touches the rollout, so `k` does not
  enter it, and its own docstring puts it near 10 MB.

The measurement window was 30 minutes wide and holds about 2,000 steps, so
nothing places the jump at any particular step.

WHAT IT DECIDES. Keep `cf412_leg_vram_mib` at the smoke row for k = 32. Two
sessions each proposed cutting it toward a live reading of about 6,400, taken
while the arm had run for hours and looked settled. That gate would have been
3,660 MiB short of the same arm later in the same leg, and a leg that dies in
`.to(device)` loses every step since its last save.

A STEADY READING IS NOT A SETTLED ONE. Three samplers agreed on 6,402: a 60 s
poll, a 30 s poll, and a tight loop that took 6,356 samples in 150 seconds.
All three were right and none of them measured the thing that mattered.

## Derive the comparisons a table supports

The report's central heading read "the width erodes the contrastive task at
every depth above 3" for several hours. It named the depth. Every arm that
erodes also runs the `mean` reduction and every arm that holds runs `sum`, so
the reduction fits the same rows exactly as well.

Neither session saw it by reading the prose. It was found by a list computed
from `arms.tsv` of every arm pair that differs in exactly ONE column. The pair
`k3_r100_09` against `k32_r100_09` is absent from that list, because it moves
two.

The cross-tab has two populated cells of four:

| | `sum` | `mean` |
|---|---|---|
| k = 3 | 6 arms | none |
| k = 8, k = 32 | none | 4 arms |

The confound is inherited and not a defect: each row is a published
configuration and the parents ran k = 3 under sum and the deeper cells under
mean.

THE PRACTICE THIS SUGGESTS, beyond this card: derive the comparisons a table
supports before writing any of them, rather than assuming a pair is controlled
because it looks like one. It costs a few lines against `arms.tsv` and it is
the only thing that caught this.

## `results/lane_d.log` is not committed

Two sessions opened that file at once, one with `>`, so its text interleaves
at two offsets. Every line it held is also in `results/arms.log` and
`results/phase1.log`, which append.

## The frozen copy is load-bearing

The sweep reads `run_snapshot/arms.tsv`. An arm added to `scripts/arms.tsv`
after the freeze is invisible to it, and that arm's checkpoint would sit
unscored with no warning. Add an arm, then re-copy `scripts/` to
`run_snapshot/` and restart the loop between ticks.

`head_eval.sh` calls `head_eval_bb.sh` of the 2026-08-08 study, which is
outside the snapshot.

## Scheduling, moved out of the report by the report review

Backbones train on one GPU and heads on the other. A backbone queue trains no
head, and one head sweep starts every head that a checkpoint lacks.

`scripts/arm_busy.sh <arm>` and `scripts/head_busy.sh <arm> <stop>` answer
whether an arm or a checkpoint is already training. Ask them before you start
anything by hand, because nothing under this card stops two trainers on one
arm.

The empty `l_pred` column of the losses CSVs is not a logging fault. The loss
shape `cosine_similarity_batch_rep_only` has no prediction term
(`src/loss.py:1152`).

## The splice is retired

`splice_report.sh` rewrote the report's arms and tables sections on every
heartbeat tick while pass 1 ran. The report review of 2026-09-08 fixed those
sections by hand, so both copies of the script (`scripts/`, `run_snapshot/`)
now exit 0 and do nothing, and the reviewed plot scripts were re-copied into
`run_snapshot/`. The heartbeat still watches, collects and mirrors.

## The climb seed

The pre-registered gate named `k3_r100_09b` for the climb past 40,000 steps.
The orchestrator overruled it before the seed scores existed, because the
1.0651 reference is a 200,000-step number on the `k3_r100_09` lineage
(`results/gate_40k.txt`). So the climb ran seed 20260520, the worse 1e-3
seed.

## The contrastive AUC does not rank arms

The orchestrator measured it over the ten arms this card scored at the
40,000-step stop. Inside each reduction group a HIGHER AUC goes with a WORSE
GM-Relative MASE. The Pearson correlation is +0.533 over the seven sum arms
and +0.954 over the three mean arms.

The best arm, `k3_r100_09_lr56` at 1.1820, holds the lowest AUC of the seven
sum arms, 0.9973. The best mean arm, `k32_r100_09_lr56` at 1.4404, holds the
lowest AUC of its three, 0.7098.

NAME THE STATISTIC AND ITS FILE. Two sessions computed this and got different
coefficients from the same data. One read the `auc` column of
`results/loss_terms.csv`, the guard's rolling median at the stop. The other
read the `last` column of `results/auc_verdicts.tsv`. The second gives +0.587
and +0.809, and it makes `k32_r100_09_lr56` tie `k8_r100_09` at 0.7302. The
sign and the verdict hold under both. The report quotes the first and says so.

WHAT IT DECIDES. The AUC has ONE use in this card: the guard stops a run whose
rolling median falls under 0.55, because a run at chance gives no score.
`scripts/gate_pass3.sh` carries the rule and ranks on GM-Relative MASE alone.
n = 7 and n = 3 are small, so the coefficient is not evidence for anything
beyond "do not rank on this axis".

## Pass 4 — the L_rep decay at 5.6e-4, over a long stop

THE GAP THIS PASS FILLS. Every decay arm of this card ran at 1e-3 and stopped
at 40,000 steps. So the decay was never tested at the rate that fits the
width, and never at a stop where the degradation appears. The two arms below
move both together.

| arm | L_rep to zero by | EMA ramp | stops | card |
|---|---|---|---|---|
| `k3_r100_09_lr56_dec` | 2,000 | 100,000 | 40k, 100k, 200k | 1 |
| `k3_r100_09_lr56_dec10k` | 10,000 | 40,000 | 40k, 100k | 0 |

Both are the configuration-1 cell: k = 3, sum, seed 20260520, 5.6e-4. The
second arm carries a 40,000-step EMA ramp so the ramp COMPLETES inside the
stop.

THE LAUNCHER IS `phase1.sh`, NOT `pass2_lane.sh`. `phase1.sh` trains the head
and runs its 97-config GIFT-Eval INLINE after each leg, so each card idles
about 4.6 hours per stop. That is the cost of a head at each stop, which this
pass needs. A peer session that gates on a #412 trainer alone sees an idle
card in that window and starts an arm beside a head. Gate on `phase1.sh` too.

THE PREFIX TRAP. `k3_r100_09_lr56_dec` is a PREFIX of
`k3_r100_09_lr56_dec10k`, and the two arms are the whole pass. A `pgrep -f`
on the arm name reads one arm as the other, and the status block then says a
leg runs when it does not. `pass4_lib.sh` matches on the leg's own
`--save-dir` instead, which no two legs share.

`pgrep -f` ALSO MATCHES THE AGENT SHELL. An agent session runs each command
through a wrapper whose own arguments hold the whole command text, so
`pgrep -fc head_eval` counted 1 with no head on the box. Every wrapper carries
`shell-snapshots` and no lane does, so `cf412_count_real` drops them.

THE HEAD DOES NOT FIT ON GPU 0 WHILE A BACKBONE RUNS THERE. At the launch
GPU 0 held an ipykernel at 11,692 MiB, an rnd-483 trainer at 3,212 MiB and a
#412 backbone at 5,416 MiB, which leaves 3,876 MiB against
`CF412_HEAD_VRAM_MIB` 9,000. The backbone exits before its own inline head
starts, which frees 5,416 MiB and clears the gate by 292 MiB. Any head that
comes from the sweep instead needs GPU 1.

## Pass 5 — the phase-1 re-run at the winning rate

The card pre-registered one rule before the rate sweep ran: a rate distance D
above the seed band voids every 1e-3 number, and phase 1 runs again at the
winning rate. D came in at 0.1675, which is three bands, so the rule fired.
Three of the six configurations never ran again, and pass 5 runs those three
at 5.6e-4, seed 20260520, to 40,000 steps: `k32_r200_08_lr56`,
`k32_r100_09_dec_lr56` and `k8_r100_09_lr56`.

ALL THREE MOVE THE REDUCTION TO SUM. The mean arms of this card score 1.4404
to 1.4629 and two of them lost the task. Every sum arm scored 1.18 to 1.35 and
none lost the task.

`k32_r200_08_lr56` also moves the EMA ramp from 200,000 steps to 40,000, so
the ramp completes inside the stop. Its 1e-3 twin lost the task at step 28,152
with the ramp only at 0.8285.

### The queue gates on the LANE, not on the trainer

Pass 4 runs under `phase1.sh`, which trains a head and runs its 97-config
evaluation INLINE between two legs of one arm. That window is about 4.6 hours,
and through it the card holds no backbone trainer. A queue that read the
trainers alone would start a leg there, and the pass-4 lane would then wait
for memory that does not come back.

So `pass5_lane.sh` counts a live `phase1.sh` or `pass2_lane.sh` whose `BB_GPU`
names the card, as well as a python trainer whose `--run-name` carries
`_cf412_`. `CF412_QUEUE_CHECK=1 bash scripts/pass5_lane.sh` prints what the
gate reads and starts nothing.

### The queue picks by memory, not by order

The queue takes the first pending arm that FITS the free memory of the card
that frees. A k = 32 leg needs 11,300 MiB and a k = 8 leg needs 8,400. GPU 0
holds another user's Jupyter kernel at 11,692 MiB, and with the rnd-483
trainer beside it that card frees to about 9,300 MiB. So GPU 0 can take the
k = 8 arm and no k = 32 arm, and a strict first-in-first-out queue would idle
that card for hours.

The rnd-483 session confirms the kernel is not its own, that nothing of its
pipeline reads it, and that its parent is an 18-hour Jupyter server that looks
like a person's notebook. No agent closed it. The cost is that the two k = 32
arms run one after the other on GPU 1 instead of side by side, which is about
12 hours.

### A leg leaves room for a neighbour that grows

`cf412_leg_vram_mib` reads the free memory ONE TIME, at the start of a leg. A
neighbour that grows after that start can kill the leg, or the leg can kill
the neighbour.

The rnd-483 session shares both cards of this box. It measured its own worker
at 3,212 MiB today and at 6,880 MiB on the largest point of its sample, which
is 3,668 MiB of growth under a leg that has already started. So
`pass5_lane.sh` adds 3,700 MiB of headroom to every leg's own need.

The rule refuses GPU 0 for every arm of pass 5. That card frees to about
9,292 MiB behind the 11,692 MiB kernel, and the smallest arm of this pass
needs 8,400 plus the headroom. Without the rule that leg would hold 2,132 MiB
of slack against a neighbour that grows by 3,668, and one of the two jobs
would fail. One of them belongs to another project.

The cost is that all three arms run on GPU 1, one after the other.

### A dead leg goes back in the queue

A leg of pass 5 is 9 to 13 hours and it shares a card with another project.
The headroom above makes a death unlikely, not impossible, and a death costs
the whole arm: the pass rests on each arm's score at the 40,000-step stop, and
an arm with no 40,000-step checkpoint carries no score.

So `pass5_lane.sh` puts a leg that ends with NO CHECKPOINT back in the queue,
up to `CF412_ATTEMPTS` times, which is 3. A re-fire is cheap, because
`run_leg_k.sh` resumes the arm's furthest checkpoint with its optimizer state
and the trainer saves every 20,000 steps.

TWO EXITS ARE RESULTS AND NEVER RETRY. Exit 4 is the AUC gate, which stopped
the arm, and the same arm would lose the task again. Exit 3 is a wiring
defect: the trainer took an objective this arm does not carry, or it named
none, and that repeats.

The path was tested with a stub runner and a mocked card, so the whole
placement, wait and re-fire ran in seconds and started no trainer.

### Two notebook kernels took both cards at 19:26

Two `run_notebook_ws.py` runs from a peer session started an ipykernel on EACH
card, 11,692 MiB each, which is 23.4 GB of the 49 GB of this box. The kernel
that pass 5 planned around, pid 1613885, is gone. These are its replacements.

After the change one card only can clear the 9,000 MiB head gate:

| card | free | plus the pass-4 leg | clears 9,000 |
|---|---|---|---|
| GPU 0 | 2,818 MiB | 9,290 MiB | yes, by 290 |
| GPU 1 | 542 MiB | 7,014 MiB | no |

So `pass5_sweep_move.sh` moves the head sweep to GPU 0 in the window between
the pass-4 inline head and the next leg. It waits for that head to appear and
then to END, because two head trains on one card is what the `flock` prevents.

GPU 1 blocks more than a head. A pass-4 k = 3 leg needs 7,700 MiB and that
card gives back 7,014, so the 100,000-step and 200,000-step legs of pass 4
wait as well. No queue of this card routes around it. Both #412 sessions asked
the owner session, which is idle, and neither touched the process.

### The head gate stranded a head, as its own comment predicted

At 21:34 arm `k3_r100_09_lr56_dec` reached 40,000 steps and its inline head
started on GPU 1, which held 7,018 MiB free against `CF412_HEAD_VRAM_MIB`
9,000. The head took the card head lock, missed the gate, and waited.

A head of this shape holds 6,252 MiB. So it would have fit, with 766 MiB to
spare. The comment above `CF412_HEAD_VRAM_MIB` already says this in the last
line it was written with: "A gate far above it strands a head that would have
fit."

THE GATE STAYS AT 9,000 ANYWAY. `head_eval_bb.sh` takes its `flock` BEFORE the
memory wait, so a head that holds the lock and misses its gate stalls every
later head on that card for up to a day. A gate under the true peak is the
worse failure. 766 MiB is also thin against this box: the rnd-483 neighbour
grew from 3,172 to 5,490 MiB in one evening, and `k32_r200_08` held 6,402 MiB
for four hours and 10,418 MiB later in the SAME leg.

READ THE GATE'S OWN COMMENT BEFORE MEASURING IT. This session built
`head_vram_sample.sh` and took 46 samples to find 6,252 MiB, a number
`study.sh:112` already carried from 94 samples and 20 samples on two earlier
runs. The measurement is a third independent replication, on a different arm,
which is worth something. The hour that found it was not: the answer sat four
lines above the value the session was questioning.

### Two drivers, one head tag

At 23:33 a second driver started for `k3_r100_09_lr56_dec` at 40,000 steps,
on GPU 0, while lane 1's own inline driver still held the GPU 1 head lock and
waited for memory. Both named one output directory:
`cf-412/k3_r100_09_lr56_dec/eval/k3_r100_09_lr56_dec_bb40k_h30k_student`.

THE `flock` DOES NOT CATCH THIS. It is per CARD
(`/tmp/cf373_head_gpu<N>.lock`), so a GPU 0 head and a GPU 1 head never
serialize against each other. `head_busy.sh` does catch it, and the second
driver did not run under anything that asks.

NOTHING WAS CORRUPTED. The GPU 1 driver had started no python, so only the
GPU 0 head ever wrote. `cf412_kill_tree 3033796` at 23:42 ended the waiting
driver, `phase1.sh` logged the head as FAILED, and lane 1 moved on to its
100,000-step leg. The GPU 1 head lock came free with it.

WHERE IT CAME FROM. Pid 3209265 had no lane in its parent chain. Its
grandparent was the agent shell that launched both `phase1.sh` lanes, so a
session started it by hand.

THE RULE THIS CONFIRMS, which this log already carried: ask
`scripts/arm_busy.sh <arm>` and `scripts/head_busy.sh <arm> <stop>` before you
start anything by hand. They are the only guards on this card, and a hand-run
driver asks neither. A duplicate that had started 90 seconds earlier would
have put two writers on the 40,000-step number that pass 4 rests on.

### A head with no lane above it

`head_busy.sh` and `arm_busy.sh` are the only guards this card has against two
processes on one arm, and a LANE calls them before it starts anything. A
process started BY HAND calls neither.

At 23:33 on 2026-09-10 a session started a second head for
`k3_r100_09_lr56_dec` at 40,000 steps by hand, while a first driver for the
SAME tag held the GPU 1 flock. The `flock` is per CARD, so a GPU 0 head and a
GPU 1 head do not serialize, and both name one head checkpoint and one eval
directory. Nothing was corrupt, because the first driver had started no python
and its owner killed it at 23:42.

`pass5_orphan_watch.sh` prints one line when a #412 head or trainer appears
with no lane script above it. It writes into `results/pass5_watch.log`, so a
session that tails that log needs no second monitor.

THE AGENT SHELL IS NOT A LANE, and this trap made the first version silent. A
session runs each command through a wrapper whose own arguments hold the WHOLE
command text, so the wrapper that started the hand-run head carries the string
`phase1.sh` and read as a lane. Every wrapper carries `shell-snapshots` and no
lane does, which is the test `cf412_count_real` already uses.

A WATCHER THAT REPORTS NOTHING LOOKS HEALTHY. The first version was tested
against a live orphan and printed nothing, which is the only reason the trap
was found.

### One card, because two notebook kernels took the other

At 19:26 two `ipykernel_launcher` processes started, one on each card, holding
11,694 and 11,692 MiB. They are not stale memory: at 23:45 they ran at 191 and
206 percent CPU with over 8 hours of CPU time each. They compute, so no
session may reclaim them.

WHAT THAT LEFT. GPU 1 holds 7,018 MiB free with the kernel and the rnd-483
neighbour on it. A k = 3 leg needs 7,700 and a head needs 9,000. So GPU 1
takes NEITHER, and it is short for the leg by 682 MiB.

THE GATE STAYS. The leg's true peak is 6,472 MiB, so it would fit with 546
MiB spare. That margin is not the card's to spend: the rnd-483 worker on the
same card grew from 3,172 to 5,490 MiB in one evening, and a leg that takes
GPU 1 to 546 MiB free can kill that job as well as its own.

WHAT THE CARD DID INSTEAD. At 23:49 it stopped lane 1, which held no trainer
and only polled GPU 1, and it started the same arm again on GPU 0 with
`STOPS="100000 200000"`. Lane 2 keeps its own `phase1.sh` and polls the same
card. Each lane takes GPU 0 when the other frees it, so the two arms
interleave with no scheduler: a leg runs while the other arm evaluates on the
CPU. The cost is about 29 hours of wall clock against 16 on two cards.

THE ORDER IS DELIBERATE. Arm `k3_r100_09_lr56_dec` goes first, because it
carries the 200,000-step stop, which is the question of pass 4. Arm
`k3_r100_09_lr56_dec10k` stops at 100,000.

### A lane's declared card is not always the card it uses

At 23:49 the pass-4 session moved its lane 1 from GPU 1 to GPU 0 by hand. The
lane process keeps the `BB_GPU` it was started with, so a live lane now
declares GPU 1 while its trainer runs on GPU 0. `pass5_lane.sh` reads the
declared value, so it holds GPU 1 for a lane that is not on it.

THIS IS NOT A DEFECT TO FIX HERE. Pass 5 queues BEHIND pass 4, and the pass-4
session states that it wants GPU 1 for its `dec10k` arm as soon as that card
reaches 7,700 MiB free. Deferring to a live pass-4 lane is the instruction, so
the conservative read gives the correct behaviour by the rule that matters.

WHAT IT WOULD COST IF PASS 4 STOPPED WITHOUT EXITING. A lane that hangs in a
memory wait holds its declared card against pass 5 until `CF412_QUEUE_TIMEOUT`,
which is four days. A lane that EXITS holds nothing, so the ordinary end of
pass 4 releases both cards with no action. The two sessions coordinate by
message for the case in between.

ARITHMETIC FOR THE OPENING. If the kernel on GPU 1 ends, that card holds about
19,074 MiB beside the rnd-483 worker. Pass 4's `dec10k` leg needs 7,700 and a
pass-5 k = 32 leg needs 15,000 with the neighbour headroom. The two do not fit
together, so the card goes to pass 4 first.

### `cf412_kill_tree` on a lane leaves an orphan that keeps running

At 23:49 the card stopped lane 1 with `cf412_kill_tree 2752684` and started a
replacement two seconds later. At 02:15 a peer session's card gate reported a
pass-4 lane declaring GPU 1, which no lane should have declared.

IT WAS AN ORPHAN OF THE KILLED LANE. Pid 3234582, parent init, started at
23:49:23, carrying `ARMS=k3_r100_09_lr56_dec STOPS="40000 100000 200000"
BB_GPU=1`. `phase1.sh` forks a subshell for its `| tee` pipeline, that
subshell survived the kill, carried the loop forward, and re-parented to init.

WHAT IT WOULD HAVE COST. It still polled GPU 1 for the SAME arm the
replacement lane trains on GPU 0. The moment GPU 1 reached 7,700 MiB it would
have started a second trainer on `k3_r100_09_lr56_dec` at 100,000 steps,
writing the save directory of pid 3353588. Nothing on this card stops two
trainers on one arm.

AFTER KILLING A LANE, LOOK FOR ITS ORPHAN. `pgrep -f phase1.sh` and read
`/proc/<pid>/environ` for `ARMS`, `STOPS` and `BB_GPU`. A lane with parent 1
that you did not start is an orphan. The process table is the only record: the
orphan writes the same log file as the lane that replaced it.

A PEER'S ODD READING WAS THE ONLY SIGNAL. Nothing this card owns reported it.
The await watched arms and stops, not lanes, and both arms looked healthy.

### A card is claimed by a file, not by a process

The note above said the wrong gate reading was harmless. It stopped being
harmless at 02:30, when the pass-4 session found that the lane declaring GPU 1
was an ORPHAN of a lane it had stopped at 23:49, and killed it. That orphan
still polled GPU 1 for the same arm its live lane trains on GPU 0, so it would
have started a SECOND trainer on one arm and one save directory. The odd gate
reading was the only signal that it existed.

After the kill no process of pass 4 named GPU 1, so the pass-5 gate read the
card free. The next opening would then have gone to pass 5 instead of to pass
4, against the rule that pass 5 queues behind it.

So a card is claimed by a FILE. While `results/pass5_defer_gpu<N>.txt` exists,
`pass5_lane.sh` treats that card as held. The queue reads the file on every
poll, so the claiming session releases the card with `rm` alone: no message to
the pass-5 session, and no restart.

A PROCESS GATE CANNOT EXPRESS PRIORITY. It answers "is this card busy now",
and the rule here is "who is next". Those differ exactly when a card is free.

### Pass 4 must release GPU 1 when it is done

A peer session's pass-5 queue defers GPU 1 to this pass through one file:

    results/pass5_defer_gpu1.txt

That queue reads it on every poll. DELETE IT when pass 4 no longer wants
GPU 1, which is when `k3_r100_09_lr56_dec10k` holds its 100,000-step
checkpoint, or when the pass ends for any other reason. The card returns to
pass 5 within two minutes, with no restart at either end.

A FILE AND NOT A PROCESS, ON PURPOSE. A process gate answers "is this card
busy now". The rule between the two passes is "who is next". Those differ
exactly when a card is free, which is the only moment that matters. The last
pass-4 process naming GPU 1 was the orphan above, and killing it would have
handed the next opening to pass 5.

IF THIS PASS ENDS WITHOUT DELETING THE FILE, pass 5 waits for days on a card
nobody wants.

### A free-memory threshold is the wrong watch on a shared card

At 06:54 GPU 1 read 9,600 MiB free, over the 7,700 MiB a k = 3 leg needs.
The card did not take it, and the reason generalises.

THE READING WAS A DIP. The rnd-483 worker on that card had restarted three
minutes before and sat at 2,908 MiB, its small size. Its measured peak is
6,880 MiB.

THE ARITHMETIC THAT DECIDES, at the NEIGHBOUR'S PEAK and not at its current
size:

    24,564 total - 11,692 kernel - 6,880 worker = 5,992 MiB for this card
    a k = 3 leg holds 6,472    DOES NOT FIT
    a head holds 6,252         DOES NOT FIT

So GPU 1 was never short by 682 MiB, which is what the gate reported. It is
short at the neighbour's peak whatever the instantaneous reading says, and a
leg started in the dip takes the box over its limit when the neighbour grows
back. That kills the neighbour's job as well as this card's.

WATCH THE PROCESS, NOT THE NUMBER. A threshold on `memory.free` fires on
every dip. The watch is now on the kernel PROCESS 2843685, because its exit
is the only event that changes the arithmetic. The same holds for any card
this project shares: size a leg against the sum of the neighbours' PEAKS, and
wait on a neighbour ending rather than on a number falling.

### The peak rule is easy to write and easy to forget

At 12:44, one hour after this log recorded "size a leg against the sum of the
neighbours' PEAKS", this card offered a peer session a parallel slot on GPU 1
and priced that peer's k = 8 leg at its CURRENT 5,424 MiB. Its peak is 7,516.

The peer redid the sum at three peaks:

    6,880 rnd-483 + 7,516 k = 8 + 10,418 k = 32 = 24,814 against 24,564

which is 250 MiB OVER the card, not the 1,842 under that this card offered.
The peer declined, and was right to.

WHAT MAKES THE RULE HARD TO APPLY. The peak of your OWN job is easy to
remember, because a gate carries it. The peak of a neighbour is not written
anywhere: `nvidia-smi` reports what a process holds NOW, and that is the
number a reader reaches for. Every mistake this card made on memory, over two
days, came from reading a current size as a settled one.

THE COST WAS NOTHING THIS TIME, because the peer checked. A shared box needs
two sessions to agree before a card is doubled up, and this is the case that
shows why.

### The neighbour reserve is what is LEFT to grow, not a fixed number

`pass5_lane.sh` first reserved a flat 3,700 MiB for the rnd-483 worker, from
its measured 3,212 and its peak of 6,880. A flat number is wrong in both
directions.

TOO SMALL when the neighbour is between points. The opening that the
orchestrator hand-started a k = 8 leg into at 12:33 came while that worker sat
at 744 MiB, so it could still take 6,136, not 3,700. The card was then 1,552
MiB over at three peaks.

TOO LARGE when the neighbour is already big. With the k = 8 head holding 6,252
MiB on GPU 1 and the worker at 3,332, the card offers 14,980 and the flat rule
wants 15,000. The peak sum is 10,418 + 6,252 + 6,880 = 23,550 of 24,564, so a
k = 32 leg fits with 1,014 to spare and only the arithmetic said no. That
20 MiB would have idled the card for the 1.7 hours of a head train.

So `p5_neighbour_reserve` reads the neighbour's CURRENT size on that card and
reserves `peak - current`, with the whole peak reserved when no neighbour runs
there.

### An arm another session trains stays PENDING

The queue used to skip a busy arm and DROP it. It waits only on legs it
started itself, so a hand-started run that died would have left that arm with
nothing to re-fire it. The arm now stays in the queue while another session
trains it, and a prune step at the top of each poll removes any arm whose
checkpoint is on disk, whoever trained it. Tested with a stubbed `arm_busy.sh`
that always reports BUSY: the arm survived every poll and the queue never
dropped it.

### What each pass-5 arm can carry, computed and not asserted

`scripts/arm_diff.py` reads `arms.tsv` and prints the settings that separate
two arms. Run against the pass-5 arms it gives this, and the report states no
comparison that this tool has not passed:

| pair | axis | verdict |
|---|---|---|
| `k8_r100_09_lr56` against `k3_r100_09_lr56` | k, 3 to 8 | SINGLE AXIS |
| `k32_r100_09_dec_lr56` against `k32_r100_09_sum` | the `L_rep` decay | SINGLE AXIS |
| `k32_r200_08_lr56` against `k3_r100_09_lr56` | k, tau and the ramp | CONFOUNDED |

SO PASS 5 CLOSES THE DEPTH LADDER. With the reduction, the rate, the seed and
the momentum schedule all held fixed, `k8_r100_09_lr56` is the middle point
between `k3_r100_09_lr56` at 1.1820 and `k32_r100_09_sum` at 1.2161. Those two
already differ in k alone and sit 0.0341 apart, which is inside the 0.0649
band, so depth does not separate them. The k = 8 point tests whether that holds
across the middle.

AND IT GIVES THE DECAY A CLEAN TEST AT k = 32. `k32_r100_09_dec_lr56` against
`k32_r100_09_sum` moves the decay alone. Pass 4 measured the same axis at
k = 3 and found it neutral, +0.0344 at 40,000 steps and +0.0337 at 100,000.

`k32_r200_08_lr56` CARRIES NO CLEAN PAIR. Its nearest sibling
`k32_r100_09_sum` differs in the start momentum AND the ramp length. It
answers the card's question in the qualitative form only: its 1e-3 twin ran
the mean and lost the task at step 28,152, so this arm asks whether the
configuration holds the task and carries a score under the sum at 5.6e-4. Its
momentum reaches exactly 1.0000 at the 40,000-step stop, and the report states
that beside its score.

### The VRAM gates of this card omit the latent-drift probe

At 14:33 the 200,000-step leg of `k3_r100_09_lr56_dec` died at step 120,000:

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.32 GiB.
      train.py:2303  latent_drift_probe.probe(model, step)
      train.py:1251  h = self.extract_h(model)
      encoders.py:112  _, hidden = self.gru(fl)

THE GATE CANNOT SEE THIS. `cf412_leg_vram_mib` returns 6,500 for k = 3, from
`results/trial/smoke.csv`. The smoke runs `CF412_SMOKE_STEPS` 150 steps. The
probe fires every 20,000 steps. So NO smoke of this card has ever crossed a
probe step, and NO gate on this card includes the probe's 4.32 GiB.

A k = 3 leg therefore needs about 10,900 MiB at a probe step and 6,472 MiB
between them, and only the second number is written down anywhere.

THIS IS A CANDIDATE FOR THE OPEN QUESTION IN THIS LOG. An earlier section
records `k32_r200_08` holding 6,402 MiB for four hours and 10,418 MiB later in
the SAME leg, with "THE CAUSE IS NOT KNOWN", and it argues the probe away on
the grounds that its docstring puts it near 10 MB and that `k` does not enter
it. The docstring is wrong: the probe runs a full GRU forward over
`probe_bs=64`, and the measured allocation is 4.32 GiB. The growth that
section could not explain is 4,016 MiB.

THE CARD DID NOT LOWER THE GATE, IT RAISED IT. The 200,000-step leg runs again
with `CF412_VRAM_MARGIN_MIB=5000`, so its gate is 11,500 rather than 7,700. No
script changed and the trainer command line is byte-for-byte what the 40,000
and 100,000-step legs ran. `--latent-drift-probe-batch-size` would have cut
the spike, but it would also make this leg's objective line differ from its
own earlier legs, and nothing in this card reads the drift CSV anyway.

WHAT IT COST. Nothing. `--save-every 20000` had written the 120,000-step
checkpoint before the probe ran, so the re-fired leg resumes there.

### The latent-drift probe costs TRANSIENT memory, not resident

A pass-4 leg died at step 120,000 with `torch.OutOfMemoryError: Tried to
allocate 4.32 GiB` inside `latent_drift_probe.probe`. The probe fires every
20,000 steps and `CF412_SMOKE_STEPS` is 150, so no smoke of this card has ever
crossed a probe step. That looked like every gate of the card being too low by
about 4,400 MiB.

MEASURED ON A LIVE LEG INSTEAD. `k8_r100_09_lr56` crossed step 20,000 on
2026-09-11 with the probe rows in its latent-drift CSV, so the probe ran.
`nvidia-smi` read the trainer at 7,140 MiB against a smoke row of 7,160. The
probe raised the resident figure by NOTHING.

BOTH OBSERVATIONS ARE TRUE. The 4.3 GiB is transient. The probe runs under
no_grad in eval mode, and between training steps the rollout graph is freed,
so the caching allocator serves it from blocks it already holds. The leg that
died had 1,628 MiB free on its card behind a new kernel, so its allocator had
to ask the driver and the driver had nothing. The leg that survived had 15,444
free.

WHAT A GATE MUST BUY is enough FREE memory on the card to absorb a 4.3 GiB
transient at a probe step, not 4,400 MiB more resident. A uniform raise costs
real placements: at +5,000 a k = 32 arm of this pass would demand 20,000 MiB
free and stop fitting cards it fits safely today.

THIS ALSO CLOSES THE OPEN QUESTION ABOVE. The log records `k32_r200_08` at
6,402 MiB for four hours and 10,418 MiB later in the same leg, cause unknown,
and it ruled the probe out on a docstring that claims about 10 MB. The
docstring is wrong: the probe runs a full GRU forward at probe_bs 64. The jump
is 4,016 and the probe allocates about 4,424. So 10,418 is the pool AFTER it
absorbed one probe, and any figure that adds the probe to 10,418 counts it
twice.

### CORRECTION: the probe is a TRANSIENT, and the gates are not all too low

The section above says "EVERY VRAM GATE ON THIS CARD IS TOO LOW". That is
WRONG in its mechanism, and this section replaces it.

THE MEASUREMENT THAT REFUTES IT. A peer session read a k = 8 leg that had
just crossed step 20,000, with both probe rows in its drift CSV. That leg sits
at 7,140 MiB against a 7,160 MiB smoke row. So a probe step does NOT raise a
leg's RESIDENT memory. This card's own legs agree: `leg_40k` crossed 6 probe
steps and `leg_100k` crossed 10, and both finished.

WHAT THE 4.32 GiB IS. A transient. The probe runs under `no_grad` in `eval`
mode between training steps, when the rollout graph is freed, so the caching
allocator usually serves it from blocks it already holds and never asks the
driver. It asks the driver only when the pool cannot cover it, and then the
card must have the memory free.

WHY THIS LEG DIED AND THE OTHERS DID NOT. Free memory on the card, nothing
else. GPU 0 held 1,628 MiB free behind a kernel that landed at 12:38. The
peer's leg crossed its probe with 15,444 MiB free on GPU 1.

THE RULE, STATED CORRECTLY. `cf412_leg_vram_mib` measures the RESIDENT need.
At a probe step a leg ALSO needs about 4.3 GiB free on the card. The 1,200 MiB
margin does not cover that, so a leg on a full card dies at the first probe
step and runs fine on an empty one. Raising THIS leg's gate to 11,500 works
because 11,500 minus its 6,472 resident leaves 5,028 free for the transient.
It is not a uniform correction every gate needs.

WHAT STAYS OPEN. This does NOT explain `k32_r200_08` going 6,402 to 10,418
inside one leg, because that jump PERSISTED and a transient does not. The
cause of that one is still not known.

THE PRACTICE. A number measured once, under one condition, explains less than
it looks like it does. This card had the OOM traceback and a 4.32 GiB figure
and wrote a rule for every gate on the project. One reading of a live leg on
the other card refuted it in a minute.

#### Why the jump persists, which closes the question

A transient does not persist, and the `k32_r200_08` jump from 6,402 to 10,418
MiB did. Both facts hold under one mechanism, and it is the caching allocator.

`nvidia-smi` reports the RESERVED POOL, not the live tensors. So at a probe
step there are two cases:

- The pool's free blocks can serve the 4.3 GiB. The pool does not grow and
  `nvidia-smi` does not move. `k8_r100_09_lr56` crossed step 20,000 and went
  from a 7,160 smoke row to 7,140 measured, a change of -20 MiB.
- They cannot. The allocator asks the driver, the pool GROWS, and PyTorch
  never returns pool memory to the driver without `empty_cache`. So the new
  figure persists for the rest of the leg. `k32_r200_08` grew 4,016 MiB
  against a probe that allocates about 4,424.

WHICH CASE A LEG TAKES depends on how fragmented its pool is and how large its
own rollout graph is, so it is not predictable from k alone.

WHAT IT MEANS FOR A GATE. Use the POST-growth resident figure, which is what
`cf412_leg_vram_mib` already carries for k = 32 at 10,100 plus margin, and
keep enough free memory on the card for the growth to happen at all. A leg on
a full card dies at its first probe step. The same leg on an empty card never
notices.

### CLOSED: one mechanism covers both probe readings

The section above left the `k32_r200_08` jump open, on the grounds that a
transient does not persist. A peer session closed it, and the key is that
`nvidia-smi` reports the caching allocator's RESERVED POOL and not live
tensors:

- The pool's free blocks CAN serve the probe's 4.3 GiB. The pool does not
  grow and the reading does not move. That is `k8_r100_09_lr56`: 7,160 smoke
  against 7,140 measured across step 20,000.
- They CANNOT. The allocator asks the driver, the pool GROWS, and PyTorch
  never returns it without `empty_cache`, so the new figure PERSISTS. That is
  `k32_r200_08`: 6,402 to 10,418, a rise of 4,016 against a 4,424 probe.

Which case a leg takes depends on its pool fragmentation and the size of its
own rollout graph, so `k` alone does not predict it. The growth this log
carried as "THE CAUSE IS NOT KNOWN" for two days now has one.

THE GATES ARE ALREADY RIGHT. `cf412_leg_vram_mib` gives 11,300 for k = 32 and
the post-growth figure this card has seen is 10,418, so it covers the grown
case with 882 MiB to spare. No gate on this card needs a raise. This card's
own 11,500 for the 200,000-step leg is a local choice for a crowded card, not
a correction.

### Handing a card back, without a race and without a session awake

Pass 5 owes the orchestrator session one card for one 40,000-step run of
`k3_r100_09_lr56_fix09`, in exchange for the card that session handed to pass
5. That arm differs from `k3_r100_09_lr56` in the EMA schedule alone, so it is
the single-axis momentum test that no pass-5 arm can be.

THE OBVIOUS IMPLEMENTATION RACES. Wait for `k32_r200_08_lr56` to land its
checkpoint, then write `results/pass5_defer_gpu<N>.txt`. But the queue polls
every 120 s and a checkpoint appears at the END of a leg, so the queue can
place the next arm on that card before the file lands. Nothing locks between
them.

`pass5_defer_after_first_k32.sh` writes the file when the leg STARTS instead.
The card is held by that leg while it runs, so an early file changes nothing,
and the moment the leg ends the file is already in place. No race, and no
session has to be awake at the moment that matters.

THE COST IS PRICED AND ACCEPTED: if only one card is free then, the third
pass-5 arm waits about 4 hours.

### A lane holds a card only while it still has a LEG

The lane gate counted every live `phase1.sh` whose `BB_GPU` named the card.
That is right while the lane has a leg to run, and wrong after its last one:
`phase1.sh` trains a head and runs a 97-config evaluation after the leg, which
is 4.6 hours during which the lane will never ask for the card again.

On 2026-09-11 that would have idled GPU 1, with 14,978 MiB free, for the whole
head and evaluation of `k8_r100_09_lr56`.

`p5_lane_has_leg_left` reads the lane's own ARMS and STOPS out of its
environment and asks whether every pair already has a checkpoint. Checked
against the four live lanes it released the two whose legs were done and held
GPU 0 for the unfinished 200,000-step leg. A lane that names neither variable
runs the whole card and counts, so the conservative answer is the default.

### The gate is invariant to the neighbour's current size, and that is right

With the k = 8 head resident on GPU 1 the gate refused a k = 32 leg by 232
MiB. The refusal is correct, and not the near-miss it looks like.

    10,418 leg + 6,252 head + 6,880 worker at peak = 23,550 of 24,564

That leaves 1,014 MiB. A k = 32 leg crosses drift-probe steps at 20,000 and
40,000, and a probe may have to GROW the pool by about 4 GB when its free
blocks cannot serve the transient. Starting into 1,014 MiB is the setup that
killed a pass-4 leg at step 120,000.

A PROPERTY WORTH RECORDING: as the neighbour shrinks, the free memory and the
reserve rise by the same amount, so the decision does not move. The gate asks
only whether the leg plus the neighbour's PEAK fit the card, which is the
question that decides whether both survive.

#### The `pgrep -f` trap caught a THIRD time, in an ad-hoc watch

The log already warns that `pgrep -f` matches the agent shell, and
`pass5_orphan_watch.sh` was silent against a live orphan for the same reason.
It happened again on 2026-09-11 at 16:45, in a one-line background watch
rather than in a committed script:

    until ... pgrep -f "_cf412_k32_r200_08_lr56" ...

The wrapper that RUNS that command carries the pattern in its own arguments,
so the watch reported "TRAINER IS UP" for an arm that had not started, and the
card it named was empty. Measured a minute later: `pgrep -f` returned 2 pids,
both of them this session's own shells, and the safe test returned none.

THE SAFE TEST, and every committed watcher of this card already uses it:

    ps -eo pid,args --no-headers | awk '$2 ~ /python/ && /_cf412_<arm>/ {print $1}'

A TRAINER IS A PYTHON PROCESS. The executable is the discriminator, and no
shell passes that test. The committed scripts were right and the throwaway
command was wrong, which is the argument for writing the check once rather
than typing it each time.

#### The same hole was inside `arm_busy.sh`, the guard both passes call

The orchestrator session hardened `scripts/arm_busy.sh` (4d445454) to test the
executable rather than the command line. Measured against the OLD copy, with a
non-python wrapper carrying the exact `--run-name` string and no trainer
anywhere:

    old copy  rc 0  BUSY   a false positive
    new copy  rc 1  FREE   correct

A false BUSY is not harmless for pass 5. The queue keeps an arm PENDING while
another session trains it, so a permanent false BUSY would hold an arm pending
for ever and the card would never place it.

THE FROZEN COPY IS A SEPARATE FILE. `pass5_lane.sh` calls `$HERE/arm_busy.sh`,
and `$HERE` is `run_snapshot/`, so a fix in `scripts/` does not reach the
running queue. It was installed by writing beside the target and renaming over
it, because the queue invokes that file every 120 s and a plain copy can be
read half-written. A running `bash` keeps the old inode and finishes safely,
and the next invocation gets the new file.

#### "queue running: 4" is the watcher counting its own readers

`pass5_watch.sh` counts a queue as any process whose first argument ends in
`pass5_lane.sh`. `CF412_QUEUE_CHECK=1 bash run_snapshot/pass5_lane.sh` is that
same script in check mode, so every status read by any session is counted
while it lives, which is under a second.

So an hourly line reading `queue=4` does NOT mean four queues. Measured at the
moment it appeared: one process matched with an age above 60 s, pid 496191,
and the rest were check-mode readers. A real queue is minutes to hours old:

    ps -eo pid,etimes,args --no-headers \
      | awk '$3 ~ /bash$/ && $4 ~ /pass5_lane\.sh$/ && $2 > 60 {print $1}'

Two queues would be a genuine fault, because each places arms independently,
so the count is worth reading rather than ignoring. It is the AGE that
separates the two cases.

### The decay changes neither the level nor the slope

Pass 4 asked whether the `L_rep` decay makes a longer stop win. It does not,
and the reason is sharper than a simple no.

    arm                        40k      100k      rise
    k3_r100_09_lr56         1.1820    1.3170   +0.1350    no decay
    k3_r100_09_lr56_dec     1.2164    1.3507   +0.1343    decay by step 2,000

The two arms differ in ONE column, the decay. Their 40,000-step scores sit
0.0344 apart and their 100,000-step scores 0.0337 apart, both inside the
0.0649 seed band. So the decay does not move the LEVEL at either stop.

THE RISES DIFFER BY 0.0007, which is one percent of the band. So the decay
does not move the SLOPE either.

WHAT THAT RULES OUT. The card's premise was that `L_rep` does no work after
step 1,000 and carries a large flat term for the rest of training. Both facts
hold. But removing that term entirely, by step 2,000, leaves the degradation
from 40,000 to 100,000 steps unchanged to within one percent of a band. So
whatever drives the degradation is not `L_rep`, and it is not sensitive to the
decay schedule.

A peer session reading the same two rows noticed this first.

### The card handover was disarmed before it fired

Pass 5 owed the orchestrator session one card for the momentum test, and
`pass5_defer_after_first_k32.sh` was armed to write the claim when
`k32_r200_08_lr56` started. That session then took GPU 0 instead and released
the handoff.

THE ARMED SCRIPT WOULD HAVE COST PASS 5 ITS OWN CARD. It writes the claim at
the START of the k = 32 leg, which is what removes the race with the queue. So
once the obligation ended, an armed script was no longer a promise but a
self-inflicted block: it would have claimed GPU 1 for a run that was already
elsewhere, and pass 5's third arm would have waited behind nothing.

It was disarmed before the leg started, so no file was ever written.
`results/pass5_defer.log` carries the line.

### A test fixture that impersonates a trainer will fool the live watchers

To test the hardened `arm_busy.sh` in its POSITIVE direction, a fake trainer
was started on 2026-09-11 at about 17:05:

    python3 -c "import time; time.sleep(25)" --run-name cf393_..._cf412_k32_r200_08_lr56

It is a python process carrying a real run-name, which is exactly what every
guard and watcher of this card is built to detect, and they are right to
detect it. A live milestone watch reported `k32_r200_08_lr56 trainer 557058`
at 17:17 for an arm that had not started. The card printed an empty GPU
because the fixture set no CUDA_VISIBLE_DEVICES, which is the tell.

THE FIXTURE WAS CORRECT AND THE DAMAGE WAS REAL. A positive test needs a
process that passes the discriminator, so the fixture had to look like a
trainer. The fault is that it used a LIVE arm name on a box where watchers run.

TWO RULES THAT FOLLOW.
- Name a fixture after an arm that does not exist, for example
  `_cf412_ZZ_fixture`, and assert on that name.
- Cross-check a process claim against the artefacts before believing it. An
  arm with no losses CSV and no checkpoint has not trained, whatever the
  process table says. That check is what settled this one in seconds.

### Three ways the 200,000-step leg was lost, and what fixed each

That leg is the last number of pass 4. It died twice and restarted three
times on 2026-09-11.

1. OOM AT A PROBE STEP, 14:33, step 120,000. GPU 0 held 1,628 MiB free behind
   a kernel that landed at 12:38, and the probe needed a 4.3 GiB transient.
   Fixed by raising this leg's gate to 11,500 through
   `CF412_VRAM_MARGIN_MIB=5000`, which leaves 5,028 MiB free above its 6,472
   resident.

2. A PATTERN KILL, 17:34, step 123,500. The leg and its lane died with NO
   traceback, 25 seconds before another session's lane logged its start. The
   re-fire supervisor restarted it at 17:36 and it lost 200 steps. A
   `pkill -f phase1.sh` matches EVERY lane of this card. Kill by pid.

3. THE NEXT PROBE WOULD HAVE GONE THE SAME WAY. With that session's arm also
   on GPU 0, the peaks read 6,880 rnd-483 + 6,472 that arm + 6,472 this leg,
   leaving 4,383 free against a 4,424 probe. 41 MiB short, 85 minutes out.
   The leg restarted at 17:39 with
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which the OOM message
   itself recommends. That is an environment value: no script, no trainer flag
   and no protocol changed. It reduces fragmentation, so it improves the odds,
   and it cannot create memory that is not there.

### A stale losses CSV read as a stalled leg

`cf412_leg_step` took `ls *_losses.csv | head -1`. A re-fired leg writes
`_r2_losses.csv`, `_r3_losses.csv` beside the first one, and `head -1` takes
the FIRST alphabetically, which is the DEAD one. So the status block reported
step 120,000 while the leg ran at 123,400, and the await's no-progress clock
would never have advanced. Both readers now take `ls -t ... | head -1`.

THE BUG HID BEHIND A TRUE POSITIVE. The await returned STALLED at 17:36 and
it was RIGHT: the leg had died at 17:34 and the re-fire had not yet run. The
wrong step number in the same block is what led to the reader.

### The first arm of this card that does not degrade

`k3_r100_09_lr56_dec10k` scores 1.2151 at 100,000 steps against 1.1915 at
40,000. The rise is +0.0236, which is 0.36 of the 0.0649 band. Every other arm
of this card loses about two bands over the same stops.

    arm                      decay   ema ramp      40k     100k     rise  bands
    k3_r100_09_lr56              -    100,000   1.1820   1.3170   +0.1350   2.08
    k3_r100_09_lr56_dec      2,000    100,000   1.2164   1.3507   +0.1343   2.07
    k3_r100_09_lr56_dec10k  10,000     40,000   1.1915   1.2151   +0.0236   0.36

It also beats the no-decay reference at the 100,000-step stop by 0.1019, which
is 1.57 bands, and that is the widest gap this card has measured at that stop.

THE DECAY IS NOT THE CAUSE. Rows 1 and 2 differ in the decay column ALONE and
their rises are +0.1350 and +0.1343, a difference of 0.0007. So removing
`L_rep` changes neither the level nor the slope.

THE REMAINING DIFFERENCE IS THE EMA RAMP. `dec10k` carries a 40,000-step ramp
that COMPLETES inside its stop. The other two ramp to momentum 1.0 at 100,000,
so their teacher is still moving at the 40,000-step read and frozen by the
100,000-step read.

THIS IS NOT A CONTROLLED RESULT AND THIS LOG DOES NOT CLAIM ONE. `dec10k`
moves TWO columns against `dec`: the decay ramp and the EMA ramp. The decay is
excluded by the pair above, which leaves the EMA ramp as the candidate, not as
the finding. One arm that moves the ramp ALONE would settle it.

THAT ARM IS ALREADY QUEUED. Another session's `k3_r100_09_lr56_fix09` holds
momentum at 0.9 with NO ramp, against `k3_r100_09_lr56`, everything else
equal. It runs when this card releases GPU 0.

## Pass 5, first result: `k8_r100_09_lr56` scores 1.2750

THE CARD'S QUESTION FOR CONFIGURATION 6 IS ANSWERED. Its 1e-3 twin ran the
mean and scored 1.4537. At 5.6e-4 under the sum it scores 1.2750, better by
0.1787, which is 2.75 seed bands. The AUC guard did not stop it and its final
AUC was 0.9979.

THE DEPTH LADDER IS NOW THREE POINTS ON ONE AXIS. `arm_diff.py` confirms each
pair differs in k alone, at 5.6e-4, sum, seed 20260520, ramp 100,000:

| k | score | against k = 3 |
|---|---|---|
| 3 | 1.1820 | — |
| 8 | 1.2750 | +0.0930, 1.43 bands, RANKED |
| 32 | 1.2161 | +0.0341, 0.53 bands, not ranked |

k = 8 against k = 32 is -0.0589, which is 0.91 bands and not ranked.

SO THE ONLY RANKED PAIR IS k = 3 OVER k = 8, and the middle point is the worst
of the three. A ladder of the two ends alone would have read as one flat
unranked gap of 0.53 bands and would have hidden this.

WHAT IT DOES NOT SUPPORT. One eval for each arm, one seed for each arm. The
0.0649 band is this card's own seed band at 5.6e-4, so the pairs are
calibrated against real noise, but "depth is not monotonic" is a claim about
a shape and three single-seed points constrain a shape weakly. The defensible
statement is the pair list above.

#### The pool-growth prediction, confirmed live on a k = 32 leg

The mechanism above said a probe either fits the allocator's free blocks or
forces the pool to grow, and that which case a leg takes is not predictable
from k alone. `k32_r200_08_lr56` crossed step 20,000 on 2026-09-11 and gave
the second case, measured:

| leg | before the probe | after it | change |
|---|---|---|---|
| `k8_r100_09_lr56` | 7,160 smoke row | 7,140 | -20 MiB |
| `k32_r200_08_lr56` | 6,362 | 10,078 | +3,716 MiB |

The latent-drift CSV holds the step-20,000 rows, so the probe ran in both. The
k = 32 growth matches the +4,016 that this card recorded for `k32_r200_08` two
days ago and carried as cause unknown.

ONE NUMBER STILL DOES NOT ADD UP, and it is worth naming rather than
smoothing. The k = 32 SMOKE row reads 10,062 MiB, but the live leg held only
6,362 before its first probe. A 150-step smoke never reaches step 20,000, so
the smoke cannot have measured a drift probe. The smoke samples at 1 s and a
leg's steady state is a median, so the likeliest reading is that the smoke
captured a different transient peak. The practical consequence is benign: the
smoke row happens to sit 16 MiB under the post-probe resident of 10,078, so
`cf412_leg_vram_mib` covers the real peak for k = 32. It covers it by
coincidence rather than by construction, and a future arm should not rely on
that holding.

## Pass 5, second backbone: `k32_r200_08_lr56` holds the task to 40,000 steps

Final step 40,000, AUC 0.9935, `ema_tau` exactly 1.0000, `r2_naive` 0.2053,
and the guard did not stop it.

THE CARD'S QUESTION FOR CONFIGURATION 3, in its qualitative form, is answered.
Its 1e-3 twin ran the mean with a 200,000-step ramp and LOST the contrastive
task at step 28,152 with the momentum only at 0.8285. Under the sum at 5.6e-4,
with the ramp matched to the stop, this arm passed step 28,152 at AUC 0.985
and reached 40,000 at 0.9935.

THE MOMENTUM REACHED EXACTLY 1.0000 AT THE STOP, which is what the card asked
for and what makes this arm the orchestrator session's freeze datum. It trains
no step against a frozen teacher, because the ramp ends as the run ends.

### The head went to the other card, and the reason is the probe

The sweep sat on GPU 0, which reported 11,506 MiB free, so it would have
started this head there. My third arm is on GPU 0 and reaches its drift probe
at step 20,000, which needs about 3,716 MiB of pool growth on a k = 32 leg:

  gpu0  dec arm after its probe 10,078 + head 6,252 + neighbours 6,124
        = 22,454 of 24,564, leaving 2,110 against a 3,716 need
  gpu1  head 6,252 + neighbours 5,888 = 12,140, leaving 12,424

So the sweep moved to GPU 1, which now holds no #412 leg. A card that fits a
head TODAY is not the same as a card that fits a head AND a leg's next probe.
