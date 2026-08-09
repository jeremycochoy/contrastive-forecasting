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
