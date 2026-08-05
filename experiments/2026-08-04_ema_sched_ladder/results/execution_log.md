# #393 execution log

Operational events only. The report carries the science; this carries what
happened to the machines, so the report does not have to.

Times are elisa's. The vast.ai containers run one hour behind, so a log
line stamped `03:12` on a vast box is `04:12` here. Nothing depends on the
skew, but do not diff timestamps across machines without correcting it.

## 2026-08-05 00:49 — arm5_combab_alignT died at step 0 on vast A

`torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or
unavailable`, raised inside `.to(device)` at model init. The box is in
`Exclusive_Process` compute mode and `arm5_combab_alignS` already held the
GPU, so the second CUDA context could not be created. The cell was dead
one second after launch and stayed dead for three hours: it consumed no
GPU time, so nothing downstream looked wrong.

Fixed by `scripts/gpu_gate.sh`, which every CUDA process in `run_leg.sh`
and `eval_stop.sh` now passes through. A queued cell waits for the device
instead of dying on it. The cell was restarted on its own box, cf393-c.

No other cell died this way. `arm6_v2_combab_alignT` also exited rc=1 at
01:30, but for an unrelated reason — `--align-target teacher` needed the
trainer change in `adde0cc` — and was relaunched at 01:34.

## 2026-08-05 03:57 — scp corrupted a running eval_stop.sh on vast A

Deploying the gate with `scp` overwrote `eval_stop.sh` **through its
existing inode** while a copy of it was running. bash reads a script
lazily, holding a byte offset into that inode; the insertions shifted
everything after them, so when the 15,000-step head finished and bash read
its next command it landed mid-word inside `--head-num-layers`:

    eval_stop.sh: line 97: ad-num-layers: command not found
    [03:12:58] head-train rc=127

`ladder.py` raises `SystemExit` on a non-zero worker, so the
`arm5_combab_alignS` driver died. The head itself had trained cleanly
(`rc=0`, `_final.pth` written), so the loss was the GIFT-Eval only, not
the 38 minutes of head training.

Three consequences, all handled:

- **vast B was armed with the same fault** and 90 seconds from hitting it.
  Restoring the original bytes *through the same inode* (`cat >`, not
  `mv`) put the running bash's saved offset back over the text it was
  parsed from, and its eval continued. The new version went on after.
- **vast A started a duplicate cell.** The 02:22 session's
  `queue_remote.sh` had been parked in its `while pgrep ladder.py` wait
  since 00:49; the driver's death released it and it began
  `arm5_combab_alignT` — a cell already running on cf393-c. Killed at
  ~1,400 steps, before any checkpoint, along with the queue. Its partial
  directory on vast A was removed so the box holds no stale copy.
- **elisa was untouched.** The local Edit tool writes a new inode and
  renames over the old one, so the three running cells kept reading the
  bytes they started with. That contrast is the fix: `scripts/deploy_scripts.sh`
  uploads to a temporary name and `mv`s it into place, so a remote deploy
  behaves the way the local edit already did.

## 2026-08-05 04:00 — four boxes added, and why

The 02:22 layout had two cells queued per vast box and two more behind a
waiter on elisa GPU 0 that was watching for the #390 session to release
the device. The spend order on the PR is "all ten cells to bb100k first",
and that layout could not deliver it: two cells had not started at all,
and the waiter's condition was outside this experiment's control.

cf393-c/d/e/f took `arm5_combab_alignT`, `arm6_v2_nse_alignT`, `arm1_nse`
and `arm4_combab`, one cell each. The waiter was killed so it could not
double-launch the two cells it was holding. `arm4_combab` moved off elisa
because it was queued *inside* `--cells arm6_v2_nse_alignS,arm4_combab`,
and that driver predates `HOLD_ABOVE`: it will be refused its 200k leg by
`run_leg.sh` and exit, taking the queued cell with it.

## 2026-08-05 04:50 — GIFT-Eval is the dominant cost, and it needs no GPU

Measured on cf393-a: 4 of 97 configs in 777 s, so 194 s per config, or about
5.2 h per GIFT-Eval by flat extrapolation and 4.9 h weighted by dataset
bytes. cf393-b agrees independently. The first four configs are all
`loop_seattle` and the rate improved from 5T to daily frequency, so the true
figure is likely nearer 4 h; `electricity` is 42% of config bytes and had not
been reached.

Four evals per cell (two stops, two heads) puts a cell at 23 to 28 h, and ten
cells at 234 to 282 machine-hours. That is more than double the ~110 the
brief allowed, and $98 to $119 against a $80.26 envelope.

While evaluating, the process holds 100% of one core, loadavg 1.00 on 8
cores, GPU at 30%, 696 MiB of VRAM. It is single-core CPU bound. On a vast
box that is expensive: `Exclusive_Process` means the eval owns the only CUDA
context, so a 5090 is rented and left 30% busy for most of a cell's life.

Elisa has 32 cores at loadavg 4 and costs nothing. Splitting `eval_stop.sh`
so head training stays on the GPU and GIFT-Eval runs on elisa leaves 74
machine-hours (~$31) on vast and moves ~208 core-hours to free compute.

Not implemented in this dispatch: four evals were in flight and about to
produce the study's first numbers. Recorded here and raised on the PR.

## 2026-08-05 05:10 — the 5.2 h figure was wrong by about 3x

**Superseding the section above.** 194 s/config was extrapolated from the
first four of 97 configs, and config cost is not remotely uniform: it spans
0.4 s to 1537 s, and twenty configs carry 89% of the work. The four that
were measured are among the most expensive there are.

A **completed** 97-config eval of this same B4 protocol totals **13,303 s =
3.70 h** (2026-07-09, `sync_2026-07-03_b1024_traj_ckpts`, a wider backbone
at forecast-len 128). Per-config times from that run are now committed as
`results/config_costs.csv`.

Measured against it on this study's `d_model=64` backbone, six configs at a
time:

| where | wall | projected 97 configs |
|---|---|---|
| elisa, one 4090 (contended) | 58.7 s | 1.73 h |
| elisa, 4 CPU threads | 81.4 s | 2.39 h |
| elisa, 1 CPU thread | 97.3 s | **2.86 core-h** |
| vast 5090 (from live per-config times) | — | ~1.1 to 1.5 h |

So phase 1 is 40 evals x 2.86 = **114 core-hours**, not 208, and had it
stayed on vast it would have been ~52 machine-hours (~$23), not $98 to $119.
The decision to move it still holds — it frees ~$23 of a $76.91 balance and
takes the eval off the GPUs entirely — but it was never the difference
between fitting the envelope and not.

**Recorded because it changed a decision.** A four-point extrapolation over
a distribution with an 89%-in-20-configs tail is not a measurement, and this
one was quoted to a decision-maker as though it were.

## 2026-08-05 05:05 — GIFT-Eval moved to elisa's CPUs

Option 1 from the PR, implemented. Head training stays on the rented GPU;
only the eval moves.

`results/EVAL_PLACE` selects where, read fresh at every stop like
`HOLD_ABOVE`, so it reaches drivers that started hours earlier: `inline`
(this box's GPU, the old behaviour and the default), `local_cpu` (here,
sharded across cores) and `broker` (on elisa). elisa is `local_cpu`, all six
vast boxes are `broker`.

`scripts/eval_local.sh` splits the 97 configs into four cost-balanced shards
(`scripts/shard_configs.py`, LPT over `config_costs.csv`, exact partition
within 0.1% of ideal) and runs them at one thread each on CPU. The shards'
CSVs are merged and the aggregate comes from re-running the official script
with `--resume`, which finds all 97 done and writes `summary.txt` through
the same code an unsharded eval would have used.

`scripts/eval_broker.sh` on elisa polls the six boxes. A vast container has
no route back here, so the direction is always elisa -> vast: the box leaves
an `EVAL_REQUEST` on its own disk and waits; the broker pulls the 5.2 MB
backbone and 0.45 MB head with `safe_pull.sh`, evaluates, then pushes
`summary.txt`, `all_results.csv` and last of all the score, whose appearance
is what releases the waiter.

The cap is **3 concurrent evals x 4 shards = 12 of elisa's 32 cores**
(`scripts/eval_slot.sh`, a flock counting semaphore). That leaves 20 cores
free against the eight the brief requires, and GPU headroom is total rather
than partial: the shards run `--device cpu`, so both 4090s stay with the
cells training on them.

The five evals already in flight at 05:00 were left alone, on the GPUs, per
the brief.

## 2026-08-05 05:05 — two cells were about to become one cell twice

`ladder.py` reaches `HOLD_ABOVE` by *returning* from `climb()`, deliberately,
so the cells behind it in a `--cells a,b` invocation still run. The elisa
driver was started as `--cells arm6_v2_nse_alignS,arm4_combab`, so at
nse_alignS's 100k stop it would have begun `arm4_combab` on elisa — a cell
vast F had been training since 03:11. The 04:10 layout note asserted the
driver would exit instead; it would not have.

Two copies of one cell write the same `<cell>/leg_<N>k/` filenames into two
roots the sync loops merge, and the second copy's checkpoints are
indistinguishable from the first's.

`results/cell_claims.txt` now names the owning machine for each of the ten
cells and `results/MACHINE` names each box. `run_leg.sh` checks both on
every leg and exits non-zero on a mismatch, which stops the driver rather
than letting it move on. The check lives in `run_leg.sh` and not in
`ladder.py` because `run_leg.sh` is a new process on every leg: it reaches a
driver that has been running for four hours, and `ladder.py` in memory does
not.

## 2026-08-05 05:05 — an interrupted eval could never be retried

Killing the broker mid-eval left three shards with a zero-byte
`all_results.csv`: `eval_gift_eval_official.py` reads that file for
`--resume` and then reopens it `"w"`, so a kill in that window truncates it.
Every subsequent `--resume` then died on `next(reader)` with
`StopIteration` — permanently, on heads that had already cost 15,000 GPU
steps each.

`eval_local.sh` now removes an empty shard CSV and drops short rows before
launching, so an interrupted eval resumes instead of failing forever. Found
by killing a worker on purpose; the repair was verified on the four
truncated files it produced.
