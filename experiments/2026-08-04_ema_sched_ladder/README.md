# EMA schedule + backbone ladder (#393)

Retrain the ten cells that placed in the two parent reports, with one
change to the EMA teacher, and train each as long as the data allows.
Measure GM-Relative MASE at every stop, once through the student encoder
and once through the teacher encoder.

The report goes in `reports/2026-08-04_ema_sched_ladder/`. This directory
holds the scripts, `results/` and `plots/`.

## The change

α is the weight the teacher keeps on its own previous value in
θ_T ← α·θ_T + (1−α)·θ_S. Every run before this one held α at 0.9. Here it
rises linearly from 0.9 at step 0 to 1.0 at step 100k, then holds at 1.0:
the teacher stops moving from 100k on.

The ramp is anchored to step 100k, not to a run's budget
(`--ema-tau-ramp-steps 100000`). The ten runs stop at different steps, and
a budget-relative ramp would put each of them on a different α curve.

![α against training step](plots/alpha_schedule.png)

α at each stop, from `results/alpha_schedule.csv`:

| stop | α |
|---|---|
| 40k | 0.9400 |
| 100k | 1.0000 |
| 200k and beyond | 1.0000 |

At bb40k the teacher is still moving. Record the number anyway.

## The ten cells

`arm` is the #379 recipe
([`run_arm.sh`](../2026-07-21_split_pred_rep_small/scripts/run_arm.sh));
`align` is L_align's target. The `arm6_v2 combab` pair runs first: that
cell leads both parent reports, and the pair gives the head-to-head first.

| # | cell | arm | align target |
|---|---|---|---|
| 1 | `arm6_v2_combab_alignS` | `arm6_v2_combab` | student |
| 2 | `arm6_v2_combab_alignT` | `arm6_v2_combab` | teacher |
| 3 | `arm5_combab_alignS` | `arm5_combab` | student |
| 4 | `arm5_combab_alignT` | `arm5_combab` | teacher |
| 5 | `arm6_v2_ncpc_alignS` | `arm6_v2_ncpc` | student |
| 6 | `arm6_v2_ncpc_alignT` | `arm6_v2_ncpc` | teacher |
| 7 | `arm6_v2_nse_alignS` | `arm6_v2_nse` | student |
| 8 | `arm6_v2_nse_alignT` | `arm6_v2_nse` | teacher |
| 9 | `arm4_combab` | `arm4_combab` | no `L_align` |
| 10 | `arm1_nse` | `arm1_nse` | no `L_align` |

`arm1_nse` never reads the teacher in its loss. It is the control: the
schedule should not change it. Its teacher still exists and still updates,
so it still gets a teacher head.

Every cell starts fresh at step 0, once. No cell resumes a #379 or #388
checkpoint: α matches those runs at step 0 only, and rises from step 1, so
their trajectories are not on this schedule.

## The ladder

Each cell is one continuous run. It trains to a stop, checkpoints,
evaluates, then resumes from that checkpoint with its saved optimizer
state. Stops are 40k and 100k unconditionally, then 100k at a time.

A stop is a process boundary, so three things have to survive it, and
[`tests/test_393_resume_leg.py`](../../tests/test_393_resume_leg.py) runs
two real legs on CPU to show they do. The step counter is global
(`start_step` comes from the checkpoint, and the loop runs
`start_step+1 … total_steps`), so α keeps climbing across the seam instead
of restarting at 0.9 each leg. `--total-steps` is an absolute step, not a
number of extra steps. `--resume` restores AdamW's moments, the step
counter and the RNG state from the optimizer companion file.

Each leg writes into its own `leg_<target>k/` directory. That is not
tidiness: train.py renames a run to `<name>_r2` when the save dir it is
given already holds `<name>_*.pth`, so a shared directory would move every
checkpoint past the first leg out from under the ladder and the eval.

At every stop the checkpoint gets two quantile heads, trained separately:
one on the student encoder, one on the teacher's. Each is evaluated
through the encoder it was trained on. Head budget is 15,000 steps at
bb40k and 30,000 from bb100k on.

The extend rule compares each head against its own value at the previous
stop, and first applies to the 100k-to-200k decision:

| previous → current | backbone | heads evaluated from then on |
|---|---|---|
| both heads down | extend | both |
| one head down | extend | that head only |
| neither down | stop | — |

Dropping a head changes nothing about the backbone: same loss, same run,
same schedule.

## The step cap

A run extends only while every sample it has seen was shown once.

`small_v1` holds **42,571,692 rows** across 4,274 shards
(`small_v1/manifest.json`, read by
[`scripts/confirm_row_count.py`](scripts/confirm_row_count.py) into
`results/dataset_rows.json`, which states this basis in full).

The cap is `total_rows // hf_rows_per_step`, and `hf_rows_per_step` counts
**real dataset rows only**. At batch size 64 that is the whole batch:

| block | rows | from |
|---|---|---|
| real | 64 | the dataset |
| synthetic | 0 | forked-ARMA |
| crossfade rows | 0 | `--crossfade-ratio 0` |
| crossfade triplets | 3 | blended from the 64 real rows |

`--mix-ratio 0.0078125` looks like it should take 1/128 of the batch, but
train.py takes `synth_bs = int(round(batch_size * mix_ratio))`, and
`64 × 0.0078125` is exactly 0.5, which Python rounds half-to-even to **0**.
The synthetic fraction rounds away at this batch size. The 3 triplet rows
are blended from the same 64 real rows and appended on top, so the model
sees 67 rows per step while the dataset gives up 64. One pass is therefore

    42,571,692 / 64 = 665,182 steps.

Were any part of the batch genuinely synthetic, `hf_bs` would drop below
64 and the cap would go **up**. Re-run `confirm_row_count.py` if the batch
size or the mix ratio ever changes.

**The cap survives the resumes.** A cell is six or more `train.py`
processes, so "one pass" only means one pass if each leg picks the stream
up where the previous one left it. It does. train.py counts consumed rows
into `hf_rows_consumed`, saves it in every checkpoint's optimizer
companion, restores it on `--resume`, and hands it to the dataloader as
`skip_rows`; the loader then opens the stream at that absolute row —
seeking to the shard that contains it rather than replaying from the head.
Past one pass it wraps modulo the dataset and starts repeating rows, which
is the behaviour the cap exists to stay ahead of. Pinned by
`TestOnePassSurvivesTheResume` and `TestTheStreamOpensAtTheRowOffset` in
[`tests/test_393_resume_leg.py`](../../tests/test_393_resume_leg.py).

**Cross-check.** The 2026-05-03 run called 167,000 steps at batch size 256
one full epoch of the same `small_v1` at mix-ratio 0.0. The same row count
gives `42,571,692 / 256 = 166,295`, so the two **agree** to 0.4% and that
figure was its rounded-up form. It does not transfer here: batch 64 means
four times the steps per epoch.

## Running it

```bash
export WT=$HOME/workspaces/contrastive-forecasting        # a persistent checkout
export RUNS=/home/jupyter/checkpoints_backup/cf-393       # outside the checkout
export BB_GPU=0

python3 scripts/ladder.py --print-cap
python3 scripts/ladder.py --cells arm6_v2_combab_alignS,arm6_v2_combab_alignT
```

`ladder.py` drives the whole loop and is resumable: it skips a stop whose
checkpoint or score is already on disk, so a crashed run picks up where it
stopped. `--max-stop` caps a session when splitting a cell across
machines, and writes a `session_end` row to `decisions.csv` so the record
distinguishes a paused cell from a finished one.

`results/HOLD_ABOVE` is the same cap, read fresh at every stop instead of
fixed when the driver starts, so a spend order decided after the cells are
already climbing still reaches them. Both `ladder.py` and `run_leg.sh`
read it; `run_leg.sh` is what reaches a driver already running, because it
is a new process on every leg. Delete the file to lift the cap, then
re-run `ladder.py` for the cell — it replays from step 0, skipping every
stop already on disk.

## One CUDA context per device

A vast.ai box comes up in `Exclusive_Process` compute mode and the
container cannot change it, so a second cell started on a busy GPU does
not queue: it dies inside `.to(device)` with "CUDA-capable device(s) is/are
busy or unavailable", one second after launch, having cost no GPU time.
That is what makes it quiet, and it is how `arm5_combab_alignT` sat dead
for three hours on 2026-08-05.

`scripts/gpu_gate.sh` sits in front of every CUDA process in `run_leg.sh`
and `eval_stop.sh`. It takes an exclusive `flock` on a per-GPU file, held
on fd 9 for the life of the leg or the eval, then waits for any compute
app it did not launch to leave the device. The lock orders the cells this
experiment starts; the drain covers another agent session's processes and
anything left over from an earlier attempt, which the lock cannot see.

On a `Default`-mode GPU it returns immediately. That is the point, not a
fallback: elisa runs two cells per 4090 deliberately, and gating there
would halve the box's throughput.

`RUNS` holds everything that cost GPU time — backbone checkpoints, the
quantile heads, the GIFT-Eval outputs, the per-stop scores:

```
$RUNS/<cell>/leg_<N>k/         backbone + optimizer + losses CSV per leg
$RUNS/<cell>/eval/bb<N>k_<enc>/  head, its encoder marker, gift/, eval.log
$RUNS/<cell>/eval/score_bb<N>k_<enc>.txt
```

It must be outside the checkout and outside `/tmp`; both launchers refuse
anything else. `git worktree remove --force` deletes every untracked file
under the checkout, which is how an 80 MB backbone was lost in Apr 2026.
Only `results/*.csv` and the plots live in the repo.

Elisa carries two RTX 4090s; run one cell per GPU with `BB_GPU`. A vast.ai
instance takes more cells on top — walk through
[`REMOTE_LAUNCH_CHECKLIST.md`](../REMOTE_LAUNCH_CHECKLIST.md) first, and
use `vastrun-kit` commands only.

## Syncing a remote run

Every remote run needs `sync/sync_loop.sh` running for its full duration,
on the machine that owns the persistent checkout:

```bash
REMOTE_HOST=elisa SSH_USER=jupyter \
REMOTE_DIR=~/workspaces/contrastive-forecasting/experiments/2026-08-04_ema_sched_ladder \
REMOTE_RUNS=/home/jupyter/checkpoints_backup/cf-393 \
LOCAL_DIR=/abs/path/experiments/2026-08-04_ema_sched_ladder \
  nohup setsid bash sync/sync_loop.sh > sync/sync_loop.log 2>&1 &
```

15-minute ticks, atomic `.tmp` → `mv` pulls with per-class size floors,
optimizer files alongside every backbone. A cell's stop list is not known
in advance, so each tick asks the remote what exists rather than guessing
filenames. **Verify the first tick by `ls`, not by reading the log.**

## Files

| path | what |
|---|---|
| `scripts/ladder.py` | the driver: stops, the extend rule, the step cap |
| `scripts/run_leg.sh` | one backbone leg of one cell, up to a target step |
| `scripts/eval_stop.sh` | one head + GIFT-Eval B4 at one stop, one encoder |
| `scripts/gpu_gate.sh` | wait for a free device rather than die on a busy one |
| `scripts/leg_paths.sh` | durable root, per-leg dirs, checkpoint-by-step, the score read |
| `scripts/confirm_row_count.py` | reads the dataset manifest, derives the cap |
| `scripts/alpha_schedule.py` | the α-vs-step record and its plot |
| `scripts/smoke_e2e.sh` | CPU end-to-end check: ramp, resumed leg, both encoders |
| `scripts/seed_replicates.sh` | the bb100k heads again at two more seeds, for the six sub-noise cells |
| `scripts/seed_replicates_bb40k.sh` | the same for the bb40k end, for the six replicated cells |
| `scripts/bb40k_supervisor.sh` | keeps that pool fed across passes until every job is scored |
| `scripts/t1_watch.sh` | one event line per job scored or failed; exits when all 24 are in |
| `scripts/deploy_seed_box.sh` | one rented box, bootstrapped and given its cell |
| `scripts/seed_spread.py` | mean, sd and branch survival across the three seeds |
| `scripts/paired_delta.py` | the delta with a standard error from **both** of its ends |
| `scripts/plot_paired_delta.py` | that delta as a forest plot, next to the denominator it replaces |
| `scripts/verify_align_target_port.sh` | the four teacher cells run #392's objective, checked by diff |
| `sync/sync_loop.sh` | 15-min pull of every artefact from a remote host |

### Which file is the experiment

Eight machines each ran their own `ladder.py`, and each kept its own
`results/ladder.csv` and `results/decisions.csv` holding **only the cells
that machine was given** — elisa's copy carries 41 of the 45 result rows.
Replaying a finished cell re-appends the decision it already wrote, so elisa's
40 decision rows carry 29 distinct (cell, stop, branch) decisions; every repeat
agrees with the row it repeats, and `merge_pooled.sh` collapses them on that
key. Elisa holds 29 of the 31 pooled decisions because it did not run every
run, not because it holds anything the pool does not. The per-machine files
keep the line endings of the machine that wrote them. Those two
are gitignored: a per-machine file under a shorter name, at the same level as
the pooled file, reads like the experiment and is not. What is
committed:

| path | what |
|---|---|
| `results/ladder_all.csv` | **the results table.** 45 rows: ten runs × two heads at bb40k and bb100k, plus the five bb200k rows |
| `results/stop_reason.csv` | **the decision table.** One row per cell: the branch, and what actually ended it |
| `results/seed_spread.csv` | the three-seed spread of the **bb100k** end, per cell per head |
| `results/paired_delta.csv` | **the uncertainty table.** One row per (cell, head): three paired deltas, their mean, SE, t and sign count |
| `results/paired_branches.csv` | the branch re-derived at each seed with both ends at that seed |
| `results/audit_scores_paired.txt` | the bb40k replicates, each traced to its summary |
| `results/decisions_all.csv` | the pooled decision *log*, several rows per stop, each marked `rule` or `park` (23 and 8) |
| `results/per_machine/*.csv` | what each machine contributed, verbatim |
| `results/audit_scores_all.txt` | every published row traced to the GIFT-Eval summary that produced it |
| `results/align_target_port.txt` | the teacher-align port, diffed against PR #392 |
| `results/parent_seed_spread.csv` | where the imported 0.0908 noise figure comes from |

## The teacher-align port

The four `alignT` cells need `--align-target teacher` while the main
contrastive loss is on. That combination is rejected on `experiments`; the
capability sits in open PR #392, and #390's teacher runs — the parent this
card compares against — were trained on it. All four cells died at step 0
until the minimal piece was ported here in `adde0cc`.

If the port differs from #392 anywhere that touches the objective, the four
teacher cells are not comparable with the parent and the head-to-head this
card exists to make is not one. So it is checked by diff rather than by
reading: [`scripts/verify_align_target_port.sh`](scripts/verify_align_target_port.sh)
asserts `src/loss.py` is byte-identical to #392's and that every `train.py`
line naming `align_target` matches, and writes
[`results/align_target_port.txt`](results/align_target_port.txt). It is
re-runnable, so the claim can be checked again rather than believed.

## Protocol constants

Unchanged from the 2026-08-04 protocol: 97 GIFT-Eval configs, official B4
strategy, forecast horizon 16, protocol head seed 20260722, backbone
`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, batch_size=64,
seed=20260520`, dataset `gift-pretrain-full-4096 / small_v1`.

The head keeps `--grad-clip 1.0`. The project rule bans grad clipping; the
previous study kept it for comparability, so this one does too, and the
report says so.

## Caveats for the report

- The head budget changes at bb100k, 15k → 30k. Any statement across that
  boundary moves two things at once.
- The extend rule is a strict `<` on a difference whose noise it does not
  know. The imported bound is **0.0908**, the largest head-seed range in the
  #390 parent report — `arm5_nse`, backbone 200k, teacher, values 1.7979 /
  1.8655 / 1.8887 across seeds 20260722 / 20260723 / 20260724. That row and
  the seven beside it are copied verbatim into
  [`results/parent_seed_spread.csv`](results/parent_seed_spread.csv) with
  the commit they came from, so the number is not a figure in a comment.
  Six of the ten cells moved less than it on **both** heads, so the run
  measures its own spread rather than borrowing that one: the bb100k heads
  are retrained at seeds 20260723 and 20260724 (the parent's two extra
  seeds), giving three seeds per cell per head. See
  [`scripts/seed_replicates.sh`](scripts/seed_replicates.sh) and
  [`results/seed_boxes.txt`](results/seed_boxes.txt). Every other head
  setting is held: same bb100k backbone, 30,000 steps, same encoder
  pairing, same 97 configs, same denominator. A cell's replicates run on
  the same GPU model its first seed did, so the spread measures the seed
  and not the hardware.

- **Never divide a paired delta by the spread of one of its ends.** The
  quantity the extend rule reads is `bb100k − bb40k`. Replicating the
  bb100k end bounds that end only, and the two ends do not even measure the
  same thing: every bb40k row is a 15,000-step head and every bb100k row is
  30,000. The parent's own 15k-step bb40k heads reach sd 0.0380, which is
  2× to 20× the in-study bb100k sds, so a σ taken against the bb100k spread
  is a property of an unmeasured quantity rather than of the data. The
  bb40k heads are therefore retrained at the same two seeds and the same
  15,000-step budget — [`scripts/seed_replicates_bb40k.sh`](scripts/seed_replicates_bb40k.sh) —
  and every delta is stated per seed, `bb100k(s) − bb40k(s)`, with the SE
  taken from the spread of those three paired deltas
  ([`scripts/paired_delta.py`](scripts/paired_delta.py)). Either both ends
  carry a measured spread, or the report states the delta with no
  significance claim.

- The rule is a **sign test**, not a t-test: `ladder_decision` asks only
  whether each head fell. So `n_down`, the number of seeds at which a head
  went down with both ends at the same seed, is reported next to every t. A
  head at 3/3 or 0/3 has a reproducible direction whatever the t says; one
  at 1/3 or 2/3 does not, and the branch it produced was decided by the
  head seed.

- **Which cells carry replicates.** Six cells were replicated at the
  **bb100k** end: `arm6_v2_combab_alignS`, `arm6_v2_combab_alignT`,
  `arm5_combab_alignS`, `arm5_combab_alignT`, `arm6_v2_nse_alignT`,
  `arm6_v2_nse_alignS`. The bb40k end was planned at the same six cells,
  which is 24 jobs; the card specifies one head seed, so the pool was
  stopped at **9 of 24**. The measured census, from `paired_delta.csv`:
  four head-rows carry three seeds at **both** ends
  (`arm6_v2_combab_alignS` and `arm6_v2_combab_alignT`, both heads), one
  carries two (`arm5_combab_alignS` student), and the other **fifteen**
  carry a one-seed bb40k end. Only the first five get an SE from their own
  spread. The fifteen are stated as measured once, with **no significance
  claim** and no σ; `paired_delta.csv` marks them `n_seeds=1` and its
  verdict column says so on every such row.

- **Three cells split their bb40k seeds across GPU models.**
  `arm5_combab_alignS`, `arm5_combab_alignT` and `arm6_v2_nse_alignT` trained
  seed 20260722 on a rented RTX 5090 and seeds 20260723/24 on elisa's RTX
  4090, so at those two seeds a 4090 bb40k head is paired with a 5090 bb100k
  head. The study does not measure that term: a 4090 and a 5090 are the same
  PyTorch on the same consumer architecture with a seeded head. It is stated
  rather than corrected — `gpu_split` in `paired_delta.csv` marks exactly
  these three cells, and no score anywhere is adjusted for the card that
  produced it. A GPU term, were it real, would enter as a common shift on the
  two replicate deltas and not on the third, which inflates the delta sd and
  the SE — the resulting t is conservative, not optimistic.

- `arm4_combab` carries `--tau 1.0` in its loss flags and does not run at
  τ=1.0. The shared `--tau 0.10` is passed after the per-cell flags and
  argparse keeps the last value, so the cell trains at τ=0.10. This is not
  a defect introduced here: #379's `run_arm.sh` orders the two the same
  way, so its `arm4_tr1` ran at τ=0.10 too. It is reproduced deliberately,
  because a cell that differed from its parent would not be comparable to
  the parent reports. Do not describe this cell as a τ=1.0 arm.
