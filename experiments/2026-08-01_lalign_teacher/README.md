# L_align on the EMA teacher — the 10 L_align cells of #379, rerun (#390)

Parent: [`experiments/2026-07-21_split_pred_rep_small/`](../2026-07-21_split_pred_rep_small/)
· Parent report: [`reports/2026-07-21_split_pred_rep_small/small_long.md`](../../reports/2026-07-21_split_pred_rep_small/small_long.md)

## What changes vs #379

One flag. #379 trained

```
L_align = 2 − 2·cos(f_t, stopgrad(h_{t+1}))          # h = student encoder
```

The intended term targets the EMA teacher:

```
L_align = 2 − 2·cos(f_t, h_teacher_{t+1})
```

`--align-target teacher` selects it. Every #379 run already trains an EMA
teacher (`--ema-embedding --ema-encoder --ema-tau 0.9`), so the teacher
latent is already computed and already inside `contrastive_latent_loss`.
The change reads it instead of the student's — no new teacher code, and no
measurable per-step cost (measured below).

Everything else is #379's command line verbatim: `d_model=64, n_heads=8,
num_encoder_layers=3, num_layers=3`, `batch_size=64`, `seed=20260520`,
dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`.
`tests/test_390_launcher_shape.py` checks that claim against #379's
launcher, arm by arm.

## The 10 cells

Only `arm5` and `arm6_v2` pass `--align-loss-weight 1.0`. Each runs in the
five #379 settings:

| arm       | loss recipe                                                        |
|-----------|--------------------------------------------------------------------|
| `arm5`    | `cosine_similarity_batch_rep_only --align-loss-weight 1.0`          |
| `arm6_v2` | the same `+ --moco-rep-keys`                                        |

| setting  | added flags                                        |
|----------|----------------------------------------------------|
| base     | —                                                  |
| `tr1`    | `--tau-rep 1.0`                                    |
| `nse`    | `--sigreg-embedding-weight 0.0`                    |
| `ncpc`   | `--cpc-infonce-weight 0.0`                         |
| `combab` | `--tau-rep 1.0 --cpc-infonce-weight 0.0`           |

`arm1`, `arm3`, `arm4` and `bimoco` carry no L_align term. Their numbers
cannot change, so #379's 20 other cells are copied, not rerun.

## GPU evidence

`arm6_v2` is the harder recipe: `--moco-rep-keys` puts the teacher latent
into the main loss's keys *and* into L_align at once. It is the cell to run
before a wave commits to it.

200 backbone steps at the real cell config (T=4096, `d_model=64`, 3+3
layers, bs=64, SIGReg, CPC aux, EMA teacher, `seed=20260520`), on one elisa
4090, the two targets back to back. Both figures are the trainer's own
step-200 line.

| `arm6_v2` `--align-target` | loss @200  | sps |
|----------------------------|------------|-----|
| `student` (#379)           | **8.0994** | 3.5 |
| `teacher` (#390)           | **7.9426** | 3.5 |

The loss moves, so the flag reaches the objective on real data in the
recipe where the teacher latent feeds two terms at once. The step time does
not: L_align reads a latent the EMA path already computed, so there is no
extra forward pass and no measurable per-step cost.

`student` reproduces #379 exactly. #379's own `arm6_v2` run logs
`loss=8.0994 ema_loss=9.8252 gap=0.3047 ema_gap=0.0039 cpc=0.6986` at step
200 — every field identical. The default is byte-for-byte #379 on GPU, not
only in the unit tests.

`arm5` (no `--moco-rep-keys`) smokes clean at 2.9 sps on the second GPU.
Both GPUs were shared with other jobs, so absolute sps is not comparable
across rows measured at different times; the two rows above are. Raw
trainer output: [`results/smoke_arm6_v2_cost_ab.log`](results/smoke_arm6_v2_cost_ab.log).

Reproduce the smoke with:

```bash
ARM=arm6_v2 GPU=1 bash scripts/smoke.sh
```

It checks more than step time: the config loads, the `--extra-save-steps`
snapshot lands beside the save-every one, and `_losses.csv` carries the
`ff` / `u_batchtime` / `u_batchtime_e` columns the plot scripts read.

## Running the waves

The issue's schedule is three waves, 40,000 → 100,000 → 200,000 backbone
steps, with a q-head and a GIFT-Eval measurement after each, and the third
wave restricted to cells whose GM-Relative MASE fell from wave 1 to wave 2.

`scripts/pipeline.sh` is all three waves, both stages of each, in one
background process. It is what the actual run used:

```bash
# the watchdog covers the WHOLE pipeline, not one wave — CLAUDE.md requires a
# sync loop for the full duration of every run
WT=/home/jupyter/wt-cf-390-train \
  nohup setsid bash scripts/watchdog.sh > /dev/null 2>&1 &

WT=/home/jupyter/wt-cf-390-train \
  nohup setsid bash scripts/pipeline.sh > /dev/null 2>&1 &

bash scripts/status.sh          # one screen, read off disk, safe any time
```

Every stage is idempotent, so re-running `pipeline.sh` after a crash resumes:
`run_arm.sh` short-circuits on a checkpoint already at the wave's target, and
`eval_arm.sh` short-circuits on a trained head and resumes a partial
GIFT-Eval from its own CSV.

### Two slots per GPU

The pipeline runs four cells at once, two per 4090, not two. Measured on
elisa before the launch — 400 backbone steps of `arm6_v2`, same GPU, minutes
apart, both GPUs shared with unrelated jobs
([`results/probe_concurrency.txt`](results/probe_concurrency.txt)):

| trainers on one 4090 | sps each      | aggregate |
|----------------------|---------------|-----------|
| 1                    | 3.2           | 3.2       |
| 2                    | 2.9 + 2.8     | **5.7**   |

A second trainer costs the first ~10% of its step rate and buys 78% more
aggregate throughput. On a sweep of this size that is more than a day. A
third slot does not fit: two trainers already hold ~10.7 GB of the 24 GB
card, and the box's other tenants held ~7 GB.

`scripts/gpu_pool.sh` is the slot pool; `scripts/orchestrate_pool.sh` is the
backbone stage on top of it. The reviewed pair-at-a-time `orchestrate.sh` is
unchanged and still valid — the tests pin its behaviour:

`orchestrate.sh` runs the 10 cells two at a time, one per GPU, and picks
`TARGET_STEPS` / `SAVE_EVERY` from `WAVE`. Wave 3 takes the surviving
subset:

```bash
WAVE=3 ARMS="arm5_ncpc arm6_v2_tr1" bash scripts/orchestrate.sh
```

A single arm, by hand:

```bash
WT=$HOME/workspaces/contrastive-forecasting BB_GPU=1 \
  TARGET_STEPS=40000 SAVE_EVERY=10000 EXTRA_SAVES=2500,40000 \
  bash scripts/run_arm.sh arm5
```

Checkpoints land in this directory's `runs/`, named with the
`_alignteacher` suffix. Neither the directory nor the name can collide
with a #379 artefact.

Staged waves, as in #379: a wave trains to `TARGET_STEPS`, `_FINAL.pth` is
written only when `TARGET_STEPS ≥ FINAL_STEPS`, and the next wave resumes
from the newest `_<N>k.pth`.

Before any launch, walk
[`../REMOTE_LAUNCH_CHECKLIST.md`](../REMOTE_LAUNCH_CHECKLIST.md).

## The student control, and why all ten cells needed one

The comparison this experiment exists to make is a #390 teacher cell against
its #379 student counterpart. That comparison spans a code boundary. Running
arm5's exact #379 command line — `--align-target student`, seed 20260520 — on
THIS branch measures the size of it:

| arm5, backbone 40k, 97/97 configs | GM-Relative MASE |
|-----------------------------------|------------------|
| student target, #379's sweep       | 1.5478 |
| student target, this branch        | **1.4501** |
| teacher target, this branch        | 1.3515 |

0.0977 of the number moves between code snapshots under an identical command
line. The cross-experiment delta of -0.1963 is therefore about half snapshot
and half flag: inside one snapshot the flag is worth -0.0986 here.

So the control was extended to the other nine cells. Each runs its own #379
command line with the one flag flipped, to step 40 000 only, then the same
15 000-step head and the same 97-config GIFT-Eval:

```bash
WT=/home/jupyter/wt-cf-390-train SLOTS_PER_GPU=2 \
  nohup bash scripts/run_student_control_batch.sh \
  > results/student_batch_driver.log 2>&1 &
```

`run_arm_student.sh` does not restate the command line. It derives the
launcher from `run_arm.sh` by three textual substitutions and refuses to run
if any of them matched nothing, so the control cannot drift from the teacher
arms by anything except `--align-target`.
`tests/test_390_student_control.py` pins the transformation.

Nothing was re-run past step 40 000, so 40k is the only controlled row. The
100k and 200k comparisons stay cross-experiment.

## Scripts

| script                | role                                                                 |
|-----------------------|----------------------------------------------------------------------|
| `scripts/pipeline.sh` | **the whole experiment**: three waves, backbone + head + GIFT-Eval each, the wave-3 gate between them. |
| `scripts/run_arm.sh`  | one arm, one wave. #379's command line plus `--align-target teacher`. |
| `scripts/eval_arm.sh` | one cell's measurement: fresh q-head, then GIFT-Eval B4 over all 97 configs. Never writes a summary for a partial. |
| `scripts/smoke.sh`    | 200-step backbone smoke: config, checkpoint naming, dynamics columns, step time. |
| `scripts/orchestrate.sh` | one wave of the 10 cells, two GPUs, one cell per GPU at a time.    |
| `scripts/orchestrate_pool.sh` | the same wave with `SLOTS_PER_GPU` cells per GPU. What the run used. |
| `scripts/eval_wave.sh`| the measurement stage of one wave, over the same slot pool.           |
| `scripts/gpu_pool.sh` | the slot pool itself: keeps N jobs alive per GPU, refills on exit.    |
| `scripts/run_arm_student.sh` | `run_arm.sh` with `--align-target student`, derived by substitution rather than restated. |
| `scripts/run_student_control.sh` | one cell's control end to end: backbone 0→40k, head, GIFT-Eval. |
| `scripts/run_student_control_batch.sh` | the nine remaining controls over the slot pool, one generated launcher per cell. |
| `scripts/run_head_seeds.sh` | the same frozen backbone, extra head seeds, full eval — the seed-spread measurement. |
| `scripts/controlled_delta.py` | the headline table: teacher minus student at 40k with both sides on this branch, dataset-level paired bootstrap. |
| `scripts/seed_spread.py` | the range a cell moves under nothing but the head seed, and the gaps read against it. |
| `scripts/select_wave3.py` | the wave-3 gate — the cells whose GM-Relative MASE fell from 40k to 100k. |
| `scripts/monitor.sh`  | 15-min watchdog for ONE wave: copies the CSVs into `sync/`, shouts on NaN or a dead trainer. |
| `scripts/watchdog.sh` | the same, for the whole multi-wave pipeline. Exits only when `pipeline.pid` is gone. |
| `scripts/status.sh`   | one-screen status, read off disk. `ONELINE=1` gives a heartbeat line. |
| `scripts/arm_names.sh`| the arm → run-name mapping and the three-wave table, derived once and pinned against `run_arm.sh` by the tests. |
| `sync/sync_loop.sh`   | 15-min pull of checkpoints, optimizers, CSVs and logs into a persistent off-host checkout. |

`monitor.sh` runs on the training host, so its `runs/` → `sync/` copy is
same-disk: a watchdog and a stable point-in-time copy, not machine-death
protection. Surviving the machine is `sync/sync_loop.sh`, which runs where
the durable checkout lives and pulls everything, checkpoints and optimizer
state included. Verify a sync by `ls`-ing the target after a tick, never by
reading its log.

`monitor.sh` runs for the whole wave, not just while a trainer is up:
`orchestrate.sh` runs the cells as five sequential pairs, so nothing is
training at each phase boundary. It publishes its PID to
`results/orchestrate_wave<N>.pid`, and the monitor exits only once that
process is gone.
