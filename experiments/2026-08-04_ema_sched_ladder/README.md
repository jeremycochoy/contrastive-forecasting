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
machines.

Elisa carries two RTX 4090s; run one cell per GPU with `BB_GPU`. A vast.ai
instance takes more cells on top — walk through
[`REMOTE_LAUNCH_CHECKLIST.md`](../REMOTE_LAUNCH_CHECKLIST.md) first, and
use `vastrun-kit` commands only.

## Files

| path | what |
|---|---|
| `scripts/ladder.py` | the driver: stops, the extend rule, the step cap |
| `scripts/run_leg.sh` | one backbone leg of one cell, up to a target step |
| `scripts/eval_stop.sh` | one head + GIFT-Eval B4 at one stop, one encoder |
| `scripts/confirm_row_count.py` | reads the dataset manifest, derives the cap |
| `scripts/alpha_schedule.py` | the α-vs-step record and its plot |
| `scripts/smoke_e2e.sh` | CPU end-to-end check of the ramp and both encoders |
| `results/ladder.csv` | GM-Relative MASE per cell, per stop, per head |
| `results/decisions.csv` | which branch of the extend rule fired, per stop |

## Protocol constants

Unchanged from the 2026-08-04 protocol: 97 GIFT-Eval configs, official B4
strategy, forecast horizon 16, one head seed (20260722), backbone
`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, batch_size=64,
seed=20260520`, dataset `gift-pretrain-full-4096 / small_v1`.

The head keeps `--grad-clip 1.0`. The project rule bans grad clipping; the
previous study kept it for comparability, so this one does too, and the
report says so.

## Caveats for the report

- The head budget changes at bb100k, 15k → 30k. Any statement across that
  boundary moves two things at once.
- One head seed per cell. The 2026-08-04 report measured head-seed ranges
  up to 0.0908, so the extend rule will sometimes fire on noise. Report
  the raw per-stop changes.
