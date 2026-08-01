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

```bash
# always start the watchdog alongside the wave — CLAUDE.md requires a sync
# loop for the FULL duration of every run, short or long
WT=$HOME/workspaces/contrastive-forecasting WAVE=1 \
  nohup setsid bash scripts/monitor.sh > /dev/null 2>&1 &

WT=$HOME/workspaces/contrastive-forecasting WAVE=1 \
  nohup setsid bash scripts/orchestrate.sh > /dev/null 2>&1 &
```

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

## Scripts

| script                | role                                                                 |
|-----------------------|----------------------------------------------------------------------|
| `scripts/run_arm.sh`  | one arm, one wave. #379's command line plus `--align-target teacher`. |
| `scripts/smoke.sh`    | 200-step backbone smoke: config, checkpoint naming, dynamics columns, step time. |
| `scripts/orchestrate.sh` | one wave of the 10 cells, two GPUs.                               |
| `scripts/monitor.sh`  | 15-min watchdog on the training host: copies the CSVs into `sync/`, shouts on NaN or a dead trainer. |
| `scripts/arm_names.sh`| the arm → run-name mapping, derived once and pinned against `run_arm.sh` by the tests. |
| `sync/sync_loop.sh`   | 15-min pull of checkpoints, optimizers, CSVs and logs into a persistent off-host checkout. |

`monitor.sh` guards the small CSVs where the runs happen; `sync/sync_loop.sh`
runs on the machine that owns the durable copy and pulls everything,
checkpoints and optimizer state included. Verify a sync by `ls`-ing the
target after a tick, never by reading its log.
