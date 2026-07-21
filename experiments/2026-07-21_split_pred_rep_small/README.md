# Small-model long-training sweep — 6 arms of #374 × 200k steps (#379)

Report: [`reports/2026-07-21_split_pred_rep_small/small_long.md`](../../reports/2026-07-21_split_pred_rep_small/small_long.md).

## What this experiment adds vs #374

The six contrastive-loss-shape arms from
[`experiments/2026-07-10_split_pred_rep/`](../2026-07-10_split_pred_rep/)
re-run on a much smaller backbone for ≥4× longer. **Backbone training
only** — no downstream q-head training, no GIFT-Eval MASE. The
deliverable is the training-dynamics trajectories:

1. Does bimoco / arm 6 v2's `1 − cos(f̂, h_{t+1})` continue to climb
   through 200k, or plateau, or reverse?
2. Does arm 5's alignment plateau (`1 − ff ≈ 0.4` at 50k in #374)
   break through at 100k or 200k?

Same six loss recipes (loss-shape + MoCo flags + alignment weights)
copied verbatim from #374; only the backbone architecture and training
length change.

## Backbone architecture (identical across the six arms)

| Field                 | #374           | #379 (this run) |
|-----------------------|----------------|-----------------|
| `d_model`             | 384            | **64**          |
| `n_heads`             | 6 (head_dim=64)| **8 (head_dim=8)** |
| `num_encoder_layers`  | 3              | 3               |
| `num_layers`          | 6              | **3**           |
| `T`, `C`              | 4096, 1        | 4096, 1         |
| `encoder_type`        | gru            | gru             |
| `rev_norm_kind/span`  | ewma / 128     | ewma / 128      |
| Params                | ~17 M          | ~1–2 M          |

## Training schedule (identical across the six arms)

| Field                 | #374                     | #379 (this run)          |
|-----------------------|--------------------------|--------------------------|
| `batch_size`          | 512                      | **64**                   |
| `total_steps`         | 12,500                   | **200,000**              |
| `save_every`          | 2,500                    | **25,000**               |
| `extra_save_steps`    | —                        | **2500** (`_2k.pth` early snapshot) |
| `lr`, `wd`, `betas`   | 1e-3, 0.1, (0.9, 0.98)   | 1e-3, 0.1, (0.9, 0.98)   |
| `seed`                | 20260520                 | 20260520                 |
| SIGReg λ_e, λ_h       | 1.0, 1.0                 | 1.0, 1.0                 |
| EMA teacher τ         | 0.90                     | 0.90                     |
| contrastive τ         | 0.10                     | 0.10                     |
| CPC auxiliary weight  | 1.0                      | 1.0                      |

At 200k × 64 = 12.8M samples ≪ 42.7M rows in
`gift-pretrain-full-4096` → no sample revisit.

## Arm table (loss flags copied from #374)

| Arm      | Extra flags added to `--loss-shape`                                 |
|----------|---------------------------------------------------------------------|
| arm 1    | `cosine_similarity_batch_split_pred_rep` — no extras                |
| arm 3    | `cosine_similarity_batch_split_pred_rep --moco-negatives`           |
| arm 4    | `cosine_similarity_batch_full_hh_negs_xshh_allt --pos-in-denominator --subtract-contrastive-floor --moco-negatives` |
| arm 5    | `cosine_similarity_batch_rep_only --align-loss-weight 1.0`          |
| arm 6 v2 | `cosine_similarity_batch_rep_only --align-loss-weight 1.0 --moco-rep-keys` |
| bimoco   | `cosine_similarity_batch_split_pred_rep --moco-negatives --moco-rep-keys` |

## Metrics recorded during training

`train.py` writes one `_losses.csv` row per logged step. Plot scripts
read the columns:

- `loss` — total training loss (per-arm floors documented in `_make_per_run_loss.py`)
- `ff` — `⟨cos(f̂, h_{t+1})⟩`, the headline signal (`1 − ff` = log perplexity)
- `u_batchtime`, `u_batchtime_e` — dim usage of `h_t` and `e_t`

## How to launch

1. **Backbone-only smoke test first** (~3 min per arm, 200 steps). Validates
   the training config + checkpoint naming + logged training-dynamics
   columns BEFORE the ~15-20 h backbone runs commit compute:
   ```bash
   ARM=arm1 GPU=0 bash scripts/smoke.sh
   # SMOKE OK — arm=arm1
   ```
   Then repeat with `ARM=bimoco` to exercise `--moco-negatives +
   --moco-rep-keys` (the harshest recipe).

2. **Sync loop** (from the machine that owns the local persistent
   checkout, in a separate shell — CLAUDE.md § Remote Machine
   Monitoring):
   ```bash
   REMOTE_HOST=elisa \
   REMOTE_DIR=~/workspaces/contrastive-forecasting/experiments/2026-07-21_split_pred_rep_small \
   LOCAL_DIR=$HOME/workspaces/contrastive-forecasting/experiments/2026-07-21_split_pred_rep_small \
     nohup setsid bash sync/sync_loop.sh > sync/sync_loop.log 2>&1 &
   ```
   Verify the first tick by `ls`-ing the local `runs/` after ~15
   minutes; do not trust `sync_loop.log`'s absence of `✗` lines
   (CLAUDE.md — a missing `✗` line can mean the pattern didn't match).

3. **Orchestrator** (on elisa, once smoke passes):
   ```bash
   WT=$HOME/workspaces/contrastive-forecasting \
     nohup setsid bash scripts/orchestrate.sh > results/orchestrate.log 2>&1 &
   ```
   Pipelines the six backbones across 2 × 4090 in three phases (arm 1
   + arm 3 | arm 4 + arm 5 | arm 6 v2 + bimoco).

To relaunch a single arm outside the orchestrator (e.g. after a
crash):
```bash
WT=$HOME/workspaces/contrastive-forecasting BB_GPU=0 \
  bash scripts/run_arm.sh arm4
```

`WT` MUST be an absolute path under a persistent checkout — the
launcher aborts if `WT` is under `/tmp` (CLAUDE.md § Checkpoint Safety
Rule 4, Apr-2026 incident).

## Restart

`train.py`'s `safe_run_name` (see [`docs/restart_protocol.md`](../../docs/restart_protocol.md))
appends `_r2`, `_r3`, … to the run name whenever it finds existing
checkpoints for the same base name. `run_arm.sh` picks up the newest
`_*k.pth` under `${NAME}` when it launches, and passes it to
`--resume`, so a crashed arm re-runs from the last snapshot.

## Reusable code changes

- `experiments/2026-04-27_freq-embedding/scripts/train.py` gains
  `--extra-save-steps` (comma-separated) so a run can snapshot off the
  `--save-every` cadence. Parsed at argument-validation time; rejects
  non-positive entries and entries that share a 1000-block (the
  filename `_{step // 1000}k.pth` would silently overwrite).
  Covered by `tests/test_extra_save_steps.py`.

## Test coverage

- `tests/test_extra_save_steps.py` — schedule union semantics and
  malformed-input rejection for `parse_extra_save_steps` /
  `should_snapshot`.
- `tests/test_small_long_launcher_shape.py` — pins the per-arm case
  block in `run_arm.sh`, the shared backbone-config literals, the
  save-cadence defaults, the backbone-only shape (no downstream
  wiring), the under-`/tmp` guard, the orchestrator phase layout, and
  the sync loop's coverage of all arms and file classes.

## Plots

Committed under [`reports/2026-07-21_split_pred_rep_small/plots/`](../../reports/2026-07-21_split_pred_rep_small/plots/):

- `_make_cos_error.py` — **headline**: `1 − ff` per arm across
  training steps, all six on one axes, log-x temporal, linear-y.
- `_make_dim_usage.py` — `u_batchtime` for `h_t` and `e_t` per arm.
- `_make_per_run_loss.py` — training-loss deviation from strict-min
  floor, log-log.

Each script reads directly from this experiment's `runs/` (losses
CSVs); adapted from the #374 plot scripts.
