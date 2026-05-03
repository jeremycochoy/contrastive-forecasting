# contrastive-forecasting handoff

**Date**: 2026-05-03. **Audience**: future Claude session resuming this work
on a fresh machine (typically elisa) after laptop migration.

## Current status (as of 2026-05-03)

- Active vast.ai instance: `36055545` = `ssh8.vast.ai:15544`, RTX 5090,
  $0.65/h, label `contrastive-forecasting-machine10`.
- Active training: **FRESH run from random init**, bs=256, MOIRAI HP, target
  step 167000 (1 epoch on the 42.57M-row dataset). Currently ~step 11.8k.
  ETA ~16 h backbone, ~10 h qhead+eval.
- Credit ~$11.18 — only enough for backbone, **not** qhead+eval. Top up
  before launching qhead.
- Local sync target: `sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/`.
- Sync_loop driving the FRESH run was on the user's laptop (PID 95662); will
  be killed during migration. Elisa takes over.
- Open PRs (don't auto-merge): #94, #89 (both from earlier work, possibly
  obsolete).

## Key decisions already made

- **#6 vs #9 winner: #9 MOIRAI HP.** GM-MASE 1.6391 vs 1.8043, GM-MAPE_SN
  1.1850 vs 1.3698, GM-CRPS_SN 1.0155 vs 1.1000. Locked via PR #104.
- **#10 attempts:**
  - **v1** (resumed from #9 30k, bs=96): abandoned at step 188k after std-
    jump diagnostic. Archived in `sync_realonly_full4096_moirai_hp_FINAL_run1/`.
  - **v2** (resumed from #9 30k, bs=96, post PR #110): same std jump
    reproduced. Archived in `sync_realonly_full4096_moirai_hp_FINAL_run2/`.
  - **v3 = current FRESH run** (no resume, bs=256, target 167k).

## Gotchas already fixed (don't reintroduce)

- **PR #106**: ported #28 learnable-τ infrastructure to `experiments` branch
  (was only on `feat/composite-synth`). Without it, train.py fails with
  `TypeError` on `tau_override` kwarg.
- **PR #110**: RNG-restore CPU cast bug — `torch.load(..., map_location=cuda)`
  moves the saved CPU ByteTensor to CUDA; need `.cpu()` before
  `set_rng_state`. Also removed in-iter `np.random.shuffle` from local-
  parquet sampler (HF bundles are pre-shuffled at upload; in-iter shuffle
  adds non-determinism).
- **PR #112**: replaced `_shard_row_counts` 18-min sweep with shard-0-only
  metadata read.
- **Dataset path naming.** `jeremycochoy/gift-pretrain-full-4096` path
  `small_v1` is the **full** data despite the name suggesting "small".
  4274 shards × 10000 rows ≈ 42.57M total.
- **HF token MUST be set** as `HF_TOKEN` AND `HUGGING_FACE_HUB_TOKEN` env
  vars in run scripts, otherwise HF rate-limits anonymous access (0.5–1.5
  sps vs 5–9 sps with token).

## Task list snapshot

| ID  | Subject                                                                                | Status      | BlockedBy |
|-----|----------------------------------------------------------------------------------------|-------------|-----------|
| #6  | 30k-step learnable-τ baseline on gift-pretrain-full-4096                               | completed   | —         |
| #7  | #34 Date-prefix all experiment dir names (git mv pass)                                  | completed   | —         |
| #9  | #31 MOIRAI-style optimizer (single arm) on learnable-τ winner config                    | completed   | —         |
| #10 | FINAL retrain — 1 full epoch on gift-pretrain-full-4096                                | in_progress | —         |
| #11 | sync_loop length-guard for append-only files (Layer 2 of SYNC_PROTOCOL_REVIEW)         | completed   | —         |
| #12 | push_resume_bundle.sh + preflight in run_*_resume.sh / run_eval_only.sh                 | completed   | —         |
| #13 | 3-panel comparison plot #6 vs #9 (final, post-#9-ALL-DONE)                              | completed   | —         |

## Active cron — recreate on elisa via CronCreate

Schedule: `37 * * * *`. **Verbatim prompt for cron `2fd024e2` was not
exported across migration — parent agent must supply.** Paths are relative
(start with `sync_realonly_*`), portable as long as Claude is launched in
`~/contrastive-forecasting`.

Structure (faithful reconstruction):

1. **Status**: `vastrun-show 36055545` (alive? credit OK?); confirm sync_loop
   process alive (`ps aux | grep sync_loop`).
2. **Sync freshness**: `sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/
   sync.log` mtime < 30 min; tail expects ✓ lines per tick.
3. **Training progress**: latest periodic ckpt step in `.../checkpoints/`;
   tail `run.log` for step, sps, GPU util.
4. **NaN watch**: `grep -E "nan|NaN|inf" run.log` — escalate if matched.
5. **std@[25k,30k) diagnostic**: when step crosses 25k, compute per-batch
   loss std over [25k, 30k); compare to v1/v2 archives. FRESH should NOT
   show the +52% jump.
6. **Decision tree**: credit-low (alert, don't auto-bid); ALL-DONE (step ≥
   167000 + gift_eval done → copy final ckpts to permanent names, write
   REPORT.md, stop cron); NaN (stop training, don't destroy instance, pull
   ckpts, diff vs last good, alert user).

## Restart on a fresh machine

1. `ssh jupyter@elisa`
2. `vastrun-balance` (must be > $0; ideally > $30 if qhead+eval still pending)
3. `cd ~/contrastive-forecasting && git pull` (verify branch = `experiments`)
4. `ls ~/contrastive-forecasting_backup/sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/checkpoints/ | wc -l`
   should be > 5
5. `bash scripts/restore_from_elisa.sh` (auto-detects: on elisa it copies
   from `~/contrastive-forecasting_backup/`; otherwise rsyncs over SSH)
6. Smoke test:
   `python3 -c "import sys; sys.path.insert(0, '.'); from src.dataloader import HFStreamingLoader; from src.models import ConfigurableModel; print('imports OK')"`
7. `wc -c experiments/hf_token.txt` (~40 chars, content starts with `hf_`)
8. `tmux new-session -d -s jeremy_claude_contrastive-forecasting -c ~/contrastive-forecasting`
9. Inside tmux: `claude --dangerously-skip-permissions --effort max`
10. To Claude: "please continue the ongoing work" — Claude reads CLAUDE.md
    + this handoff, recreates the cron via CronCreate, resumes monitoring.

## Sync_loop on elisa

The sync_loop script for the FRESH run lives at
`sync_realonly_full4096_moirai_hp_FRESH/sync_loop.sh`. Start with:

```
nohup bash sync_realonly_full4096_moirai_hp_FRESH/sync_loop.sh \
  ssh8.vast.ai 15544 36055545 moirai_hp_FRESH \
  >> sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/sync.driver.log 2>&1 &
```

Verify alive: `ps aux | grep sync_loop | grep -v grep`. If not running,
restart with the command above. Pulls every 15 min from the live vast.ai
instance — critical for crash recovery, since latest periodic save (every
10k steps) is preserved locally if the instance dies.

## Mirror scripts

- `scripts/backup_to_elisa.sh` — rsync laptop → elisa. Skips if running on elisa.
- `scripts/restore_from_elisa.sh` — pull from elisa to current host (auto-detects).
- `scripts/mirror_with_elisa.sh` — bidirectional, takes `--push` or `--pull`
  (default `--pull`).
