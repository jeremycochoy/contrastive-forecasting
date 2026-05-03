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

Schedule: `37 * * * *`. Paths are relative (start with `sync_realonly_*`),
portable as long as Claude is launched in `~/contrastive-forecasting`. To
recreate: invoke `CronCreate(cron="37 * * * *", prompt=<<<below>>>,
recurring=true)`.

```
Hourly check-in for #10 — **FRESH run from random init** (no resume). Investigation agent could not pinpoint the std-jump root cause; per user directive, switched from resume-from-30k to fresh-from-init to definitively isolate whether the resume mechanism causes the issue.

**Context**:
- v1 (`_FINAL_*`) and v2 (`_FINAL_v2_*`) BOTH resumed from #9's 30k → both showed identical std=0.351 over 1300+ samples (vs #9's 0.23 baseline). Refuted: optimizer binding, hidden buffers, PR #106 port, RNG state. The cause remained unclear.
- Fresh run started with random init at step 0, target step 167000 (1 epoch at bs=256). Same MOIRAI HPs. First 100 batches match #9's exactly (deterministic).

**Paths**:
- Sync: `sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/`
- Run names: `tiny_full4096_moirai_hp_FRESH` (BB), `R1q_full4096_moirai_hp_FRESH` (qhead)
- Remote launcher: `/workspace/app/run_fresh.sh`
- Run.log marker for completion: `=== run_full4096_moirai_hp_FRESH: ALL DONE ===`
- Archives (do not touch): `sync_realonly_full4096_moirai_hp_FINAL_run1/` (v1) and `sync_realonly_full4096_moirai_hp_FINAL_run2/` (v2)

**Instance**: 36055545 = ssh8.vast.ai:15544, RTX 5090, $0.65/h.

REPORT under 25 lines via Bash.

1. `vastrun-status | head -3` and `vastrun-balance | head -2`. Note credit and burn.
2. Sync freshness: `tail -5 sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/sync.log` and mtime via `stat -c '%y %n' sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/sync.log`. (On macOS use `stat -f '%Sm %N'`.) If >30 min stale: restart `nohup bash sync_realonly_full4096_moirai_hp_FRESH/sync_loop.sh ssh8.vast.ai 15544 36055545 moirai_hp_FRESH >> sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/sync.driver.log 2>&1 &`.
3. Training progress: `tail -3 sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/run.log` and `wc -l sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/checkpoints/tiny_full4096_moirai_hp_FRESH_losses.csv 2>/dev/null`. Highest [N] step from log tail.
4. NaN/crash watch: `grep -i 'nan\|killed\|cuda.*error\|traceback' sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/run.log 2>/dev/null | tail -3`.
5. Completion: `grep -E 'run_full4096_moirai_hp_FRESH:.*ALL DONE' sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/run.log`.
6. **Crucial diagnostic at step 25k+**: when fresh CSV reaches >25k rows, compute `python3 -c "import pandas as pd; df=pd.read_csv('sync_realonly_full4096_moirai_hp_FRESH/moirai_hp_FRESH/checkpoints/tiny_full4096_moirai_hp_FRESH_losses.csv'); m=(df.step>=25000)&(df.step<30000); s=df.loc[m,'loss']; print(f'fresh [25k,30k): n={len(s)} mean={s.mean():.3f} std={s.std():.3f}')"`. Compare with #9 same window (mean ≈ 5.18, std ≈ 0.23). NOTE: fresh is bs=256 vs #9 bs=96 — loss values not directly comparable due to cross-batch term scaling, but std behavior is. If fresh std ≈ 0.23 → confirms resume mechanism is the culprit. If fresh std ≈ 0.35 too → cause is something pervasive.
7. Decision tree:
   - **Credit < $1.50** AND no FRESH ALL DONE → graceful shutdown.
   - **FRESH ALL DONE** → completion (read summary.txt, compute GM, pause-before-destroy task check, then vastrun-destroy 36055545 contrastive-forecasting-machine10, mark #10 completed, CronDelete this job).
   - **NaN/CUDA error** → STOP, surface, no auto-relaunch.
   - Normal → no-op summary.
8. Output: `step=<N>/167000 (<P>%); credit=$<C>; eta-cred=<H>h; sync=fresh|stale[; std@[25k,30k)=<X> (vs #9=0.23) — when applicable]`.

Constraints: vastrun-kit only. Pause-before-destroy. No --force.
```

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
