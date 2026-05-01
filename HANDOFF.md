# HANDOFF — May 1, 2026

Self-contained pickup doc. The previous HANDOFF.md was deleted in this commit
window; this replaces it. Cover everything needed to resume work on the
in-flight pipeline.

## Quick orientation

- **Working branch / PR**: `feat/composite-synth` → PR #89 against `experiments`. All committed work in this session lives there. HEAD = `ecac7bf` (or later — push the worktree before reading).
- **Active worktree**: `/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/worktrees/feat+composite-synth/`. Run all `git`, `Edit`, `Read`, `Write` commands rooted there for code work.
- **Shell cwd at session start**: usually `…/.claude/worktrees/feat+source-id-freq-plumb/` (a different worktree). Doesn't matter — use absolute paths.
- **Vast.ai is shared with concurrent agents** — only destroy instances whose contract ID came from MY OWN `vastai create` IN THIS SESSION (CLAUDE.md ownership rule).

## Task list (live at writing — re-read with `TaskList`)

```
#16 [completed] Phase 4: combine pulse + more-primitives
#17 [completed] Phase 5: explosive-trend env_gain bump
#18 [completed] Add SN-normalized MAPE/CRPS to eval pipeline
#19 [in_progress] Real-data only training on gift-pretrain-small-4096
#20 [in_progress] Smaller-arch sweep: L=6 H=384 nhead=6 vs current Tiny
#21 [pending] Full single-pass training on gift-pretrain-base with winning arch    [BLOCKED — dataset doesn't exist on HF]
#22 [in_progress] EWMA span sweep on the best architecture from #19/#20            [span=64 NaN'd at step 1 — see Open Issues]
#23 [pending] Train-to-completion on the most promising configs
#24 [completed] Write REPORT.md for #19
#25 [in_progress] Write REPORT.md for #20 (partial — RevIN-smaller now in)
#26 [in_progress] Write REPORT.md for #22 (partial — span=64 broken; investigate)
#27 [pending] Tau sweep (0.05 + 0.20 around the existing 0.07)                     [blocked by #20, #22]
#28 [pending] Learnable tau (CLIP-style log_inv_tau, clamped post-step)            [blocked by #27]
#29 [pending] Write REPORT.md for #28
#30 [pending] Write REPORT.md for #27
```

#19, #20 EWMA-smaller arm, #22 span=32 + span=128 are DONE. #20 RevIN-smaller just landed (GM-MASE 2.53, MAPE_SN 1.85, CRPS_SN 1.55). #22 span=64 crashed. #22 span=256 auto-chained on EWMA box (in flight).

## Headlines so far

| arm                                      | GM-MASE | GM-MAPE_SN | GM-CRPS_SN |
|------------------------------------------|--------:|-----------:|-----------:|
| Tiny + EWMA-128 (#19)                    | 1.805   | 1.432      | 1.083      |
| Tiny + RevIN (#19)                       | 2.448   | 1.887      | 1.510      |
| **smaller + EWMA-128 (#20)**             | **1.783** | **1.243** | **1.082**  |
| smaller + RevIN (#20)                    | 2.533   | 1.849      | 1.548      |
| smaller + EWMA-128 span=32 (#22)         | 1.739   | 1.277      | 1.076      |
| smaller + EWMA-128 span=128 (= #20)      | 1.783   | 1.243      | 1.082      |
| (Aksu Moirai-Small reference)            |   —     | 0.882      | 0.642      |
| v3-prim + EWMA-128 (phase 3 winner)      | 1.621   |  n/a       |  n/a       |

**Two findings**:
1. **Synth was load-bearing in phases 1–5** (realonly EWMA 1.78 vs phase v3-prim 1.62, ~10% worse). NOT just regularising.
2. **smaller arch wins on EWMA**, basically ties Tiny on RevIN. EWMA also clearly beats RevIN at this scale.

## Live resources

### Vast.ai instances (mine)

| ID         | Address              | Label                                | Role |
|------------|----------------------|--------------------------------------|------|
| 35892408   | ssh6.vast.ai:12408   | `compositesynth-v5-ewma128-0430`     | "EWMA box" — currently running #22 span=256 (started after span=32 finished) |
| 35927139   | ssh9.vast.ai:17138   | `realonly-4096-revin-r3-0501`        | "RevIN box" — currently span=64 CRASHED at step 1 — see Open Issues |

**Don't destroy these yet** — both are mid-pipeline and have local-only state we'd lose.

### Background watchers / chains (in-memory, in this session)

| id          | role |
|-------------|------|
| `b6t1vf5lb` | wait-and-pull span=64 results (will never fire — span=64 NaN'd) |
| `bujyep1s0` | wait-and-pull span=256 results (active) |
| `bw4hqvluy` | wait-and-pull span=512 results (gated by span=64 chain → won't fire) |
| `bzj5olxkw` | wait for span=64 ALL DONE → launch span=512 (won't fire — span=64 didn't ALL DONE) |

The span=64 NaN broke the chain on the RevIN box. **The RevIN box is currently idle** while span=256 finishes on the EWMA box.

### Sync_loops (running)

```
pgrep -af sync_realonly_4096
```
Should show 2 active loops:
- `sync_realonly_4096_smaller/sync_loop.sh ssh6.vast.ai 12408 35892408 ewma128` — pulls #20/#22 EWMA artifacts
- `sync_realonly_4096_smaller/sync_loop.sh ssh9.vast.ai 17138 35927139 revin` — pulls #20 RevIN artifacts

The `sync_realonly_4096/{ewma128,revin}/` (no-`_smaller` suffix) loops were #19 — now stopped.

## Open issues

### 1. span=64 NaN at step 1 (the most recent, important)

`run_span64.log` first line after init:
```
*** NaN/Inf DETECTED at step 1 ***
  -> Saved checkpoints/tiny_realonly_4096_smaller_ewma_span64_EMERGENCY_1.pth
```

**Significance**: this is exactly the design-defect signal you wanted preserved
when you banned grad-clip. The float64 cumsum fix was sufficient at span=32
and span=128 but NOT at span=64 on this dataset. Hypothesis (un-verified):
`RevEWMNorm`'s eps clamp at `1e-12` is too small — a window with a near-constant
first patch (`var ≈ 0`) gets `stdev = 1e-6`, and dividing real data values by
that produces z-scores in the 1e5+ range. The contrastive loss with τ=0.07 then
overflows in `exp(sim / τ)`.

**To investigate / fix**:
1. Reproduce locally with the failing data (what does the first batch look like?
   Pull a few rows of `gift-pretrain-small-4096/small_v1/shard_00000.parquet`
   and try the model forward.).
2. Likely fix: increase `RevEWMNorm.eps` from `1e-12` to `1e-5` (matches
   foundation-model conventions). This is at `src/norm.py:138`.
3. Re-launch span=64 on RevIN box.
4. Then chain to span=512.

DO NOT add `--grad-clip` — banned per project rule (see `MEMORY.md` →
`feedback_no_grad_clip.md`). Fix the underlying defect.

### 2. #21 dataset blocker

`jeremycochoy/gift-pretrain-base` does NOT exist on HF (verified 2026-04-30).
Need user clarification: upload a base-4096 companion dataset? Use
`Salesforce/GiftEvalPretrain` directly with custom T=4096/C=1 stream? Regress
to T=1024 `contrastive-training-base-bundles`?

## Where things live

### Git branches / worktrees

| branch                      | worktree path                                          | purpose |
|-----------------------------|--------------------------------------------------------|---------|
| `experiments`               | `…/contrastive-forecasting/` (main)                    | merge target, what gets PR'd to |
| `master`                    | n/a                                                    | stable; usually behind `experiments` |
| `feat/composite-synth`      | `…/.claude/worktrees/feat+composite-synth/`            | **this session's work** — PR #89 |
| `worktree-feat+handoff`     | `…/.claude/worktrees/feat+source-id-freq-plumb/`       | shell start dir, irrelevant for the task work |
| `paper/arxiv-prep-pdf-output` | `…/.claude/worktrees/paper+arxiv-prep/`              | unrelated (paper writing) |
| `paper/add-pdf-to-master`   | `…/.claude/worktrees/paper-pdf/`                       | unrelated (paper writing) |

### Sync dirs (in main checkout, NOT worktree)

- `sync_compositesynth_v5envboost/{ewma128,revin}/` — phase 5 historical
- `sync_realonly_4096/{ewma128,revin}/` — #19
- `sync_realonly_4096_smaller/{ewma128,revin,ewma_span32}/` — #20 + first span sweep arm
- (future) `sync_realonly_4096_smaller_tau_sweep/`, `sync_realonly_4096_smaller_learnable_tau/` — for #27/#28

These hold real-time-synced checkpoints + logs from the remote instances.
The worktree's `experiments/exp_*/results/` only contains the FINAL CSVs
copied in at end of run.

### Experiment dirs in worktree (`.claude/worktrees/feat+composite-synth/experiments/`)

Phase 1–5 (synth recipe iteration, master result = v3-prim+EWMA-128 GM 1.621):
- `exp_compositesynth_2arm/` — phase 1 baseline composite synth
- `exp_compositesynth_v2pulse_2arm/` — phase 2 (pulse primitive)
- `exp_compositesynth_v2bseasheavy_2arm/` — phase 2B (seas-heavy ablation, regressed)
- `exp_compositesynth_v3primitives_2arm/` — phase 3 (more primitives — winner)
- `exp_compositesynth_v4combined_2arm/` — phase 4 (combined v2+v3 — regressed)
- `exp_compositesynth_v5envboost_2arm/` — phase 5 (env-bump — neutral)
- `exp_dualemb_3arm/` — earlier baseline (phase 0 reference)
- (older) `exp_csb_pair_revin/`, `exp_csb_pair/`, `exp_freq_emb/`, etc. — pre-phase-1

This session (real-only training on gift-pretrain-small-4096):
- `exp_realonly_4096_2arm/` — #19 (Tiny, T=4096, C=1, mix=0.0). REPORT.md complete.
- `exp_realonly_4096_smaller_2arm/` — #20 (smaller arch sweep). REPORT.md partial.
- `exp_realonly_4096_smaller_span_sweep/` — #22 (EWMA span sweep). REPORT.md partial; span=64 broken.

Pending dirs:
- `exp_realonly_4096_smaller_tau_sweep/` — for #27 (NOT created yet; will be when #22 finishes)
- `exp_realonly_4096_smaller_learnable_tau/` — for #28 (NOT created yet)

### Key code files (worktree)

- `src/norm.py` — RevEWMNorm with float64-cumsum-when-T>2048 fix. **`.eps = 1e-12` here is the suspected span=64 NaN cause.**
- `src/dataloader.py` — has `t_raw=` arg threaded through `HFStreamingLoader`, `ShardDataset`, factories.
- `src/loss.py` — `contrastive_latent_loss`. Uses `tau = train_config.get('contrastive_divergence_temperature', 1.0)` — default 1.0 fallback if dict missing the key. Trainer always sets 0.07.
- `src/models.py` — `ConfigurableModel`. Where `log_inv_tau` will go for #28.
- `experiments/freq-embedding/scripts/train.py` — backbone trainer. Has `--t-raw`, `--n-channels`, `--d-model`, `--n-heads`, `--num-layers` flags. `LOSS_SPEC.contrastive_divergence_temperature = 0.07` at line 58.
- `experiments/gift-eval/scripts/train_forecasting_head.py` — qhead trainer. Same set of CLI flags.
- `experiments/gift-eval/scripts/eval_gift_eval_official.py` — eval. Has SN-normalized columns (task #18). CLI: `--t-raw`, `--backbone-c`, `--d-model`, `--n-heads`, `--num-layers`.

### Key docs

- `docs/HANDOFF_COMPOSITE_SYNTH_2026_04_30.md` — yesterday's master handoff. More detail than this file on phases 1–5.
- `docs/PHASE5_FOLLOWUP_IDEAS.md` — comprehensive future-work doc.
- `experiments/exp_realonly_4096_2arm/REPORT.md` — full report for #19.
- `experiments/exp_realonly_4096_smaller_2arm/REPORT.md` — partial report for #20.
- `experiments/exp_realonly_4096_smaller_span_sweep/REPORT.md` — partial report for #22.

## Project rules (from `~/.claude/projects/.../memory/MEMORY.md`)

1. **No grad-clip** (`feedback_no_grad_clip.md`). Forbidden in this project. Fix
   the underlying defect; if data needs curation, curate at the data layer.
2. **Don't destroy a partial-result vastai instance without considering reuse**
   (`feedback_vastai_instance_reuse.md`). Pause and ask if the next task could
   use the same hardware.
3. **Sync dirs in MAIN checkout, never in a worktree** (CLAUDE.md). Worktree
   may be `git worktree remove --force`-d, deleting untracked state.
4. **EVERY remote training run must have a sync_loop running** for crash
   recovery (CLAUDE.md). 15-min cadence, atomic .tmp → mv.
5. **NEVER raw-scp a checkpoint** — use `safe_pull.sh` (atomic with .prev
   backup) (CLAUDE.md).
6. **macOS case-insensitive FS** for `_FINAL.pth` vs `_final.pth` — pre-rename
   the canonical to `_FINAL_safe.pth` on remote before pulling lowercase.
7. **HF token** at `experiments/hf_token.txt` (gitignored) — every cloud run
   must export `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`.

## Cost so far (rough)

| phase | wall hours | cost  |
|-------|-----------:|------:|
| #19 EWMA  | ~5h    | $1.85 |
| #19 RevIN (incl. r2 sunk) | ~6h | $3.50 |
| #20 EWMA-smaller | ~3h | $1.10 |
| #20 RevIN-smaller | ~3h | $2.00 |
| #22 span=32 | ~3h | $1.10 |
| #22 span=64 (crashed at step 1) | ~0.1h | $0.06 |
| #22 span=256 (in flight) | ~3h | $1.10 |
| total so far | | **~$10.70** |

## Pending work — order

1. **Investigate / fix span=64 NaN-at-step-1** (likely `RevEWMNorm.eps`).
   Re-launch span=64 + chain span=512.
2. **Wait for span=256 to finish** (in flight on EWMA box, ~2h).
3. **Run span sweep plot** + finalize #22 REPORT.md.
4. **Launch #27** (tau sweep, 0.05 + 0.20) on the freed boxes. Best span from
   #22, smaller arch, EWMA.
5. **Update #20 REPORT.md** with RevIN-smaller numbers + finalize.
6. **Implement learnable tau** (#28) — `log_inv_tau` as `nn.Parameter` on
   `ConfigurableModel`, clamped to `[0, log(100)]` after `optimizer.step()`,
   init from #27 winner.
7. **#23 train-to-completion** on the overall winning config.
8. **#21** still blocked. Pingthe user whenever they're back.
9. **Date-prefix experiment dirs** (deferred, post-everything-else): user
   asked for `2026-MM-DD_exp_*` naming via `git mv` in MAIN checkout. Do AFTER
   #22/#27/#28 finish to avoid disrupting in-flight scripts.

## Picking back up

To resume cold:
1. `cd /Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/worktrees/feat+composite-synth`
2. `git status` → should be clean. `git log --oneline -10` → see recent.
3. `vastai show instances` → confirm 35892408 + 35927139 still running. If
   either is gone, look at `sync_realonly_4096_smaller/<arm>/` for the latest
   synced checkpoints and decide whether to re-provision + resume or accept
   partial results.
4. Read this file + `docs/HANDOFF_COMPOSITE_SYNTH_2026_04_30.md`.
5. `TaskList` → see what's pending.
6. Investigate span=64 NaN if not already fixed.
