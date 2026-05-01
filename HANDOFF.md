# HANDOFF — successor pickup, May 2 2026

This file replaces every prior `HANDOFF.md`. It is written so that a fresh
agent can resume this work cold without reading any prior conversation.

PR #89 (`feat/composite-synth`) holds all the experimental code, results,
and REPORTs for the composite-synth follow-on work. This handoff document
itself ships via a separate PR against `experiments`.

---

## 1 — Active task list (verbatim)

```
#27 [in_progress] Tau sweep at bs=96: 0.05, 0.07, 0.20 on smaller-EWMA-128
#29 [pending]     Write REPORT.md for #32 (learnable tau)        [blocked by #32]
#30 [pending]     Write REPORT.md for #27 (tau sweep)            [blocked by #27]
#31 [pending]     MOIRAI-style optimizer hyperparams (single arm) on τ-sweep winner   [blocked by #27, #32]
#32 [in_progress] Learnable tau (CLIP-style log_inv_tau, clamped after optimizer.step)
#33 [pending]     Retrain winner on jeremycochoy/gift-pretrain-full-4096 (FINAL run)  [blocked by #27, #31, #32]
```

Always re-read with `TaskList`. Use `TaskGet <id>` for full descriptions.

### Pipeline topology

```
                #27 ──┐
                      ├─→ pick best τ-policy ─→ #31 ─→ #33 ─→ END
                #32 ──┘                              ↑
                                             (gift-pretrain-full-4096
                                              dataset — coming from
                                              user soon, may already
                                              be ready when #31 starts)
```

Reports interleave: #30 fires once #27 lands, #29 once #32 lands, plus
new reports for #31 and #33 (TODO: add report tasks when those start).

---

## 2 — Monitor / sync / chain policy

### Cron (hourly check-in)

There is **one** session-only cron job firing at `:47` every hour
(`CronList` will show its ID). It runs an autonomous self-prompt with
the current state — when it fires, the agent reads it and goes through:

1. `vastrun-status` to see live instances + spend.
2. SSH into each active instance, `tail` the relevant `run_*.log`, look
   for STAGE markers, NaN/Traceback, and approximate ETA.
3. Refresh the cron prompt with the new state (CronDelete + CronCreate
   with updated text).

The prompt grows stale fast. **Always overwrite it after acting** so the
next firing reads accurate state.

### sync_loop.sh — 15-min cadence per arm

Located at `/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_<exp>/<arm>/sync_loop.sh`. Each is a shell script that:

- scp's backbone .pth + optimizer + losses CSV
- scp's qhead .pth + optimizer + losses CSV
- scp's any periodic `_Nk.pth` checkpoints
- scp's run.log + results/all_results.csv + results/summary.txt
- Atomic `.tmp → mv` with one-deep `.prev` backup (CLAUDE.md rule)
- Per-class size thresholds (BB ~80MB, optimizer ~155MB, head ~2.4MB,
  CSV any size); for the smaller arch (11M params) BB ≈ 46MB so the
  threshold needs to be lower (40MB) — see `sync_realonly_4096_smaller/sync_loop.sh`.
- Sleeps 900s, repeats
- Exits cleanly when "ALL DONE" marker appears in run.log

Active sync_loops via `pgrep -af sync_realonly_4096`. Every active
training arm MUST have one — it's the crash-recovery insurance per
CLAUDE.md.

**Critical**: sync dirs live in the **main checkout**, NEVER in a
worktree. `git worktree remove --force` deletes untracked state, so any
in-flight sync data inside a worktree is one rm away from gone.

### Chain-launchers — background bash watchers

Spawned via `Bash run_in_background:true` with an `until grep -q
"<MARKER>" <log>` loop. When the marker appears (e.g. `ARM <name> ALL
DONE`), the watcher fires the next launch (chain) or pulls results
(auto-pull). Track via `Bash` task IDs or grep `pgrep -af "until ssh"`.

There's no first-class scheduler — these are session-bound. If the
session dies, restart watchers manually (run the same `until ssh ...; do
sleep 60; done` lines).

### When to manually intervene

- A run.log shows `Traceback` or `NaN/Inf DETECTED` → diagnose root
  cause; do NOT add `--grad-clip` (banned, see §5).
- vast.ai-side host stop (instance moves to `intended_status: stopped`,
  status_msg "success, running pytorch/pytorch...") → re-provision; the
  sync_loop will have the latest checkpoint locally for resume.
- An instance's SSH refuses connection but `vastrun-status` says
  `running` → usually transient (1-3 min) on the SSH proxy, retry; the
  python process keeps going.
- A `run.sh` is rewritten on disk during execution → bash re-reads
  per-line and may EOF on the now-shorter file (one-time bug seen
  May 1). Use `cp .tmp + mv` if editing during a run, or just don't
  edit run.sh while it's running.

---

## 3 — Live machines (mid-flight at write time)

| machine | id        | label                             | SSH                   | $/h    | running                        |
|---------|-----------|-----------------------------------|-----------------------|--------|--------------------------------|
| 1       | 35892408  | contrastive-forecasting-machine1  | ssh6.vast.ai:12408    | $0.37  | #27 τ=0.20 qhead 4k/30k        |
| 2       | 35927139  | contrastive-forecasting-machine2  | ssh9.vast.ai:17138    | $0.64  | #27 τ=0.05 qhead 25k/30k       |
| 3       | 35970433  | contrastive-forecasting-machine3  | ssh4.vast.ai:10432    | $0.62  | #27 τ=0.07 backbone in flight  |
| 4       | 35970908  | contrastive-forecasting-machine4  | ssh8.vast.ai:10908    | $0.77  | #32 learnable τ backbone       |

Spend so far: ~$24.40. Budget headroom: another ~$25 should comfortably
cover #31 + #33.

These labels are neutral — re-use freely. Do NOT destroy any unless ALL
THREE CLAUDE.md ownership conditions hold (contract from your call, label
match, GPU class match).

---

## 4 — Per-task chapters

### #27 — τ sweep at bs=96 (0.05 / 0.07 / 0.20)

**What**: 3-arm grid finding the best fixed τ for the contrastive loss
under the bs=96 regime. A baseline τ=0.07 result already exists from
`exp_realonly_4096_smaller_2arm/ewma128/` but at bs=24 — the bs change
invalidates that anchor (different gradient noise scale → different
absolute loss → not directly comparable), so this sweep includes a
re-run of τ=0.07 at bs=96.

**Where**:
- Code: `experiments/exp_realonly_4096_smaller_tau_sweep/`
- Run: `bash experiments/exp_realonly_4096_smaller_tau_sweep/run.sh
  <005|007|020>` on the remote.
- Results: `results/gift_eval_tau<005|007|020>/`
- Sync: `sync_realonly_4096_smaller_tau_sweep/tau<005|007|020>/`
- Plot: `scripts/plot_tau_sweep.py` (4-panel, expects all 3 CSVs).

**Setup**: smaller arch (L=6 H=384 nhead=6, ~11.4M params), EWMA-128,
bs=96, 30k steps, T=4096, C=1, mix=0.0, NO grad-clip. `--tau <value>`
CLI flag overrides `LOSS_SPEC.contrastive_divergence_temperature`.

**Watch out**:
- bs=96 makes throughput ~2 sps (vs ~6 sps at bs=24). Each arm
  takes ~5h end-to-end. Don't be surprised by long ETAs.
- All 3 arms must finish before #30 (REPORT) and #31.
- Within-sweep loss values ARE comparable (only τ varies). Across to
  bs=24 #19/#20/#22 numbers, NO direct comparison — note in any
  cross-experiment claim.

**Acceptance**: pick the best τ across GM-MASE / GM-MAPE_SN /
GM-CRPS_SN. If 0.07 wins, fixed-τ optimum confirmed; #32 then tests
whether learnable beats it.

### #32 — Learnable τ (CLIP-style)

**What**: replace the fixed τ with a trainable scalar parameter
`log_inv_tau`. The loss uses τ = exp(-log_inv_tau). After every
`optimizer.step()`, `log_inv_tau.clamp_(0, log(100))` keeps τ in
[0.01, 1.0]. Init at log(1/0.07) so initial τ = 0.07.

**Where**:
- Code: src/models.py (ConfigurableModel.learnable_tau, log_inv_tau,
  tau(), clamp_log_inv_tau()), src/loss.py (tau_override arg),
  experiments/freq-embedding/scripts/train.py (--learnable-tau flag,
  τ logging in loss-print line, post-step clamp), and head + eval
  scripts auto-detect log_inv_tau in the checkpoint.
- Run: `bash experiments/exp_realonly_4096_smaller_learnable_tau/run.sh`
- Results: `results/gift_eval/`
- Sync: `sync_realonly_4096_smaller_learnable_tau/learnable/`

**Setup**: same as #27 plus `--tau 0.07 --learnable-tau`. NOT gated on
#27 — runs in parallel (machine4 currently). Init at the established
0.07 baseline; the user explicitly said don't wait for #27's winner.

**Watch out**:
- Verify log lines show `τ=0.07XX` drifting (we've already seen
  0.0704 → 0.0706 over 700 steps). If τ stays exactly fixed, the
  gradient isn't reaching log_inv_tau — check `model.tau()` is passed
  as `tau_override` (must be a tensor, not a Python float).
- Auto-detect in head/eval: `if "log_inv_tau" in sd: BACKBONE_CONFIG
  ["learnable_tau"] = True`. Without this, load_state_dict crashes on
  the unknown key.

**Acceptance**: GM-MASE / GM-MAPE_SN / GM-CRPS_SN ≥ #27's best fixed-τ
result. Plus a τ-trajectory plot showing where it converges to.

### #29 — REPORT.md for #32

Standard report shape (template: `exp_realonly_4096_2arm/REPORT.md`).
Must include:

- The τ-trajectory plot. The losses CSV doesn't have a τ column — pull
  τ values via `grep "τ=" /workspace/app/run_learnable_tau.log` and
  pair with step numbers.
- GM-MASE / GM-MAPE_SN / GM-CRPS_SN headline.
- Comparison vs #27 fixed-τ winner.
- Note on the bs=96 caveat.

### #30 — REPORT.md for #27

Same shape. Three τ values, comparison plot, pick the winner. Plus the
caveat that the sweep was at bs=96 and the existing #20 EWMA-smaller
result at bs=24 with τ=0.07 is NOT a clean reference.

### #31 — MOIRAI-style optimizer (single arm)

**Conditional shape** depending on dataset readiness at launch time.

Check first:
```
curl -s -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/api/datasets/jeremycochoy/gift-pretrain-full-4096 \
  | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('exists' if d.get('id') else 'missing')"
```

#### Path A — gift-pretrain-full-4096 IS on HF

Run **2 arms on the FULL dataset, 15k steps each**:
- Arm 1: winner config WITHOUT MOIRAI hyperparams
- Arm 2: winner config WITH MOIRAI hyperparams (AdamW lr=1e-3, wd=0.1, β1=0.9, β2=0.98)

Both at bs=96. Compares MOIRAI effect at the data-rich regime AND tells
us full-dataset effect for free. Mark #33 as redundant in this case
(Arm 1 is what #33 was going to do).

#### Path B — full-4096 still missing

Original single-arm plan: winner config WITH MOIRAI hyperparams, on
gift-pretrain-small-4096, 30k steps. Compare to the τ-winner baseline.
#33 stays as planned for when full-4096 arrives.

**MOIRAI hyperparams** (both paths, same):
- AdamW (already)
- lr = 1e-3 (vs current 1e-4)
- weight_decay = 0.1 (vs 0)
- β1 = 0.9, β2 = 0.98 (vs 0.999)

**Code changes needed before #31 launches**:
- Add `--weight-decay`, `--adam-beta1`, `--adam-beta2` CLI flags to
  train.py + train_forecasting_head.py.
- Pass to `torch.optim.AdamW(...)`.
- Currently NOT yet implemented — the next agent does this.

**Watch out**:
- 10× lr is a big jump; numerical instability is a real risk. Watch
  the first 1k steps for NaN. (Per project rule, NO grad-clip — fix
  the underlying issue if NaN.)
- Path A's "15k steps each" is half the budget. Verify 15k is enough
  for convergence on the bigger dataset before publishing the verdict.

### #33 — Retrain winner on gift-pretrain-full-4096

**What**: clean reference run on the full dataset using whatever wins
from #31.

**Status if path A in #31 fired**: redundant — Arm 1 of #31 IS this. Just
delete #33 or rescope to "longer training on full-4096 with the #31
winner".

**Status if path B**: single arm, 1 full epoch through full-4096
(compute step count from manifest: ⌈total_rows / batch_size⌉). bs=96.
Same arch / norm / τ-policy / optimizer as #31 winner. NO grad-clip.

**Watch out**:
- Verify the dataset's split name first (`small_v1` was the convention
  for gift-pretrain-small-4096; full's may be `full_v1` or similar).
- Save-every: dependent on step count. For sub-1k steps, every 100;
  for 1k–10k, every 1000.

---

## 5 — Project-wide rules / invariants

### From `~/.claude/projects/.../memory/`

1. **NO grad-clip in this project, ever** (`feedback_no_grad_clip.md`).
   Forbidden. Grad-clip masks design defects we want to see. AdamW's v
   moving average already attenuates outlier gradients. If the data
   needs taming, curate the data — don't paper over training.
   - Source incident: realonly+T=4096+EWMA NaN'd at step 1697 from
     float32 cumsum overflow (real bug, fixed by float64 promotion in
     RevEWMNorm for T>2048). I added grad-clip on top defensively;
     user reverted it. The float64 fix alone was sufficient.

2. **Vastai instance reuse before destroy**
   (`feedback_vastai_instance_reuse.md`). Before destroying an instance,
   pause and ask: is the next queued task something that could reuse it?
   Idle instance time at $0.37–0.77/h is small compared to the latency
   of re-provisioning + the spawn-3 bug.

### From CLAUDE.md (project-checked-in)

3. **Sync dirs in MAIN checkout, never in a worktree**. Worktree may be
   `git worktree remove --force`-d, deleting untracked state.
4. **EVERY remote training run must have a sync_loop running** for
   crash recovery. 15-min cadence, atomic `.tmp → mv`, per-class size
   thresholds.
5. **NEVER raw-scp a checkpoint** — use `safe_pull.sh` (atomic with
   `.prev` backup).
6. **macOS case-insensitive FS** for `_FINAL.pth` vs `_final.pth`. Both
   exist on the Linux remote (run.sh's `cp _best_loss.pth _FINAL.pth`
   plus the trainer's auto `_final.pth`). On macOS they are the SAME
   file. Pre-rename remote `_FINAL.pth → _FINAL_safe.pth` before
   pulling lowercase, or just pull whichever you want and accept
   you'll lose the other locally.
7. **HF token** at `experiments/hf_token.txt` (gitignored). Every
   cloud run must export both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`.

### From CLAUDE.md (user-global)

8. **Vast.ai is a shared account**. Only destroy an instance if (a)
   contract ID came from your own provision call this session, (b)
   label matches your call, AND (c) GPU class matches. Otherwise leave
   it alone — could be another agent's mid-training work.

9. **Use vastrun-kit, not raw vastai** for provisioning + destruction.
   Read-only inspection (vastai show offers, search) via raw vastai
   is fine.

### Known vastrun-kit bugs (filed by us this session)

- **#315**: vastrun-provision spawns 2-3 duplicate instances per call
  (SSH-attach race). Mitigation: clean up duplicates immediately after
  via raw `vastai destroy instance <id> -y`. Keep the one that's
  `running` with `ssh_host`.
- **#330**: no `vastrun-label` command. Fall back to
  `vastai label instance <id> <name>`.
- **#333**: vastrun-cancel returns warning ("API did not confirm
  termination") when destroy doesn't actually fire. Fall back to raw
  `vastai destroy instance <id> -y`.

---

## 6 — Code conventions / file layout

### Worktree: `feat+composite-synth`

```
…/.claude/worktrees/feat+composite-synth/
├── src/
│   ├── models.py          ← ConfigurableModel + learnable_tau (#32)
│   ├── loss.py            ← contrastive_latent_loss + tau_override
│   ├── norm.py            ← RevEWMNorm with float64-cumsum-when-T>2048
│   └── dataloader.py      ← t_raw arg threaded through factories
├── experiments/
│   ├── freq-embedding/scripts/train.py            ← --t-raw, --n-channels, --d-model, --n-heads, --num-layers, --tau, --learnable-tau, --grad-clip (DON'T USE), --mix-ratio
│   ├── gift-eval/scripts/train_forecasting_head.py ← matching CLI flags + auto-detect log_inv_tau
│   ├── gift-eval/scripts/eval_gift_eval_official.py ← --t-raw, --backbone-c, SN-normalized columns (#18), auto-detect log_inv_tau
│   ├── exp_realonly_4096_2arm/                    ← #19 (Tiny + EWMA + RevIN at bs=24, mix=0.0, T=4096)
│   ├── exp_realonly_4096_smaller_2arm/            ← #20 (smaller arch sweep at bs=24)
│   ├── exp_realonly_4096_smaller_span_sweep/      ← #22 (span 32/128/256 at bs=24, smaller)
│   ├── exp_realonly_4096_smaller_tau_sweep/       ← #27 (τ=0.05/0.07/0.20 at bs=96, smaller, EWMA-128)
│   ├── exp_realonly_4096_smaller_learnable_tau/   ← #32 (CLIP-style learnable τ at bs=96)
│   ├── (TODO) exp_realonly_4096_moirai_optim/     ← #31 — to be created
│   └── (TODO) exp_realonly_full4096/              ← #33 — to be created when full-4096 arrives
└── docs/
    ├── HANDOFF_COMPOSITE_SYNTH_2026_04_30.md      ← prior handoff (April 30)
    └── PHASE5_FOLLOWUP_IDEAS.md
```

### Main checkout (sync dirs, NOT in worktree)

```
…/contrastive-forecasting/
├── HANDOFF.md  ← THIS FILE
├── sync_compositesynth_v5envboost/   ← phase 5 historical
├── sync_realonly_4096/{ewma128,revin}/   ← #19
├── sync_realonly_4096_smaller/{ewma128,revin,ewma_span32,ewma_span256}/  ← #20 + #22
├── sync_realonly_4096_smaller_tau_sweep/{tau005,tau007,tau020}/   ← #27
├── sync_realonly_4096_smaller_learnable_tau/learnable/   ← #32
├── (future) sync_realonly_4096_moirai_optim/   ← #31
└── (future) sync_realonly_full4096/  ← #33
```

---

## 7 — Headline numbers so far

(All on gift-pretrain-small-4096, T=4096, C=1, mix=0.0; bs as noted.)

| arm                               | bs   | GM-MASE | GM-MAPE_SN | GM-CRPS_SN |
|-----------------------------------|-----:|--------:|-----------:|-----------:|
| Tiny + EWMA-128 (#19)             |   24 |   1.81  |   1.43     |   1.08     |
| Tiny + RevIN (#19)                |   24 |   2.45  |   1.89     |   1.51     |
| **smaller + EWMA-128 (#20)**      |   24 | **1.78** | **1.24**  | **1.08**   |
| smaller + RevIN (#20)             |   24 |   2.53  |   1.85     |   1.55     |
| smaller + span=32 (#22)           |   24 |   1.74  |   1.28     |   1.08     |
| smaller + span=128 (=#20)         |   24 |   1.78  |   1.24     |   1.08     |
| smaller + span=256 (#22)          |   24 |   1.91  |   1.52     |   1.14     |
| (Aksu Moirai-Small reference)     |    — |    —    |   0.882    |   0.642    |
| v3-prim + EWMA-128 (phase 3)      |   24 |   1.62  |    n/a     |    n/a     |

**Two findings to remember**:
1. Synth was load-bearing in phases 1–5. realonly underperforms
   v3-prim by ~10% on GM-MASE.
2. smaller arch (L=6 H=384 nhead=6, 11.4M) beats Tiny (L=6 H=512
   nhead=8, 20M) on EWMA-128 — likely because 30k×bs=96 ≈ 47 epochs
   on 61k-row dataset means we're over-training, and smaller has
   less capacity to memorise.

---

## 8 — Picking up cold (10-step recipe)

1. `cd /Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/worktrees/feat+composite-synth`
2. `git status` → should be clean (or near-clean). `git log --oneline -10` for recent commits.
3. Read this HANDOFF + `docs/HANDOFF_COMPOSITE_SYNTH_2026_04_30.md` (prior handoff for phase 1-5 context).
4. `vastrun-status` → see live instances. Match labels to the table in §3.
5. SSH each instance, `tail -5 /workspace/app/run_*.log`, check progress.
6. `pgrep -af sync_realonly_4096` → confirm all expected sync_loops alive.
7. `pgrep -af "until ssh"` → confirm chain-launchers / auto-pulls if any are still relevant.
8. `TaskList` → see open tasks. Read full descriptions with `TaskGet <id>`.
9. `cat ~/.claude/projects/-Users-jeremycochoy-Desktop-workspace-trading-contrastive-forecasting/memory/MEMORY.md` (and the linked feedback files) — read project rules.
10. Refresh the `:47` cron if its prompt is stale (`CronList`, then CronDelete + CronCreate).

When a notification fires (sync completion, watcher fire, cron tick), it brings you back. The pipeline is largely self-driving — your job is to be there when something deviates from plan.

Good luck.
