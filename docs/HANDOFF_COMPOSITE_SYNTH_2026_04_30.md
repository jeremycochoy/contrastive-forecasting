# Handoff — composite-synth multi-phase experiment + next steps (2026-04-30)

This is a self-contained handoff covering: where Phase 5 stands, what is left
to do for it, and the two follow-on experiments the user has prioritised
(SN metrics + real-data-only training on `gift-pretrain-small-4096`).

Current cwd is the `feat+source-id-freq-plumb` worktree
(`/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/worktrees/feat+source-id-freq-plumb`),
**but all phase 5 code/docs live in the `feat+composite-synth` worktree**
(`/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/worktrees/feat+composite-synth`,
branch `feat/composite-synth`, current head `bf8719d`). PR #89 is open against
`experiments`. Switch to the composite-synth worktree to wrap up phase 5.

## Task list at compact time

| ID  | Status         | Subject                                              |
| --- | -------------- | ---------------------------------------------------- |
| #16 | completed      | Phase 4: combine pulse + more-primitives             |
| #17 | **in_progress** | Phase 5: explosive-trend env_gain bump (wrap-up)    |
| #18 | pending        | Add SN-normalized MAPE/CRPS to eval pipeline         |
| #19 | pending        | Real-data only training on gift-pretrain-small-4096  |

## #17 — Phase 5 wrap-up (IN PROGRESS)

### Working dir

`/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/.claude/worktrees/feat+composite-synth/experiments/exp_compositesynth_v5envboost_2arm/`

Has `README.md` + `run.sh` already committed (commit `2d7bd9c`). `results/`,
`scripts/`, `plots/` are all empty in the worktree.

### Sync dir (live data)

`/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_compositesynth_v5envboost/`
* `ewma128/`: full sync, 97/97 configs, has `summary.txt`
* `revin/`:   partial sync (segfault at config 72)

### v5 results vs prior winners

```
arm                   | EWMA-128 GM | RevIN GM
----------------------|-------------|----------
phase 0 baseline      |    1.659    |  1.859
phase 2A v2pulse      |    1.670    |  1.782   ← prev RevIN winner
phase 3  v3-prim      |  1.621      |  1.807   ← prev EWMA winner
phase 4  v4 combined  |    1.655    |  1.807
phase 5  v5 env100    |    TBD      |   TBD (partial — 72/97)
```

GMs not yet computed for v5 — DO compute when wrapping up (use only the
72-config intersection for revin vs prior winners to make it fair).

### v5-ewma128 (instance 35892408 @ ssh6.vast.ai:12408)

* ALL DONE — full 97/97 configs.
* Local CSV: `sync_compositesynth_v5envboost/ewma128/results/all_results.csv` (98 lines).
* Local summary.txt present.
* Local checkpoints present:
  - `tiny_compsyn_v5_ewma128_FINAL.pth`
  - `tiny_compsyn_v5_ewma128_best_loss.pth` + optimizer
  - `tiny_compsyn_v5_ewma128_best_gap.pth` + optimizer
  - `R1q_compsyn_v5_ewma128_best.pth` + optimizer
  - `R1q_compsyn_v5_ewma128_losses.csv`
  - periodic 2k…30k saves
* **Spot-check vs explosive-trend hypothesis**:
  - `covid_deaths/D/short` MASE = **69.26** (was 67.8 at v3 — env-bump did NOT help; basically unchanged within seed noise)
  - `bizitobs_application/10S/medium` MASE = **15.71** (was 16.3 — marginal)
  - `bitbrains_rnd/5T/long` MASE = **5.26** (was 5+ — unchanged)
  - `electricity/15T/long` MASE = 2.32 — unchanged
  - The headline finding: **env_gain (0.01, 100) does not address explosive-trend failure modes.** Phase 5 hypothesis falsified — covid-style growth is a *shape* problem (saturation/logistic), not a *gain* problem. See PHASE5_FOLLOWUP_IDEAS.md item B (tanh/logistic envelope).

### v5-revin (instance 35892708 @ ssh4.vast.ai:12708)

* SEGFAULTED at config 72/97 during eval. Last successful: `kdd_cup_2018/D/short` MASE=1.87. The crash happened *between* configs (gluonts-side, not training-side).
* Local CSV: `sync_compositesynth_v5envboost/revin/results/all_results.csv` (73 lines = header + 72 configs).
* Local summary.txt: **DOES NOT EXIST** — eval crashed before generating it. Don't pull it; write a partial-summary locally if needed.
* Local checkpoints present:
  - `tiny_compsyn_v5_revin_FINAL.pth` + best_loss + best_gap (+ optimizer for both)
  - `R1q_compsyn_v5_revin_FINAL.pth` + best (+ optimizer)
  - `R1q_compsyn_v5_revin_losses.csv`
* **Decision** (per user "no more synth experiments after phase 5 for now"):
  accept the partial 72-config result, document the segfault in REPORT.md, do
  NOT restart the eval. Note in REPORT that the missing 25 configs include
  `m4_yearly` (alphabetically after kdd) which would have been one of the key
  explosive-trend datapoints — slight asymmetry in the comparison.

### What's left for #17

1. **Copy CSVs to worktree**
   ```
   cp sync_compositesynth_v5envboost/ewma128/results/all_results.csv \
      .claude/worktrees/feat+composite-synth/experiments/exp_compositesynth_v5envboost_2arm/results/gift_eval_ewma128/
   cp sync_compositesynth_v5envboost/ewma128/results/summary.txt \
      .claude/worktrees/feat+composite-synth/experiments/exp_compositesynth_v5envboost_2arm/results/gift_eval_ewma128/
   cp sync_compositesynth_v5envboost/revin/results/all_results.csv \
      .claude/worktrees/feat+composite-synth/experiments/exp_compositesynth_v5envboost_2arm/results/gift_eval_revin/
   ```
   (mkdir -p the gift_eval_<arm> subdirs first.)

2. **Clone v4 plotter**
   - Source: `experiments/exp_compositesynth_v4combined_2arm/scripts/plot_compare_2arm.py`
   - Dest: `experiments/exp_compositesynth_v5envboost_2arm/scripts/plot_compare_2arm.py`
   - Edit: swap arm labels (v3-prim vs v5-ewma128, v2pulse vs v5-revin), swap CSV paths.
   - For revin comparison, restrict to the 72-config intersection (drop rows
     not present in v5-revin partial CSV).
   - Output to `experiments/exp_compositesynth_v5envboost_2arm/plots/`.

3. **Pull missing checkpoint files (if any)**: spot-check by ssh-ls against
   remote vs local. macOS case-insensitive FS protocol: pre-rename remote
   `*_FINAL.pth` → `*_FINAL_safe.pth`, scp lowercase `*_final.pth` if needed,
   md5-verify, then `mv` to canonical. Most files are already synced — only
   pull what's missing.

4. **md5-verify** local vs remote for the synced checkpoints (especially
   FINAL — this is the irreplaceable file). Use:
   ```
   ssh -p <port> root@<host> "cd /workspace/app/checkpoints && md5sum tiny_compsyn_v5_<arm>_FINAL.pth"
   md5 sync_compositesynth_v5envboost/<arm>/checkpoints/tiny_compsyn_v5_<arm>_FINAL.pth
   ```

5. **Shutdown both vast.ai instances** (only the two I created, by label and
   contract ID — see `vastai show instances` above):
   - `35892408` label `compositesynth-v5-ewma128-0430`
   - `35892708` label `compositesynth-v5-revin-r2-0430`
   - Use `vastai destroy 35892408 35892708` (or vastrun-cancel).
   - Kill the local sync_loops:
     `pkill -f "sync_compositesynth_v5envboost/sync_loop.sh"`

6. **Write `REPORT.md`** at
   `experiments/exp_compositesynth_v5envboost_2arm/REPORT.md`:
   - Headline: **env_gain bump (0.01, 100) does NOT improve explosive-trend
     extrapolation.** covid_deaths unchanged, bizitobs marginal.
   - GM table (use 72-config intersection for revin).
   - Per-config diff vs v3-ewma128 / v2pulse-revin for the explosive-trend
     candidates: covid_deaths, m4_yearly (only ewma128 has this), saugeen,
     bizitobs_application.
   - Note v5-revin segfault at config 72, eval crashed at next-after-kdd_cup,
     suspected gluonts-side issue (not training).
   - Conclusion: the env knob is *shape-limited*, not dynamic-range-limited.
     Future work should add a saturating/logistic envelope (PHASE5_FOLLOWUP_IDEAS
     item B), not push env_gain wider.
   - Cost: ~$3.40 × 2 arms ≈ $6.80 (one cycle ran 2.5h × $0.37/h, plus failed re-launch).

7. **Commit + push to PR #89**
   ```
   cd .claude/worktrees/feat+composite-synth
   git add experiments/exp_compositesynth_v5envboost_2arm/
   git commit -m "Phase 5: env_gain (0.01, 100) does not help explosive trends"
   git push
   ```

8. **Update task #17 to completed.**

## Cross-cutting context for phases 1–5 (already completed)

### Best-of-breed at end of phase 4 (entering phase 5)

* **EWMA-128 winner**: `--more-primitives` (v3 = sin/sq/saw/triangle/half_sin pool). GM = 1.621.
* **RevIN winner**: `--enable-pulse` (v2pulse = sin/sq/saw/pulse pool). GM = 1.782.

### Lessons that should not be forgotten

* **Diversity beats redundancy**: phase 2B (more seas-tied waves at same period)
  hurt; phase 3 (new shapes) helped. Phase 4 (combine 2pulse + 3prim into 6
  primitives) regressed both norms — pool-size ceiling is 5.
* **74/97 configs still worse than seasonal naive** at v3-EWMA-128 (the
  best). Top failure modes: explosive trends, spike-driven CloudOps,
  M4 short-history, long-horizon energy.
* **EWMA-128 wins outright on real data** at all phases, despite RevIN being
  the more popular reversible normaliser in the literature.

### Useful files (composite-synth worktree)

* `src/synthetic_composite.py` — main synth module. Knobs: `enable_pulse`,
  `enable_more_primitives`, `n_free_waves` (def 2), `n_seas_tied_waves` (def 1),
  `env_gain_range` (def (0.1, 10)).
* `src/dataloader.py` — `MixedCompositeLoader`, `create_mixed_composite_dataloader`.
* `src/norm.py` — RevEWMNorm with float32 + cached buffers (33% CPU speedup,
  max 3e-6 relative error vs float64).
* `experiments/freq-embedding/scripts/train.py` — backbone trainer; flags
  `--synth-kind {periodic,composite}`, `--enable-pulse`, `--seas-heavy`,
  `--more-primitives`, `--env-gain-max <X>`.
* `experiments/gift-eval/scripts/train_forecasting_head.py` — qhead trainer;
  same synth flags.
* `experiments/gift-eval/scripts/eval_gift_eval_official.py` — eval pipeline.
* `docs/PHASE5_FOLLOWUP_IDEAS.md` — comprehensive future-work doc, items A–U.
* `experiments/exp_compositesynth_*` — six experiment dirs (phases 1–5).

## #18 — SN-normalized MAPE/CRPS (PENDING)

### Working dir

To create when starting:
`experiments/gift-eval/scripts/eval_gift_eval_official.py` (modify in place,
no new dir needed).

### Why

Aksu et al. (GIFT-Eval paper) report **SN-normalized** skill scores:
* GM-MAPE target = **0.882** (Moirai-Small-on-GiftEvalPretrain reference)
* GM-CRPS target = **0.642** (same)

Right now our pipeline emits raw MASE per config and averages geometrically
(GM-MASE). The SN-normalized version divides each per-config metric by the
seasonal-naive baseline's metric for that config, giving "skill score over
SN" rather than absolute MASE.

### What

Modify `eval_gift_eval_official.py`:
1. For each config, additionally fit `gluonts.SeasonalNaivePredictor` (no
   training; the predictor is parameter-free) and emit SN's MAPE and WQL on
   the same forecast windows.
2. Add CSV columns: `eval_metrics/MAPE_SN`, `eval_metrics/WQL_SN`,
   `SN_MAPE_ratio = MAPE / MAPE_SN`, `SN_WQL_ratio = WQL / WQL_SN`.
3. In aggregator: GM of `SN_MAPE_ratio` across configs gives Aksu-comparable
   GM-MAPE. Same for CRPS via WQL ratio.
4. Note: gluonts `WeightedQuantileLoss` is the right CRPS proxy (Aksu paper
   uses `mean_weighted_sum_quantile_loss` already in our CSV — divide by SN's).

### How

* Reference: ts-forecasting-wiki PR #12 (https://github.com/redstone-solution-ou/ts-forecasting-wiki/pull/12) — user said "you can ask a sub agent". Spawn a guide subagent on this URL if any ambiguity remains about exact metric definitions.
* `gluonts.model.seasonal_naive.SeasonalNaivePredictor` is the canonical predictor; it takes `season_length` from the config metadata.
* The forecast horizon (16) and quantile levels (9) must match the contrastive predictor for a fair ratio.
* Run on existing checkpoints (no re-train needed) — pick v3-ewma128 best
  and v2pulse-revin best as the canonical references.

### Acceptance

Re-run eval on the v3-ewma128 and v2pulse-revin checkpoints; compare resulting
GM-MAPE and GM-CRPS to 0.882 / 0.642. Document delta in a new
`experiments/exp_sn_metrics/` report (or just add to PR #89's REPORTs).

## #19 — Real-data only on `gift-pretrain-small-4096` (PENDING)

### What & why

Train the two best per-norm checkpoints (v3-ewma128 + v2pulse-revin recipes)
on **real data only** (`mix_ratio=0.0`). Answers the question:
**how much of our gain comes from the synth recipe vs from the real-data
pretraining?** If real-data-only is competitive at GM-MASE, synth was a
useful regulariser; if not, synth was the main lever.

This is also the planned basis for upcoming **architecture-search** experiments
(scaling Tiny → Small → Base, depth/width sweeps).

### Dataset

`jeremycochoy/gift-pretrain-small-4096` (HuggingFace dataset).

* T_raw = **4096** (was 1024 in all phases 1–5)
* C = **1** (single-channel; was C=4 in phases 1–5)
* Contains the full GIFT-Eval pretrain corpus (Salesforce-released real time
  series, mixed domains).

User has confirmed the dataset is ready (or will be by start time — verify
before launching).

### Architecture changes needed

These are the **breaking changes** from the phases 1–5 setup. Get them right
before launching, otherwise the run will OOM or crash.

| Knob              | Phase 1–5            | Phase #19            | Notes                                                                |
| ----------------- | -------------------- | -------------------- | -------------------------------------------------------------------- |
| T_raw             | 1024                 | **4096**             | 4× longer raw sequence                                               |
| W (patch size)    | 16                   | 16                   | unchanged                                                            |
| T_patches (= T/W) | 64                   | **256**              | **16× more attention compute** (quadratic in seq len for some heads) |
| C (channels)      | 4                    | **1**                | single-channel input                                                 |
| batch_size        | 24                   | **start at 8**       | reduce to fit; benchmark on 5090 first                               |
| Norm              | revin / ewma-128     | revin / ewma-128     | keep both arms                                                       |
| `--mix-ratio`     | 0.5                  | **0.0**              | NO synth                                                             |
| `--synth-kind`    | composite            | omit / set to none   | flag may need to be optional                                         |
| Loss              | cosine_similarity    | unchanged            |                                                                      |
| total_steps       | 30k                  | **full single-pass** | with safety checkpoints                                              |

* **Verify training script supports `--mix-ratio 0.0`**: it should load real
  data only (the bundle's HFStreamingLoader path) and skip the
  `MixedCompositeLoader` synth gate. Test locally with a 100-step smoke run
  before launching cloud.
* **C=1 input plumbing**: backbone embedder probably hardcodes C=4 in places.
  Audit `src/encoder.py` (or wherever the patch embedding lives) for
  hardcoded channel counts. Either generalise to dynamic C or set a config
  flag.
* **Position embedding range**: if the encoder's posenc has a
  `max_seq_len=64` cap (matching phase 1–5's T_patches=64), bump to 256.
  Common pitfall — easily missed.
* **Memory**: T_patches=256 + C=1 vs T_patches=64 + C=4 → 256×1=256 tokens
  vs 64×4=256 tokens. Roughly the same total tokens, BUT attention is
  quadratic in seq_len (not in tokens-per-batch), so attention scales 16×.
  Linear/MLP layers scale 1×. Net memory bump probably 2–4×; bs=8 should
  fit on 5090 (32GB). Benchmark first.

### Setup

* New experiment dir:
  `experiments/exp_realonly_4096_2arm/`
  - `README.md` — hypothesis, knobs table above, expected outcomes
  - `run.sh` — 2 arms (revin + ewma128), each on a fresh 5090
  - `scripts/plot_compare_2arm.py` — vs phase 1–5 winners, vs SN baseline
* 2 fresh Vast.ai instances, 5090 (NOT 4090 — the 5090 is faster, user prefers
  these for 4096-seq runs).
* Use raw `vastai create` (vastrun-provision SSH-attach bug; documented in
  CLAUDE.md). Label format: `realonly-4096-<arm>-<MMDD>`.
* Sync dir in MAIN checkout (NOT worktree per CLAUDE.md):
  `/Users/jeremycochoy/Desktop/workspace/trading/contrastive-forecasting/sync_realonly_4096/<arm>/`
* sync_loop.sh per CLAUDE.md "Remote Machine Monitoring" rules:
  - 15 min cadence
  - Atomic .tmp → mv
  - Per-class size thresholds (BB ~80M, optimizer ~150M, head ~2.4M)
  - **Always sync optimizer files** (resume needs them)

### Checkpoints to preserve

Per user's explicit request:
1. **30k checkpoint** (`*_30k.pth` + optimizer)
2. **Last checkpoint** (`*_FINAL.pth` at end of full pass)
3. **Best loss checkpoint** (`*_best_loss.pth` + optimizer)
4. **Best gap checkpoint** (`*_best_gap.pth` + optimizer)
5. **Periodic safety checkpoints**: every 5–10k steps (`*_5k.pth` …
   `*_50k.pth` etc.). The trainer's `--save-every` flag handles this; tune it
   so we get one every ~1 hour of wall clock at minimum.

These will be used downstream by architecture-search experiments. Do NOT
delete from local sync dir until after #19's eval is done AND the architecture
search has the checkpoints it needs.

### Eval & report

* Run GIFT-Eval B4 on the 30k checkpoint AND the FINAL checkpoint for each arm.
* Compare 4 ways: (v3-ewma128 phase 3) vs (realonly-ewma128 30k) vs
  (realonly-ewma128 FINAL); same for revin.
* If #18 is done by then, also report SN-MAPE and SN-CRPS.
* REPORT.md headline: which lever — synth recipe or real-data pretraining —
  drove the bulk of the phase 1–5 gains?

### Cost estimate

* T=4096 vs T=1024 → 16× attention cost.
* Dataset is "small" → maybe 10–50× phase 5's data → say 30× steps for full
  pass.
* Per arm: 30× × 16× = ~480× phase 5's compute. But bs=8 vs bs=24 = 1/3
  effective compute per second. So ~160× phase 5's wall time.
* Phase 5 was 5h × $0.37/h. So #19 could be 800h × $0.37/h × 2 arms ≈ $590.
* **DOUBLE-CHECK with user before launching** — this is much bigger than phase
  1–5's $20 total budget.
* If it's too expensive: drop to single-pass on a *subset*, or single-arm
  first (ewma128, the better one).

### Order of operations

The user asked us to do these in this order:
1. Phase 5 wrap-up (#17) — almost done.
2. SN metrics (#18).
3. Real-data-only training (#19) — gated on dataset readiness.

So **finish #17 first, then #18, then #19**. Don't start #19 in parallel
with the others — it's the biggest budget item and benefits from #18 being
in place (so we can report SN-comparable metrics on day 1).

## Operational notes (don't forget these)

### Vast.ai is shared — DO NOT touch instances I don't own

* Only act on instances whose contract ID came from MY OWN `vastai create` (or
  `vastrun-provision`) call IN THIS SESSION.
* Verify by label match before any `vastai destroy`.
* For Phase 5: my instances are `35892408` (label `compositesynth-v5-ewma128-0430`)
  and `35892708` (label `compositesynth-v5-revin-r2-0430`).

### macOS case-insensitive FS protocol for `_FINAL.pth`

Linux remote has both `*_final.pth` (wherever the trainer wrote it) and
`*_FINAL.pth` (canonical alias copy). macOS treats them as the same file. To
sync without collision:

```bash
# On remote:
mv /workspace/app/checkpoints/<name>_FINAL.pth /workspace/app/checkpoints/<name>_FINAL_safe.pth
# On local (via safe_pull.sh):
bash experiments/periodic-synth-mix/scripts/safe_pull.sh \
  ssh4.vast.ai 12708 \
  /workspace/app/checkpoints/<name>_FINAL_safe.pth \
  sync_xxx/checkpoints/<name>_FINAL.pth 70000000
# md5-verify, then on remote restore the canonical name:
ssh ... "mv <name>_FINAL_safe.pth <name>_FINAL.pth"
```

### Sync dir always in MAIN checkout, never in a worktree

Per CLAUDE.md (learned the hard way April 2026): `git worktree remove --force`
deletes ALL untracked files. Sync dirs MUST be at
`/Users/.../trading/contrastive-forecasting/sync_<run_name>/`.

### NEVER raw-scp a checkpoint

Use `experiments/periodic-synth-mix/scripts/safe_pull.sh` — atomic .tmp + mv,
preserves a `.prev` backup, fails loud on size mismatch.

### sync_loop must run for the duration of every remote run

Even short runs. SSH drops on the final pull are common. The sync_loop pulls
periodic snapshots so a dead instance still leaves a recent local copy.

### HF token

`experiments/hf_token.txt` (gitignored). Every cloud run must:
```bash
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```
Without it: HF rate-limits → GPU idles at 0.5–1.5 sps instead of 5–9 sps.

## Cron / monitoring

Already running: `327c9b48` — every hour at :47, hourly check-in for phase 5.
Reuse for #18 and #19; update the prompt as each task starts. The 7-day
session cap means it auto-expires; refresh if the experiment runs longer.

## Where this doc lives

In the composite-synth worktree under `docs/`. Once committed it goes to
PR #89 with phase 5. Future-me (or any successor) reading this should be able
to pick up #17, #18, #19 cold.
