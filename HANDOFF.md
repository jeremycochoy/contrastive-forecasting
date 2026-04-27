# Session Handoff — late April 2026 (continuation, day 2)

This handoff supersedes the previous late-April handoff. The previous
session's three queued items are all done, plus one user-added experiment
(patch-stats), plus a synth-only redo, plus a real-data and synth-only
span sweep. Branch: `feat/periodic-synth-mix`.

## TL;DR — what to do next

1. **Re-introduce the within-time negative in the contrastive loss.**
   See [`experiments/freq-embedding/FOLLOWUP.md`](experiments/freq-embedding/FOLLOWUP.md)
   for the full proposal. Quick version: `loss_shape='cosine_similarity_batch_no_time_neg'`
   currently drops the `h[b, t-1, c] vs h[b, t, c]` push-apart term. That
   was tuned for ARMA training data (high lag-1 autocorrelation made it
   counter-productive). On periodic data, adjacent latents *should* be
   pushed apart. Run as a single-axis ablation on the best arm we have:
   **fe+mu, mix=1.0 synth-only, 30k, span=512, ewma**. Cost ~$0.30.

2. **Validate the span=512 finding on real data.** The synth sweep
   peaked at span=512 (GM-MASE 0.848 — best of every arm we ran). The
   real-data sweep (mix=0.0, 20k steps) showed only ~6% loss
   improvement going from span=32 to span=128, with span=256 *worse*
   than span=128. Either (a) real data doesn't benefit from longer
   spans, or (b) 20k steps wasn't enough to reveal the effect. A
   30k-step real-data sweep at the synth winner (span=512) is a
   useful disambiguation — ~$0.50, ~83 min.

3. **Patch-stats operator follow-up.** The current
   `compute_patch_stats(kind='diff')` uses `(mean[t]-mean[t-1])/std[t-1]`
   for `dmean`. User flagged this can spike on series that move
   between very different absolute scales within one context window
   (e.g., 10000 → 0.01). Even with std-normalisation the dmean can
   briefly hit O(10) before std catches up. Better operators (asinh-
   diff, log-of-abs + sign-channel) noted in
   `~/.claude/projects/.../memory/feedback_patch_stats_dmean_op.md`.
   Patch-stats was a wash on synth (slightly *worse* than baseline at
   30k and 60k); the operator question and span=512 may both unlock
   that capacity. ~$0.30 for one re-run with a better operator.

After those three, the experimental thread is in a clean state —
candidates for the next round in
[`experiments/freq-embedding/FOLLOWUP.md`](experiments/freq-embedding/FOLLOWUP.md).

## What this session ran (chronological)

| # | What | Result | Where |
|---|---|---|---|
| EXP1 | RevIN reproduction (#28 redo) | Reproduced; backbone gap=0.469, qhead loss=0.052; synth grid still shows the patch-boundary issue (RevIN doesn't fix it). | `experiments/freq-embedding/plots/synth_qhead_grid_revin.png` |
| EXP4 (a) | Patch-stats on mix=0.5 + GIFT-Eval | Backbone gap=**0.626** (+33% over RevIN), qhead loss=0.071 (worse than RevIN). On the 23 with-SN-baseline configs: pstats slightly *worse* than fe+mu+qh and RevIN+qh. Synth grid unchanged. | `experiments/freq-embedding/results/comparison_with_sn.csv`, `plots/synth_qhead_grid_pstats.png` |
| Synth-only round | fe+mu and fe+mu+pstats × {30k, 60k} on mix=1.0 + held-out 1024-sample synth eval | 30k→60k helps both arms ~1-2%. patch-stats was 1-3% *worse* than baseline at both step counts. | `experiments/freq-embedding/results/synth_eval.csv`, `plots/synth_qhead_grid_*.png` |
| Real-data span sweep | span ∈ {32, 64, 128, 256}, mix=0.0, 20k steps each, backbones-only | ema_loss U-shaped: 32→3.00, 64→2.89, 128→**2.83**, 256→2.92. Gap monotonically *decreases* with span: 32→0.33, 256→0.30. Metrics disagree. | `experiments/freq-embedding/plots/span_sweep_real.png` |
| Synth span sweep | span ∈ {32, 64, 128, 256, 512, 1024}, mix=1.0, 30k bb + 30k qhead per span + 1024-sample synth eval | **Inverted-U with peak at span=512** (GM-MASE 0.848, MASE skill -71%). span=1024 falls back to 0.921. | `experiments/freq-embedding/plots/span_skill_synth.png`, `plots/span_compare_synth.png` |
| RevIN-synth | RevIN backbone + qhead on mix=1.0 (60k bb, 30k qh) | GM-MASE 2.230 — best of the 4 *original* synth arms but dominated by every span≥64 EWMA arm. | `synth_eval.csv` row "RevIN-synth @ 60k" |

## Final aggregate table (1024-sample synth eval, sorted best first)

SN baseline uses the *known* period and is essentially optimal here.

| Arm | GM-MASE | GM-WQL | MASE skill | WQL skill |
|---|---:|---:|---:|---:|
| **fe+mu @ 30k span=512** | **0.848** | **0.413** | **−71%** | **−20%** |
| fe+mu @ 30k span=1024 | 0.921 | 0.452 | −85% | −31% |
| fe+mu @ 30k span=256 | 1.049 | 0.517 | −111% | −50% |
| fe+mu @ 30k span=128 | 1.192 | 0.600 | −140% | −74% |
| fe+mu @ 30k span=64 | 1.761 | 0.918 | −254% | −167% |
| RevIN-synth @ 60k | 2.230 | 1.201 | −348% | −249% |
| fe+mu @ 60k (span=32) | 2.366 | 1.293 | −376% | −276% |
| fe+mu @ 30k (span=32) | 2.394 | 1.306 | −381% | −280% |
| fe+mu+pstats @ 60k | 2.411 | 1.319 | −385% | −283% |
| fe+mu+pstats @ 30k | 2.485 | 1.368 | −400% | −298% |
| fe+mu peak_gap (matched qhead) | 2.622 | 1.443 | −427% | −319% |
| Seasonal Naive | 0.497 | 0.344 | 0% | 0% |

## What we KNOW (single-seed; treat as data points, not verdicts)

1. **RevEWMNorm span=32 was the wrong default for periodic data.**
   On synth periodics with periods up to ~256, the optimal span on this
   sweep is 512. That's a >2.8× improvement on GM-MASE over the previous
   default. The trend was monotonic up to 512 and reverses at 1024.

2. **EWMA at the right span beats RevIN on synth.** RevIN-synth = 2.230.
   span=512 = 0.848. The user's earlier "RevIN-vs-EWMA" comparison
   (#28 in the previous session) was confounded by the bad span — both
   arms in that comparison used span=32.

3. **30k → 60k matters less than span.** Both fe+mu @ 30k (2.394) and
   fe+mu @ 60k (2.366) on span=32 are dominated by every span ≥ 64. The
   architecture knob (span) wins by a much wider margin than the
   compute knob (60k vs 30k) on this data.

4. **Patch-stats didn't help on synth.** fe+mu+pstats @ 30k = 2.485 vs
   fe+mu @ 30k = 2.394 — pstats is 4% *worse*. The contrastive backbone
   gap improvement (0.85 → 0.85, basically tied; on mix=0.5 it was
   +33% but didn't transfer to forecasts either) doesn't translate to
   downstream forecast quality. The user's later feedback about the
   `dmean` operator (asinh-diff vs std-normalised diff) is a plausible
   reason worth chasing in a follow-up.

5. **Best_gap.pth saturates very early on synth-only.** Both fe+mu
   30k and 60k runs hit gap=0.842 at step 1600 in deterministic
   training (same seed, same data). The "peak gap" model is *worse*
   than 30k or 60k end-of-training models — early-stopping on gap is
   the wrong signal here.

6. **Real-data span doesn't transfer cleanly from synth.** mix=0.0
   sweep at 20k steps: span=128 best on loss (5.9% better than 32),
   span=256 *worse* than 128. Could be insufficient steps or genuinely
   different optimum on real data — to be disambiguated.

## Open questions / follow-ups

- **#FOLLOWUP-1**: re-introduce the within-time contrastive negative
  (see `experiments/freq-embedding/FOLLOWUP.md`).
- **#FOLLOWUP-2**: real-data span sweep at 30k+ steps to validate the
  span=512 finding outside synth.
- **#FOLLOWUP-3**: patch-stats `dmean` operator comparison (current
  `(Δmean)/std` vs asinh-diff vs log-abs+sign).
- **#FOLLOWUP-4**: with span=512 backbone, retry quantile-head training
  on the GIFT-Eval setup — does the better backbone improve OOD MASE/WQL?

## Operational template — what worked this session

The pattern that worked end-to-end after fixing the bugs we hit:

```bash
# 1. Local code changes in a worktree (not the user's main checkout!).
git worktree add ../contrastive-forecasting-cf feat/periodic-synth-mix
cd ../contrastive-forecasting-cf
# ... edit ...

# 2. Provision a single 4090 via vastrun-provision.
vastrun-provision --label cf-multiexp-... --gpu-model RTX_4090 \
    --num-gpus 1 --min-vram 20 --min-reliability 0.99 --max-bid 0.5 \
    --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# (The kit's SSH-key-attach failure is a known issue — the instance
# can still be alive even when vastrun-kit thinks it lost it; check
# `vastai show instance <id>`.)

# 3. Upload code + token; launch via nohup with tee to run_all.log.

# 4. Local sync_loop running for the duration. PR #47 atomicity fixes
#    + the bumped 130MB optimizer threshold (this session) prevent
#    partial-transfer corruption from leaking into local copies.

# 5. Multiple training stages can run in parallel on the same GPU.
#    A 4090 can comfortably hold 2 backbones (~6 GB each) + their
#    qheads simultaneously. Use that.

# 6. Pull losses CSVs eagerly (they're small) for live analysis.
#    Pull .pth files only when needed for plotting/eval.

# 7. CRITICAL: clean up checkpoints. We hit disk-full (60GB) twice;
#    the optimizer.pth files are ~155 MB each and accumulate fast.
#    Delete them after a run completes — we keep model weights only.

# 8. Use a held-out synth eval set (synth_eval.py) for fast in-distribution
#    quality checks during the run; full GIFT-Eval is ~5h and not
#    necessary for arch comparison.
```

## Bugs caught and fixed this session

1. **`train.py::forward_step` reimplemented patching manually** and
   silently dropped the patch-stats concat. EXP4 stage 1 crashed at
   first batch because the encoder expected wider input. Routed
   through `ConfigurableModel.prepare_encoder_input` and added a
   regression test in `tests/test_norm.py`.

2. **`create_mixed_periodic_dataloader(mix_ratio=0.0, ...)` ignored
   `emit_freq_ids`.** The short-circuit returned `create_hf_dataloader`
   directly, which yields a Tensor (not a tuple). Caused the real-data
   span sweep to crash with "too many values to unpack". Fixed in
   `src/dataloader.py` — only short-circuit when `emit_freq_ids=False`.

3. **`_FINAL.pth` was set from `_best_gap.pth`** which saturated at step
   1600 in deterministic synth-only training. 30k and 60k runs produced
   byte-identical FINAL backbones. Caught by md5sum + per-sample
   forecast diff. Repointed FINAL to end-of-training snapshots
   (`_30k.pth` / `_60k.pth`) and re-trained the qheads against the
   correct backbones.

4. **`sync_loop.sh` 70 MB optimizer threshold** let through a 78 MB
   partial transfer that overwrote a 155 MB good copy. PR #47 rotation
   saved us (good copy in `.prev`); bumped the threshold to 130 MB
   so partial transfers fail outright.

5. **`synth_eval.py` C=4 spp shape bug.** `meta["spp"]` is shaped
   `[batch_size * C]` (flattened); my code treated it as `[bs, C]`.
   Switched to C=1 single-channel synth samples (matching
   `plot_synth_qhead.py`) — simpler and matches the intent.

6. **Disk full on remote.** Periodic snapshots (`_*k.pth`) plus
   `*_optimizer.pth` files filled the 60GB image partition. Cleared
   them mid-run. Going forward, prune optimizer files after each
   completed run.

## Cost summary

This session: ~$10 of vast.ai spend (4090 single-GPU, ~25h
wall-time). The user topped up the budget mid-run when it was running
low; final balance ~$22 unspent.

## Branch / artefact pointers

- Branch: `feat/periodic-synth-mix` (pushed to origin).
- Worktree path I used: `~/Desktop/workspace/trading/contrastive-forecasting-cf`
  (separate from the user's main checkout, which stayed on whatever
  branch they were on).
- All 8 backbones + heads on remote at `/workspace/app/checkpoints/`
  (will be lost when the instance is destroyed; pull anything you
  want preserved).
- Plots: `experiments/freq-embedding/plots/` (synth_qhead_grid_*,
  synth_compare_*, span_sweep_real, span_compare_synth, span_skill_synth).
- Eval CSVs: `experiments/freq-embedding/results/synth_eval.csv` (synth)
  and `comparison_with_sn.csv` (GIFT-Eval).
- New scripts: `synth_eval.py`, `synth_compare_grid.py`,
  `run_synth_only.sh`, `run_span_sweep_real.sh`, `run_span_sweep_synth.sh`.
