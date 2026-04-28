# exp_csb_pair_span512 — clean A/B of contrastive losses on the span=512 best arm

**Status: complete (single seed each).**

## Why

Two reasons converge here:

1. **The span=512 baseline (no_time_neg) checkpoint was lost.** During
   the wrap-up of `exp_csb_synth`, an auxiliary git worktree was torn
   down with `git worktree remove --force`, which deletes all untracked
   files in the worktree directory, including `sync_multiexp/checkpoints/`
   where the only local copy of `tiny_femu_span512_synth30k_FINAL.pth`
   lived. The corresponding remote vast.ai instance had been destroyed
   shortly before, so the checkpoint was unrecoverable. CLAUDE.md rule 4
   (PR #82) was added to prevent this happening again.

2. **`exp_csb_synth` had a multi-resume confound.** That CSB run
   suffered three remote-instance failures during backbone training; its
   FINAL checkpoint was the result of `8k → resume → 24k → resume → 30k`,
   not a clean single-shot training. A clean rerun was needed before
   treating that result as definitive.

Plus, two unintended differences had crept into the historical
comparison: the lost baseline used `_best_gap → FINAL.pth` while the
CSB run used `_best_loss → FINAL.pth`. Different selectors mean the two
FINAL checkpoints aren't sampling the same point in training. This
experiment retrains both arms cleanly, single-shot, and with **the same
selector (`_best_loss`)**, so the only axis between them is the loss
flag.

## Setup

| Knob | Value (both arms) |
|---|---|
| Steps (backbone) | 30000 |
| Steps (qhead) | 30000 |
| Batch size | 24 |
| LR (bb / qh) | 1e-4 / 3e-4 |
| Mix ratio | 1.0 (synth-only) |
| Freq emb dim | 3 |
| Mixup | 0.3 |
| Reversible norm | RevEWMNorm span=512 |
| Backbone selector | `_best_loss → _FINAL.pth` (matched between arms) |
| Qhead selector | `_best.pth → _FINAL.pth` (qhead val loss) |
| Save-every (bb / qh) | 2000 / 1000 |
| `--resume` | NOT used (single-shot) |
| Eval | 1024 held-out synth samples (`synth_eval.py`, seed=99999999) |

Single-axis difference between the arms:

| Arm | `--loss-shape` |
|---|---|
| A | `cosine_similarity_batch_no_time_neg` |
| B | `cosine_similarity_batch` (paper-matching, includes within-time and cross-time negatives) |

## Results (single seed each, clean)

| Arm | GM-MASE | GM-WQL | MASE skill vs SN | WQL skill vs SN |
|---|---:|---:|---:|---:|
| **A** (`no_time_neg`) | 0.924 | 0.449 | −86% | −31% |
| **B** (`cosine_similarity_batch`) | **0.883** | **0.432** | **−77%** | **−26%** |
| Seasonal Naive (oracle period) | 0.497 | 0.344 | 0% | 0% |

**B beats A by 4.5% on MASE and 3.9% on WQL.** Plot of medians on 12
random fixed-seed synth samples: `plots/synth_compare_pair.png`.

### Comparison with historical rows on the same eval

| Arm | GM-MASE | GM-WQL | Selector | Continuity | Notes |
|---|---:|---:|---|---|---|
| `fe+mu @ 30k span=512` (lost baseline, exp_span_sweep_synth) | 0.848 | 0.413 | `_best_gap` | clean | — |
| `fe+mu @ 30k span=512 +cosine_similarity_batch` (exp_csb_synth) | 0.886 | 0.434 | `_best_loss` | multi-resume | 8k → 24k → 30k |
| **A** `pair span=512 ntn (clean, best_loss)` (this run) | 0.924 | 0.449 | `_best_loss` | clean | |
| **B** `pair span=512 csb (clean, best_loss)` (this run) | 0.883 | 0.432 | `_best_loss` | clean | |

Decomposition of the four numbers:

- **A vs lost baseline (same loss, different selector):** clean
  `_best_loss` (0.924) is **9% worse on MASE** than `_best_gap` (0.848)
  for the no_time_neg loss. On synth, `_best_gap` saturates around step
  ~1600 deterministically, so what looked like a "30k baseline" was in
  practice an early-stopped checkpoint, and that early-stopped checkpoint
  forecasts better than the full-30k version under the same loss. Caveat:
  single seed on each side, which could partly be sampling noise, but a
  9% gap on a single axis change is large.
- **B vs exp_csb_synth (same loss + same selector, different
  continuity):** 0.883 vs 0.886, within noise. The multi-resume runs in
  exp_csb_synth turned out to be approximately a clean run for the CSB
  arm.
- **B vs A (the headline):** 0.883 vs 0.924. `cosine_similarity_batch`
  beats `cosine_similarity_batch_no_time_neg` on the matched-protocol
  comparison. The within-time and cross-time negatives that the
  paper-matching loss adds are net helpful at span=512 + synth-only.

### What this changes about the original conclusion

The original `exp_csb_synth` finding ("CSB is 4.5% worse than baseline")
was **measuring three differences at once**: loss flag, selector, and
continuity. Holding the latter two fixed (this experiment), CSB is
**4.5% better**, not worse. The earlier conclusion was inverted by the
selector confound.

`_best_loss` was the right call for this comparison (gap is not a useful
selector on synth at span=512), but it cost roughly 9% MASE on the
no_time_neg arm relative to early-stopping at the gap-saturation point.
That's a separate, equally interesting finding: **on this data, the
contrastive-loss objective and the downstream forecasting objective are
imperfectly aligned**. Minimum-loss is not minimum-forecast-error.

## What was measured (no interpretation beyond the data)

- Both arms ran clean single-shot 30k bb + 30k qh on the same instance
  (no `--resume`). Same hyperparameters except `--loss-shape`.
- Same eval: byte-identical SN baseline values (sn_gm_mase=0.49723,
  sn_gm_wql=0.34394) confirm the held-out 1024 samples are bit-identical
  across all rows in the aggregate CSV.
- B is better than A on every reported metric and on most of the 12
  visual panels.

## Speculation (single seed; not validated)

1. **Cross-time negatives help on periodic data.** The paper-matching
   loss `cosine_similarity_batch` includes negatives between `h[b, t-1, c]`
   and `h[b, t, c]` (within-channel, within-batch, different times) and
   `h[b, t-1, c1]` vs `h[b, t, c2]` (cross-channel). On periodic synth,
   adjacent latents walk a non-trivial manifold; pushing them apart
   sharpens the representation. The earlier ARMA-era tuning that dropped
   these terms may have been a false economy.

2. **Single-seed gap of 4.5% is comfortably outside the within-arm noise
   we saw on the multi-resume vs clean comparison (~0.3% on B).** Likely
   real, but a second seed of either arm would be the standard
   confirmation.

3. **The `_best_loss` selector hurts forecasting on this data.** The 9%
   gap between A@best_loss (0.924) and the lost baseline @best_gap
   (0.848) suggests there's an over-fitting-like phenomenon in the
   contrastive backbone after step ~1600 when the gap stops climbing.
   The qhead is then trained against a less-useful representation. This
   should be re-examined when we have a more general-purpose evaluation
   harness (real data, multi-domain).

## Caveats

- Single seed per arm.
- The "9% selector cost" on the no_time_neg arm is two single-seed
  numbers compared against each other (lost baseline 0.848 vs new clean
  ntn 0.924). Without a re-run with `_best_gap` for both arms, we can't
  cleanly attribute it to selector vs seed sensitivity.
- The CSB arm at `_best_gap` was never measured (the lost-baseline regime
  pre-dated CSB). If we re-eval the periodic snapshots from this run we
  could fill that in cheaply: `_best_gap.pth` and `_2k.pth` ... `_30k.pth`
  are all preserved in `sync_csb_pair_ewma/checkpoints/`.

## Open questions

- Does the `_best_gap` selector also help the CSB arm? Re-evaluating the
  CSB arm with `_best_gap.pth → FINAL.pth` would settle whether this is
  a loss-specific or general phenomenon.
- Does the same loss-flag effect (B beats A) hold for **RevIN** as the
  reversible normaliser? Tracked as the next experiment in the queue
  (`exp_csb_pair_revin`).
- Across multiple seeds, is the 4.5% gap stable?

## Provenance

- `run.sh` is the actual driver, committed.
- The first two Vast.ai instances each died mid-run (lingering destroy
  from a failed `vastrun-provision` SSH-attach earlier propagated with
  delay). The third attempt succeeded by using raw `vastai create` plus
  manual SSH key attach to bypass the vastrun-provision destroy bug.
  Total run time ~50 min on a single 5090 (23.8 sps backbone, 57 sps
  qhead). Cost burned across the three attempts: ~$2.50.
- `feedback_vastrun_provision_destroy_bug.md` saved to project memory.

## Artefacts

- Backbones + qheads (~165 MB total per arm in checkpoints, optimisers
  excluded from this size) under
  `<repo>/sync_csb_pair_ewma/checkpoints/` (in the main checkout, NOT
  in any worktree, per CLAUDE.md rule 4):
  - `tiny_pair_span512_ntn_FINAL.pth` (Arm A backbone)
  - `tiny_pair_span512_csb_FINAL.pth` (Arm B backbone)
  - `R1q_pair_span512_ntn_FINAL.pth` (Arm A qhead)
  - `R1q_pair_span512_csb_FINAL.pth` (Arm B qhead)
  - `_best_gap.pth` and periodic `_2k.pth … _30k.pth` snapshots also
    preserved for both arms.
- Local results CSV: `results/synth_eval.csv` (the 2 new rows).
- Aggregate CSV: 2 new rows in
  `../_aggregate/results/synth_eval.csv`.
- Plot: `plots/synth_compare_pair.png`, the 12-panel × 2-arm forecast grid.
- Training loss CSVs: `tiny_pair_span512_{ntn,csb}_losses.csv`,
  `R1q_pair_span512_{ntn,csb}_losses.csv` (in the sync dir).
