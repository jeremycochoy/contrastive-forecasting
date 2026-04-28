# exp_csb_pair_revin — RevIN counterpart of the contrastive-loss A/B

**Status: complete (single seed each).**

## Why

Companion to `exp_csb_pair_span512`. That experiment showed that, with
matched protocol on RevEWMNorm span=512, `cosine_similarity_batch` (CSB)
beats `cosine_similarity_batch_no_time_neg` by 4.5% on MASE. The
question this answers: does the loss-flag direction generalise to
**RevIN** as the reversible normaliser, or is it specific to EWMA?

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
| Reversible norm | RevIN (no span parameter) |
| Backbone selector | `_best_loss → _FINAL.pth` |
| Qhead selector | `_best.pth → _FINAL.pth` |
| Save-every (bb / qh) | 2000 / 1000 |
| `--resume` | NOT used (single-shot) |
| Eval | 1024 held-out synth samples (`synth_eval.py`, seed=99999999) |

Single-axis difference between arms:

| Arm | `--loss-shape` |
|---|---|
| C | `cosine_similarity_batch_no_time_neg` |
| D | `cosine_similarity_batch` |

`synth_eval.py` was patched in this PR to accept `--rev-norm-kind
{ewma,revin}`. Default is unchanged (ewma); RevIN drops the span knob.

## Results (single seed each)

### 2-arm (RevIN only)

| Arm | GM-MASE | GM-WQL | MASE skill vs SN | WQL skill vs SN |
|---|---:|---:|---:|---:|
| **C** (`revin + no_time_neg`) | 1.072 | 0.531 | -116% | -54% |
| **D** (`revin + cosine_similarity_batch`) | **0.936** | **0.453** | **-88%** | **-32%** |
| Seasonal Naive (oracle period) | 0.497 | 0.344 | 0% | 0% |

**D beats C by 12.7% on MASE and 14.7% on WQL.** Plot:
`plots/synth_compare_pair_revin.png`.

### 4-arm cross-cutting grid (combined with `exp_csb_pair_span512`)

| Arm | Norm | Loss | GM-MASE | GM-WQL | MASE skill |
|---|---|---|---:|---:|---:|
| A | EWMA span=512 | no_time_neg | 0.924 | 0.449 | -86% |
| **B** | EWMA span=512 | **csb** | **0.883** | **0.432** | **-77%** |
| C | RevIN | no_time_neg | 1.072 | 0.531 | -116% |
| D | RevIN | csb | 0.936 | 0.453 | -88% |
| Seasonal Naive | — | — | 0.497 | 0.344 | 0% |

Plot: `plots/synth_compare_grid_4arm.png`.

## Findings

1. **CSB beats no_time_neg under both normalisers, in the same direction.**
   - EWMA: 0.924 → 0.883 (4.5% MASE improvement).
   - RevIN: 1.072 → 0.936 (12.7% MASE improvement).
   The within-time and cross-time negatives that the paper-matching loss
   adds are net helpful regardless of how the input is reversibly
   normalised. The earlier ARMA-era tuning that dropped these terms was
   a false economy.

2. **EWMA span=512 beats RevIN under both losses.**
   - no_time_neg: 0.924 (EWMA) vs 1.072 (RevIN), EWMA 13.8% better on MASE.
   - csb: 0.883 (EWMA) vs 0.936 (RevIN), EWMA 5.7% better on MASE.
   Consistent with `exp_span_sweep_synth`'s finding that span=512 is the
   right operating point on synth periodics with periods log-uniform in
   [8, 256]. RevIN normalises with the input window's own statistics
   (effectively span = T_RAW = 1024), which is past the optimal point on
   this distribution.

3. **The best single configuration of the four is EWMA span=512 + CSB
   at GM-MASE 0.883.** It does not beat the lost baseline at `_best_gap`
   (0.848 MASE), but `_best_gap` and `_best_loss` are not directly
   comparable selectors on this data, as discussed in
   `../exp_csb_pair_span512/REPORT.md`. Among the comparable
   `_best_loss` rows, this experiment's Arm B is the best so far on
   synth held-out.

## What was measured (no interpretation beyond the data)

- Both RevIN arms ran clean single-shot 30k bb + 30k qh on the same
  instance that ran the EWMA pair. Same hyperparameters except
  `--loss-shape` between C and D, and `--rev-norm-kind ewma/revin`
  between {A,B} and {C,D}.
- Same eval script, same seed, same byte-identical SN baseline values
  (0.49723 / 0.34394) confirm the held-out test set is identical to
  every prior arm in `_aggregate/results/synth_eval.csv`.
- D is better than C on every reported metric and on most of the 12
  visual panels. Same for B vs A.

## Speculation (single seed)

- **Effect-size gap between EWMA and RevIN.** The loss-flag delta is
  roughly 3x larger on RevIN (12.7%) than on EWMA (4.5%). With a single
  seed per cell the magnitude itself is unidentifiable, but if the
  direction holds across seeds, one possible mechanism: RevIN normalises
  away the global mean and scale of the input window, removing
  low-frequency signal that the simpler loss might otherwise rely on for
  contrastive separation. The CSB loss compensates by putting more
  structure into the time-axis pairs.
- **EWMA-vs-RevIN gap.** EWMA span=512 keeps a wider context's mean
  (half-life ~352 steps, comparable to 2 x the longest synth period in
  [8, 256]), preserving enough low-frequency signal that even the
  simpler no_time_neg loss can work. RevIN strips this entirely, so the
  simpler loss has less to exploit.

## Caveats

- Single seed per arm.
- Loss-flag effect size differing by 3x between EWMA and RevIN is a
  comparison of two single-seed deltas; could be noise. A second seed
  on each pair would settle it.
- We did not measure RevIN at `_best_gap`. Given the earlier finding
  that `_best_gap` over-fit-protects on EWMA, this could change the
  ordering. Snapshots are preserved in `sync_csb_pair_revin/checkpoints/`
  for cheap re-eval.

## Open questions

- Across multiple seeds, are the 4.5% and 12.7% loss-flag effects
  stable? Is the 3x ratio between them real?
- Does the same selector finding (`_best_gap` better than `_best_loss`)
  hold for RevIN on the no_time_neg arm? Quick to test from preserved
  snapshots.
- Real-data eval: which of the 4 configurations transfers best
  off-distribution? The synth eval is in-distribution by construction;
  the right next move is to evaluate the 4 backbones on the GIFT-Eval
  slice.

## Provenance

- `run.sh` is the actual driver, committed.
- The Phase 2 instance was reused (raw `vastai create` plus manual SSH
  key attach to bypass the vastrun-provision destroy bug, see
  `feedback_vastrun_provision_destroy_bug.md`). Saved ~10 min of
  provisioning + setup. Total RevIN-pair training ~50 min on a single
  5090 (~24 sps backbone, ~57 sps qhead). Cost for Phase 3: ~$0.40.

## Artefacts

- Backbones + qheads (~80 MB + 2.5 MB each) under
  `<repo>/sync_csb_pair_revin/checkpoints/` (in the main checkout, NOT
  in any worktree, per CLAUDE.md rule 4):
  - `tiny_pair_revin_ntn_FINAL.pth`, `tiny_pair_revin_csb_FINAL.pth`
  - `R1q_pair_revin_ntn_FINAL.pth`, `R1q_pair_revin_csb_FINAL.pth`
  - `_best_gap.pth` snapshots also preserved for both arms.
- Plots (committed):
  - `plots/synth_compare_pair_revin.png`: 12-panel × 2-arm forecast grid
    for the RevIN pair on its own.
  - `plots/synth_compare_grid_4arm.png`: 12-panel × 4-arm grid covering
    EWMA × RevIN × ntn × csb.
- Local results CSV: `results/synth_eval.csv` (the 2 RevIN rows).
- Aggregate CSV: 2 new rows appended in
  `../_aggregate/results/synth_eval.csv`.
- Patched script: `../freq-embedding/scripts/synth_eval.py` now accepts
  `--rev-norm-kind`.
