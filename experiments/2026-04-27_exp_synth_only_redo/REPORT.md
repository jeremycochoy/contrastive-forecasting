# Synth-only redo — patch-stats vs baseline on mix=1.0

## Why

The previous patch-stats run (see `REPORT_patch_stats_mix05.md`) used
mix=0.5 + GIFT-Eval. Two problems:
1. The GIFT-Eval is OOD relative to training data; an architectural
   change can help on training distribution but lose on OOD, confounding
   the comparison.
2. The 5h GIFT-Eval was the bottleneck for iteration speed.

User reframed the experiment: train on synth-only (mix=1.0) and eval on
held-out synth. This is in-distribution, runs fast, and isolates the
architectural change from OOD transfer.

Comparison includes 30k vs 60k step counts to disentangle "needs more
training" from "architecture limit".

## Setup

4 backbones × 30k or 60k:

| Arm | Mix ratio | Span | Patch stats | Steps |
|---|---|---|---|---|
| fe+mu @ 30k | 1.0 | 32 | none | 30000 |
| fe+mu @ 60k | 1.0 | 32 | none | 60000 |
| fe+mu+pstats @ 30k | 1.0 | 32 | diff | 30000 |
| fe+mu+pstats @ 60k | 1.0 | 32 | diff | 60000 |

Then a quantile head per backbone (also synth-only, 30k steps).

Eval: held-out synth via `experiments/freq-embedding/scripts/synth_eval.py`,
1024 samples, seed=99999999 (never used during training). Single-channel
(C=1) samples per the existing `plot_synth_qhead.py` convention.

Loss: `cosine_similarity_batch_no_time_neg` (the established default for
this arm family — to be revisited in the within-time-neg follow-up).

## Results

| Arm | GM-MASE | GM-WQL | MASE skill | WQL skill |
|---|---:|---:|---:|---:|
| **fe+mu @ 60k** | **2.366** | **1.293** | **−376%** | **−276%** |
| fe+mu @ 30k | 2.394 | 1.306 | −381% | −280% |
| fe+mu+pstats @ 60k | 2.411 | 1.319 | −385% | −283% |
| fe+mu+pstats @ 30k | 2.485 | 1.368 | −400% | −298% |
| Seasonal Naive | 0.497 | 0.344 | 0% | 0% |

(SN uses the *known* period and is essentially optimal on this clean
synth data.)

## Findings (single seed each)

1. **30k → 60k helps both arms ~1-2%.** Modest but monotonic on both
   metrics for both architectures. Compute knob has small diminishing
   returns at this scale.

2. **patch-stats is consistently 1-3% worse than baseline** at both step
   counts and on both metrics. The +33% backbone gap improvement we saw
   on mix=0.5 didn't translate to forecast-quality improvement here.

3. **All arms still ~5× worse than SN.** Synth grids show the models
   track periodic structure in a damped, shifted way but don't match
   SN-with-known-period. Patch-boundary phase loss appears structural
   to the W=16 patching, not solvable by architectural input changes
   at this scale.

4. **Synth grids look very similar across arms.** Visual differences
   between fe+mu and fe+mu+pstats grids are subtle on the 12-panel
   view; the metric differences come from systematic but small
   improvements/regressions, not from any arm "cracking" a class of
   panels the others miss.

## Bugs caught and fixed during this run

1. **`_FINAL.pth = best_gap.pth`** made 30k and 60k synth backbones
   byte-identical (gap saturates at step 1600 in deterministic synth-only
   training; same seed → same step-1600 weights). Caught by md5sum diff
   showing 30k_FINAL == 60k_FINAL. Fixed by repointing to end-of-training
   `_30k.pth` / `_60k.pth` snapshots and re-training the qheads against
   the correct backbones.

2. **`synth_eval.py` C=4 spp shape bug.** `meta["spp"]` is shaped
   `[batch_size * C]` (flattened); my code treated it as `[bs, C]`.
   Switched to C=1 single-channel synth samples to match
   `plot_synth_qhead.py`.

3. **Disk full on remote** during this run. ~60GB filled with periodic
   snapshot + optimizer files from many concurrent runs. Cleared 50GB
   by deleting `*_optimizer.pth` and old `_*k.pth` periodic snapshots.

## Caveats

- Single seed per arm.
- patch-stats `dmean = (mean[t]−mean[t−1])/std[t−1]` operator can spike
  on series with very different absolute scales within a context window
  (user-flagged). asinh-diff or log-of-abs+sign would be a cleaner
  follow-up.
- The 60k arm's `best_gap` step was identical to the 30k arm's because
  the gap metric saturates very early on synth-only. Going forward,
  best_loss is the right selector (per HANDOFF rationale).

## Conclusion (single seed)

On synth-only, **fe+mu @ 60k (no patch-stats)** is the marginal best
of the 4 arms tested here, but the differences are small (1-4%). The
real lever found later in this session was the EWMA span (see
`../2026-04-27_exp_span_sweep_synth/REPORT.md`) which dwarfs every architectural
knob in this report.

## Artefacts

- Backbones: `checkpoints/tiny_femu_synth{30k,60k}_FINAL.pth`,
  `tiny_femu_pstats_synth{30k,60k}_FINAL.pth` (not tracked in git).
- Heads: `checkpoints/R1q_femu_synth{30k,60k}_FINAL.pth`,
  `R1q_femu_pstats_synth{30k,60k}_FINAL.pth` (not tracked in git).
- Eval CSV: `../2026-04-27__aggregate/results/synth_eval.csv` (rows 1-4).
- Synth grids in this dir: `plots/synth_qhead_grid_{synth30k,synth60k,pstats_synth30k,pstats_synth60k}.png`.
- Run script: `run.sh`.
