# RevIN-synth (mix=1.0)

## Why

EXP1 reproduced the previous-session RevIN result on mix=0.5, but the
mix=0.5 + GIFT-Eval setup is OOD relative to training. To test whether
RevIN actually helps on the in-distribution synth, we trained a
RevIN backbone+qhead on mix=1.0.

## Setup

| Knob | Value |
|---|---|
| Steps | 60k backbone, 30k qhead |
| Mix ratio | 1.0 |
| Freq emb | dim=3, mixup=0.3 |
| Reversible norm | RevIN (per-instance z-score) |
| Patch stats | none |
| Loss | `cosine_similarity_batch_no_time_neg` |
| Eval | 1024 held-out synth samples |

The 60k backbone choice (vs 30k for the fe+mu / pstats arms) was to
give RevIN extra training compute given its lower default gap on
synth — plus the 4-row plot the user had asked for at this point
included a "60k" arm for comparison.

## Results

### Backbone (`tiny_femu_revin_synth60k`)

- Wallclock: ~1.2h (single GPU shared with span sweep)
- Best gap: ~0.77 (lower than fe+mu+ewma's 0.85)
- Best loss: progressively decreased through 60k

### Quantile head (`R1q_femu_revin_synth60k`)

- 30k steps, ~20 min on synth-only (28 sps).

### Synth eval

| Arm | GM-MASE | GM-WQL | MASE skill | WQL skill |
|---|---:|---:|---:|---:|
| **RevIN-synth @ 60k** | **2.230** | **1.201** | **−348%** | **−249%** |
| fe+mu @ 60k (span=32) | 2.366 | 1.293 | −376% | −276% |
| fe+mu @ 30k (span=32) | 2.394 | 1.306 | −381% | −280% |
| fe+mu+pstats @ 60k | 2.411 | 1.319 | −385% | −283% |
| fe+mu+pstats @ 30k | 2.485 | 1.368 | −400% | −298% |

## Findings (single seed)

1. **RevIN-synth was the best of the original 4 arms on synth eval**
   (~5.7% better MASE than fe+mu @ 60k). Confirmed the user's
   earlier suspicion that RevEWMNorm `span=32` was over-smoothing
   the periodic structure on synth.

2. **But RevIN-synth was later dominated by EWMA span=64+** when the
   synth-only span sweep (see `../2026-04-27_exp_span_sweep_synth/REPORT.md`)
   revealed span=512 hits GM-MASE 0.848. The "RevIN better than EWMA"
   claim needs to be qualified to "RevIN better than EWMA *at the
   wrong span*". At the right span, EWMA wins by a wide margin on
   synth.

3. **Synth grid plot** (`synth_qhead_grid_*` for RevIN variants)
   visually similar to other arms — same amplitude damping and phase
   drift. Single-seed visual comparison; not strong evidence on its
   own, but consistent with the pattern across normalisers in this
   sequence.

## Caveats

- Single seed.
- This RevIN run used the established `cosine_similarity_batch_no_time_neg`
  loss. The follow-up cosine-similarity-batch comparison is queued
  next: train a RevIN-csb arm with the paper-matching loss to see if
  the within-time negatives interact with RevIN differently.

## Artefacts

- Backbone: `checkpoints/tiny_femu_revin_synth60k_FINAL.pth` (not
  tracked in git; 80MB).
- Qhead: `checkpoints/R1q_femu_revin_synth60k_FINAL.pth` (not tracked
  in git).
- Eval CSV row: "RevIN-synth @ 60k" in `../2026-04-27__aggregate/results/synth_eval.csv`.
- Run script: not preserved. The run was launched ad-hoc on remote
  (`/tmp/run_revin_synth.sh` on the vast.ai instance, lost when the
  instance was destroyed). The setup table above plus `run_synth_only.sh`
  as a template (substitute `--rev-norm-kind revin`, drop the span
  flag, set 60k bb steps) is enough to reproduce.
