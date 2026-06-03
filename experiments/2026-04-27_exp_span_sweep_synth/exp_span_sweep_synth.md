# Synth-only RevEWMNorm span sweep

## Why

Optional follow-up after the user topped up the budget. The real-data
sweep ([`exp_span_sweep_real.md`](../2026-04-27_exp_span_sweep_real/exp_span_sweep_real.md)) showed metrics disagreeing
(loss U-shaped at 128, gap monotonically lower at 256). Run the same
sweep on synth-only to (a) test on the model's training distribution
where the freq embedding has perfect info, and (b) get a downstream
held-out eval signal in addition to training loss/gap.

## Setup

| Knob | Value |
|---|---|
| Spans tested | 32, 64, 128, 256, 512, 1024 (extended after monotonic improvement at 256) |
| Mix ratio | **1.0** (pure synth) |
| Steps per backbone | 30000 |
| Architecture | fe+mu (freq_emb=3, mixup=0.3) |
| Patch stats | none |
| Loss | `cosine_similarity_batch_no_time_neg` |
| Eval | held-out 1024-sample synth eval (`synth_eval.py`, seed=99999999) |
| Per arm | 30k bb + 30k qhead + 1024-sample synth eval (full pipeline) |

`span=32` is covered by the existing `tiny_femu_synth30k_FINAL.pth`
from the synth-only redo (same config).

## Results

| Span | GM-MASE | GM-WQL | MASE skill | WQL skill |
|---|---:|---:|---:|---:|
| 32 (existing baseline) | 2.394 | 1.306 | −381% | −280% |
| 64 | 1.761 | 0.918 | −254% | −167% |
| 128 | 1.192 | 0.600 | −140% | −74% |
| 256 | 1.049 | 0.517 | −111% | −50% |
| **512** | **0.848** | **0.413** | **−71%** | **−20%** |
| 1024 | 0.921 | 0.452 | −85% | −31% |
| Seasonal Naive | 0.497 | 0.344 | 0% | 0% |

(Plots: `plots/span_skill_synth.png` — inverted-U on a log-2 x-axis.
`plots/span_compare_synth.png` for the 6-arm × 12-sample comparison
grid.)

## Findings (single seed each)

1. **Inverted-U with peak at span=512.** Both MASE and WQL agree on
   the optimum — same span wins on both metrics, unlike the real-data
   sweep where they disagreed.

2. **Span=512 is 2.8× better on MASE than the previous span=32 default**
   (0.848 vs 2.394). On WQL, 3.2× better (0.413 vs 1.306). This is by
   far the biggest single-knob improvement we found in the late-Apr
   2026 sequence.

3. **Span=512 beats RevIN-synth.** RevIN-synth `mix=1.0, 60k steps`
   scored GM-MASE 2.230 (see `../2026-04-27_exp_revin_synth/exp_revin_synth.md`); span=512
   scores 0.848. The earlier "RevIN better than EWMA on periodic data"
   finding (#28 from previous session) was confounded by both arms
   using `span=32`.

4. **Past the peak, span=1024 falls back** to GM-MASE 0.921. The EWMA
   at span=1024 has half-life ~352 steps — wider than the longest
   period in the synth sampler (256). At that point the EMA
   under-tracks the local mean and we lose information.

## Interpretation (speculative)

The synth periodic data has periods log-uniform [8, 256]. The
optimal span (512) is roughly 2× the longest period — wide enough
that the EMA can track even the slowest cycles' mean drift without
also tracking the cycle itself. Below 256 the EMA over-smooths the
periodic content of long cycles; above 512 it under-tracks the mean
of even short cycles.

## Caveats

- Single seed per arm.
- The optimum (512) is specific to the synth sampler's spp range
  [8, 256]. On real data with potentially different period
  distributions, the optimum may differ. Real-data sweep at 20k steps
  (`../2026-04-27_exp_span_sweep_real/exp_span_sweep_real.md`) hinted the answer is in the
  span ∈ {64, 128} range there but with insufficient compute to be
  confident.

## Open questions

- Real-data span sweep at 30k+ steps with downstream eval — does
  the real-data optimum also climb to 256 or 512 with more compute?
- Validate span=512 finding with second seed.
- The within-time contrastive negative was dropped during ARMA-era
  tuning; with span=512 as the new best, redo the comparison
  cosine_similarity_batch (with the negative) vs no_time_neg.
  Tracked separately as the cosine-similarity-batch follow-up
  (in flight at time of writing).

## Artefacts

- 5 backbones (span 64/128/256/512/1024):
  `checkpoints/tiny_femu_span*_synth30k_FINAL.pth` (not tracked in
  git). Plus `tiny_femu_synth30k_FINAL.pth` for span=32 (also from
  `2026-04-27_exp_synth_only_redo`).
- 5 qheads: `checkpoints/R1q_femu_span*_synth30k_FINAL.pth` (not
  tracked in git).
- Eval CSV rows: `../2026-04-27__aggregate/results/synth_eval.csv`.
- Plots in this dir:
  - `plots/span_skill_synth.png` — skill curves.
  - `plots/span_compare_synth.png` — 6-arm × 12-sample forecast grid.
- Run script: `run.sh` (covers spans 64/128/256; span=512 and 1024
  were launched ad-hoc on remote and the run scripts were not preserved
  here — the same run.sh template extends trivially by adjusting the
  for-loop range).
