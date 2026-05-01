# exp_realonly_4096_smaller_span_sweep — EWMA-128 span sweep on the smaller arch

## Question

#19 and #20 both used `--rev-norm-span 128` for the EWMA arms — that's
the value we landed on in phase 5 (and the only one we've evaluated at
T=4096). The realonly + T=4096 setting has very different
dynamic-range characteristics than the phase 1-5 mix=0.5 + T=1024
runs, so the span optimum may have shifted.

This experiment sweeps span ∈ {32, 64, 256, 512} with the **smaller
arch winner** from #20 (L=6 H=384 ffn_mult=4 nhead=6, 11.4M params).
Span=128 is already covered by exp_realonly_4096_smaller_2arm so it's
the reference point.

## Architecture choice

Smaller (L=6 H=384) won the #20 sweep on:
* GM-MASE: 1.783 (vs Tiny 1.805, ~1.2% better)
* GM-MAPE_SN: **1.243** (vs Tiny 1.432, **13% better**)
* GM-CRPS_SN: 1.082 (vs Tiny 1.083, basically tied)
* Plus 43% fewer params and ~1.5× faster training.

So #22 fixes arch=smaller and sweeps span only.

## Spans

| span | alpha=2/(span+1) | typical "memory" |
| ----:| ----------------:| ---------------- |
| 32   | 0.061            | ~32 timesteps    |
| 64   | 0.031            | ~64 timesteps    |
| 128  | 0.0155           | ~128 timesteps (already done in #20) |
| 256  | 0.0078           | ~256 timesteps   |
| 512  | 0.0039           | ~512 timesteps   |

At T=4096, span=512 means the EWM stats average over 12.5% of the
window, which is already a fairly slow-moving normaliser. Going past
that probably stops being a "norm" and becomes a "global" stat.

## Setup

Identical to exp_realonly_4096_smaller_2arm except `--rev-norm-span`
varies. Single arm per span (EWMA only — RevIN doesn't have a span knob).

Machines:
- ssh6.vast.ai:12408 (35892408, freed after #20 EWMA-smaller done) —
  start with span=32 ASAP
- ssh9.vast.ai:17138 (35927139, busy with #20 RevIN-smaller until ETA) —
  start span=64 once free, then span=512 in series
- EWMA box runs span=32, then span=256

## Acceptance

97-config GIFT-Eval × 5 spans (4 new + the existing span=128) →
plot showing per-config MASE / MAPE_SN / CRPS_SN as a function of span.
The minimum of the GM curve is the new optimum.

## Status

- [x] arch winner picked (smaller, L=6 H=384 nhead=6)
- [x] run.sh built (parameterised on span)
- [ ] sync_loops + scripts deployed
- [ ] span=32 launched
- [ ] span=64 launched
- [ ] span=256 launched
- [ ] span=512 launched
- [ ] Plot + REPORT.md
