# exp_compositesynth_v5envboost_2arm — wider exp envelope (covid-style trends)

*Written: 2026-04-30. Date-stamp added: 2026-05-02.*

## Question

Phase 1–4 converged to a best-of-breed:
* **v3 (more-primitives) wins EWMA-128**: GM 1.621
* **v2pulse (enable-pulse) wins RevIN**: GM 1.782

74/97 configs are still worse than seasonal naive at the EWMA-128 winner.
The dominant remaining failure mode is **explosive-trend extrapolation**:
covid_deaths/D/short MASE 67.8, m4_yearly 8.4, saugeen 5.0,
healthcare growth 1.25, etc.

The composite synth has an `exp(λt)` envelope applied to 30% of channels,
but its `env_gain_range=(0.1, 10)` only spans **10× growth or decay**
across T=1024. covid_deaths actually grew ~200× over its observed window,
m4_yearly series grow even more.

Bumping the env range exposes the model to the kind of explosive
dynamics the failing configs need.

## What's new

A single new flag: `--env-gain-max <X>` (default 10.0). The env range
becomes `(1/X, X)` — log-symmetric around 1.0 (no growth/decay).

Phase 5 sets `--env-gain-max 100` → range `(0.01, 100)`. 100× growth or
decay across T. About 10000× wider in dynamic range than phase 1.

Two arms use best-of-breed flags from phase 2/3:
| arm | flags | rationale |
|---|---|---|
| revin | `--enable-pulse --env-gain-max 100` | adds env-bump on top of v2pulse winner |
| ewma128 | `--more-primitives --env-gain-max 100` | adds env-bump on top of v3 winner |

This is an **additive** test: does extending the env range improve
*on top of* the per-norm winner, or does it conflict?

## Predicted outcomes

* **Best case**: covid_deaths drops from 67.8 to single digits or below 1
  at EWMA-128, m4_yearly drops from 8.4 to ~2, GM drops by 5–10%.
* **Neutral**: covid stays high (env applies to some channels but covid
  needs *all-channel* aggressive trend), bulk MASE roughly unchanged.
* **Worst**: float32 precision strain at large gain values produces NaN
  spikes during training; loss explodes. Will monitor closely on first
  cycle and revert to (0.1, 10) if so.

## Setup

Identical to phase 1–4 except `--env-gain-max 100` plus the per-norm
best flag. Two arms in parallel, two fresh Vast.ai instances, 30k bb +
30k qhead + GIFT-Eval.

## Risk: numerical stability at extreme env

Worst-case input value:
* scale_max = 1000, env_gain_max = 100, wave_amp = 1, ARIMA target_std = 3
* peak per-channel pre-mix: 1000 × 100 × (1 + 3 + 1 + 1) = 6e5
* RevIN/EWMA-128 z-scores by std → normalised values stay in ~[-3, +3]
* float32 max representable: 3.4e38 → safe with 30+ orders of margin

The recently-optimised RevEWMNorm runs in float32 with cumsum tricks;
verified safe for `span=128` and `T=1024` even at peak ~6e5 values.

## Status

- [x] Code (single `--env-gain-max` flag added; env_gain_range plumbed through)
- [x] All 325 tests still pass
- [ ] 2 fresh Vast.ai instances provisioned
- [ ] Both arms launched
- [ ] Plotted vs phase 2/3/4 winners
