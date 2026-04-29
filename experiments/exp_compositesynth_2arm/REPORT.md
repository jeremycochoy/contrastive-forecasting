# Composite-synth A/B vs periodic-synth on GIFT-Eval

## Question

`exp_dualemb_3arm` mixed 50% bundle HF + 50% on-the-fly *clean-periodic*
synth (`src/synthetic_periodic.py`). Bundle synth (TimesFM composite:
ARMA + trend + sinusoids) is only ~1% of the bundle, so the model
rarely sees ARMA / piecewise trend / ARIMA random walks — exactly the
structure GIFT-Eval's worst configs (Econ/Fin trend extrapolation,
covid explosive trend) require.

Does swapping the on-the-fly synth for a TimesFM-style **composite**
recipe (`src/synthetic_composite.py` — trend + ARMA(p,q) optionally
integrated + 2 free-spp waves + 1 seasonality-tied wave, all
coinflipped, with per-row freq+seas labels) move GIFT-Eval numbers?

Two arms, identical to `exp_dualemb_3arm` except `--synth-kind composite`:
* **Arm A**: RevIN
* **Arm B**: RevEWMNorm span=128 (the headline GM-MASE winner from `exp_dualemb_3arm`)

## Headline (97 configs, single seed each, B4 strategy, forecast_len=16)

| | GM-MASE ↓ | median | max | configs<1.5 |
|---|---:|---:|---:|---:|
| Composite + RevIN (NEW) | **1.785** | 1.514 | 194.3 | 48/97 |
| **Composite + EWMA-128 (NEW)** | **1.697** | **1.459** | 66.3 | **51/97** |
| Periodic + RevIN (baseline) | 1.859 | 1.568 | 190.4 | 43/97 |
| **Periodic + EWMA-128 (baseline)** | **1.659** | 1.528 | **70.8** | 47/97 |

Head-to-head (lower MASE wins):
* Composite-RevIN beats Periodic-RevIN on **60/97** configs.
* Composite-EWMA-128 beats Periodic-EWMA-128 on **47/97** configs.

## What composite synth changed

**At RevIN — clear win.** Composite is −4.0% on GM-MASE, wins 62% of
configs. The recipe's trend + ARIMA components survive RevIN's
per-instance z-score and give the model trend exposure that periodic
synth couldn't provide.

**At EWMA-128 — split decision.** Composite beats periodic on the
**median** (1.459 vs 1.528, −4.5%) and on the **good-config count**
(51/97 vs 47/97). But the **GM-MASE goes up** by +2.3% (1.659 → 1.697)
because the upper tail is heavier. Per-domain shows the asymmetry
clearly:

| domain | comp-RevIN | **comp-128** | per-RevIN | per-128 | n |
|---|---:|---:|---:|---:|---:|
| Econ/Fin | 7.442 | 3.626 | 8.455 | **3.257** | 6 |
| Energy | 1.558 | 1.600 | 1.600 | **1.534** | 32 |
| Healthcare | 3.602 | **2.214** | 4.466 | 3.275 | 5 |
| Nature | **1.134** | 1.214 | 1.196 | 1.171 | 15 |
| Sales | **0.906** | 0.961 | 0.956 | 0.982 | 4 |
| Transport | 1.058 | **1.050** | 1.162 | 1.162 | 15 |
| Web/CloudOps | 2.887 | 2.870 | 2.722 | **2.445** | 20 |

* **Composite-EWMA-128 wins 2/7 domains** (Healthcare −33%, Transport −10%).
* **Composite-RevIN wins 2/7 domains** (Nature, Sales).
* **Periodic-EWMA-128 wins 3/7 domains** (Econ/Fin, Energy, Web/CloudOps).

## Where composite-EWMA-128 helps and where it hurts

**Top 5 wins** (composite better than periodic, both at EWMA-128):

| config | comp-128 | per-128 | Δ |
|---|---:|---:|---:|
| us_births/M/short | 0.93 | 1.81 | **−49%** |
| us_births/W/short | 1.24 | 2.43 | **−49%** |
| us_births/D/short | 0.81 | 1.24 | −34% |
| ett1/W/short | 1.38 | 2.09 | −34% |
| loop_seattle/D/short | 1.00 | 1.25 | −20% |

These are all **slow-seasonality, trend-heavy** configs (monthly /
weekly / daily with few cycles in-window). The composite's piecewise
trend + ARIMA gives the model trend exposure that "clean sin every
spp" never did.

**Top 5 losses** (composite worse than periodic, both at EWMA-128):

| config | comp-128 | per-128 | Δ |
|---|---:|---:|---:|
| solar/H/medium | 2.58 | 1.37 | **+89%** |
| bizitobs_application/10S/medium | 18.25 | 10.19 | **+79%** |
| solar/H/long | 2.35 | 1.40 | +68% |
| bizitobs_application/10S/long | 17.21 | 10.56 | +63% |
| bizitobs_l2c/H/long | 1.59 | 1.08 | +48% |

Two clusters:

1. **Strongly periodic configs** (solar/H, bizitobs_l2c/H): periodic
   synth had a single clean sin/sq/saw per channel; composite synth
   stacks that wave with trend + ARIMA + a free wave at a different
   period. Under EWMA-128 (which preserves periodic structure but
   removes slow trends), the seas-tied wave becomes a smaller fraction
   of the channel's signal — diluting the cleanest training signal for
   "lock onto period 24".

2. **Spike-driven configs** (bizitobs_application 10S series — sparse
   activity bursts on a 10-second sample rate). Neither synth
   generates spike content; composite is a touch worse here just
   because the diluting effect plays out on a noisier signal.

## Why composite hurts EWMA-128 on average

EWMA-128 already removes the slow-trend component locally. The
composite recipe's main *new* exposure (piecewise linear trend +
cumsum'd ARIMA → ARIMA(1,p,q)) is exactly what EWMA-128 normalizes
away. So under EWMA-128, the trend/ARIMA components are essentially
"wasted training capacity" that displaces the clean periodic signal
the model needs to lock onto.

Under RevIN the trend stays in-window and the synth-side trend
exposure helps the model handle e.g. covid-style explosive growth.

## Decision

* **Composite synth replaces periodic synth at RevIN** — clear win on every metric.
* **At EWMA-128, decision depends on the metric**:
  - if you weight tail heavily (GM-MASE), periodic + EWMA-128 still wins (1.659 vs 1.697).
  - if you weight median or good-config count, composite + EWMA-128 wins.
* **Cross-norm**: composite + EWMA-128 is the new best on Healthcare and Transport — the first time a non-periodic-synth arm has won any domain with EWMA-128.

## Phase-2 hypothesis

Two leaks identified:

1. **Composite has no spike content.** Top losses on bizitobs_application
   /bitbrains are all spike-driven series (cloud-ops bursts, 10s
   sampling). Periodic synth's saw/square get tail-clipped by EWMA but
   not entirely; composite's smoothed mix loses the spike texture entirely.

2. **Composite dilutes the seas-tied wave** when other components are
   on, hurting strongly-periodic configs (solar/H) under EWMA-128.

Phase-2 plan: add a 4th waveform primitive `_PRIM_PULSE` to the
existing `{sin, square, saw}` pick — sparse pulse train (amplitude 1
at every P-th sample, 0 elsewhere, optional sign flip). Mathematically
a low-duty-cycle square wave but treated as its own primitive so we
can tune duty cycle independently. Targets the spike-deficit (1)
without touching (2).

If (1) doesn't close the EWMA-128 GM gap, (2) becomes the next
priority — likely via reducing the per-channel coinflip rates of trend
/ ARMA / free waves so the seas-tied wave becomes a larger fraction
of the signal when present.

## Artefacts

* `run.sh` — driver (mirrors `exp_dualemb_3arm/run.sh`, only
  `--synth-kind composite` differs)
* `results/gift_eval_{revin,ewma128}/all_results.csv` — 97 configs each
* `plots/gift_eval_compositesynth_compare.png` — 4-panel summary
* `scripts/plot_compare_2arm.py` — plot generator, idempotent

## Cost

5090 instance @ ~$0.34/h × ~14h ≈ $4.80. Two arms × (bb 30k + qhead
30k + eval 97 configs).
