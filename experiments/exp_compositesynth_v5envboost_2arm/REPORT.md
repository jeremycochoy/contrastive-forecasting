# Phase 5 — env_gain_max=100 (explosive-trend test): REPORT

## Headline

**The env-bump (env_gain_max 10 → 100) does NOT improve explosive-trend
extrapolation.** Best-of-breed remains **v3-prim + EWMA-128** at full-97 GM
1.621.

| arm                                | configs | GM-MASE  | vs prior winner |
| ---------------------------------- | ------- | -------- | --------------- |
| v3-prim + EWMA-128 (prior winner)  | 97      | **1.621**| —               |
| v5envboost + EWMA-128 (env100)     | 97      | 1.661    | **+2.4% worse** |
| v2pulse + RevIN (prior winner)     | 97      | 1.782    | —               |
| v5envboost + RevIN (env100)        | 72*     | 1.653    | n/a (partial)   |

\* v5-revin eval segfaulted at config 72/97 — see "Eval segfault" below.

## 72-config intersection (apples-to-apples)

For the configs where all six arms have data:

| arm                                      | GM-MASE | median | max  | configs<1.5 |
| ---------------------------------------- | ------- | ------ | ---- | ----------- |
| **v3-prim + EWMA-128 (prior EWMA winner)** | **1.591** | 1.409 | 67.8 | 43/72 |
| v5envboost + EWMA-128 (env100)           | 1.633   | 1.459  | 69.3 | 41/72       |
| periodic + EWMA-128 (phase 0)            | 1.591   | 1.487  | 70.8 | 37/72       |
| v5envboost + RevIN (env100)              | 1.653   | 1.402  | 175.1| 37/72       |
| v2pulse + RevIN (prior RevIN winner)     | 1.673   | 1.440  | 152.1| 39/72       |
| periodic + RevIN (phase 0)               | 1.717   | 1.508  | 190.4| 34/72       |

* **EWMA-128**: v5 (env100) regresses by 0.04 GM (+2.6%) vs v3 (env10).
* **RevIN**: v5 improves by 0.02 GM (-1.2%) vs v2pulse — within seed noise
  (cross-seed variance ~3-5% per recovery-head search).

**Head-to-head per-config**:
* v5 EWMA vs v3 EWMA: v5 wins **25/72** (v3 wins 47/72). v3 dominates.
* v5 RevIN vs v2pulse RevIN: v5 wins **45/72**. Marginal.

## Per-config: explosive-trend hypothesis test

The whole reason for phase 5 was to test if exposing the synth to 100×
growth/decay would help configs like covid_deaths (which actually grew
~200× over its window). Per-config diff at EWMA-128 (v5 - v3):

| config                          | v3 MASE | v5 MASE | delta  |
| ------------------------------- | ------- | ------- | ------ |
| covid_deaths/D/short            | 67.79   | 69.26   | **+1.48** (worse) |
| m4_yearly/A/short               | 8.40    | 8.76    | +0.36 (worse)     |
| saugeen/D/short                 | 5.04    | 4.89    | -0.15 (marginal)  |
| bizitobs_application/10S/short  | 6.65    | 7.47    | +0.82 (worse)     |
| bizitobs_application/10S/medium | 16.31   | 15.71   | -0.60 (marginal)  |
| bizitobs_application/10S/long   | 15.26   | 15.88   | +0.62 (worse)     |
| electricity/15T/long            | 2.31    | 2.32    | +0.00 (no change) |

Signal is mixed and within seed noise on each config individually. **None
of the explosive-trend offenders meaningfully improved.** covid actually
regressed.

## Why env-bump didn't help — interpretation

Real explosive trends (covid, viral content, infectious disease) follow
**logistic / saturating curves**: accelerate, then plateau. Our envelope
is `exp(λt)` — pure monotonic exponential. Widening the gain range gives
longer monotonic runs without changing the shape.

A model trained on monotonic exponentials that have only seen 10× total gain
might still extrapolate badly on a real series doing 200× gain *in part of
the window*, especially if the rest of the window saturates. Going to 100×
just gives the model wider monotonic exemplars — same shape mismatch.

**The env knob is shape-limited, not gain-limited.** Future work (see
PHASE5_FOLLOWUP_IDEAS.md item B) should add a saturating envelope:
`y(t) = scale × tanh(steep × (t - t_mid))`, fired with its own coinflip
alongside the existing `exp(λt)` envelope.

## Eval segfault on v5-revin

Eval crashed mid-run at config 72/97 with a segmentation fault. Last
successful: `kdd_cup_2018/D/short` MASE=1.87. The crash happened *after*
that config's metrics were written and *before* the next config's eval
began — most likely a gluonts-side issue on one of the next 25 configs
(alphabetical: kdd_cup_2018/H/{short,medium,long}, m4_*, restaurant,
saugeen/{D,M,W}, solar/*, temperature_rain, us_births).

Per the "no more synth experiments after phase 5" decision, we accepted
the partial result rather than re-running the eval. The 72-config
intersection above is sufficient to show that the env-bump does not help
RevIN beyond seed-noise levels, and the EWMA-128 result (full 97) is
already decisive on its own.

The 25 missing configs include several heavy-MASE explosive-trend offenders
(m4_yearly, saugeen, solar series). If we extrapolate v5-revin's 72-GM
ratio of 1.6534 / v2pulse-72-GM 1.6730 = 0.988 to the full 97 (where
v2pulse-97-GM is 1.7817), v5-revin full GM ≈ 1.760 — still nowhere near
v3-EWMA-128's 1.621.

## Best-of-breed unchanged

Final best for the entire phase 1–5 sweep:

* **EWMA-128 winner**: `--more-primitives` (v3 = sin/sq/saw/triangle/half_sin pool).
  GM-MASE 1.621 on full 97. Single arm beats periodic baseline (1.659).
* **RevIN winner**: `--enable-pulse` (v2pulse = sin/sq/saw/pulse pool).
  GM-MASE 1.782 on full 97.

Phase 5 falsifies the explosive-trend dynamic-range hypothesis. **No
further synth experiments planned for now** per user decision; next steps
are SN-normalized metrics (#18) and real-data-only training on
`gift-pretrain-small-4096` (#19).

## Cost

| arm     | wall hours | rate | cost |
| ------- | ----------:| ----:| ----:|
| ewma128 | 18         | $0.37/h | $6.66 |
| revin   | 8.7 + relaunch | $0.32/h | ~$3.50 |
| total   |            |         | **~$10.20** |

(One v5-revin retry needed after the first instance had an SSH-key
propagation failure during init.)

## TODO — SN-normalized metrics (gated on task #18)

Once `eval_gift_eval_official.py` emits `SN_MAPE` and `SN_WQL` columns
(task #18), re-eval the v5 checkpoints (no retraining needed) and fill in
the table below. Targets from Aksu et al. (Moirai-Small-on-GiftEvalPretrain):
GM-MAPE = 0.882, GM-CRPS = 0.642.

| arm                              | GM-MASE | GM-MAPE_SN | GM-CRPS_SN |
| -------------------------------- | ------- | ---------- | ---------- |
| v3-prim + EWMA-128 (prior best)  | 1.621   | TODO       | TODO       |
| v5envboost + EWMA-128 (env100)   | 1.661   | TODO       | TODO       |
| v2pulse + RevIN (prior best)     | 1.782   | TODO       | TODO       |
| v5envboost + RevIN (env100, 72*) | 1.653   | TODO       | TODO       |

Re-eval should use the saved checkpoints in
`sync_compositesynth_v5envboost/<arm>/checkpoints/` plus the v3 / v2pulse
checkpoints from earlier phases.

## Plot

`plots/gift_eval_v5envboost_compare.png` — 4-panel comparison on 72-config
intersection: aggregate bars, MASE CDF, per-domain GM, head-to-head
scatter.

## Files

* `results/gift_eval_ewma128/all_results.csv` — full 97 configs
* `results/gift_eval_ewma128/summary.txt` — formatted per-config table
* `results/gift_eval_revin/all_results.csv` — partial 72 configs
* `scripts/plot_compare_2arm.py` — comparison plotter
* `plots/gift_eval_v5envboost_compare.png` — output plot
* `README.md` — pre-experiment hypothesis & setup
* `run.sh` — vast.ai run script

## Checkpoints (preserved locally for downstream use)

In `sync_compositesynth_v5envboost/<arm>/checkpoints/`:
* `tiny_compsyn_v5_<arm>_FINAL.pth` — backbone (final, used for eval)
* `tiny_compsyn_v5_<arm>_best_loss.pth` (+ optimizer)
* `tiny_compsyn_v5_<arm>_best_gap.pth` (+ optimizer)
* `R1q_compsyn_v5_<arm>_best.pth` (+ optimizer)
* periodic 2k…30k saves
* `R1q_compsyn_v5_<arm>_losses.csv`

Not committed (too large), but kept locally for any retro analysis.
