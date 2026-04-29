# exp_compositesynth_v2bseasheavy_2arm — composite synth + 2 seas-tied waves

## Question

Phase-1 (`exp_compositesynth_2arm`) revealed a "wave-dilution" leak at
EWMA-128: composite synth's seas-tied wave (1 of 4 wave/non-trend
slots) gets crowded out by ARMA + trend + 2 free waves. Top losses
under composite-EWMA-128 vs periodic-EWMA-128 included strongly
periodic configs (solar/H/medium +89%, solar/H/long +68%,
bizitobs_l2c/H/long +48%) — exactly the configs where periodic synth's
single-wave-per-channel design was a clean periodic prior the model
could lock onto.

Does **doubling the seas-tied wave count** (and reducing free waves by
one to keep the wave-slot count constant) fix the dilution without
losing phase-1's wins?

## What's new

In `src/synthetic_composite.py`, `_build_one_channel` now accepts
`n_free_waves` and `n_seas_tied_waves` parameters (defaults 2 and 1 =
phase-1 behaviour). Phase-2B uses `--seas-heavy` flag which sets
`n_free_waves=1, n_seas_tied_waves=2`. Both seas-tied waves draw from
the same row's bucket (so they share period bucket but have independent
waveforms, phases, and exact spp within the bucket range), giving
richer harmonic content at the matched seasonality.

Recipe stays identical otherwise:
* trend always on, ARMA / wave / seas-tied each Bernoulli(0.5)
* trend application 50% mult / 50% additive
* env p=0.3, scale log-uniform [0.1, 1000]
* per-row `(freq_id, seas_id)` labels (seas_id=0 if seas-tied off)

## Setup

| Knob | Value |
|---|---|
| Recipe | composite + `--seas-heavy` (2 seas-tied + 1 free wave) |
| Backbone | Tiny (H=512, L=6, GRU encoder, W=16, 20M params) |
| Loss | `cosine_similarity_batch` |
| Mix ratio | 0.5 |
| `freq_emb_dim` / `seasonality_emb_dim` | 3 / 3 |
| `mixup_p` | 0.3 |
| Backbone steps | 30 000 |
| Quantile head steps | 30 000 (R1 forecaster reconstruction, fl=16) |
| Selector | `_best_loss` → `FINAL.pth` |
| Eval | GIFT-Eval official, 97 configs, B4 |
| Seed | 42 (single-seed) |

Two arms run **in parallel** on two separate Vast.ai instances:
* **Arm A**: RevIN — `bash run.sh revin`
* **Arm B**: EWMA span=128 — `bash run.sh ewma128`

Runs concurrently with `exp_compositesynth_v2pulse_2arm` (4 instances total).

## Hypotheses being tested

1. **Seas-heavy beats composite-v1 on strongly periodic configs**
   (solar/H, bizitobs_l2c/H, electricity/H), under EWMA-128 specifically.
2. **Seas-heavy preserves composite-v1's gains on trend-heavy configs**
   (us_births, m4_monthly) — trend + ARMA components are unchanged.
3. **Seas-heavy helps both norms** (RevIN and EWMA-128) because cleaner
   periodic signal is generally useful, not norm-specific.

## Why pulse and seas-heavy are orthogonal

* **Pulse** (phase-2A `v2pulse`): adds a *new modality* (sparse spike
  bursts). Targets configs where the truth has rare large excursions
  (bizitobs_application, bitbrains).
* **Seas-heavy** (phase-2B `v2bseasheavy`): doubles *coverage of the
  existing periodic modality* in seas-tied-on rows. Targets configs
  where periodic structure already exists but the model fails to lock
  onto it (solar/H, bizitobs_l2c/H).

If both work, phase 3 combines them: `--enable-pulse --seas-heavy`.

## Files

| Item | Path |
|---|---|
| Recipe | `src/synthetic_composite.py` (gated by `n_free_waves`/`n_seas_tied_waves`) |
| Train flag | `--seas-heavy` on both `train.py` and `train_forecasting_head.py` |
| Driver | `experiments/exp_compositesynth_v2bseasheavy_2arm/run.sh` |
| Local sync | `sync_compositesynth_v2bseasheavy/{revin,ewma128}/` |

## Status

- [x] Code + tests landed (314/314 tests pass).
- [ ] 2 Vast.ai instances provisioned (alongside the 2 phase-2A instances = 4 total).
- [ ] Arm A (RevIN) + Arm B (EWMA-128) running concurrently.
- [ ] Results plotted vs phase-1 / phase-0 baseline.
