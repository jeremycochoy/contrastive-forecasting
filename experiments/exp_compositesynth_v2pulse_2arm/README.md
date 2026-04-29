# exp_compositesynth_v2pulse_2arm — composite synth + PULSE primitive

## Question

Phase-1 (`exp_compositesynth_2arm`) result: composite synth beats periodic
synth at RevIN (−4% GM) but loses GM-MASE at EWMA-128 (+2.3%) despite
winning on median and good-config count. Top losses concentrated in
**spike-driven Web/CloudOps configs** (bizitobs_application,
bitbrains/bizitobs_service) — neither composite nor periodic synth
generates spike content, so models trained on either fail on series
that consist of mostly-flat baseline plus rare large excursions.

Does adding a 4th waveform primitive — a sparse pulse train — close
the spike gap without disturbing the other phase-1 wins?

## What's new

A new `_PRIM_PULSE` primitive in `src/synthetic_composite.py`:

* sparse pulse train at every spp-th sample, ±1 amplitude (random sign)
* duty cycle = `pulse_width / spp` (default `pulse_width=1` ⇒ Dirac comb)
* gated behind `--enable-pulse` flag; when off, synth is byte-identical to phase-1

Pulse joins `{sin, square, saw}` as the 4th option in the wave-primitive
pick (uniform over 4 instead of 3). The seas-tied wave and both free
waves share the same pool, so ~25% of any wave slot will draw pulse.

This survives EWMA-128 normalisation (a brief ±1 burst barely shifts
the local EWMA mean) where the slow-moving trend / ARIMA components
get smoothed away. So pulse specifically targets the EWMA-128 case
where composite v1 currently has nothing extra over periodic.

## Setup

Same recipe as `exp_compositesynth_2arm`, only `--enable-pulse` differs.

| Knob | Value |
|---|---|
| Backbone | Tiny (H=512, L=6, GRU encoder, W=16, 20M params) |
| Loss | `cosine_similarity_batch` |
| Mix ratio | 0.5 (50% bundle base_mixed_v1 + 50% on-the-fly composite **with pulse**) |
| `freq_emb_dim` / `seasonality_emb_dim` | 3 / 3 |
| `mixup_p` | 0.3 |
| Backbone steps | 30 000 |
| Quantile head steps | 30 000 (R1 forecaster reconstruction, 9 quantiles, fl=16) |
| Selector | `_best_loss` → `FINAL.pth` |
| Eval | GIFT-Eval official, 97 configs, B4 |
| Seed | 42 (single-seed) |

Two arms run **in parallel** on two separate Vast.ai instances:
* **Arm A** (instance 1): RevIN — `bash run.sh revin`
* **Arm B** (instance 2): RevEWMNorm span=128 — `bash run.sh ewma128`

Each arm writes its own `run_${ARM}.log` so a per-instance sync_loop
pulls just that arm's artefacts.

## Hypotheses being tested

1. **Pulse closes the EWMA-128 GM gap.** Phase-1 composite-EWMA-128 was
   1.697 vs periodic-EWMA-128 1.659 (+2.3%). Pulse should remove the
   tail-pulling spike-deficit configs (bizitobs_application
   /bitbrains), bringing GM under 1.659.
2. **Pulse keeps phase-1's median + good-config wins.** Phase-1
   composite-EWMA-128 had median 1.459 vs periodic-EWMA-128 1.528.
   Pulse only adds capacity; it shouldn't undo wins on slow-trend
   configs (us_births, ett1/W, m4_monthly).
3. **Pulse helps RevIN too** (smaller magnitude, since RevIN already
   benefits more from trend/ARIMA which is unchanged), or at minimum
   doesn't regress.

## Files

| Item | Path |
|---|---|
| Recipe (with pulse) | `src/synthetic_composite.py:_sample_wave` (gated by `enable_pulse`) |
| Loader | unchanged: `src/dataloader.py:MixedCompositeLoader` (forwards `synth_kwargs`) |
| Train flag | `--enable-pulse` on both `train.py` and `train_forecasting_head.py` |
| Driver | `experiments/exp_compositesynth_v2pulse_2arm/run.sh` (single-arm, takes ARM as $1) |
| Local sync | `sync_compositesynth_v2pulse_revin/`, `sync_compositesynth_v2pulse_ewma128/` (in main checkout) |

## Status

- [x] Code + tests landed.
- [ ] 2 Vast.ai instances provisioned in parallel.
- [ ] Arm A (RevIN) + Arm B (EWMA-128) running concurrently.
- [ ] Results plotted vs `exp_compositesynth_2arm` (phase-1) and `exp_dualemb_3arm` (phase-0 baseline).
