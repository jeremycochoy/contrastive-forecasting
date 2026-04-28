# exp_compositesynth_2arm — composite synth A/B vs periodic baseline

## Question

The dual-axis-embedding training in `exp_dualemb_3arm` mixes 50% bundle
HF + 50% on-the-fly *clean-periodic* synth (`src/synthetic_periodic.py`).
Bundle synth (TimesFM composite: ARMA + trend + sinusoids) is only ~1%
of the bundle, so the model rarely sees ARMA / piecewise trend / ARIMA
random walks — exactly the structure GIFT-Eval's worst configs (Econ/Fin
trend extrapolation, covid explosive trend) require.

Does swapping the on-the-fly synth for a TimesFM-style **composite**
recipe (`src/synthetic_composite.py` — trend + ARMA + 2 free-spp waves +
1 seasonality-tied wave, all coinflipped, with per-row freq+seas labels)
move GIFT-Eval numbers?

## Setup

| Knob | Value |
|---|---|
| Backbone | Tiny (H=512, L=6, GRU encoder, W=16, 20M params) |
| Loss | `cosine_similarity_batch` (won the pair A/B in `exp_csb_pair_*`) |
| Mix ratio | **0.5** (50% bundle `base_mixed_v1` + 50% on-the-fly **composite** synth) |
| `freq_emb_dim` / `seasonality_emb_dim` | 3 / 3 |
| `mixup_p` | 0.3 |
| Backbone steps | 30 000 |
| Quantile head steps | 30 000 (R1 forecaster reconstruction, 9 quantiles, forecast_len=16) |
| Selector | `_best_loss` → `FINAL.pth` |
| Eval | GIFT-Eval official, 97 configs, B4 strategy |
| Seed | 42 (single-seed) |

Two arms:
* **Arm A — RevIN** (per-instance z-score)
* **Arm B — RevEWMNorm span=128** (best on `exp_dualemb_3arm`'s GM-MASE)

## Composite-synth recipe (defaults)

Per-row state:
* `seas_id ~ U{0..9}`, `freq_id ~ U{1..9}` (independent).
* Bernoulli(0.5) per row whether the seas-tied wave is on; emitted
  `seas_id = drawn` if on, else `0`.

Per-channel state (independent per (b,c)):
* Trend: piecewise linear, always on, `slope_std=0.003`.
* ARMA(p,q): Bernoulli(0.5). Sampled via the polynomial-root method
  reused verbatim from `src/synthetic.py:sample_arma_from_roots` (the
  same function `../rnd/scripts/training_data_prep` calls into when it
  builds the bundle). When on, integrate to ARIMA(1,p,q) with
  Bernoulli(0.5). Post-(optional)integration the component is rescaled
  to a target std drawn log-uniform from `[0.5, 3]` so it mixes against
  bounded waves without dominating an order of magnitude.
* Free wave 1: Bernoulli(0.5). Primitive ∈ {sin, square, saw} uniform;
  spp log-uniform in `[4, T/2]`; phase uniform; sign flip on square/saw
  with prob 0.5.
* Free wave 2: independent of free wave 1, same distribution.
* Seas-tied wave: presence is row-level (above). When on, every channel
  gets one with primitive uniform from {sin, square, saw} and spp drawn
  log-uniformly from the row's `SEASONALITY_BUCKET_SPP_RANGES[seas]`.
* "≥1 non-trend on" rule: if ARMA, free1, free2 all coinflip off and
  the row's seas-tied is off, force-on uniformly from {ARMA, free1, free2}.

Combination:
* Non-trend components summed with U[0,1] weights (no per-component
  pre-mix unit-std normalisation; ranges chosen to keep components in
  similar amplitude band).
* Trend application: 50% multiplicative `(non_trend * trend)` else
  additive with U[0,1] weight.
* Final exponential envelope applied per-(b,c) with prob 0.3, total
  gain log-uniform `[0.1, 10]`.
* Final scale per-(b,c) log-uniform `[0.1, 1000]`.

## Files

| Item | Path |
|---|---|
| Recipe | `src/synthetic_composite.py` |
| Loader | `src/dataloader.py:MixedCompositeLoader` |
| Driver | `experiments/exp_compositesynth_2arm/run.sh` |
| Plot | `experiments/exp_compositesynth_2arm/scripts/plot_compare_2arm.py` |
| Local sync | `sync_compositesynth/` (in main checkout, not this worktree) |

## Status

- [x] Code + tests landed in PR #89.
- [ ] Phase-1 A/B running on Vast.ai.
- [ ] Phase-1 results plotted vs `exp_dualemb_3arm` baseline.
- [ ] Phase-2 HP iteration (conditional).
