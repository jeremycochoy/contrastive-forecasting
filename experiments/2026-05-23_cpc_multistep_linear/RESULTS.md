# #316 — Does predicting 12 steps ahead beat predicting 1?

**No.** Every 12-step variant transfers worse than β, and none of the seed noise
changes that. The one solid mechanism: 12-step prediction makes the latent
*diffuse* (uses ~10× more dimensions), not richer.

![experiment](plots/experiment.png)

## Question

β's forecaster predicts the **next** latent (k=1). Contrastive Predictive Coding
(van den Oord 2018) predicts **several** steps ahead, on the theory that it packs
more forecastable structure into the latent. Holding β's encoder, negatives, and
training fixed, does predicting **k = 12** ahead improve transfer?

Metric: **GM-Relative MASE** — model MASE ÷ seasonal-naive MASE, geometric mean
over 97 GIFT-Eval configs; **lower is better**. β = 1.327; champion v11c = 1.292.

## Answer: no — every k=12 variant is worse than β

![gm summary](plots/gm_summary.png)

All four k=12 runs land at **1.478 – 2.014**, above β's **1.327**. To *improve* β
a run would have to drop below 1.327; none comes close.

## Is the trend reliable? Direction yes, size no

![k trend](plots/k_trend.png)

k=12 is worse in all three forecaster families. **But** two seeds of one arm land
**0.49 apart** (1.524 vs 2.014) — wider than any k=1→k=12 gap — despite
near-identical pretraining loss, forecast gap, and AUC. So the **direction**
(k=12 worse) is reliable; the **exact penalty** and the small within-family
orderings are within seed noise.

## Why: k=12 spreads the latent out — it does not collapse it

![dim usage](plots/dim_usage.png)

The k=12 encoder latent uses **~50 effective dimensions**; the k=1 latent
collapses to **~3–5** over training — a 10–17× gap, identical in both head types.
This **refutes** the natural guess that multi-step prediction *collapses* the
latent onto a low-rank subspace: the opposite happens.

*Hypothesis (not proven):* a diffuse, high-dimensional latent is harder for the
downstream head to use than β's tight, low-dimensional one — consistent with
k=12's worse GM-MASE.

## Protocol

- **Only the forecast horizon changes** (see schematic): identical 6-layer causal
  encoder, identical InfoNCE negatives (β's pool), identical training (τ=0.10,
  50k steps, batch 256, fp32). At k=1 the loss is byte-identical to β.
- **Forecaster head** tested two ways — transformer-1L (= β's) and linear — under
  two negative sets (β's and CPC-canonical), giving the three families plotted.
- **Downstream:** freeze the backbone, train a quantile head, score GIFT-Eval
  full-97. The headline transformer arm was also run with a 6-layer head (1.421).

## What we learned

1. **No improvement (robust).** Every k=12 run — every family, every seed — is
   worse than β.
2. **Mechanism (robust).** k=12 makes the latent diffuse (~50 dims) vs k=1's tight
   ~3, refuting "multi-step collapses the latent."
3. **Method caution.** GM-MASE seed variance here is ~0.5 — larger than the
   k-effect — so single-seed comparisons at this scale are unreliable; ≥2 seeds
   are needed to quote a magnitude.

## Annex

- **Two-seed check.** CPC-neg k=12 arm: seed A = 1.5240, seed B = 2.0137. Both
  trained the full 50k and converged to near-identical loss (≈0.06) / gap (≈0.76)
  / AUC (≈1.0); seed B's mid-run resume was clean (no loss spike or step
  discontinuity), so the 0.49 spread is genuine seed variance, not a damaged run.
- **References.** full-97 GM-Relative MASE: β = 1.3272, v11c = 1.292, (B) = 1.3572.
- **Code.** Branch `experiment/2026-05-23-cpc-multistep-linear`. `forecaster_kind`
  ∈ {transformer (β), cpc (K transformer-1L heads), linear_cpc (K linear heads)};
  scripts `diagram.py`, `plot_results.py`, `dim_usage.py`, and `test_cpc.py`
  (verifies k=1 loss ≡ β = 7.468332).
