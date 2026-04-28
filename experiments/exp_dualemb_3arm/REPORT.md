# Dualemb 3-arm norm comparison on GIFT-Eval

## Question

Two prior results disagreed:

* `exp_csb_pair_span512` (synth held-out): RevEWMNorm **span=512** wins.
* `exp_span_sweep_real` (real-data contrastive ema_loss): RevEWMNorm **span=128** wins.

Does the disagreement carry into downstream forecasting? Run all three
norm variants (RevIN, EWMA span=512, EWMA span=128) through the same
training pipeline with the new dual-axis label embedding (frequency +
seasonality), then benchmark on full GIFT-Eval (97 configs).

## Setup

| Knob | Value |
|---|---|
| Backbone | Tiny (H=512, L=6, GRU encoder, W=16, 20M params) |
| Loss | `cosine_similarity_batch` (won the pair A/B) |
| Mix ratio | **0.5** (50% bundle `base_mixed_v1` + 50% on-the-fly periodic synth) |
| `freq_emb_dim` / `seasonality_emb_dim` | 3 / 3 (concatenated to every patch) |
| `mixup_p` | 0.3 (mixed across both label embeddings) |
| Backbone steps | 30 000 |
| Quantile head steps | 30 000 (R1 forecaster reconstruction, W=16, 9 quantiles) |
| Selector | `_best_loss` → `FINAL.pth` |
| Eval | GIFT-Eval official, 97 configs, strategy B4 (latent rollout, W-value head) |
| Single seed | 42 (no multi-seed validation) |

Per-task labels at eval time: `freq_id = gluonts_freq_to_id(dataset.freq)`,
`seasonality_id = seasonality_to_id(get_seasonality(dataset.freq))`.

## Headline

| Arm | GM-MASE ↓ | median MASE | max MASE | beats SN (<1.0) | configs <1.5 |
|---|---:|---:|---:|---:|---:|
| RevIN          | **1.859** | 1.568 | 190.4 | 20 / 97 | 43 / 97 |
| EWMA span=512  | **1.725** | **1.476** | 80.3 | 20 / 97 | **51 / 97** |
| **EWMA span=128**  | **1.659** | 1.528 | **70.8** | 18 / 97 | 47 / 97 |

`gift_eval_3arm_compare.png` (this directory): aggregate bar, MASE CDF
across configs, per-domain bars, head-to-head scatter.

## Findings

1. **EWMA span=128 wins on GM-MASE.** It is 4 % better than span=512 and
   11 % better than RevIN. The geometric mean is dominated by the upper
   tail, so a smaller worst-case (max MASE 70.8 vs 80.3 vs 190.4) lifts
   span=128 into first place.

2. **EWMA span=512 wins on median and on the `<1.5` count.** It is the
   most consistent across configs, with 51 / 97 below 1.5 vs 47 / 97 for
   span=128. The two metrics disagree exactly as in the original
   `exp_span_sweep_real` sweep, but here both winners are EWMA, not RevIN.

3. **RevIN is dominated on every metric.** Worst GM-MASE, worst median,
   and a 2.5 × heavier upper tail (190 vs ~75 for EWMA). RevIN's
   per-instance z-score throws away the slow-moving local mean that EWMA
   span=128 / 512 tracks, so non-stationary forecasts blow up at horizon
   end.

4. **Head-to-head wins**:
   - EWMA-128 beats EWMA-512 on **59 / 97** configs (61 %).
   - EWMA-128 beats RevIN on **67 / 97** configs (69 %).
   - EWMA-512 beats RevIN on **53 / 97** configs (55 %).

5. **Per-domain** GM-MASE (`gift_eval_3arm_compare.png`, bottom-left):
   EWMA-128 wins 5 / 7 domains (Econ/Fin, Healthcare, Nature, Transport,
   Web/CloudOps), EWMA-512 wins 2 / 7 (Energy, Sales). EWMA-128's
   biggest advantage is **Econ/Fin** (3.26 vs 4.93 for span=512, a 34 %
   gap on 6 configs) and **Web/CloudOps** (2.45 vs 2.62, a 7 % gap on
   20 configs). EWMA-512's two wins are close (Energy by 1.3 %, Sales
   by 4 %). RevIN is dominated in every domain.

## Settles the span paradox

`exp_span_sweep_real` showed loss preferred span=128 but gap preferred
span=32. Neither was checked against downstream. **Downstream now agrees
with loss**: span=128 produces the lowest GM-MASE on GIFT-Eval. Gap on
real data is therefore a misleading selector: it favoured short spans
that left more periodic structure in the patch values for the
contrastive objective to discriminate, but didn't actually help the
forecasting head.

## Caveats

* **Single seed.** The ~4 % gap between EWMA-128 and EWMA-512 is in the
  same ballpark as cross-seed variance observed in the recovery-head
  search (Mar 2026). A second seed would either confirm the ordering
  or move EWMA-512 ahead.

* **No SN reference in the eval CSV.** The Salesforce GIFT-Eval bundle
  used at inference time did not ship a Seasonal Naive baseline CSV
  (the `summary.txt` shows `N/A` for SN_MASE). Skill scores below are
  raw MASE only. A separate run with the SN reference re-attached can
  produce skill-percent figures, but the relative ordering across arms
  is unaffected.

* **`gift` rows in training stay `freq_id=0` / `seasonality_id=0`.**
  The bundle build pipeline did not preserve the upstream sub-dataset
  name, so 73.5 % of bundle rows tagged unknown. Wiki rows (25.5 %) and
  on-the-fly synth (50 % of every batch via mix_ratio=0.5) carry real
  labels, so coverage of non-zero buckets is ~62 % of every batch.
  Re-emitting the bundle with sub-dataset metadata would lift this.

* **Original run.sh forgot `GIFT_EVAL` env var.** The first eval pass
  produced empty CSVs because gift_eval's data loader needs a path env
  var; we re-ran the eval stage with `GIFT_EVAL=/workspace/gift-eval-data`
  (in `reeval_dualemb.sh`) and the numbers above come from that re-eval.
  Backbones and heads were trained correctly the first time and reused
  unchanged for the re-eval.

## Open questions

1. **Multi-seed**: re-run the EWMA-128 vs EWMA-512 pair with seeds
   {7, 13, 99} to confirm the 4 % GM-MASE gap is real, not noise.
2. **Bundle freq plumb**: rewrite `base_mixed_v2` with per-row
   `(freq_id, seasonality_id)` columns derived from the gift sub-dataset
   path, then re-run EWMA-128. Coverage of non-zero buckets jumps from
   ~62 % → 100 %.
3. **EWMA span=64 / span=256**: the U-curve from `exp_span_sweep_real`
   suggested an optimum near 128. Filling in the U with full downstream
   eval (1-2 more arms) would localise the minimum.

## Artefacts

* `run.sh`, `reeval_dualemb.sh`: drivers
* `results/gift_eval_{revin,ewma512,ewma128}/all_results.csv`: 97 configs each
* `plots/gift_eval_3arm_compare.png`: aggregate + CDF + per-domain + head-to-head
* `scripts/plot_compare_3arm.py`: plot generator, idempotent, re-runs
  from CSVs

* Backbone / head FINAL checkpoints in `sync_dualemb_3arm/checkpoints/`
  (not tracked in git: 80 MB backbones, 2.4 MB heads)

## Cost

5090 instance for ~10 h ($0.47/hr × 10h ≈ $4.70). Main run ~5 h, eval +
re-eval ~5 h (the slow `electricity/H/medium` and `electricity/15T/medium`
configs each take ~10 min on the B4 strategy at horizon ~720).
