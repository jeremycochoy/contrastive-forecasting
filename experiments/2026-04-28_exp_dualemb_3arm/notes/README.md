# Three-arm norm comparison with dual freq+seasonality embeddings

> **Results landed in [`REPORT.md`](../exp_dualemb_3arm.md).** This README is the
> pre-run protocol; REPORT.md has the headline numbers and analysis.

## Why

Two prior results to combine:
* `2026-04-28_exp_csb_pair_span512` showed RevEWMNorm span=512 wins on synth held-out.
* `2026-04-27_exp_span_sweep_real` showed RevEWMNorm span=128 wins on real-data
  contrastive ema_loss. The two metrics disagree on which span is best.

This experiment trains all three norm variants on the SAME data mix with
the new dual-axis label embedding (frequency + seasonality), then
benchmarks them on GIFT-Eval to break the tie on real downstream
performance.

## Arms

| Arm | Norm | Span | Reason |
|-----|------|-----:|--------|
| A | RevIN | n/a | per-instance z-score, stationary by design |
| B | RevEWMNorm | 512 | best on synth held-out |
| C | RevEWMNorm | 128 | best ema_loss on real-data sweep |

Shared knobs:
* `cosine_similarity_batch` loss (won the pair A/B by ~5–13% MASE)
* `mix_ratio=0.5` (50% bundle + 50% on-the-fly periodic synth, so all
  seasonality buckets get training signal — wiki rows alone cover only
  buckets 2 and 4)
* `freq_emb_dim=3` and `seasonality_emb_dim=3` — both label embeddings
  are concatenated to every patch
* mixup_p=0.3 across both label embeddings
* 30k backbone + 30k qhead (R1 reconstruction-forecaster, W=16, 9 quantiles)
* Selector: `_best_loss → FINAL.pth` (gap saturates early)

## Eval

Full GIFT-Eval (97 configs, official evaluator). The backbone reads
`dataset.freq` and `get_seasonality(dataset.freq)` per task and tags
itself with the matching `(freq_id, seasonality_id)` so the embeddings
carry meaningful per-task labels at eval time. No FFT required.

## Source → label table at training time

| source_id | source             | freq_id      | seasonality_id    |
|-----------|--------------------|--------------|-------------------|
| 0         | gift (train)       | 0 (unknown)  | 0 (unknown)       |
| 1         | wiki_hourly        | 7 (1h)       | 4 (≤32, for 24)   |
| 2         | wiki_daily         | 8 (1d)       | 2 (≤8, for 7)     |
| 3-5       | wiki_stl_*         | 7 (1h)       | 0 (unknown)       |
| 6         | synthetic (bundle) | 0            | 0                 |
| —         | on-the-fly synth   | uniform 1..9 | from spp          |

Coverage with `mix_ratio=0.5`: ~62% of rows get a non-zero seasonality
tag (50% from synth + 25.5% × 50% from wiki).

## Files

* `run.sh` — driver, runs the 3 arms sequentially
* `REPORT.md` — written after the experiment completes
* `results/gift_eval_{revin,ewma512,ewma128}/all_results.csv` — per-task
  MASE, WQL, skill scores
* `plots/` — comparison plots

## Cost estimate

3 × (30k bb ≈ 1h on 5090, 30k qh ≈ 30min, gift_eval ≈ 10min) plus ~1h
setup ≈ 6h on a 5090 ≈ $4 at ~$0.7/hr.
