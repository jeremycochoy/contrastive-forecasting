# Session handoff — late April 2026 (dual-axis embeddings landed)

This handoff supersedes the previous one. Three PRs landed in this
session: dual freq+seasonality embedding plumb (#85), worst-config
forecast plots (#86), and joint (freq, seas) coverage in synth + a
formal coverage doc (#87). Branch is back on `experiments`.

## Two best architectures

Both produced by `experiments/exp_dualemb_3arm` (PR #85). 30k backbone +
30k R1 reconstruction-forecaster qhead, `csb` contrastive loss,
`mix_ratio=0.5` (HF base-bundles + on-the-fly periodic synth),
`freq_emb_dim=3 + seasonality_emb_dim=3`, `mixup_p=0.3`. Selector:
`_best_loss → FINAL.pth`.

| arm | norm | GM-MASE | median MASE | wins (of 97) |
|-----|------|--------:|------------:|-------------:|
| **EWMA span=128** | RevEWMNorm span=128 | **1.659** | 1.528 | 55 |
| **EWMA span=512** | RevEWMNorm span=512 | 1.725 | **1.476** | 23 |

Both beat RevIN (GM 1.859, never wins a domain). The two metrics
disagree on which EWMA is best; `span=128` wins GM, `span=512` wins
median. Per-domain breakdown and full 97-config table in
[`experiments/exp_dualemb_3arm/REPORT.md`](experiments/exp_dualemb_3arm/REPORT.md);
worst-case forecast plots with seasonal-naive baseline in
[`experiments/exp_dualemb_3arm/plots/gift_eval_worst_configs.png`](experiments/exp_dualemb_3arm/plots/gift_eval_worst_configs.png)
and `gift_eval_all_failures.png` (all 73 fail-all configs).

## Checkpoints

Stored under the **main checkout** (not in any worktree) at
`sync_dualemb_3arm/checkpoints/`:

```
tiny_dualemb_revin_FINAL.pth      80 MB   backbone (RevIN)
tiny_dualemb_ewma512_FINAL.pth    80 MB   backbone (EWMA span=512)
tiny_dualemb_ewma128_FINAL.pth    80 MB   backbone (EWMA span=128)
R1q_dualemb_revin_FINAL.pth       2.5 MB  qhead (RevIN)
R1q_dualemb_ewma512_FINAL.pth     2.5 MB  qhead (EWMA span=512)
R1q_dualemb_ewma128_FINAL.pth     2.5 MB  qhead (EWMA span=128)
```

Each backbone auto-loads via `ConfigurableModel` because the freq and
seasonality embedding dimensions are detectable from the state-dict
(`freq_embedding.embedding.weight`,
`seasonality_embedding.embedding.weight`). Eval auto-detects in
`experiments/gift-eval/scripts/eval_gift_eval_official.py`.

## Frequency and seasonality embeddings (PR #87)

Two parallel categorical axes. Spec lives in
[`docs/FREQ_SEASONALITY_COVERAGE.md`](docs/FREQ_SEASONALITY_COVERAGE.md):

* **Frequency** (10 buckets): wall-clock sample rate (`unknown`,
  `10s`..`1w`). Out-of-vocab freqs (monthly, yearly, sub-second)
  collapse to `unknown`.
* **Seasonality** (10 buckets): doubling buckets on
  samples-per-period. Bucket **0** is the no-info sentinel and also
  catches `spp = 1` (the gluonts default for daily/weekly), so trivial-
  seasonality eval rows share the embedding with truly-unknown
  training rows.

GIFT-Eval has **14 distinct (freq, seas) pairs** across its 97 configs.
The on-the-fly synth (`src/synthetic_periodic.py:generate_periodic_batch`
with `return_labels=True`) covers all **90 cells** of the 9-freqs ×
10-seasonalities grid. Sampling is independent (freq sampled uniformly
from `{1..9}`, seasonality from `{0..9}`), then spp is drawn from the
seasonality bucket's range. Coverage verified at batch=5000: every cell
appears at least once with marginals within ±10% of uniform.

Notable synth-only pairs that GIFT-Eval doesn't include: weekly-on-
daily `(freq=1d, seas=2)`, weekly-on-hourly `(freq=1h, seas=7)`,
yearly-on-daily `(freq=1d, seas=8)`. These let the model learn the
real-world periodic structures GIFT-Eval doesn't directly test.

## Data generation pipeline

Two sources, blended by `MixedPeriodicLoader` at training time:

1. **Bundle (HuggingFace)** — `jeremycochoy/contrastive-training-base-bundles`,
   path `base_mixed_v1`. Built by the `training_data_prep` pipeline in
   [`jeremycochoy/rnd`](https://github.com/jeremycochoy/rnd) under
   `scripts/training_data_prep/`. Each parquet shard row carries
   `(series, source_id, meta)`:

   | source_id | source             | mix ratio | freq_id | seas_id |
   |----------:|--------------------|----------:|--------:|--------:|
   | 0 | gift (train)       | 0.735 | 0 (unknown — meta dropped) | 0 |
   | 1 | wiki_hourly        | 0.140 | 7 (1h) | 4 (`seas=24`) |
   | 2 | wiki_daily         | 0.075 | 8 (1d) | 2 (`seas=7`) |
   | 3 | wiki_stl_residual  | 0.020 | 7 (1h) | 0 |
   | 4 | wiki_stl_seasonal  | 0.014 | 7 (1h) | 0 |
   | 5 | wiki_stl_trend     | 0.006 | 7 (1h) | 0 |
   | 6 | synthetic (bundle) | 0.010 | 0 | 0 |

   The label lookup is `SOURCE_ID_TO_LABELS` in `src/freq_embedding.py`.
   Bundle synth is tagged unknown because the build pipeline flattens
   the per-row spp metadata; only on-the-fly synth carries true labels.

2. **On-the-fly periodic synth** — `src/synthetic_periodic.py`. Three
   primitives (sin / square / saw) at `spp ∈ bucket-range`, optional
   exponential envelope (p=0.3), log-uniform scale `∈ [0.1, 1000]`,
   random sign flip on square/saw. Each batch row carries `(freq_id,
   seas_id)` sampled jointly per the coverage spec above; channels
   share the row's bucket but draw spp independently.

`MixedPeriodicLoader` concatenates a real-data sub-batch and a synth
sub-batch per training step, with row-level labels passed through.

## What lives where

| topic | file |
|-------|------|
| Architecture: model + dual embedding | `src/models.py`, `src/freq_embedding.py` |
| Embedding spec & coverage | `docs/FREQ_SEASONALITY_COVERAGE.md` |
| Synth | `src/synthetic_periodic.py` |
| Dataloader (HF + synth + label plumb) | `src/dataloader.py` |
| Training (backbone) | `experiments/freq-embedding/scripts/train.py` |
| Training (qhead) | `experiments/gift-eval/scripts/train_forecasting_head.py` |
| Eval (GIFT-Eval official) | `experiments/gift-eval/scripts/eval_gift_eval_official.py` |
| 3-arm experiment driver | `experiments/exp_dualemb_3arm/run.sh` |
| 3-arm REPORT | `experiments/exp_dualemb_3arm/REPORT.md` |
| Per-config plots | `experiments/exp_dualemb_3arm/plots/` |
| Result CSVs (97 configs × 3 arms) | `experiments/exp_dualemb_3arm/results/` |
| Experiment index | `experiments/INDEX.md` |
| FINAL checkpoints (local only) | `sync_dualemb_3arm/checkpoints/` |

## Open follow-ups (not done)

* **Multi-seed validation** of the EWMA-128 vs EWMA-512 4% gap. Single
  seed with cross-seed variance ~3–5% in past runs leaves the ranking
  marginal.
* **Bundle freq plumb** (sub-dataset → freq_id for gift train rows).
  Currently 73.5% of bundle rows are tagged unknown because the
  `training_data_prep` pipeline drops the upstream sub-dataset name.
  Re-emitting `base_mixed_v2` with per-row metadata would lift this.
* **Re-train the 3 arms with the new joint-coverage synth** (PR #87)
  to see whether the better synth distribution closes more of the
  worst-config failures (Econ/Fin trend extrapolation, spike-driven
  Web/CloudOps).
* **Seasonal-naive sidecar CSV** for the eval summary. The
  `summary.txt` shows `N/A` for SN_MASE because the sidecar wasn't
  shipped to the instance. The MASE values themselves are correct
  (gluonts's `evaluate_model` computes them with the right
  seasonality); only the auxiliary skill-score column is missing.
