# Experiments Index

All experiments for the contrastive forecasting project, organized chronologically by when they were started.

## Architecture and Training Foundations

| Experiment | Status | Key Result |
|---|---|---|
| [Contrastive ARMA](contrastive-arma/report/README.md) | Complete | GRU encoder +58% gap over MLP; 12L H=1024 backbone peak gap 0.203 at 2M steps; best recovery 6.96x (GRU h128 l2). ~433 GPU-hours total. |
| [Encoder Comparison](encoder-comparison/REPORT.md) | Complete | GRU encoder +12% gap over patch (residual SiLU) on real-world data at 200k steps. Advantage narrows vs synthetic but persists. |
| [Window Size Comparison](window-size-comparison/report.md) | Complete | W=16 beats W=32: +13% gap, 37% less VRAM. Optimal batch size bs=24 on Tiny. |
| [RevEWMNorm Span Search](revnorm-span-search/report.md) | Complete | span=32 optimal (gap 0.235 vs 0.020 baseline). RevEWMNorm essential for non-stationary data. |
| [RMSNorm Comparison](rmsnorm-comparison/report.md) | Complete | No significant difference between LayerNorm and RMSNorm at Tiny scale. Keep LayerNorm. |

## Training Infrastructure

| Experiment | Status | Key Result |
|---|---|---|
| [Tiny Training](tiny-training/README.md) | Complete | Backbone v2 training on HF data. NaN crash root-caused to all-NaN rows; checkpoint state completeness overhauled (PRs #13-#16). |

## Evaluation and Forecasting

| Experiment | Status | Key Result |
|---|---|---|
| [GIFT-Eval](gift-eval/README.md) | Complete | GM-Relative MASE ~1.26 (below seasonal naive). Flat scaling curve traced to unshuffled dataset. Per-domain analysis identifies energy (32 configs) as highest-leverage gap. |
| [Head / Rollout Comparison](head-rollout-comparison/README.md) | Complete | Value-space rollout (A1=1.275) initially beat latent rollout (B1=1.258). Led to reconstruction head experiment. |
| [Reconstruction Head](reconstruction-head/README.md) | Complete | Reconstruction heads fix latent rollout. R1 (forecaster recon W=16) achieves MASE 1.121, a 12% improvement over value-space baseline. Key insight: head should reconstruct, not predict. |

## Experiment Timeline

```
Feb 2026    Contrastive ARMA early experiments (H=512, DeepGRU)
Mar 18-20   Architecture search (5 phases, 47+ runs)
Mar 21-26   2M backbone training, checkpoint improvements
Mar 27-30   Recovery head search (47+ experiments)
Mar 30-     Scaling search (12L/16L/20L), window/norm comparisons
Apr 1-12    20L full training (2M+ steps)
Apr 13-15   GIFT-Eval evaluation, LR sweep, data ordering diagnosis
Apr 15-17   Head/rollout comparison (6 variants on Vast.ai)
Apr 17-18   Reconstruction head experiment (R1-R4), failure mode analysis
```

## Architecture Summary (Tiny v2)

| Component | Choice | Source |
|---|---|---|
| Encoder | Bidirectional GRU, 2L h=128 | [contrastive-arma](contrastive-arma/report/README.md) |
| Transformer | 6 layers, 8 heads, FFN 4x, Pre-LayerNorm | [contrastive-arma](contrastive-arma/report/README.md) |
| Hidden dim | H=512 | [contrastive-arma](contrastive-arma/report/README.md) |
| Patch size | W=16 | [window-size-comparison](window-size-comparison/report.md) |
| Input norm | RevEWMNorm span=32 | [revnorm-span-search](revnorm-span-search/report.md) |
| Layer norm | LayerNorm (not RMSNorm) | [rmsnorm-comparison](rmsnorm-comparison/report.md) |
| Prediction head | Reconstruction R1 (forecaster, W=16) | [reconstruction-head](reconstruction-head/REPORT.md) |
| Total params | ~20M backbone + ~626K head | |
