# Experiments Index

All experiments for the contrastive forecasting project, organized chronologically by when they were started.

Cross-cutting empirical learnings, hyperparameter defaults, and project
conventions live in [`LEARNED.md`](LEARNED.md).

## Architecture and Training Foundations

| Experiment | Status | Key Result |
|---|---|---|
| [Contrastive ARMA](contrastive-arma/report/README.md) | Complete | GRU encoder +58% gap over MLP; 12L H=1024 backbone peak gap 0.203 at 2M steps; best recovery 6.96x (GRU h128 l2). ~433 GPU-hours total. |
| [Encoder Comparison](2026-04-19_encoder-comparison/REPORT.md) | Complete | GRU encoder +12% gap over patch (residual SiLU) on real-world data at 200k steps. Advantage narrows vs synthetic but persists. |
| [Window Size Comparison](2026-04-12_window-size-comparison/report.md) | Complete | W=16 beats W=32: +13% gap, 37% less VRAM. Optimal batch size bs=24 on Tiny. |
| [RevEWMNorm Span Search](2026-04-12_revnorm-span-search/report.md) | Complete | span=32 optimal (gap 0.235 vs 0.020 baseline). RevEWMNorm essential for non-stationary data. |
| [RMSNorm Comparison](2026-04-12_rmsnorm-comparison/report.md) | Complete | No significant difference between LayerNorm and RMSNorm at Tiny scale. Keep LayerNorm. |

## Training Infrastructure

| Experiment | Status | Key Result |
|---|---|---|
| [Tiny Training](tiny-training/README.md) | Complete | Backbone v2 training on HF data. NaN crash root-caused to all-NaN rows; checkpoint state completeness overhauled (PRs #13-#16). |

## Evaluation and Forecasting

| Experiment | Status | Key Result |
|---|---|---|
| [GIFT-Eval](gift-eval/README.md) | Complete | GM-Relative MASE ~1.26 (below seasonal naive). Flat scaling curve traced to unshuffled dataset. Per-domain analysis identifies energy (32 configs) as highest-leverage gap. |
| [Head / Rollout Comparison](head-rollout-comparison/README.md) | Complete | Value-space rollout (A1=1.275) initially beat latent rollout (B1=1.258). Led to reconstruction head experiment. |
| [Reconstruction Head](2026-04-17_reconstruction-head/README.md) | Complete | Reconstruction heads fix latent rollout. R1 (forecaster recon W=16) achieves MASE 1.121, a 12% improvement over value-space baseline. Key insight: head should reconstruct, not predict. |

## Freq-Embedding Sequence (Apr 2026)

Aggregate report and cross-cutting artefacts: [`2026-04-27__aggregate/REPORT.md`](2026-04-27__aggregate/REPORT.md).
Shared scripts and design: [`freq-embedding/README.md`](freq-embedding/README.md).

| Experiment | Status | Key Result |
|---|---|---|
| [2026-04-27_exp_revin_repro](2026-04-27_exp_revin_repro/README.md) | Success (reproduction) | RevIN backbone + qhead on mix=0.5; gap=0.469, qh loss=0.052 — matches previous-session #28 within noise. |
| [2026-04-27_exp_patch_stats_mix05](2026-04-27_exp_patch_stats_mix05/README.md) | Superseded | Backbone gap +33% but downstream 1-3% worse than fe+mu+qh / RevIN+qh on the 23-config SN slice. |
| [2026-04-27_exp_synth_only_redo](2026-04-27_exp_synth_only_redo/README.md) | Success | fe+mu @ 60k marginal best of 4 arms on synth-only (GM-MASE 2.366); patch-stats 1-3% worse at both step counts. |
| [2026-04-27_exp_span_sweep_real](2026-04-27_exp_span_sweep_real/README.md) | Partial | 20k steps, mix=0.0; loss U-shaped at span=128, gap monotonically decreasing — metrics disagree, open question. |
| [2026-04-27_exp_span_sweep_synth](2026-04-27_exp_span_sweep_synth/README.md) | Success | Inverted-U with peak at span=512 (GM-MASE 0.848 — 2.8× over previous span=32 default). |
| [2026-04-27_exp_revin_synth](2026-04-27_exp_revin_synth/README.md) | Complete | Best of original 4 synth arms (GM-MASE 2.230) but dominated by EWMA span=64+ once the right span was found. |
| [2026-04-27_exp_csb_synth](2026-04-27_exp_csb_synth/README.md) | Complete (single seed) | cosine_similarity_batch loss on span=512 best arm — GM-MASE 0.886, ~4.5% worse than the no_time_neg baseline (0.848). Multi-resume run; needs second seed to confirm. |
| [2026-04-28_exp_csb_pair_span512](2026-04-28_exp_csb_pair_span512/README.md) | Complete (single seed each) | Clean A/B retrain with matched `_best_loss` selector. CSB **0.883** vs no_time_neg **0.924**: CSB is **4.5% better** on MASE, 3.9% better on WQL, flipping the original conclusion (selector + multi-resume confound). |
| [2026-04-28_exp_csb_pair_revin](2026-04-28_exp_csb_pair_revin/README.md) | Complete (single seed each) | RevIN counterpart of the loss A/B. CSB **0.936** vs no_time_neg **1.072**: CSB is **12.7% better** on MASE, 14.7% better on WQL. Same direction as EWMA pair, ~3x larger effect. EWMA span=512 still beats RevIN under both losses. 4-arm grid in `plots/synth_compare_grid_4arm.png`. |
| [2026-04-28_exp_dualemb_3arm](2026-04-28_exp_dualemb_3arm/REPORT.md) | Complete (single seed each) | First downstream GIFT-Eval test of the new dual-axis label embedding (freq + seasonality). 3 norms × 97 configs. **EWMA span=128** wins GM-MASE 1.659 vs span=512 1.725 vs RevIN 1.859. Settles the span paradox: real-data downstream agrees with loss-based span=128, not gap-based span=32. |

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
| Patch size | W=16 | [2026-04-12_window-size-comparison](2026-04-12_window-size-comparison/report.md) |
| Input norm | RevEWMNorm span=32 | [2026-04-12_revnorm-span-search](2026-04-12_revnorm-span-search/report.md) |
| Layer norm | LayerNorm (not RMSNorm) | [2026-04-12_rmsnorm-comparison](2026-04-12_rmsnorm-comparison/report.md) |
| Prediction head | Reconstruction R1 (forecaster, W=16) | [2026-04-17_reconstruction-head](2026-04-17_reconstruction-head/REPORT.md) |
| Total params | ~20M backbone + ~626K head | |
