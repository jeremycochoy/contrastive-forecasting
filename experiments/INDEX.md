# Experiments Index

All experiments for the contrastive forecasting project, organized chronologically by when they were started.

## Architecture and Training Foundations

| Experiment | Status | Description |
|---|---|---|
| [Contrastive ARMA](2026-04-12_contrastive-arma/contrastive-arma.md) | Complete | Original architecture search (encoder type, depth, width) for the ARMA-recovery contrastive task — the foundation behind the project's current Tiny backbone. |
| [v3b Continuation](2026-04-21_v3b-continuation/v3b-continuation.md) | Complete | Long-running continuation training of the v3b backbone across multiple Vast.ai instances to push it further past the architecture-search horizon. |
| [Encoder Comparison](2026-04-19_encoder-comparison/encoder-comparison.md) | Complete | Head-to-head test of the GRU encoder vs a flat residual-MLP patch encoder on real-world data, asking whether the GRU's edge survives at scale. |
| [Window Size Comparison](2026-04-12_window-size-comparison/window-size-comparison.md) | Complete | Comparison of patch widths to pick the right tradeoff between temporal resolution and attention cost. |
| [RevEWMNorm Span Search](2026-04-12_revnorm-span-search/revnorm-span-search.md) | Complete | Sweep of the EWMA span used by the reversible normaliser to find how fast it should adapt on non-stationary data. |
| [RMSNorm Comparison](2026-04-12_rmsnorm-comparison/rmsnorm-comparison.md) | Complete | Ablation testing whether replacing pre-LayerNorm with RMSNorm changes contrastive gap or training speed. |

## Training Infrastructure

| Experiment | Status | Description |
|---|---|---|
| [Tiny Training](2026-04-12_tiny-training/tiny-training.md) | Complete | First long backbone training on HuggingFace streaming data — the run that surfaced and hardened the project's checkpoint and NaN-handling infrastructure. |

## Evaluation and Forecasting

| Experiment | Status | Description |
|---|---|---|
| [GIFT-Eval](2026-04-13_gift-eval/gift-eval.md) | Complete | Setting up the GIFT-Eval benchmark harness and using it to diagnose where the Tiny backbone underperforms across domains. |
| [Head / Rollout Comparison](2026-04-16_head-rollout-comparison/head-rollout-comparison.md) | Complete | Comparing value-space vs latent-space rollout strategies to test whether the prediction head, not the backbone, was capping downstream MASE. |
| [Reconstruction Head](2026-04-17_reconstruction-head/reconstruction-head.md) | Complete | Testing the hypothesis that the head should reconstruct what each latent represents, instead of predicting the future, to fix latent rollout. |

## Freq-Embedding Sequence (Apr 2026)

Aggregate report and cross-cutting artefacts: [`2026-04-27__aggregate/aggregate.md`](2026-04-27__aggregate/aggregate.md).
Shared scripts and design: [`2026-04-27_freq-embedding/freq-embedding.md`](2026-04-27_freq-embedding/freq-embedding.md).

| Experiment | Status | Description |
|---|---|---|
| [2026-04-27_exp_revin_repro](2026-04-27_exp_revin_repro/exp_revin_repro.md) | Success (reproduction) | Reproduction of a previous-session RevIN ablation to confirm the new shared trainer matches the earlier numbers before iterating further. |
| [2026-04-27_exp_patch_stats_mix05](2026-04-27_exp_patch_stats_mix05/exp_patch_stats_mix05.md) | Superseded | First attempt at adding per-patch summary statistics to the encoder input to see whether it improves contrastive and downstream quality. |
| [2026-04-27_exp_synth_only_redo](2026-04-27_exp_synth_only_redo/exp_synth_only_redo.md) | Success | Synth-only redo of the patch-stats arms to isolate architecture effects from out-of-distribution transfer and iterate faster. |
| [2026-04-27_exp_span_sweep_real](2026-04-27_exp_span_sweep_real/exp_span_sweep_real.md) | Partial | EWMA span sweep on pure real data, asking how span affects contrastive signal away from the synthetic regime. |
| [2026-04-27_exp_span_sweep_synth](2026-04-27_exp_span_sweep_synth/exp_span_sweep_synth.md) | Success | EWMA span sweep on synth-only data to find the in-distribution optimum and check whether the prior default was leaving signal on the table. |
| [2026-04-27_exp_revin_synth](2026-04-27_exp_revin_synth/exp_revin_synth.md) | Complete | RevIN-vs-EWMA comparison on synth-only data to isolate the normaliser choice from out-of-distribution transfer effects. |
| [2026-04-27_exp_csb_synth](2026-04-27_exp_csb_synth/exp_csb_synth.md) | Complete (single seed) | First test of the paper-matching contrastive loss (with within-time and cross-time negatives) on the best synth arm. |
| [2026-04-28_exp_csb_pair_span512](2026-04-28_exp_csb_pair_span512/exp_csb_pair_span512.md) | Complete (single seed each) | Clean A/B retrain of the two contrastive losses on the EWMA best arm, to remove the multi-resume confound from the earlier CSB run. |
| [2026-04-28_exp_csb_pair_revin](2026-04-28_exp_csb_pair_revin/exp_csb_pair_revin.md) | Complete (single seed each) | RevIN counterpart of the contrastive-loss A/B, asking whether the loss-flag direction depends on the choice of normaliser. |
| [2026-04-28_exp_dualemb_3arm](2026-04-28_exp_dualemb_3arm/exp_dualemb_3arm.md) | Complete (single seed each) | First downstream GIFT-Eval test of the new dual-axis (frequency + seasonality) label embedding, comparing all three normaliser variants on real data. |

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
| Encoder | Bidirectional GRU, 2L h=128 | [contrastive-arma](2026-04-12_contrastive-arma/contrastive-arma.md) |
| Transformer | 6 layers, 8 heads, FFN 4x, Pre-LayerNorm | [contrastive-arma](2026-04-12_contrastive-arma/contrastive-arma.md) |
| Hidden dim | H=512 | [contrastive-arma](2026-04-12_contrastive-arma/contrastive-arma.md) |
| Patch size | W=16 | [2026-04-12_window-size-comparison](2026-04-12_window-size-comparison/window-size-comparison.md) |
| Input norm | RevEWMNorm span=32 | [2026-04-12_revnorm-span-search](2026-04-12_revnorm-span-search/revnorm-span-search.md) |
| Layer norm | LayerNorm (not RMSNorm) | [2026-04-12_rmsnorm-comparison](2026-04-12_rmsnorm-comparison/rmsnorm-comparison.md) |
| Prediction head | Reconstruction R1 (forecaster, W=16) | [2026-04-17_reconstruction-head](2026-04-17_reconstruction-head/reconstruction-head.md) |
| Total params | ~20M backbone + ~626K head | |
