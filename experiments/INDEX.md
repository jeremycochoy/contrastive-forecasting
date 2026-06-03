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
| [2026-04-27_periodic-synth-mix](2026-04-27_periodic-synth-mix/periodic-synth-mix.md) | Partial | Whether mixing 50% clean-periodic synthetic data into training fixes the periodic-failure datasets; a modest 3.4% periodic gain offset by a small generalisation tax. |
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

## Real-Data Scaling and the MOIRAI Recipe (early May 2026)

| Experiment | Status | Description |
|---|---|---|
| [2026-05-02_exp_realonly_4096_smaller_tau_sweep](2026-05-02_exp_realonly_4096_smaller_tau_sweep/exp_realonly_4096_smaller_tau_sweep.md) | Inconclusive | Small-data (47-epoch) sweep of contrastive temperature τ (fixed 0.05/0.07/0.20 vs learnable) to pick a default; the memorization regime makes the ranking unreliable. |
| [2026-05-02_exp_realonly_full4096_learnable_tau](2026-05-02_exp_realonly_full4096_learnable_tau/exp_realonly_full4096_learnable_tau.md) | Superseded | Full-4096 30k learnable-τ baseline asking whether broader data beats the small-data overfit at matched steps; it did not (GM-MASE 1.804), implicating step-starvation. |
| [2026-05-02_exp_realonly_full4096_moirai_hp](2026-05-02_exp_realonly_full4096_moirai_hp/exp_realonly_full4096_moirai_hp.md) | Success | Whether MOIRAI-paper optimizer HP (10× lr, 10× weight-decay) helps the full-4096 30k run; it wins on every GM metric (GM-MASE 1.639 vs 1.804). |
| [2026-05-03_exp_realonly_full4096_moirai_hp_FINAL](2026-05-03_exp_realonly_full4096_moirai_hp_FINAL/exp_realonly_full4096_moirai_hp_FINAL.md) | Complete | One full epoch (167k steps, MOIRAI HP) testing the step-starvation hypothesis and deterministic resume; reaches GM-MASE 1.183, still 17% behind seasonal naive. |
| [2026-05-05_exp_qhead_improvements](2026-05-05_exp_qhead_improvements/exp_qhead_improvements.md) | Partial | Whether forecasting-head changes (architecture, depth, schedule, loss, train/eval input match) close the gap to Moirai on a frozen backbone; best head reaches 1.029. |

## Encoder, Temperature and Loss Ablations (May 2026)

| Experiment | Status | Description |
|---|---|---|
| [2026-05-08_exp_init_u_sweep](2026-05-08_exp_init_u_sweep/exp_init_u_sweep.md) | Negative result | Whether init, encoder, patch width, or batch size lifts the encoder latent's batch-axis dimension usage at step 0; the input-concat width caps it, only widening the patch helps. |
| [2026-05-08_exp_tau_sweep](2026-05-08_exp_tau_sweep/exp_tau_sweep.md) | Complete | Fixed-τ sweep from scratch to find the representation optimum and which value matches the learned τ; τ=0.20 is the optimum, learnable τ slides to ~0.07. |
| [2026-05-09_exp_tau_encoder](2026-05-09_exp_tau_encoder/exp_tau_encoder.md) | Complete (single seed) | Whether a residual_silu encoder beats the GRU patch encoder at τ=0.10 on held-out metrics; GRU wins 5 of 6, so GRU stays the default encoder. |
| [2026-05-10_exp_transformer_encoder](2026-05-10_exp_transformer_encoder/exp_transformer_encoder.md) | Complete | Whether a 2-layer within-patch transformer encoder beats the GRU+linear-skip encoder; transformer wins on latent spread but loses on R² and ties on retrieval. |
| [2026-05-09_exp_additional_negatives](2026-05-09_exp_additional_negatives/exp_additional_negatives.md) | Complete (diagram) | Catalogues which negative-pair structures the contrastive loss already covers versus candidates on a batch-time graph, identifying five uncovered pair-types to ablate. |
| [2026-05-09_exp_loss_extensions](2026-05-09_exp_loss_extensions/exp_loss_extensions.md) | Negative result | Whether square cross-batch negatives (same-time f-f and h-h repulsion) beat the baseline loss on AUC/Top-1; baselines edge squares at every step-matched window. |
| [2026-05-09_exp_loss_extensions_failed](2026-05-09_exp_loss_extensions_failed/exp_loss_extensions_failed.md) | Negative result | Tests three loss-shape extensions (an (h_t,f_t) positive, f-side cross negatives, a skip-step f negative); all three rejected, the (h_t,f_t) positive collapsing retrieval. |
| [2026-05-10_exp_encoder_forecaster_failed](2026-05-10_exp_encoder_forecaster_failed/exp_encoder_forecaster_failed.md) | Failed | Whether a 6-layer causal transformer encoder before the forecaster improves GIFT-Eval MASE; it aces legacy contrastive metrics via a positional shortcut but fails MASE. |
| [2026-05-11_exp_encoder_forecaster](2026-05-11_exp_encoder_forecaster/exp_encoder_forecaster.md) | Complete | Full-GIFT-Eval sweep of JEPA-style encoder-forecaster backbones plus an fp16 safety test; v11c wins (GM-MASE 1.292) but no arm beats seasonal naive. |
| [2026-05-17_bottleneck_fullfh_ddp](2026-05-17_bottleneck_fullfh_ddp/bottleneck_fullfh_ddp.md) | Inconclusive | Whether the all-time forecast↔encoder (fₜ↔hₗ) negatives loss improves forecasting; not isolable (bundled with ≥4 other changes), but yields a stable fp16 recipe. |
| [2026-05-18_qhead150k_on_150kbb](2026-05-18_qhead150k_on_150kbb/qhead150k_on_150kbb.md) | Complete (single seed) | Whether an undertrained q-head causes the bottleneck GM-MASE gap, by matching head training (30k→150k) on a fixed body; only a small gain (1.409→1.382). |
| [2026-05-19_crossed_loss_ablation](2026-05-19_crossed_loss_ablation/crossed_loss_ablation.md) | Complete (single seed) | Whether the harmful all-time fₜ↔hₗ negative is special or its hₜ↔hₗ / fₜ↔fₗ siblings behave the same; the siblings differ, pinning fₜ↔hₗ as harmful. |
| [2026-05-19_crossed_loss_xbranch_ablation](2026-05-19_crossed_loss_xbranch_ablation/crossed_loss_xbranch_ablation.md) | Complete | Whether the cross-branch f↔h negative carries signal or the whole family is inert; at 3 seeds full_hh is best and only all-time f↔h is robustly harmful. |
| [2026-05-20_bottleneck_beta2_confound](2026-05-20_bottleneck_beta2_confound/bottleneck_beta2_confound.md) | Negative result | Whether tuning the forecaster bottleneck, AdamW β2, or τ on the (B) recipe closes its ~5% GM-MASE gap to champion v11c; no converged arm reaches v11c. |
| [2026-05-22_align_floor_loss_B](2026-05-22_align_floor_loss_B/align_floor_loss_B.md) | Negative result | Whether adding the BYOL alignment loss L_align to the (B) recipe closes its transfer gap to v11c; it moves the wrong way, hurting GIFT-Eval transfer. |
| [2026-05-23_cpc_multistep_linear](2026-05-23_cpc_multistep_linear/cpc_multistep_linear.md) | Negative result | Whether forecasting 12 steps ahead (CPC k=12) beats 1 step on GIFT-Eval transfer; every k=12 backbone transfers worse than k=1. |
| [2026-05-23_xseries_hh](2026-05-23_xseries_hh/xseries_hh.md) | Partial | Whether denying the positional shortcut via cross-series repulsion or forked continuations improves transfer; only a β-loss fork at ≈10% reproducibly beats β. |
| [2026-05-26_forked_6Lf](2026-05-26_forked_6Lf/forked_6Lf.md) | Inconclusive | Whether a 6-layer forecaster changes where forked-data shortcut denial helps transfer; mixed — helps the shortcut-anchor arm but hurts the prior GIFT-Eval winner. |
| [2026-05-29_forked_6Lf_b1024](2026-05-29_forked_6Lf_b1024/forked_6Lf_b1024.md) | Success | Whether quadrupling the contrastive batch to 1024 lets the forked arms beat the baseline; every arm improves and allt·10% leads at GM-MASE 1.19. |

## Generative-Parameter Recovery (companions to Contrastive ARMA)

| Experiment | Status | Description |
|---|---|---|
| [2026-05-04_contrastive-correlation](2026-05-04_contrastive-correlation/contrastive-correlation.md) | Success | Whether a head can recover the 4×4 generative correlation matrix from a backbone trained unsupervised on correlated random walks; TimeAware head r=0.918 (V5 joint-channel: 0.962). |
| [2026-05-18_contrastive-arma-correlation](2026-05-18_contrastive-arma-correlation/contrastive-arma-correlation.md) | Success | Whether one contrastive backbone supports recovery of both ARMA coefficients and 4×4 correlation matrices; ARMA always recovers, correlation only with attention channel-mixing (r≈0.74). |

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
Apr 21-28   Freq/seasonality embedding sequence (norm choice, patch-stats, CSB loss, dual-emb)
May 1-5     Real-data 4096 scaling, MOIRAI-HP optimizer recipe, q-head improvements
May 4-18    Correlation-recovery companions (contrastive-correlation V4→V5, ARMA+correlation)
May 8-23    Temperature, encoder and contrastive-loss / negatives ablations
May 23-29   Forked-continuation shortcut denial; contrastive batch scaled to 1024
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
