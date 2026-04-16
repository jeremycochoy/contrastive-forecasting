# GIFT-Eval Per-Domain Comparison: Small Models

**Date:** 2026-04-13
**Our model:** Tiny contrastive forecaster, ~20M backbone + 626K recovery head = ~21M params total.
**Eval pair:** v2 backbone, 30k-step recovery head.

## 1. Models Compared

All models below are evaluated zero-shot on the full 97-config GIFT-Eval suite (23 datasets, 7 domains).
Parameter counts are total (not activated) unless noted.

| Model | Params | Type | Source |
|---|---|---|---|
| **Ours (Contrastive Tiny)** | ~21M | Contrastive encoder + GRU head | This project |
| Moirai-2-Small | 11.4M | Decoder-only, causal attention | Salesforce, arXiv:2511.11698 |
| FlowState-9.1M | 9.1M | Flow-matching | Granite family |
| Reverso-Small | ~15M (est.) | Unknown | GIFT-Eval leaderboard |
| Kairos-23M | 23M | Unknown | GIFT-Eval leaderboard |
| Chronos-Bolt-Small | ~28M (est.) | Distilled T5, single-pass | Amazon, arXiv:2510.15821 |
| Chronos-Small | 46M | T5 encoder-decoder, AR | Amazon, arXiv:2403.07815 |
| MOIRAI-Small | 14M | Masked encoder, any-variate | Salesforce, arXiv:2402.02592 |
| TTM-R2 (pretrained) | 1-5M | TSMixer (MLP-Mixer) | IBM, arXiv:2401.03955 |
| Seasonal Naive | 0 | Repeat last seasonal cycle | Baseline |

For reference, Chronos-2 (120M) is included as the current GIFT-Eval SOTA among
foundation models, but it is 6x our parameter count and not a "small" model.

## 2. Per-Domain Results (GM-Relative MASE)

GM-relative MASE = geometric mean of per-config MASE / seasonal-naive MASE.
This is the standard GIFT-Eval aggregation convention (Sundial Table 2, Moirai-2 Table 1).
**Lower is better. Values below 1.0 beat seasonal naive.**

| Model | Sales (4) | Transport (15) | Nature (15) | Energy (32) | Web/CO (20) | Econ/Fin (6) | Health (5) | **Overall** |
|---|---|---|---|---|---|---|---|---|
| Chronos-2 (120M) | 0.679 | 0.603 | 0.723 | 0.816 | 0.611 | 0.772 | 0.551 | **0.698** |
| FlowState (9.1M) | 0.694 | 0.620 | 0.733 | 0.827 | 0.684 | 0.760 | 0.617 | **0.726** |
| Moirai-2-Small (11.4M) | 0.689 | 0.620 | 0.755 | 0.837 | 0.665 | 0.779 | 0.600 | **0.728** |
| Reverso-Small | 0.700 | 0.610 | 0.745 | 0.844 | 0.636 | 0.816 | 0.660 | **0.726** |
| Kairos-23M | 0.722 | 0.636 | 0.741 | 0.841 | 0.701 | 0.837 | 0.685 | **0.748** |
| Chronos-Bolt-Small | 0.696 | 0.692 | 0.704 | 0.864 | 1.058 | 0.816 | 0.671 | **0.822** |
| Chronos-Small (46M) | 0.733 | 0.737 | 0.852 | 0.948 | 1.144 | 0.797 | 0.607 | **0.892** |
| MOIRAI-Small (14M) | 0.731 | 0.731 | 0.807 | 1.069 | 1.136 | 0.985 | 0.848 | **0.946** |
| TTM-R2 (1-5M) | 0.977 | 0.792 | 0.851 | 1.016 | 1.254 | 1.409 | 1.176 | **1.020** |
| **Ours (21M)** | **0.831** | **1.056** | **0.948** | **1.550** | **1.272** | **1.786** | **1.117** | **1.256** |
| Seasonal Naive | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |

## 3. Raw GM-MASE (Absolute, Not Relative to Seasonal Naive)

For readers who prefer raw MASE values. These are the geometric mean of per-config
MASE within each domain. Lower is better; 1.0 = match seasonal naive in-sample.

| Model | Sales | Transport | Nature | Energy | Web/CO | Econ/Fin | Health | Overall |
|---|---|---|---|---|---|---|---|---|
| Chronos-2 (120M) | 0.739 | 0.698 | 0.900 | 0.996 | 1.072 | 1.625 | 1.373 | 0.975 |
| Moirai-2-Small (11.4M) | 0.751 | 0.718 | 0.939 | 1.022 | 1.167 | 1.640 | 1.495 | 1.018 |
| FlowState (9.1M) | 0.755 | 0.718 | 0.912 | 1.010 | 1.201 | 1.599 | 1.536 | 1.015 |
| Reverso-Small | 0.762 | 0.707 | 0.927 | 1.031 | 1.116 | 1.718 | 1.643 | 1.015 |
| Kairos-23M | 0.786 | 0.737 | 0.923 | 1.027 | 1.231 | 1.761 | 1.707 | 1.046 |
| Chronos-Bolt-Small | 0.758 | 0.801 | 0.876 | 1.055 | 1.857 | 1.717 | 1.670 | 1.149 |
| Chronos-Small (46M) | 0.798 | 0.853 | 1.060 | 1.157 | 2.008 | 1.677 | 1.511 | 1.246 |
| MOIRAI-Small (14M) | 0.796 | 0.847 | 1.004 | 1.306 | 1.995 | 2.074 | 2.112 | 1.323 |
| TTM-R2 (1-5M) | 1.064 | 0.917 | 1.059 | 1.240 | 2.201 | 2.966 | 2.929 | 1.425 |
| **Ours (21M)** | **0.905** | **1.223** | **1.179** | **1.893** | **2.233** | **3.760** | **2.781** | **1.756** |
| Seasonal Naive | 1.089 | 1.158 | 1.244 | 1.221 | 1.756 | 2.105 | 2.490 | 1.398 |

## 4. Domain-by-Domain Analysis

### Sales (4 configs): Competitive -- our best domain

Our relative MASE of 0.831 beats seasonal naive by 17%. We trail the best small models
(Moirai-2-Small at 0.689, FlowState at 0.694) by ~0.14 points, but we are closer to
Chronos-Small (0.733) and MOIRAI-Small (0.731) than to seasonal naive.

**Gap to close:** ~0.14 relative MASE to match top small models.

### Nature (15 configs): Competitive -- beats seasonal naive

Our relative MASE of 0.948 beats seasonal naive by 5%. This is a respectable showing,
though top small models reach 0.70-0.75 here. The gap is moderate.

**Gap to close:** ~0.20 relative MASE to match top small models.

### Transport (15 configs): Marginal -- barely above seasonal naive

Our relative MASE of 1.056 is 6% worse than seasonal naive. The best small models
(Reverso-Small at 0.610, Moirai-2-Small/FlowState at 0.620) are roughly 2x better
than us relative to the baseline. This is a domain where pre-trained models with
diverse real-world transport data (traffic, ride-sharing) have a clear advantage.

**Gap to close:** ~0.44 relative MASE. Significant.

### Energy (32 configs): Far behind -- worst relative gap

Our relative MASE of 1.550 is 55% worse than seasonal naive and almost 2x the best
small models (FlowState 0.827, Moirai-2 0.837). Energy is the largest domain by
config count (32 of 97 configs) and thus has outsized influence on overall GM-MASE.
Energy series often have strong diurnal/weekly seasonality and smooth trends that
well-trained foundation models capture easily, but our synthetic-data-only pretraining
does not include real energy patterns.

**Gap to close:** ~0.72 relative MASE. This is the single most impactful domain to fix.

### Web/CloudOps (20 configs): Far behind

Our relative MASE of 1.272 is 27% worse than seasonal naive. Top small models
(Reverso-Small 0.636, Moirai-2 0.665) are roughly 2x better. CloudOps data is
high-frequency, bursty, and often non-stationary -- characteristics that demand
exposure to real telemetry data during pretraining. Note that even Chronos-Small
(1.144) and MOIRAI-Small (1.136) lose to seasonal naive here, so this is a hard domain
for older/simpler models too.

**Gap to close:** ~0.64 relative MASE.

### Econ/Fin (6 configs): Far behind -- hardest domain for all models

Our relative MASE of 1.786 is 79% worse than seasonal naive. But this domain is hard
for everyone: the 6 configs are all M4 competition subsets (yearly, quarterly, monthly,
weekly, daily, hourly) which contain very short series with high noise-to-signal
ratios. Even Chronos-2 (0.772) and FlowState (0.760) only beat seasonal naive by ~23%.
TTM (1.409) also loses badly here.

The M4 datasets reward models that have seen similar macro/financial patterns in
pretraining. Our contrastive backbone, trained on synthetic ARMA + sinusoidal
composites, has no exposure to the kind of short, noisy, trend-dominated series
that characterize M4.

**Gap to close:** ~1.0 relative MASE. Extremely large, but partly inherent to
the domain difficulty.

### Healthcare (5 configs): Far behind, but dominated by one outlier

Our relative MASE of 1.117 looks bad, but healthcare is dominated by a single
extreme dataset: `covid_deaths/D/short`, which has MASE 33-47 for every model
(including Chronos-2 at 32.5 and seasonal naive at 46.9). The other 4 configs
(hospital/M, us_births/D/M/W) have MASE 0.3-1.0 for most models.

In GM terms, we score 2.781 vs Moirai-2's 1.495 and Chronos-2's 1.373. The gap
is real but substantially driven by how badly each model handles the covid_deaths
outlier. This domain has only 5 configs and is the least impactful on the overall score.

**Gap to close:** ~0.50 relative MASE.

## 5. Overall Ranking Among Small Models

Ranking by GM-relative MASE (lower = better):

| Rank | Model | Params | Rel. MASE |
|---|---|---|---|
| 1 | FlowState | 9.1M | 0.726 |
| 2 | Reverso-Small | ~15M | 0.726 |
| 3 | Moirai-2-Small | 11.4M | 0.728 |
| 4 | Kairos-23M | 23M | 0.748 |
| 5 | Chronos-Bolt-Small | ~28M | 0.822 |
| 6 | Chronos-Small | 46M | 0.892 |
| 7 | MOIRAI-Small | 14M | 0.946 |
| 8 | TTM-R2 | 1-5M | 1.020 |
| 9 | Seasonal Naive | 0 | 1.000 |
| 10 | **Ours** | **21M** | **1.256** |

We rank last among all compared models, and below seasonal naive. The 2024
generation of small models (Chronos-Small, MOIRAI-Small, TTM) generally score
0.89-1.02 on this metric. The 2025 generation (Moirai-2, FlowState, Kairos,
Reverso) has pushed that to 0.72-0.75. Our model at 1.256 is behind even the
weakest 2024 baselines.

## 6. Root Cause Analysis

### Why are we behind?

1. **Pretraining data mismatch.** Our backbone was pretrained on synthetic
   ARMA + sinusoidal composites. The GIFT-Eval benchmark spans energy meters,
   CloudOps telemetry, M4 macro-economic series, traffic sensors, hospital
   admissions, and ecological measurements. None of these distributions appear
   in our pretraining data. Models like Moirai-2 (trained on LOTSA + KernelSynth
   + CloudOps) and Chronos (trained on 28 real datasets + TSMixup + KernelSynth)
   have direct exposure to these domains.

2. **Training maturity.** Our v2 backbone was trained for only ~30k contrastive
   steps. The EXPERIMENT_STATUS.md documents that training was hampered by data
   ordering issues (unshuffled shards), and the scaling curve was flat because
   each epoch repeated the same distribution shift. The v2 dataset with proper
   shuffling may improve this.

3. **Energy dominates the benchmark.** Energy has 32 of 97 configs (33%).
   Our Energy relative MASE of 1.550 is catastrophic and drags the overall
   score. Energy is where we lose the most absolute ground.

4. **Point forecast vs. probabilistic.** Our model produces point forecasts
   via a GRU recovery head. Models like Chronos-2, Moirai-2, and Chronos
   produce probabilistic forecasts, and their median (0.5 quantile) is
   optimized during training. Our contrastive-then-recover pipeline was not
   designed for point forecasting as a primary objective.

5. **No frequency-aware tokenization.** Models like MOIRAI and Moirai-2 use
   frequency-dependent patch sizes to handle the wide range of sampling rates
   in GIFT-Eval (10-second to yearly). Our fixed window size (W=16 patches)
   may not adapt well to this diversity.

### Where do we have a path to improvement?

- **Sales and Nature** are already within striking distance. Better pretraining
  data alone could close these gaps.
- **Energy** is the highest-leverage domain: fixing 32 configs from 1.550 to
  ~0.85 would move our overall score from 1.256 to approximately 1.0.
- **Transport** is the second-highest leverage: 15 configs moving from 1.056
  to ~0.70 would be a significant overall improvement.
- **Econ/Fin** is hard for everyone and has only 6 configs; not worth
  over-optimizing for.
- **Healthcare** has only 5 configs and is dominated by the covid_deaths
  outlier; not a priority.

## 7. Key Takeaway

Our ~21M contrastive forecaster currently scores 1.256 GM-relative MASE on
GIFT-Eval, placing it below seasonal naive (1.000) and behind all small
foundation models tested (0.73-1.02 range). The primary bottleneck is not
architecture but pretraining data: our synthetic-only training set does not
cover the real-world distributions that GIFT-Eval tests. The secondary
bottleneck is training maturity (30k steps with data ordering issues).

To become competitive with the 2025 generation of small models (~0.73
relative MASE), we would need:
1. Real-world pretraining data covering energy, transport, and CloudOps
   domains (or much richer synthetic priors).
2. A properly shuffled training pipeline running to convergence.
3. Potentially, frequency-aware patch sizes or other adaptations for the
   diversity of sampling rates in GIFT-Eval.
