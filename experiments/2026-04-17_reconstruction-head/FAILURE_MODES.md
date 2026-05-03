# Failure Mode Analysis — R1 Forecaster Reconstruction

## Overview

R1 (best head, GM-Relative MASE = 1.121) beats seasonal naive on 38/97 GIFT-Eval
configs but is worse on 59/97. The GM-MASE > 1.0 means the model is worse than
naive overall, despite excellent performance on some datasets.

This document analyzes the failure modes to guide future improvements.

## Per-config breakdown

**Worst configs (model >> naive):**

| Config | R1 MASE | Naive MASE | Relative |
|--------|---------|------------|----------|
| solar/10T/medium | 3.38 | 0.93 | 3.64x |
| m4_hourly/H/short | 3.96 | 1.19 | 3.32x |
| bizitobs_service/10S/medium | 3.88 | 1.32 | 2.94x |
| electricity/15T/long | 3.15 | 1.16 | 2.71x |
| electricity/15T/medium | 2.97 | 1.15 | 2.58x |
| solar/10T/long | 2.24 | 0.87 | 2.57x |
| ett2/W/short | 1.72 | 0.78 | 2.21x |

**Best configs (model << naive):**

| Config | R1 MASE | Naive MASE | Relative |
|--------|---------|------------|----------|
| bizitobs_l2c/5T/short | 0.35 | 0.99 | 0.36x |
| us_births/D/short | 0.96 | 1.86 | 0.52x |
| m_dense/D/short | 0.92 | 1.67 | 0.55x |
| jena_weather/10T/short | 0.44 | 0.74 | 0.59x |
| loop_seattle/D/short | 1.04 | 1.73 | 0.60x |

## Three failure modes (from prediction plots)

### 1. Strong periodic patterns — model can't match frequency

**Affected:** solar, electricity, bizitobs_service, bizitobs_application

The model produces smooth curves or oscillates at the wrong frequency/amplitude.
Seasonal naive copies the last season exactly, making it unbeatable when the
periodicity is clean and fixed.

**Example:** solar/10T has clear daily cycles. The model attempts oscillation but
with wrong period and dampened amplitude. Seasonal naive reproduces the cycle
perfectly.

**Root cause:** The training data (TimesFM-style composite) includes sinusoids
with random periods sampled log-uniformly from [4, T/2]. The model learns
"there might be oscillation" but not "this is 24h-periodic at exactly 96
timesteps." It has no mechanism to detect and lock onto a fixed periodicity.

### 2. Explosive trends — model can't extrapolate growth

**Affected:** covid_deaths, m4_yearly

The model predicts a downturn or plateau while the ground truth keeps climbing
exponentially. With only 1024 timesteps of context and RevEWMNorm subtracting
the trend, the model has no information about the growth rate.

**Example:** covid_deaths has exponential growth. The model's prediction curves
downward while truth doubles. Only ~20 data points in m4_yearly — too few patches
for meaningful context.

### 3. Sharp spikes — model smooths everything

**Affected:** bitbrains_rnd, bizitobs (cloud ops data)

Cloud infrastructure metrics have sudden spikes (CPU bursts, traffic peaks).
The model outputs smooth curves, unable to produce the sharp transients.

**Root cause:** The contrastive objective + RevEWMNorm encourage smooth latent
trajectories. The reconstruction head inherits this smoothness — it can't
reconstruct what was never encoded.

## Why GM-MASE > 1 despite many good configs

The geometric mean is dominated by the large outliers. A 3.6x failure weighs much
more than a 0.36x success in log-space:

- log(3.64) = +1.29 (one bad config contributes +1.29)
- log(0.36) = -1.02 (one great config only offsets -1.02)

To push GM-MASE below 1.0, we need to either:
1. Fix the worst outliers (solar, electricity — periodic data)
2. Add many more configs where we beat naive
3. Both

## Implications for next experiments

1. **Seasonality-aware features:** Add period detection or frequency embeddings
   to help the model lock onto fixed periodicities. TimesFM uses explicit
   frequency tokens.

2. **Training data diversity:** Include real-world seasonal data in training,
   not just synthetic composites. The model needs to learn actual 24h/7d/365d
   patterns.

3. **Longer context:** 1024 timesteps (64 patches) may be insufficient for
   seasonal patterns with long periods (weekly = 672 timesteps at hourly,
   exceeding our context).

4. **Spike modeling:** RevEWMNorm dampens spikes. Consider alternative
   normalization or spike-preserving encoding for cloud ops data.
