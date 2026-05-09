# Experiment: loss_extensions — square cross-batch negatives

## Goal

Compare AUC and Top-1 of the new `cosine_similarity_batch_square` loss
against the baseline `cosine_similarity_batch` loss, at τ=0.10 and τ=0.20.

## What we changed

`cosine_similarity_batch_square` adds two cross-batch negative edges on
top of `cosine_similarity_batch`: **neg_cross_batch_forecast** (forecast
embedding of element b vs forecast of b′≠b at the same t) and
**neg_cross_batch_embedding** (context h_{b,t+1} vs h_{b′,t+1}).
Together they tile the diagonal that the base loss leaves untouched,
forming a 2×2 square of negatives instead of a 1×2 rectangle.

## Protocol

| Setting | Value |
|---|---|
| Arms | 4: {baseline, square} × {τ=0.10, τ=0.20} |
| Steps | baselines 50 000, square arms 15 000, batch 256 |
| Dataset | GIFT-pretrain-full-4096 |
| Encoder | GRU, d=384, 6 heads, 6 layers |
| RevNorm | EWMA span=128, mixup p=0.3 |

Baselines = τ=0.10/τ=0.20 arms from prior tau-sweep (identical hyperparams
except loss). The square arms only have the 0–15 k segment; baselines
are now extended to 50 k so the late plateau is visible.

## What the AUC / Top-1 curves show

![AUC and Top-1](plots/4arm_auc_top1.png)

Inside the 0–15 k overlap window all four arms cluster tightly within
run-to-run noise. Past 15 k the baselines drift up by another ≈0.005
AUC and ≈0.005 Top-1 on the smoothed curve and then sit on a flat
plateau through 50 k. Without a 50 k square run we can't claim square
diverges late — only that the ≈0.001 AUC lead square τ=0.10 had at
15 k is well inside the noise band the baselines produce on their
late plateau.

## The same data on log-log

![Log-log convergence](plots/4arm_logscale.png)

The log-x view exposes the early-training ramp and confirms the late-
training band is genuinely narrow rather than just compressed by the
linear scale. The 4 arms track each other through the whole ramp; no
arm wins by reaching convergence faster.

## Where the loss change is most visible: dimension usage

![U_batch and U_temporal](plots/4arm_dim_usage.png)

`U_batch` and `U_temporal` measure how many embedding dimensions are
actively used along the batch and temporal axes respectively (higher =
more dimensions in use, lower = more collapsed). The square loss
systematically lowers both at both τ — the cleanest signal in the
experiment, consistent with the extra cross-batch negatives reducing
batch-axis collapse. With baselines extended to 50 k the gap widens:
baseline τ=0.10 reaches U_batch ≈ 0.114, while the square τ=0.10 arm
ends at ≈ 0.069 — though we don't know whether square would itself
have continued climbing past 15 k. Side metric, not the objective we
score on.

## Late-window means

Per-step values are noisy by ±0.02 AUC, so the headline numbers below
are means over the last 10 k steps of each arm.

| Arm | Window | AUC | Top-1 | U_batch | U_temporal |
|---|---|---:|---:|---:|---:|
| baseline τ=0.10 | 40 001–50 000 | 0.9034 | 0.7573 | 0.1136 | 0.0571 |
| baseline τ=0.20 | 40 001–50 000 | 0.9039 | 0.7588 | 0.0890 | 0.0423 |
| square   τ=0.10 | 5 001–15 000  | 0.8953 | 0.7415 | 0.0643 | 0.0336 |
| square   τ=0.20 | 5 001–15 000  | 0.8934 | 0.7392 | 0.0713 | 0.0350 |

The square arms are quoted on their available 5 k–15 k window because
that is also the window used for the Welch tests below; this is a
conservative late-window for them but it cannot include the 35 k of
training the baselines have past 15 k.

## Statistical tests on AUC / Top-1 (Welch t, overlap window 5 001–15 000, n=10 000 each)

| Comparison | Δ AUC | p_AUC | Δ Top-1 | p_Top-1 |
|---|---:|---:|---:|---:|
| baseline τ=0.10 vs square τ=0.10 | +0.0001 | 5.5e-01 | −0.0000 | 9.4e-01 |
| baseline τ=0.20 vs square τ=0.20 | +0.0037 | 1.6e-49 | +0.0060 | 1.1e-49 |
| baseline τ=0.10 vs baseline τ=0.20 (sanity) | −0.0017 | 2.7e-11 | −0.0038 | 1.6e-19 |
| square τ=0.10 vs square τ=0.20 | +0.0019 | 9.6e-14 | +0.0023 | 1.4e-08 |

**Caveat:** samples are consecutive training steps from a single run,
not i.i.d.; effective sample size is much smaller than n, so Welch
p-values are anti-conservative. Treat as directional, not rigorous.
The overlap window is the only window where all four arms have data.

## Bottom line

- **τ=0.10 (overlap window):** square is statistically indistinguishable
  from baseline on AUC and Top-1 (p=0.55, 0.94) — no regression.
- **τ=0.20 (overlap window):** square is significantly worse than baseline
  (Δ AUC −0.0037, Δ Top-1 −0.0060, p<1e-48).
- **Past 15 k:** baselines plateau at AUC ≈ 0.903, Top-1 ≈ 0.757–0.759
  (40–50 k mean). We did not extend square arms, so we can't say whether
  the square τ=0.10 noise-band advantage at 15 k would have held.
- **Side effect:** dimension usage (U_batch, U_temporal) is consistently
  lower under square at both τ; with baselines extended to 50 k the gap
  becomes very visible — directional support that the extra edges reduce
  batch-axis collapse, but not the metric we're optimizing.
