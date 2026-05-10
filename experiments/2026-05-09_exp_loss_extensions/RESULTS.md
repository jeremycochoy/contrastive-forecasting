# Experiment: loss_extensions — square cross-batch negatives

## Goal

Compare AUC and Top-1 of `cosine_similarity_batch_square` against the
baseline `cosine_similarity_batch` at τ=0.10 and τ=0.20.

## What we changed

`cosine_similarity_batch_square` adds two cross-batch negative edges on
top of `cosine_similarity_batch`: **neg_cross_batch_forecast** (forecast
of element b vs forecast of b′≠b at the same t) and
**neg_cross_batch_embedding** (context h_{b,t+1} vs h_{b′,t+1}). They
tile the diagonal the base loss leaves empty, forming a 2×2 square of
negatives instead of a 1×2 rectangle.

## Protocol

| Setting | Value |
|---|---|
| Arms | 4: {baseline, square} × {τ=0.10, τ=0.20} |
| Steps | baselines 50 000, square arms 100 000, batch 256 |
| Dataset | GIFT-pretrain-full-4096 |
| Encoder | GRU, d=384, 6 heads, 6 layers |
| RevNorm | EWMA span=128, mixup p=0.3 |

Baselines = τ=0.10/τ=0.20 arms from the prior tau-sweep (identical
hyperparams except loss). Square arms extended to 100 k so we can read
their own late plateau.

## AUC / Top-1

![AUC and Top-1](plots/4arm_auc_top1.png)

Linear-x view: in the 0–15 k overlap all four arms cluster within
run-to-run noise; baselines plateau by ~40 k while squares keep ramping
and reach the same band by 90–100 k.

## Log-log

![Log-log convergence](plots/4arm_logscale.png)

Same data on log axes — all four arms ride the same scaling curve;
squares are shifted right but not on a different trajectory.

## Dimension usage

![U_batch and U_temporal](plots/4arm_dim_usage.png)

Square systematically lowers `U_batch` and `U_temporal` through 50 k.
With 100 k of training the gap narrows but does not close at τ=0.10
(see late-window table). Side metric, not what we score on.

## Late-window means

Per-step values are noisy (±0.02 AUC); means are taken over 10 k-step
windows. **Common** window (40 001–50 000) is the apples-to-apples
comparison; **own** window (90 001–100 000) is each square arm's late
plateau.

| Arm | Window | AUC | Top-1 | U_batch | U_temporal |
|---|---|---:|---:|---:|---:|
| baseline τ=0.10 | 40 001–50 000  | 0.9034 | 0.7573 | 0.1136 | 0.0571 |
| baseline τ=0.20 | 40 001–50 000  | 0.9039 | 0.7588 | 0.0890 | 0.0423 |
| square   τ=0.10 | 40 001–50 000  | 0.9024 | 0.7571 | 0.0814 | 0.0408 |
| square   τ=0.20 | 40 001–50 000  | 0.9014 | 0.7555 | 0.0891 | 0.0416 |
| square   τ=0.10 | 90 001–100 000 | 0.9040 | 0.7607 | 0.0868 | 0.0430 |
| square   τ=0.20 | 90 001–100 000 | 0.9031 | 0.7590 | 0.0930 | 0.0429 |

## Welch t-tests on AUC / Top-1 (overlap window 5 001–50 000, n=45 000)

| Comparison | Δ AUC | p_AUC | Δ Top-1 | p_Top-1 |
|---|---:|---:|---:|---:|
| baseline τ=0.10 vs square τ=0.10 | +0.0005 | 3.8e-05 | −0.0002 | 5.0e-01 |
| baseline τ=0.20 vs square τ=0.20 | +0.0031 | 7.9e-173 | +0.0046 | 2.0e-130 |
| baseline τ=0.10 vs baseline τ=0.20 (sanity) | −0.0010 | 1.3e-20 | −0.0024 | 1.4e-33 |
| square τ=0.10 vs square τ=0.20 | +0.0016 | 5.5e-47 | +0.0024 | 1.9e-36 |

Samples are consecutive training steps from a single run, not i.i.d.;
effective N is much smaller than 45 000, so p-values are
anti-conservative. Treat the Δ values as load-bearing, not the p
numbers. The window covers the squares' slow-converging region, so
Welch Δ overstates the late-plateau gap — see the 90 001–100 000 rows
above for the converged comparison.

## Bottom line

- **τ=0.10 at 50 k:** square is marginally behind baseline (Δ AUC
  −0.0010, Δ Top-1 −0.0002), inside per-step noise.
- **τ=0.20 at 50 k:** square is clearly behind baseline (Δ AUC
  −0.0025, Δ Top-1 −0.0033).
- **τ=0.10 at squares' own plateau (90–100 k):** square edges baseline's
  50 k plateau by +0.0006 AUC, +0.0034 Top-1.
- **τ=0.20 at squares' own plateau (90–100 k):** square closes to
  −0.0008 AUC, +0.0002 Top-1 vs baseline 50 k. The earlier "square
  is significantly worse at τ=0.20" verdict was a convergence-speed
  artifact, not a final-plateau gap.
- **U_batch:** square τ=0.10 still below baseline at 100 k (0.0868 vs
  0.1136, ~24 % lower); τ=0.20 has crossed baseline (0.0930 vs 0.0890).
  Lower U_batch is not desirable — baselines with higher U_batch reach
  the same AUC.
- **Practical take:** at this model scale and dataset the extra
  cross-batch negatives are a wash on AUC / Top-1 at convergence and
  cost extra compute to get there. Not a recommended default; not
  harmful at convergence.
