# Contrastive Correlation Recovery — Design

## Motivation

Mirror of the contrastive-arma experiment, but instead of recovering ARMA(p,q) coefficients
the recovery head predicts the pairwise Pearson correlation matrix between channels.

Why this is interesting:
- The "task" content is purely cross-channel structure (no per-channel temporal dynamics
  to memorize — each channel is a plain Brownian motion).
- The contrastive backbone has cross-channel and cross-batch negatives (no time term).
- Strong test of whether the backbone learns sample-level structure.

## Data

For each batch sample b:

1. Sample a 4×4 correlation matrix C_b with the factor-construction below
   (always positive off-diagonals, exactly 1 on diagonal, PSD by construction).
2. Cholesky factor L_b = chol(C_b).
3. Draw T iid increments z_t ~ N(0, I_4), apply ε_t = L_b z_t → joint covariance C_b.
4. Cumulative sum: x_t = Σ_{s≤t} ε_s (4 correlated Brownian motions).
5. Per-channel z-score within sample (preserves correlation, normalises magnitude).

Output: `x ∈ R^{B × 4096 × 4}`, `C ∈ R^{B × 4 × 4}` correlation matrices.

### Correlation sampling

Two-factor non-negative loadings:
```
L_ki = U[0, 1]               # k=1..4, i=1..2
target_norm_k = U[0.4, 0.95] # per channel
L_k *= target_norm_k / ||L_k||
diag_eps_k = 1 - ||L_k||²    # ≥ 0
C = L Lᵀ + diag(diag_eps)
```

This yields valid correlation matrices (diag = 1, off-diag in [0, ~0.9], PSD).
The 6 unique pairwise correlations cover the range [0.1, 0.9] empirically.

## Backbone

Same `ConfigurableModel` as ARMA v2:
- GRU encoder, H=1024, W=32, 12 layers, 8 heads, FFN ×4, GELU, depthwise conv k=3.
- Loss: `cosine_similarity_batch_no_time_neg` (cross-channel + cross-batch negatives only).
- Patches: 4096 / 32 = 128 patches per sample.

Why no time term? In ARMA, the next-time embedding is informative because of temporal
dynamics. For Brownian motion the increments are iid, so cross-time prediction is mostly
noise. Cross-channel and cross-batch negatives carry the signal we care about.

## Recovery head

Frozen backbone. The latent is `h ∈ R^{B × T × C × H}`. We mean-pool over T,
concatenate over C, then a single linear layer to 6 outputs:

```
h_pool = mean_t h           # [B, C, H]
flat   = h_pool.flatten(C,H) # [B, C*H]
y      = sigmoid(W flat + b) # [B, 6]
```

Targets are the 6 unique pairwise correlations from C_b (upper-triangle off-diagonal),
already in [0, 1) by construction. Trained with MSE.

The "single linear" choice is intentional: it tests whether the backbone latent
linearly encodes correlation magnitude. We also include a small MLP variant for
comparison.

## Metrics

- MSE on each of 6 pairs vs zero-baseline (mean of training targets, ≈ 0.3).
- Pearson r between predicted and true correlation, per pair.
- Mean absolute error per pair.
- Improvement ratio (mean baseline MSE / mean head MSE).

## Plots

Mirror the ARMA experiment plots:
- True vs predicted bar charts for individual samples.
- Per-pair scatter (true vs predicted) with y=x reference and Pearson r.
- Error distribution histograms.
- Side-by-side correlation matrix heatmaps for a few samples.

## Compute budget (target: overnight)

- Backbone: 12L H=1024, batch 16, 150–200k steps, ~6–8 hours on RTX 4090.
- Recovery head: 5–10k epochs, ~30 min.
- Plotting + report: ~30 min.
