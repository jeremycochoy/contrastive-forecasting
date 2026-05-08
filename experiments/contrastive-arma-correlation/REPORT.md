# V6: Joint ARMA + Correlation Recovery, Channel-Mixing in the Loss

## Question

Can a single contrastive backbone — trained with the
`Simple_channel_mixing_module` *active in the loss path* — produce a frozen
representation from which two separate heads can simultaneously recover
**ARMA coefficients** (per channel) and **pairwise channel correlations**
(per sample)?

## Design

- **Data**. 4 ARMA(p,q) channels, p,q ∈ {1..4}, T = 4096. Innovations are
  Cholesky-correlated with a per-sample 4×4 correlation matrix C drawn
  uniformly per pair (PSD rejection). Each (b, k) channel z-scored. See
  `data.py`.
- **Backbone**. `ConfigurableModel` (GRU encoder + 12-layer transformer +
  `Simple_channel_mixing_module`), H = 1024, ~290 M params. Trained with
  `model.forward(x)` so the Kronecker R, Q matrices are inside the loss path
  (R initialised to I, Q ~ N(0, 0.01²/H)). Loss
  `cosine_similarity_batch_no_time_neg`: cross-channel + cross-batch
  negatives, no cross-time negatives. 150 k steps, bs = 16, lr = 7e-5. See
  `train_backbone.py`.
- **Heads** (frozen backbone, no fine-tune):
  - **ARMA head** = `GRURecoveryHead` (h = 128, 2 layers, bidirectional)
    on per-channel encoder output `h` ∈ ℝ^{B·C × T × H}, 4 AR + 4 MA
    coefficients per channel. MSE vs. ground-truth padded coefficients.
    676 k params. See `train_arma_head.py`.
  - **Correlation head** = `JointCorrelationHead` (Linear(C·H → H) + GRU
    h = 128, mean-pool → Linear → 6) on the channel-mixed forecaster
    `h_hat` ∈ ℝ^{B × T × C·H}, 6 off-diagonal correlations per sample.
    MSE vs. ground-truth C. 4.87 M params. See `train_correlation_head.py`.

## Results

### Backbone

- 11.8 h on a single 4090.
- Best val gap **0.132** (FF − FP) at step 144 k. CB drops to ~0.005–0.015
  (cross-batch differentiation healthy throughout).
- Loss plateaus at 1.0–1.5; no collapse, no grad clipping.
- Compared to V5 (Brownian-only correlation): V5 reached gap 0.51 because
  Brownian increments make o_t / o_{t+1} naturally easy to distinguish
  *step-locally*. ARMA processes are far more autocorrelated, so a small
  gap is expected even when the representation is informative.

See `plots/training_curves_v6.png`.

### ARMA head ✅

| Coefficient | Pearson r | MAE  | n_nonzero |
|-------------|-----------|------|-----------|
| AR[0]       | 0.930     | 0.121 | 757      |
| AR[1]       | 0.948     | 0.091 | 569      |
| AR[2]       | 0.949     | 0.075 | 358      |
| AR[3]       | 0.927     | 0.088 | 181      |
| MA[0]       | 0.928     | 0.121 | 753      |
| MA[1]       | 0.944     | 0.089 | 564      |
| MA[2]       | 0.946     | 0.078 | 372      |
| MA[3]       | 0.933     | 0.083 | 187      |

- Overall MSE 0.014 vs. zero-baseline 0.098 → **7.1× improvement**.
- Sign agreement 93.9 %.
- Per-channel ARMA structure is preserved cleanly in the encoder output
  `h`, despite the joint-correlation training objective.

See `plots/arma_recovery_v6.png`.

### Correlation head ❌

| Pair | r_head  | r_diff_baseline | r_pos_baseline | MAE   |
|------|---------|-----------------|----------------|-------|
| 0    | −0.012  | 0.653           | 0.656          | 0.234 |
| 1    | 0.061   | 0.667           | 0.645          | 0.203 |
| 2    | 0.051   | 0.600           | 0.599          | 0.202 |
| 3    | 0.077   | 0.675           | 0.667          | 0.224 |
| 4    | −0.016  | 0.601           | 0.599          | 0.222 |
| 5    | −0.041  | 0.688           | 0.695          | 0.217 |

- Per-pair Pearson r essentially **zero**.
- Overall MSE 0.0645. The mean-baseline MSE (predict the marginal mean
  ≈ 0.45 for every pair) is 0.0639 — the head is **0.99×** the trivial
  mean baseline.
- The MSE table looks superficially OK: head is "4.08× better than zero
  baseline" and "1.31× better than diff(y) corrcoef baseline." But the
  per-pair Pearson r tells the real story: the head outputs ≈ the marginal
  mean for every pair, with no per-sample sensitivity. Diff(y) baseline,
  in contrast, has per-pair r ≈ 0.6–0.7 — the trivial estimator
  `corrcoef(diff(y))` correctly recovers correlation up to noise, even
  though its MSE is higher because it is unbiased and noisy rather than
  biased and constant.

See `plots/correlation_recovery_v6.png`.

## Interpretation

**ARMA succeeds, correlation fails — and the failure is informative.**

Two architectural choices interact here:

1. The ARMA head reads `h` (per-channel encoder output, before
   channel-mixing). The contrastive loss preserves per-channel temporal
   structure — that signal is what the cross-batch and cross-channel
   negatives reward — and a small head can decode AR/MA coefficients out
   of it.
2. The correlation head reads `h_hat` (post channel-mixing forecaster).
   The hypothesis was that cross-batch negatives would force `h_hat` to
   encode per-sample correlation patterns. They did not. The likely
   reason: cross-channel negatives in the loss penalise the model when
   `h_hat` for channel-i resembles `h` for channel-j of the same sample.
   To minimise this penalty, the backbone learns an `h_hat` that is
   **largely invariant** to the cross-channel correlation structure —
   exactly the opposite of what we wanted from the correlation head.

V5 (the previous joint-channel experiment) avoided this by flattening
C·H into the *transformer input dimension* and training over a single
[B, T, C·H] sequence. There is no per-channel slice for the loss to
penalise, so the joint embedding is forced to encode cross-channel
structure from the start. V5 reached r = 0.962 on Brownian-only
correlation; V6 (with the same `JointCorrelationHead` topology applied
post-hoc to a different backbone) reaches ≈ 0.

**Practical takeaway**: For correlation recovery, the backbone must
treat channels jointly *in its representation*, not just in its loss.
`Simple_channel_mixing_module` plus per-channel cross-channel negatives
is the wrong architectural pattern for per-sample correlation features.

## Artifacts

- `plots/data_samples_v6.png`
- `plots/training_curves_v6.png`
- `plots/arma_recovery_v6.png`
- `plots/correlation_recovery_v6.png`
- `plots/baseline_comparison_v6.png`
- `checkpoints/corrV6_best_gap.pth` (best gap = 0.132 @ step 144 k)
- `checkpoints/corrV6_head_arma_best.pth` (val_loss = 0.0136)
- `checkpoints/corrV6_head_corr_best.pth` (val_loss = 0.0638)
- `checkpoints/{backbone,corrV6_head_arma,corrV6_head_corr}_results.json`
- `logs/corrV6.log`, `logs/corrV6_head_arma.log`, `logs/corrV6_head_corr.log`
