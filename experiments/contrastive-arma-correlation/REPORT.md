# Joint ARMA + Correlation Recovery, Channel-Mixing in the Loss

Two runs:

- **V6** — original loss `cosine_similarity_batch_no_time_neg`, which (we
  later found) was missing a same-time cross-channel negative on the
  forecaster.
- **V7** — same architecture, same hyperparameters, with the loss bug fixed.

## Question

Can a single contrastive backbone — trained with the
`Simple_channel_mixing_module` *active in the loss path* — produce a frozen
representation from which two separate heads can simultaneously recover
**ARMA coefficients** (per channel) and **pairwise channel correlations**
(per sample)?

## Design

- **Data** (`data.py`). 4 ARMA(p,q) channels, p,q ∈ {1..4}, T = 4096.
  Innovations Cholesky-correlated with a per-sample 4×4 correlation
  matrix C drawn uniformly per pair (PSD rejection). Each (b, k) channel
  z-scored.
- **Backbone**. `ConfigurableModel` (per-channel GRU encoder + 12-layer
  transformer applied per-channel + `Simple_channel_mixing_module`),
  H = 1024, 154 M params. The transformer reshapes to `[B*C, T, H]` so
  it never sees more than one channel at a time. The channel-mixing
  module sits at the top, applying a sample-independent
  `kron(I, R) + kron(mask, Q)` linear map to mix channels in latent
  space. Loss `cosine_similarity_batch_no_time_neg`. R initialised to I,
  Q ~ N(0, 0.01²/H). 150 k steps, bs = 16, lr = 7e-5, no grad clip.
  See `train_backbone.py`.
- **Heads** (frozen backbone, no fine-tune):
  - **ARMA head** = `GRURecoveryHead` (h = 128, 2 layers, bidirectional)
    on per-channel encoder output `h` ∈ ℝ^{B·C × T × H}, 4 AR + 4 MA
    coefficients per channel. MSE vs. ground-truth padded coefficients.
    676 k params.
  - **Correlation head** = `JointCorrelationHead` (Linear(C·H → H) + GRU
    h = 128, mean-pool → Linear → 6) on the channel-mixed forecaster
    `h_hat` ∈ ℝ^{B × T × C·H}, 6 off-diagonal correlations per sample.
    MSE vs. ground-truth C. 4.87 M params.

### V6 vs V7 difference

The original `cosine_similarity_batch_no_time_neg` (V6) had only:

```
positives  = exp(cos(h_hat[t,c], h[t+1,c]))            # forecasting
neg_xx     = exp(cos(h[t,c1],   h[t,c2]))     c1 ≠ c2  # h × h, no h_hat
neg_cb     = exp(cos(h_hat[b1], h[b2]))       b1 ≠ b2  # cross-batch
```

The cross-channel negative `neg_xx` does not involve `h_hat` and so does
not gradient through the forecaster — it only spreads encoder outputs
per channel. The variant *intended* to drop cross-time negatives also
silently dropped `neg_xy_hat = exp(cos(h[t,c1], h_hat[t,c2]))` for
c1 ≠ c2, which is **same-time, cross-channel** — not cross-time —
and is the term that gives the forecaster cross-channel discriminative
pressure.

V7 re-adds `neg_xy_hat` (cross-channel only, c1==c2 diagonal masked).

## V6 results

Backbone: 11.8 h. Best val gap **0.132** (FF − FP) at step 144 k.
CB drops to ~0.005–0.015. Loss plateaus at 1.0–1.5; no collapse.

ARMA head ✅ — Per-coef Pearson r 0.93–0.95, sign agreement 93.9 %,
**7.1× improvement** vs zero baseline. Per-channel ARMA structure is
cleanly preserved in `h`.

Correlation head ❌ — Per-pair Pearson r ≈ 0 (-0.04 to 0.08).
Improvement vs zero 4.08×, vs mean 0.99× (head ≈ marginal mean), vs
trivial `corrcoef(diff(y))` 1.31× on MSE — but the diff baseline gets
per-pair r ≈ 0.6–0.7 against the head's ~0. The head learns the
unconditional correlation distribution.

See `plots/{training_curves_v6,arma_recovery_v6,correlation_recovery_v6,baseline_comparison_v6,data_samples_v6}.png`.

## V7 results (loss bug fixed)

Backbone: 11.8 h, same hyperparameters. The fixed loss has a noticeable
**delayed phase transition** — V6 had cross-batch differentiation kick in
at step ~4 k; V7 stays at FF=FP≈0.85, CB≈0.80 until step ~14 k, then
breaks through. Once it does, V7 tracks V6 with a ~10–20 k step lag and
catches up by ~step 100 k. Best val gap **0.135** at step 150 k —
slightly *higher* than V6 (0.132). CB drops to ~0.005–0.015 as in V6.

See `plots/v6_v7_compare.png` for side-by-side curves.

ARMA head ✅ — Same quality as V6:

| Coefficient | V6 r | V7 r  |
|-------------|------|-------|
| AR[0]       | 0.930 | 0.930 |
| AR[1]       | 0.948 | 0.949 |
| AR[2]       | 0.949 | 0.945 |
| AR[3]       | 0.927 | 0.931 |
| MA[0]       | 0.928 | 0.928 |
| MA[1]       | 0.944 | 0.947 |
| MA[2]       | 0.946 | 0.947 |
| MA[3]       | 0.933 | 0.945 |

7.17× improvement vs zero (V6: 7.06×), sign agreement 94.0 %.

Correlation head ❌ — **Same failure as V6**:

| Pair | V6 r_head | V7 r_head | r_diff_baseline |
|------|-----------|-----------|-----------------|
| 0    | −0.012    | −0.087    | 0.653           |
| 1    |  0.061    | −0.030    | 0.667           |
| 2    |  0.051    | −0.073    | 0.600           |
| 3    |  0.077    |  0.043    | 0.675           |
| 4    | −0.016    |  0.006    | 0.601           |
| 5    | −0.041    |  0.032    | 0.688           |

Per-pair Pearson r remains ≈ 0; improvement vs mean 0.99× (head still
predicts the marginal mean). MSE numbers are within sampling noise of V6.

See `plots_v7/{arma_recovery_v6,correlation_recovery_v6,baseline_comparison_v6,training_curves_v6,data_samples_v6}.png`.

## Interpretation

The V6 report blamed the correlation failure on cross-channel negatives
making `h_hat` "invariant to correlation." That interpretation was wrong
for two reasons: the cross-channel negative in V6 didn't actually involve
`h_hat` (the bug), and fixing the bug in V7 leaves the correlation result
unchanged. The real story is **architectural** and the loss bug is a
separate issue that mostly affects the optimization trajectory, not the
final representation's content.

The information-theoretic argument:

- The encoder + transformer operate strictly per-channel. They cannot
  observe `(y^{c1}, y^{c2})` jointly for any sample. The transformer
  output `t^{b,c}` is therefore a function of channel c's data only.
- After per-(b,k) z-scoring, channel c's marginal distribution is the
  same regardless of `C^(b)`: `Var(ε^(c)) = C[c,c] = 1` for any
  correlation matrix. There is no per-channel statistic that varies
  with the per-sample correlation.
- The `Simple_channel_mixing_module` applies a sample-independent
  linear map. Its mixing of `t^{b,c}` across channels uses the same
  R, Q for every sample. So `h_hat^{b,c}` is a deterministic linear
  function of `{t^{b,c'}}_{c'}`, which by the above does not encode
  `C^(b)`.

Therefore there is no information path from `C^(b)` to `h_hat`, no
matter how the loss is wired. V7 confirms this: with the cross-channel
pressure on `h_hat` re-added, the gap and ARMA recovery improve
slightly, but per-sample correlation is still un-recoverable.

ARMA recovery still works because per-channel ARMA dynamics (AR/MA
coefficients) **are** in channel c's marginal distribution, and the
encoder can extract them.

The loss bug *is* a real bug — V7 reaches a slightly better gap and
converges with a healthier separation between FF and FP, and the fix
should be carried forward to all future runs using this loss. But the
correlation experiment requires either:

1. Channels in the transformer input dimension, like V5
   (`JointChannelModel`, `[B, T, C·H]`), so attention can compute
   per-sample cross-channel statistics; or
2. A sample-dependent channel-mixing module (e.g. one whose R, Q are
   functions of the input).

Plain per-channel transformer + sample-independent channel-mixing
on top, regardless of loss, cannot recover per-sample correlation.

## Artifacts

V6 (kept for the loss-bug contrast):
- `plots/{training_curves,arma_recovery,correlation_recovery,baseline_comparison,data_samples}_v6.png`
- `checkpoints/corrV6_best_gap.pth` (gap = 0.132 @ 144 k)
- `checkpoints/corrV6_head_{arma,corr}_best.pth`
- `checkpoints/{backbone_corrV6,corrV6_head_arma,corrV6_head_corr}_results.json`

V7 (with loss fix):
- `plots/v6_v7_compare.png`
- `plots_v7/{training_curves,arma_recovery,correlation_recovery,baseline_comparison,data_samples}_v6.png`
  *(filenames retain the `_v6` suffix from the unparameterised plot
  script; contents are V7.)*
- `checkpoints/corrV7_best_gap.pth` (gap = 0.135 @ 150 k)
- `checkpoints/corrV7_head_{arma,corr}_best.pth`
- `checkpoints/{backbone_corrV7,corrV7_head_arma,corrV7_head_corr}_results.json`

Loss fix in `src/loss.py:133–172` (commit `7443a77`).
