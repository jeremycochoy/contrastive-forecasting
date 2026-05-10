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

## Relative-gap analysis

`analyze_ratios.py` recomputes the missing cross-channel cosine similarities
on the final backbone (`*_best_gap.pth`) of each run and plots them next
to the JSON-logged FF/FP/CB time-series.

Final-state cosine similarities:

| metric                                  | V6      | V7      |
|-----------------------------------------|---------|---------|
| FF = cos(h_hat[t,c], h[t+1,c])           | 0.624   | 0.638   |
| FP = cos(h_hat[t,c], h[t,c])             | 0.492   | 0.503   |
| TP = cos(h[t+1,c], h[t,c])               | 0.328   | 0.348   |
| CC(h,h) = cos(h[t,c1], h[t,c2]), c1≠c2   | 0.036   | 0.041   |
| CC(h,h_hat) = cos(h[t,c1], h_hat[t,c2])  | +0.017  | **−0.013** |
| CB = cos(h_hat[b1,t,c], h[b2,t+1,c])     | 0.011   | 0.011   |

Final-state ratios (FF / FP, CC / FP, FF / CC):

| ratio          | V6    | V7    |
|----------------|-------|-------|
| FF / FP        | 1.27  | 1.27  |
| CC(h,h) / FP   | 0.07  | 0.08  |
| FF / CC(h,h)   | 17.5  | 15.7  |

The loss fix worked exactly on its target metric: `CC(h,h_hat)` flipped
from +0.017 in V6 (no cross-channel pressure on the forecaster) to −0.013
in V7 (forecaster now actively anti-aligned across channels). The
encoder-side ratios that quantify the "structure" of the latent are
essentially unchanged — channels are nearly orthogonal in `h` (CC ≈ 0.04
versus FP ≈ 0.5), and samples are very orthogonal in the cross-batch
sense (CB ≈ 0.011). The forecast positive is 16–17× the cross-channel
similarity in both runs.

So the latent has the structure we wanted: tight within-channel
forecasting, near-orthogonal between channels, near-orthogonal between
samples. Yet correlation recovery still fails. The signal isn't blocked
by the architecture *or* missing from the optimisation pressure; it's
present but at a much lower order than the dominant features the head
sees, and the simple GRU correlation head doesn't tease it out.

See `plots/v6_v7_ratios.png`.

## Interpretation

The V6 report claimed cross-channel negatives made `h_hat` "invariant to
correlation," and a follow-up pass tightened that to a flat
information-theoretic impossibility ("no information path from C^(b) to
h_hat regardless of loss"). Both claims were stronger than the evidence
warrants:

1. The information-theoretic argument was based on per-channel marginals
   being correlation-invariant after z-scoring — true — but ignored that
   the **joint** distribution of channels in a single sample still
   depends on `C^(b)`. The cross-channel negative `neg_xx` in the loss
   pushes `h^{b,c1}` and `h^{b,c2}` apart while they are realisations
   of correlated processes; how easy that separation is and what
   features the encoder picks for it can carry sample-specific
   structure. The channel-mixing module's `Q · ∑_{c'≠c} t^{b,c'}` term
   then linearly combines those joint realisations, so per-sample joint
   statistics do flow into `h_hat`.

2. The relative-gap analysis confirms the cross-channel pressure works
   mechanically: V7's `CC(h, h_hat)` is −0.013 (anti-aligned) versus
   V6's +0.017. The encoder-side `CC(h, h)` is ~0.04 in both, and the
   forecast positive is 16–17× the cross-channel similarity. The
   latent has the structure we wanted — channels nearly orthogonal,
   samples very nearly orthogonal — yet correlation recovery is still
   ≈ 0 per pair.

The honest summary: we don't have a clean impossibility result. What we
have is empirical evidence that **a 4.87 M-param GRU correlation head
with mean-pool readout cannot extract per-sample correlation from this
particular `h_hat`**. Possible reasons we have not ruled out:

- The signal exists in `h_hat` but at much lower magnitude than the
  per-channel ARMA features that dominate the cross-batch contrastive
  pressure; the head allocates capacity to the loud signal and ignores
  the quiet one.
- The head architecture (Linear(C·H → H) + GRU(h=128)) collapses the
  channel dimension before the GRU sees it, losing the cross-channel
  structure too early. A head that processes channels jointly per-time
  step (e.g. attention over C tokens of size H) might do better.
- The signal is only carried in higher-order joint statistics
  (variances of cross-channel inner products over time), which a
  mean-pooled GRU is poorly suited to capture.

ARMA recovery still works because per-channel ARMA dynamics (AR/MA
coefficients) live in channel c's univariate marginal, which the
encoder can extract straightforwardly.

The loss bug is a real bug — V7 reaches a slightly better gap, FF/FP
separates more cleanly, and `CC(h, h_hat)` actually moves below zero.
The fix should be carried forward to all future runs using this loss.

For correlation recovery, the next thing to try (cheapest first) is a
**different head architecture** — keep the V7 backbone, replace the
`Linear(C·H → H) + GRU` head with one that preserves the channel
dimension into the temporal model (e.g. per-time-step attention across
C, or a small transformer over `[B·T, C, H]`). If that still fails,
the next step is V5-style joint-channel input or a sample-dependent
mixing module.

## Artifacts

V6 (kept for the loss-bug contrast):
- `plots/{training_curves,arma_recovery,correlation_recovery,baseline_comparison,data_samples}_v6.png`
- `checkpoints/corrV6_best_gap.pth` (gap = 0.132 @ 144 k)
- `checkpoints/corrV6_head_{arma,corr}_best.pth`
- `checkpoints/{backbone_corrV6,corrV6_head_arma,corrV6_head_corr}_results.json`

V7 (with loss fix):
- `plots/v6_v7_compare.png`
- `plots/v6_v7_ratios.png`, `plots/v6_v7_ratios_finals.json`
- `analyze_ratios.py`
- `plots_v7/{training_curves,arma_recovery,correlation_recovery,baseline_comparison,data_samples}_v6.png`
  *(filenames retain the `_v6` suffix from the unparameterised plot
  script; contents are V7.)*
- `checkpoints/corrV7_best_gap.pth` (gap = 0.135 @ 150 k)
- `checkpoints/corrV7_head_{arma,corr}_best.pth`
- `checkpoints/{backbone_corrV7,corrV7_head_arma,corrV7_head_corr}_results.json`

Loss fix in `src/loss.py:133–172` (commit `7443a77`).
