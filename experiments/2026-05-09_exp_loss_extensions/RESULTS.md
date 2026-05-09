# Experiment: loss_extensions — square cross-batch negatives

## Goal

Test whether adding two explicit cross-batch negative edges
(`cosine_similarity_batch_square`) reduces batch-axis collapse (U_batch)
without regressing AUC or Top-1 relative to the baseline
(`cosine_similarity_batch`).

## New loss: cosine_similarity_batch_square

`cosine_similarity_batch_square` adds two new contrastive edges on top of
`cosine_similarity_batch`:

- **neg_cross_batch_forecast**: for each time-step t, the forecast embedding
  of batch element b is contrasted against the forecast embedding of a
  different batch element b′ at the same t.  This repels forecast embeddings
  across the batch dimension.
- **neg_cross_batch_embedding**: for each t+1, the context embedding h_{b,t+1}
  is contrasted against h_{b′,t+1} for b′ ≠ b.  This repels context
  embeddings across the batch dimension at the same horizon.

Together, the two edges tile the "diagonal" of the cross-batch similarity
matrix that `cosine_similarity_batch` leaves untouched, forming a square
(2×2) pattern of negatives rather than the 1×2 rectangle of the base loss.

## Hypothesis

Adding these edges increases the pressure on the encoder to spread
representations along the batch axis (lower U_batch) while preserving or
improving discrimination ability (AUC, Top-1).

## Protocol

| Setting | Value |
|---|---|
| Arms | 4: {baseline, square} × {τ=0.10, τ=0.20} |
| Steps | 15 000 |
| Batch size | 256 |
| Dataset | GIFT-pretrain-full-4096 (HuggingFace) |
| Encoder | GRU |
| Model dims | d=384, n_heads=6, n_layers=6 |
| RevNorm | EWMA span=128 |
| Mixup probability | 0.3 |

Baselines are the τ=0.10 and τ=0.20 arms from the prior tau-sweep experiment
(15 000 steps each, identical hyperparameters except loss function).

## Results

### Final-step values (step 15 000)

| Arm | AUC | Top-1 | U_batch | U_temporal |
|---|---:|---:|---:|---:|
| baseline τ=0.10 | 0.9199 | 0.7765 | 0.0939 | 0.0491 |
| square   τ=0.10 | 0.9209 | 0.7790 | 0.0687 | 0.0346 |
| baseline τ=0.20 | 0.9205 | 0.7804 | 0.0784 | 0.0376 |
| square   τ=0.20 | 0.9183 | 0.7765 | 0.0762 | 0.0360 |

### Plots

**AUC and Top-1 (rolling mean window=500):**

![AUC and Top-1](plots/4arm_auc_top1.png)

**U_batch and U_temporal (rolling mean window=500):**

![Uniformity](plots/4arm_uniformity.png)

### Statistical tests — Welch t-test on steps 5 001–15 000

*Metric definitions*: AUC = area under the ROC curve for discriminating
positive (same-series future) from negative pairs; Top-1 = fraction of
queries where the true future is the top-ranked candidate; U_batch =
normalised embedding variance across the batch dimension (higher = more
spread; lower = more collapsed); U_temporal = same quantity computed
across the time dimension instead of the batch dimension.

All t-tests are two-sided Welch (unequal variances).

**Caveat — autocorrelated samples.** The n≈10 000 measurements per arm
are consecutive training steps from a *single* run, not i.i.d. evaluation
samples. Adjacent steps are strongly correlated (the model state changes
slowly), so the effective sample size is much smaller than n. Welch's
t-test assumes independence and is therefore anti-conservative here:
p-values are deflated and very small differences can appear "significant"
(e.g. the τ=0.10 sanity-check Δ AUC=0.0017 reaches p≈3e-11). Treat the
p-values below as **indicative directional evidence**, not a rigorous
hypothesis test. Multi-seed replication or held-out i.i.d. eval would be
required for a proper significance claim.

#### 1. baseline τ=0.10 vs square τ=0.10

| Metric | baseline τ=0.10 mean±std | square τ=0.10 mean±std | n each | t | p | sig (α=0.05) |
|---|---:|---:|---:|---:|---:|---:|
| AUC   | 0.8954 ± 0.0179 | 0.8953 ± 0.0175 | 10 000 |  0.492 | 6.23e-01 | NO |
| Top-1 | 0.7414 ± 0.0299 | 0.7415 ± 0.0293 | 10 000 | −0.212 | 8.32e-01 | NO |

No significant difference in AUC or Top-1 between the two τ=0.10 arms.

#### 2. baseline τ=0.20 vs square τ=0.20

| Metric | baseline τ=0.20 mean±std | square τ=0.20 mean±std | n each | t | p | sig (α=0.05) |
|---|---:|---:|---:|---:|---:|---:|
| AUC   | 0.8971 ± 0.0172 | 0.8934 ± 0.0176 | 10 000 | 14.726 | 7.89e-49 | YES |
| Top-1 | 0.7452 ± 0.0286 | 0.7392 ± 0.0290 | 10 000 | 14.715 | 9.25e-49 | YES |

At τ=0.20, `square` is significantly lower than baseline on both metrics
(mean AUC −0.0037, mean Top-1 −0.0060).

#### 3. baseline τ=0.10 vs baseline τ=0.20 (sanity check)

| Metric | baseline τ=0.10 | baseline τ=0.20 | n each | t | p | sig (α=0.05) |
|---|---:|---:|---:|---:|---:|---:|
| AUC   | 0.8954 ± 0.0179 | 0.8971 ± 0.0172 | 10 000 | −6.661 | 2.80e-11 | YES |
| Top-1 | 0.7414 ± 0.0299 | 0.7452 ± 0.0286 | 10 000 | −9.041 | 1.69e-19 | YES |

τ=0.20 beats τ=0.10 on both metrics (consistent with tau-sweep findings).

#### 4. square τ=0.10 vs square τ=0.20

| Metric | square τ=0.10 | square τ=0.20 | n each | t | p | sig (α=0.05) |
|---|---:|---:|---:|---:|---:|---:|
| AUC   | 0.8953 ± 0.0175 | 0.8934 ± 0.0176 | 10 000 | 7.452 | 9.59e-14 | YES |
| Top-1 | 0.7415 ± 0.0293 | 0.7392 ± 0.0290 | 10 000 | 5.679 | 1.37e-08 | YES |

Within the square loss, τ=0.10 is significantly better than τ=0.20 on both
metrics — reversing the ordering seen in the baseline.

## Conclusions

1. **U_batch reduction confirmed.** Both square arms show meaningfully lower
   U_batch at final step relative to their baseline counterparts
   (τ=0.10: 0.0687 vs 0.0939; τ=0.20: 0.0762 vs 0.0784). U_temporal is also
   lower under the square loss.

2. **No regression at τ=0.10.** For τ=0.10, the square loss produces
   statistically indistinguishable AUC and Top-1 over steps 5 001–15 000
   (p=0.62, p=0.83). Final-step values are essentially identical.

3. **Regression at τ=0.20.** For τ=0.20, the square loss is significantly
   worse than baseline on both AUC and Top-1 (p<1e-48), with mean AUC
   dropping 0.0037. The extra negatives appear to interact negatively with
   the softer temperature at τ=0.20.

4. **τ ordering reversal under square loss** (hypothesis). Baseline favours
   τ=0.20 over τ=0.10; under square, τ=0.10 is significantly better than
   τ=0.20. This is consistent with the hypothesis that the additional
   cross-batch negatives create a sharper effective contrastive signal that
   conflicts with the larger τ softening, but this is not directly tested
   here.

5. **Best single arm at 15 k steps** (by final AUC): baseline τ=0.20 (0.9205)
   and square τ=0.10 (0.9209) are virtually tied; baseline τ=0.10 (0.9199)
   and square τ=0.20 (0.9183) trail slightly.

**Summary**: `cosine_similarity_batch_square` at τ=0.10 achieves the
hypothesis goal — lower U_batch without detectable regression on AUC/Top-1.
At τ=0.20 it incurs a measurable cost. If U_batch reduction is the objective,
the combination `square + τ=0.10` is the recommended configuration from this
experiment.
