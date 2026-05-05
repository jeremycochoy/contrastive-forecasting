# Recovering Pairwise Correlations from Contrastive Embeddings of Random Walks

A companion experiment to **contrastive-arma**. Same backbone family, same loss, same
recovery-head philosophy — but the generative parameter is the **4×4 correlation matrix**
of 4 correlated Brownian motions instead of ARMA(p,q) coefficients.

## TL;DR

- Backbone: 154M-parameter transformer with GRU encoder, trained 100k steps (7.85 h on
  one RTX 4090). Reaches **FF–FP gap 0.380** at step 56k — substantially higher than the
  ARMA-V2 backbone's 0.203 at 2M steps. The cross-batch similarity collapses to ~0.02,
  meaning samples are nearly orthogonal in embedding space.
- Recovery requires the *right* probe shape:
  - The literal "single linear" head (mean-pool over time → linear → 6 outputs)
    cannot in principle compute pairwise correlations and indeed gives **Pearson
    r ≈ 0.03** — it converges to predicting the target mean.
  - A 2-hidden-layer MLP applied *after* mean-pooling does no better
    (**r ≈ 0.10**); pooling first destroys the temporal/quadratic structure
    needed for correlation.
  - A **TimeAware** head (per-timestep MLP across channels, *then* mean-pool over
    time, ending in a single linear projection to 6 outputs) recovers a clear
    signal: **mean Pearson r = 0.57** across the 6 pairs, MSE 0.0169 vs the
    mean-baseline 0.0242 (1.43× improvement) and the zero-baseline 0.166
    (9.80×). The architecture *exactly* matches the form of the empirical
    correlation operator: `(1/T) Σ_t f(channel-features at t)`.
- The trivial **diff(x) corrcoef** baseline (Pearson r = 0.997) remains far ahead —
  the contrastive head recovers ~57 % of the available linear signal. This is a
  positive but bounded result: the backbone *does* encode correlation information,
  but a non-trivial head is required to extract it, and even then the analytic
  estimator wins comfortably.

## 1. Question

Given a contrastive backbone trained without supervision on 4-channel correlated random
walks, can a head on top of the backbone recover the underlying correlation matrix that
generated the walks?

This is the cross-channel analogue of ARMA parameter recovery. There, the per-channel
generative parameters (p, q, AR/MA coefficients) sit *inside* each channel's temporal
dynamics. Here, the generative information sits *between* channels — there is no
per-channel temporal structure to memorise. Any usable signal must come from how
channels move together.

We add two analytic baselines:

- **diff(x) corrcoef** — empirical Pearson correlation of the channel **increments**.
  For Brownian motion this is essentially the maximum-likelihood estimator of the
  generative correlation, so it sets a near-optimal ceiling for any recovery method.
- **x corrcoef** — empirical Pearson correlation of the **positions** themselves.
  Naïve, biased upward by common drift.

A successful contrastive-head recovery should sit close to the diff(x) baseline and
clearly beat the position baseline.

## 2. Data

For each batch sample we sample a 4×4 correlation matrix C, draw 4096 iid 4-dim
increments with covariance C, take their cumulative sum, and z-score each channel
within sample (z-scoring is affine and so does not change Pearson correlations).

**Correlation sampling** uses a non-negative two-factor construction:

```
L_ki ~ U[0, 1],    k=1..4, i=1..2
||L_k|| := τ_k,    τ_k ~ U[0.4, 0.95]
diag_eps_k = 1 − ||L_k||²
C = L Lᵀ + diag(diag_eps)
```

This guarantees `diag(C) = 1`, off-diagonals in `[0, ~0.9]`, and PSD by construction.
Empirically the 6 unique pairwise correlations cover ≈ `[0.05, 0.9]`.

![Generated samples and their true correlation matrices](figures/data_samples.png)

## 3. Backbone

`ConfigurableModel` from `src.models`, identical configuration to the ARMA-V2 best:

- GRU encoder, H = 1024, W = 32 (128 patches per 4096-step sample)
- Transformer: 12 layers, 8 heads, FFN ×4, GELU, depthwise causal conv k=3
- Channel-mixing module (Kronecker product)
- Total parameters: **153.85 M**

**Training.** Contrastive loss `cosine_similarity_batch_no_time_neg`:

- positive: `cos(f_t, e_{t+1})` — same channel, next time step
- negatives: cross-channel (same time, different channel) and cross-batch
  (different sample)
- *no* cross-time negatives — for Brownian increments consecutive embeddings of a
  stationary process should be near-identical; only cross-channel and cross-batch
  signals carry sample-distinguishing information.

AdamW, batch 16, lr 7e−5, temperature 0.07, **100 000 steps** (~7.85 h on one
RTX 4090).

**Result.** Peak FF–FP gap = **0.3795** at step 56 000 (saved as `corrV1_best_gap.pth`).
Cross-batch similarity drops from 0.55 → 0.02 — the embedding space cleanly separates
different samples with different correlation matrices. Loss falls from 4.4 → 1.4.

For comparison, the ARMA-V2 12L backbone at 2 M steps reaches gap 0.203. This task
saturates at higher gap and **20× faster** because the cross-batch signal is much
stronger: each sample has a correlation structure that uniquely identifies it, and the
contrastive loss pushes hard on cross-batch separation.

![Training curves: contrastive gap and recovery loss](figures/training_curves.png)

## 4. Recovery heads

The backbone produces two latent tap-points: `h` (per-channel, pre-channel-mixing) and
`h_hat` (post-channel-mixing). Both have shape `[B, T, 4, 1024]`. We tested three head
families:

### 4.1 LinearHead (the literal "single linear" ask)

```
flat = mean_t(h).flatten(C, H)   # [B, 4*1024]
y    = sigmoid(W flat + b)        # [B, 6]
```

**Cannot work in principle.** A linear function over a per-channel concatenation can
only be a *sum of per-channel scalars*. Pairwise correlation requires multiplicative
interaction across channels, which is non-linear. Included as a sanity check.

### 4.2 MLPHead (mean-pool then MLP)

```
flat = mean_t(h).flatten(C, H)
y    = sigmoid(MLP(flat))         # 2 hidden layers, hidden=512
```

Adds non-linearity, so it can in principle compute multiplicative interactions. But
mean-pooling first **destroys** the time-dependence that's necessary to compute
quadratic time-averages.

### 4.3 TimeAwareHead (per-timestep MLP, then mean-pool)

```
per_t = MLP(flatten(h_t, K, H))    # [B, T, 6]   (one MLP shared across t)
y     = mean_t(per_t).clamp(0, 1)  # [B, 6]
```

Structurally matches the form of the empirical correlation operator
`(1/T) Σ_t f(x_{t,1}, ..., x_{t,K})` — non-linear cross-channel features computed per
timestep, then averaged over time. The output layer is still a single linear map to 6
outputs (final layer bias initialised at 0.4 to prevent sigmoid-style collapse).

We tested both `latent="h"` and `latent="h_hat"` for the TimeAware head.

### 4.4 Training

10 000 (LinearHead, MLPHead) or 3 000 (TimeAwareHead) epochs of AdamW, batch 8,
lr 1e−4 to 3e−4, MSE loss. Each epoch sees a freshly sampled batch — there is no
training/test set; "val" is a fixed 64-sample set that's never optimised against.

## 5. Results

### 5.1 Numerical summary (400-sample test set)

| Head | Latent | Overall MSE | vs mean baseline | vs zero-baseline | Mean Pearson r |
|------|--------|-------------|------------------|------------------|---------------|
| Linear        | h     | 0.0264 | 0.92× (worse) | 6.28× | **0.034** |
| MLP           | h     | 0.0240 | 1.01× | 6.89× | **0.101** |
| TimeAware     | h_hat | 0.0242 | 1.00× | 6.85× | **0.060** |
| **TimeAware** | **h** | **0.0169** | **1.43×** | **9.80×** | **0.573** |
| **diff(x) corrcoef** | n/a | **0.000174** | — | — | **0.997** |
| x corrcoef | n/a | 0.193 | — | — | 0.29 |

Key observations:

- The **diff(x) corrcoef baseline** essentially solves the problem (Pearson r = 0.997
  per pair, MSE 1.7e-4). Any model with raw access to the data can compute this in two
  lines. It is the ceiling for this task.
- The literal **single-linear** head and the 2-layer **mean-pool MLP** both converge
  to ≈ predicting the mean of the targets (Pearson r 0.03–0.10, vs-mean ≤ 1.01×).
  This is what a model that can't see the relevant signal *should* do under MSE:
  minimising MSE without input information drives the predictor toward the constant
  target mean.
- The **TimeAware** head on the per-channel encoder latent (`h`) clearly does better:
  per-pair Pearson r = 0.55–0.59 (mean 0.57), 1.43× improvement over the mean
  baseline. Same head on the channel-mixed latent (`h_hat`) collapses again to mean
  prediction (r ≈ 0.06), suggesting that the channel-mixing layer destroys the
  per-channel structure that the head needs to compute pairwise interactions.
- The position-correlation baseline is biased upward by the random-walk effect: it
  sees off-diagonal correlations of 0.6–0.85 even when the underlying generative
  correlation is 0.2–0.5 (mean Pearson r vs truth = 0.29).

### 5.2 Per-pair recovery

![True vs predicted, sample-level](figures/true_vs_predicted_bars.png)

![Per-pair scatter](figures/scatter_per_pair.png)

![Error distributions](figures/error_distributions.png)

![Predicted vs ground-truth correlation matrices](figures/corr_matrix_heatmaps.png)

### 5.3 Head vs analytic baselines

The diff(x) corrcoef is the optimal estimator of the generative correlation; the x
corrcoef is the naïve answer that ignores integration. We expected the contrastive
head to sit between the two.

![Head vs baselines](figures/baseline_comparison.png)

In our results the head sits on the position-corrcoef side of the picture, not the
diff(x) side, even though the diff(x) ceiling is much closer.

## 6. Discussion

### Why does only the TimeAware/h head work?

Pairwise correlation is a *quadratic* statistic of the data:
`corr(x_i, x_j) = (1/T) Σ_t z_{i,t} z_{j,t}` where `z` is the standardised channel.
Two structural ingredients are needed to compute it:

1. **Cross-channel multiplicative interaction**, which a linear function over a
   per-channel concatenation cannot produce. → rules out *LinearHead*.
2. **Sum/average over time of those interactions**, which mean-pooling-first
   destroys (you can't recover the time-average of a cross-channel product from the
   product of the time-averages). → rules out *MLPHead* with mean-pool-then-MLP.

The **TimeAware** head was designed to satisfy both: a non-linear cross-channel MLP
applied per timestep, then mean-pooled over time. It can in principle compute any
quadratic-form-then-time-average, including correlation. The empirical result
(mean Pearson r = 0.57) confirms the inductive bias matters more than capacity here:
the MLPHead has more parameters than the TimeAware head and still achieves r ≈ 0.

### Why does TimeAware/h_hat fail where TimeAware/h succeeds?

The two latents differ only by the channel-mixing module:

```
h     = (post-transformer, per-channel)        → TimeAware succeeds (r 0.57)
h_hat = channel_mix(h)                          → TimeAware collapses  (r 0.06)
```

The channel-mixing module is a learnable Kronecker product over channels. It mixes
each channel's H-dim feature into a sample-level summary, after which "channel 0"
no longer corresponds to "the encoder output for input channel 0". The TimeAware
head needs the per-channel structure to compute pairwise interactions — once the
channels are blended, the linear-then-mean operation can no longer separate
`channel i` from `channel j`. This is consistent with the same effect we see in
ARMA recovery, where the per-channel `h` is the canonical input.

### Why is recovery still well below the diff(x) ceiling?

Even at r = 0.57 the head is far below the diff(x) baseline (r = 0.997). Possible
contributors:

1. **Information-bottleneck-style invariance.** The contrastive loss suppresses
   task-irrelevant variance and the model has compressed the per-sample signal more
   than it needs to express the correlation matrix faithfully. The cross-batch
   similarity collapsing to 0.02 says "samples are nearly orthogonal", which is
   stronger than "correlations are linearly encoded".
2. **Probe under-trained.** The TimeAware head was still improving when we cut it
   off at 3 000 epochs (val loss dropped 0.026 → 0.018 over the run, with no clear
   plateau). A longer probe run would likely move r up.
3. **Probe architecture.** A bilinear pooling layer or an explicit Pearson-style
   `Σ_t z_{i,t} z_{j,t}` operator would be a sharper inductive bias than the
   per-time MLP we used.

### Comparison to ARMA recovery

| Experiment | Backbone steps | Gap | Recovery probe Pearson r |
|------------|----------------|-----|--------------------------|
| ARMA V2 12L     | 2 M    | 0.203 | 0.93–0.96 per coefficient (GRU recovery head) |
| Correlation V1  | 100 k  | 0.380 | 0.55–0.59 per pair (TimeAware head) |

A higher contrastive gap does not translate to better probe recovery. The ARMA
experiment got 0.93+ at a much lower gap because the contrastive bottleneck
naturally preserves AR/MA structure (every channel's temporal dynamics matter for
prediction). Here, the bottleneck preserves enough information to *separate*
samples but not enough to keep the full correlation matrix linearly readable.

## 7. Compute

| Stage | Wall time |
|-------|-----------|
| Backbone (100k steps, bs 16, RTX 4090) | 7 h 51 min |
| Linear head (8k epochs, bs 8) | ~1 h 20 min |
| MLP head (8k epochs, bs 8) | ~1 h 20 min |
| TimeAware head × 2 latents (3k epochs each, bs 8) | ~50 min |
| Plotting | < 5 min |
| **Total** | **~ 11.5 h** |

Single GPU (NVIDIA RTX 4090), entirely autonomous overnight run with hourly
status-check cron and a streaming log monitor.

## 8. Follow-ups worth trying

- Train the TimeAware head **longer** (it was still improving at 3 000 epochs when
  the run ended). Add LR schedule.
- Add an **explicit bilinear** layer in the head: `Σ_t z_t W z_tᵀ` where `z_t` is a
  per-channel projection. This is the natural "Pearson-like" operator and is a
  sharper inductive bias than the per-time MLP used here.
- Train a **GRU recovery head** over time, mirroring the architecture that won ARMA
  recovery. The GRU's temporal aggregation could match or exceed the per-step MLP.
- Try a backbone with **smaller batch** so the cross-batch term doesn't dominate, or
  drop cross-batch entirely and rely on cross-channel + cross-time. The current
  cross-batch dominance probably pushes toward more aggressive compression than the
  task needs.
- Train the head **alongside the backbone** with a multi-task objective (contrastive +
  small supervised correlation MSE) to bias the representation toward
  correlation-structured features.

## Appendix A. Sampling diagnostics

Sanity check that the data generator does what it claims:

```text
True C (sample 0):
 1.00 0.53 0.56 0.61
 0.53 1.00 0.43 0.47
 0.56 0.43 1.00 0.51
 0.61 0.47 0.51 1.00

corrcoef(diff(x))  → 0.519, 0.567, 0.600, 0.441, 0.475, 0.510  (matches true)
corrcoef(x)        → 0.838, 0.624, 0.611, 0.847, 0.796, 0.860  (random-walk bias)
```

The diff-of-positions empirical Pearson correlation matches the generating C closely
(MAE ≈ 0.01 at T = 4096). The position correlation is biased upward — the random-walk
effect.

## Appendix B. Files

- `src/correlation.py` — data generator and pair↔matrix helpers.
- `experiments/contrastive-correlation/train_contrastive_corr.py` — backbone training.
- `experiments/contrastive-correlation/correlation_recovery.py` — recovery heads
  (Linear, MLP, TimeAware) and training/eval logic.
- `experiments/contrastive-correlation/evaluate_and_plot.py` — figure generation.
- `experiments/contrastive-correlation/scripts/run.sh` — end-to-end pipeline.
- `experiments/contrastive-correlation/checkpoints/corrV1_best_gap.pth` — backbone
  checkpoint at the peak FF–FP gap.
- `experiments/contrastive-correlation/figures/` — figures used in this report.
