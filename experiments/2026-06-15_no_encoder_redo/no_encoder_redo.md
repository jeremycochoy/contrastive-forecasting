# Without the encoder the base loss has higher GIFT-Eval error; the + CPC and + CPC_All arms match the encoder'd backbones

We remove the encoder stack from the backbone (`--num-encoder-layers 0`), so the
forecaster reads the patch-embedding directly, and measure GIFT-Eval forecasting
error (GM-Relative MASE). We train three losses: the contrastive loss alone
(**base**); that loss plus the CPC InfoNCE auxiliary term (**+ CPC**); and the
CPC term with its candidate set widened to the full van den Oord Eq. 4 marginal —
every other sequence at every step (**+ CPC_All**). We compare against the same
recipe trained with a 3-layer and a 6-layer encoder.

*A backbone here is patch-embedding → (encoder stack) → forecaster. The
patch-embedding (a GRU) turns each window patch into one token; the encoder is a
stack of causal transformer layers; the forecaster is a separate causal
transformer trained by the contrastive objective to predict the next token's
embedding. `num_encoder_layers=0` removes the encoder stack, leaving the
forecaster to read the patch-embedding tokens directly. **GM-Relative MASE** is
the geometric mean, over GIFT-Eval's 97 tasks, of a model's error divided by the
seasonal-naive forecast's error; lower is better, 1.0 = seasonal-naive.*

## Result

![Three panels of GM-Relative MASE (GIFT-Eval full-97), grouped by head ×
checkpoint. Left: the encoder-depth ladder {0 = no encoder, 3, 6} for the base
loss. Middle: every + CPC arm side by side — no-encoder + CPC and + CPC_All
beside the enc-3 and enc-6 + CPC backbones. Right: the no-encoder loss comparison
— base vs + CPC vs + CPC_All.](plots/gm_summary.png)

GM-Relative MASE (GIFT-Eval full-97; lower is better). Encoder depth 0 = no
encoder (this work); 3 and 6 = the 3- and 6-layer-encoder backbones.

| arm | 2L head, best / last | 6L head, best / last |
|---|--:|--:|
| base, **no encoder** | **1.425 / 1.264** | **1.353 / 1.239** |
| base, enc-3 | 1.177 / 1.180 | 1.159 / 1.163 |
| base, enc-6 | 1.180 / 1.213 | 1.161 / 1.193 |
| + CPC, **no encoder** | **1.168 / 1.165** | **1.153 / 1.160** |
| + CPC, enc-3 | 1.185 / 1.153 | 1.158 / 1.144 |
| + CPC, enc-6 | 1.179 / 1.180 | 1.158 / 1.162 |
| + CPC_All, **no encoder** | **1.177 / 1.171** | **1.182 / 1.168** |

![Forest plot of the paired-bootstrap Δ = GM(left arm) − GM(right arm) on
GIFT-Eval full-97, with the 90% interval from resampling the 97-task list. One
panel per head × checkpoint; one row per comparison. A filled marker marks an
interval that excludes zero, an open marker one that spans it.](plots/deltas_forest.png)

Reading the intervals: for the **base** loss, every no-encoder − encoder'd
interval excludes zero — the no-encoder GM is higher, and more so at best-loss
than at the last checkpoint. For **+ CPC**, those intervals span zero in seven of
eight cells. Adding either CPC term to the no-encoder backbone (**+ CPC** or
**+ CPC_All** vs **base**) gives a negative interval that excludes zero in all
eight cells. **+ CPC_All** sits within ±0.03 of the no-encoder **+ CPC** — the two
best-loss cells exclude zero, the two last span it — and its intervals against the
encoder'd **+ CPC** arms span zero in six of eight cells.

## Training curves

![Eight log-log panels. Top row: contrastive reference loss, CPC InfoNCE term
value, ratio gap (1−ff)/(1−fp), U_batch. Bottom row: U_temporal, 1−R²_naive,
1−R²_random, 1−retrieval AUC. Solid = no encoder (blue base / red + CPC / green
+ CPC_All); dashed = enc-6 reference (cyan base / orange + CPC).](plots/training_dynamics.png)

In the panels, **ff** is the forecast-to-future cosine (the positive pair's
similarity) and **fp** the forecast-to-present cosine, so the ratio gap
(1−ff)/(1−fp) falls toward 0 as the positive separates; **R²_naive / R²_random**
is the variance in the next embedding the forecast explains, relative to a
copy-the-present / random-embedding baseline (1−R² is the unexplained share);
**U_batch / U_temporal** is the fraction of embedding dimensions that vary across
the batch / across time; **retrieval AUC** ranks the positive against the
negatives. The contrastive reference loss is a fixed-recipe normalised-InfoNCE
value at temperature 0.07, logged for cross-run comparison and distinct from the
τ 0.10 training objective.

The CPC term's logged value (CPC InfoNCE term value panel) is ≈3–6 for + CPC and
≈6–10 for + CPC_All across the no-encoder runs; in the enc-6 reference it drops
from ≈0.2 early into the 10⁻³–10⁻⁴ band (median ≈7×10⁻⁴ over the latter part of
training). (The + CPC and + CPC_All term values are not on the same scale: by
construction the two arms sum over candidate sets of different sizes — see *The
CPC term*.) Across training the no-encoder + CPC and + CPC_All runs record a lower
contrastive reference loss, lower 1−R² (both the naive and random panels), and
lower 1−AUC than the no-encoder base run; they also settle at a lower ratio gap
(≈0.81–0.83 vs base 1.03) and higher U_batch (≈0.34 vs 0.28), with U_temporal
slightly lower (≈0.20–0.21 vs 0.23).

From best-loss to last checkpoint, the no-encoder + CPC GM changes by ≤0.007
(2L 1.168→1.165, 6L 1.153→1.160); the no-encoder base GM is lower at the last
checkpoint than at best-loss (2L 1.425→1.264, 6L 1.353→1.239).

## Protocol

One backbone per arm, single seed, one RTX 4090. The shared recipe is a GRU
patch-embedding, d_model 384 with 6 heads, a 6-layer full-width forecaster, the
crossfade-triplet allt·0.8% data mix, qk-norm and attention-output norm, the
`xshh_allt` contrastive loss (positive-in-denominator, floor subtraction), the
encoder-side positive stop-gradient (`--stopgrad-positive-h`), τ 0.10, batch
1024, 12,500 steps, and seed 20260520. The encoder stack is removed with
`--num-encoder-layers 0`.

The arms differ only in the loss. **base** is the contrastive loss alone.
**+ CPC** adds the CPC InfoNCE term (`--cpc-infonce-weight 1.0`). **+ CPC_All** is
the same term with its candidate set widened to the full marginal
(`--cpc-infonce-negs cross`).

To score a backbone we freeze it and train a fresh quantile forecasting head,
once with two transformer layers and once with six. We evaluate on GIFT-Eval's 97
tasks at the best-loss checkpoint (lowest smoothed contrastive loss) and the last
checkpoint (step 12,500). The enc-3 and enc-6 backbones use this identical recipe
at encoder depth 3 and 6; the experiment changes only the encoder depth.

## The CPC term

The + CPC arm adds one CPC InfoNCE term (van den Oord et al. 2018, Eq. 4, horizon
k = 1): for each step *t* it predicts the next embedding `e_{t+1}` from the
forecaster context `h_t` through a learnable log-bilinear score
`f(e_j, h_t) = exp(e_jᵀ W₁ h_t)`:

```
L_cpc = − log(  exp(e_{t+1}ᵀ W₁ h_t)  /  Σ_{e_j ∈ C}  exp(e_jᵀ W₁ h_t)  )
```

The positive sits in the denominator (normalised InfoNCE ≥ 0), summed
equal-weight, no stop-gradient, no temperature (`W₁` carries the scale). The
candidate set `C` differs by arm: **+ CPC** uses the matched-step cross-batch
embeddings plus the same sequence's other-step embeddings; **+ CPC_All** uses the
positive plus every other sequence's embeddings at every step (the full marginal
`p(x_{t+1})` of van den Oord Eq. 4, whose negatives are independent of the context
`h_t`).
