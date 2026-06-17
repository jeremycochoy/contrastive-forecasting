# No-encoder backbone: GIFT-Eval forecasting error with and without the CPC term

We remove the encoder stack from the backbone (`--num-encoder-layers 0`) so the
forecaster reads the patch-embedding directly, train it under three losses — the
contrastive loss alone (**base**), the contrastive loss plus the CPC InfoNCE
auxiliary term (**+ CPC**), and that term with its candidate set widened to the
full van den Oord Eq. 4 marginal — every other sequence at every step
(**+ CPC_All**) — and measure GIFT-Eval forecasting error (GM-Relative MASE). We
compare against the same recipe trained with a 3-layer and a 6-layer encoder.

*A backbone here is patch-embedding → (encoder stack) → forecaster. The
patch-embedding (a GRU) turns each window patch into one token; the encoder is a
stack of causal transformer layers; the forecaster is a separate causal
transformer trained by the contrastive objective to predict the next token's
embedding. `num_encoder_layers=0` removes the encoder stack, leaving the
forecaster to read the patch-embedding tokens directly. **GM-Relative MASE** is
the geometric mean, over GIFT-Eval's 97 tasks, of a model's error divided by the
seasonal-naive forecast's error; lower is better, 1.0 = seasonal-naive.*

## Result

![GM-Relative MASE across encoder depth {0 = no encoder, 3, 6}, left for the base
contrastive loss, right for + CPC, at 2-layer / 6-layer heads × best-loss / last
checkpoints.](plots/gm_summary.png)

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

Paired-bootstrap Δ = GM(no encoder) − GM(encoder'd), 90% interval (resampling the
97-task list with repeats; positive ⇒ the no-encoder GM is higher):

| loss · vs | 2L best | 6L best | 2L last | 6L last |
|---|--:|--:|--:|--:|
| base · enc-3 | +0.248 (+0.197, +0.312) | +0.194 (+0.155, +0.242) | +0.084 (+0.059, +0.110) | +0.076 (+0.053, +0.104) |
| base · enc-6 | +0.245 (+0.186, +0.314) | +0.192 (+0.144, +0.249) | +0.050 (+0.023, +0.079) | +0.046 (+0.018, +0.076) |
| +CPC · enc-3 | −0.017 (−0.038, +0.004) | −0.005 (−0.027, +0.017) | +0.012 (−0.001, +0.025) | +0.016 (+0.004, +0.031) |
| +CPC · enc-6 | −0.011 (−0.030, +0.009) | −0.004 (−0.025, +0.016) | −0.016 (−0.033, +0.002) | −0.002 (−0.019, +0.015) |

For the **base** loss every interval is above zero (the no-encoder GM is higher
by 0.19–0.25 at best-loss, 0.05–0.08 at last). For **+ CPC**, seven of the eight
intervals straddle zero (the exception is +0.016 at 6L-last vs enc-3).

Adding the CPC term to the no-encoder backbone, Δ = GM(+CPC) − GM(base), 90%
interval: 2L best −0.258 (−0.325, −0.200), 6L best −0.199 (−0.250, −0.156), 2L
last −0.099 (−0.124, −0.076), 6L last −0.079 (−0.106, −0.056) — every interval
below zero.

**+ CPC_All** (no encoder, the full-marginal candidate set) vs **base**, same Δ:
2L best −0.248 (−0.316, −0.190), 6L best −0.171 (−0.222, −0.122), 2L last −0.093
(−0.114, −0.073), 6L last −0.071 (−0.094, −0.049) — every interval below zero.
vs the no-encoder **+ CPC** arm, Δ = GM(+CPC_All) − GM(+CPC): 2L best +0.009
(+0.000, +0.020), 6L best +0.029 (+0.011, +0.049), 2L last +0.007 (−0.002,
+0.015), 6L last +0.008 (−0.002, +0.019) — the two last cells straddle zero, the
two best cells are above it. vs the encoder'd **+ CPC** arms, six of eight
intervals straddle zero (the exceptions are +0.018 and +0.025 at the last
checkpoint vs enc-3).

## Training curves

![Training metrics, log-log. Solid = no encoder (blue base / red + CPC / green
+ CPC_All), dashed = enc-6 reference (blue/orange).](plots/training_dynamics.png)

The CPC term's logged value (panel 2) stays near 3 (+ CPC) and near 6 (+ CPC_All)
throughout the no-encoder runs; in the encoder'd run it falls below 10⁻³ by about
step 1,000. (The + CPC and + CPC_All term values are on different scales — their
candidate sets differ in size.) Across training the no-encoder + CPC and + CPC_All
runs record a lower contrastive reference loss, lower 1−R², and lower 1−AUC than
the no-encoder base run (panels 1, 6, 8).

From best-loss to last checkpoint, the no-encoder + CPC GM changes by ≤0.007
(2L 1.168→1.165, 6L 1.153→1.160); the no-encoder base GM is lower at the last
checkpoint than at best-loss (2L 1.425→1.264, 6L 1.353→1.239).

## Protocol

One backbone per arm, single seed, one RTX 4090. Each arm is the same recipe —
GRU patch-embedding, d_model 384 / 6 heads, a 6-layer full-width forecaster, the
crossfade-triplet allt·0.8% data mix, qk-norm, attention-output norm, the
`xshh_allt` contrastive loss with positive-in-denominator and floor subtraction,
the encoder-side positive stop-gradient (`--stopgrad-positive-h`), τ 0.10, batch
1024, 12,500 steps, seed 20260520 — with the encoder stack removed
(`--num-encoder-layers 0`). The arms differ only in the loss: **base** is the
contrastive loss alone; **+ CPC** adds the CPC InfoNCE term (`--cpc-infonce-weight
1.0`); **+ CPC_All** is the same term with its candidate set widened to the full
marginal (`--cpc-infonce-negs cross`). To score a backbone we freeze it and train a fresh quantile forecasting
head, once with two transformer layers and once with six, and evaluate on
GIFT-Eval's 97 tasks at the best-loss checkpoint (lowest smoothed contrastive
loss) and the last checkpoint (step 12,500). The enc-3 and enc-6 numbers come
from separate backbones trained with this identical recipe at encoder depth 3
and 6 — the experiment changes only the encoder depth.

## The CPC term

The + CPC arm adds one CPC InfoNCE term (van den Oord et al. 2018, Eq. 4, horizon
k = 1): for each step *t* it predicts the next embedding `e_{t+1}` from the
forecaster context `h_t` through a learnable log-bilinear score
`f(e_j, h_t) = exp(e_jᵀ W₁ h_t)`:

```
L_cpc = − log(  exp(e_{t+1}ᵀ W₁ h_t)  /  Σ_{e_j ∈ C}  exp(e_jᵀ W₁ h_t)  )
```

The positive sits in the denominator (normalized InfoNCE ≥ 0), summed
equal-weight, no stop-gradient, no temperature (`W₁` carries the scale). The
candidate set `C` differs by arm: **+ CPC** uses the matched-step cross-batch
embeddings plus the same sequence's other-step embeddings; **+ CPC_All** uses the
positive plus every other sequence's embeddings at every step (the full marginal
`p(x_{t+1})` of van den Oord Eq. 4, whose negatives are independent of the context
`h_t`).
