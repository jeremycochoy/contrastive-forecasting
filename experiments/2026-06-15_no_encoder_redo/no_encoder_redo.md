# Removing the encoder: does the patch-embedding + forecaster still transfer?

**Question.** #344's two competing setups — the **normal contrastive loss** and
that loss **+ a CPC InfoNCE term** — both ran with a causal *encoder* stack
(3 or 6 layers) between the patch-embedding and the forecaster. Removing the
encoder forces the forecaster to read the patch-embedding directly, denying it
the encoder's learned positional mixing. Does that **improve transfer** to
GIFT-Eval, and does the **CPC term still help** (and still stabilise late
training) once the encoder is gone?

*Architecture. A backbone here is patch-embedding → (optional encoder stack) →
forecaster. The **patch-embedding** is a GRU that turns each window patch into
one token. The **encoder** is a stack of causal transformer layers that mixes
those tokens; the **forecaster** is a separate causal transformer trained by the
contrastive objective to predict the next token's embedding. "No encoder"
(`num_encoder_layers=0`) deletes the middle stack: the forecaster reads the
patch-embedding tokens directly, and the contrastive target `e_t` becomes the
patch-embedding output instead of an encoder output.*

<!-- RESULTS PENDING — filled when downstream evals land -->

## Result

![pending](plots/gm_summary.png)

_TODO: GM-Relative MASE depth ladder (no-encoder vs enc3 vs enc6), for base and
+CPC, at 2L/6L heads × best/last checkpoints; paired-bootstrap Δ vs the encoder'd
arms and for the CPC term without the encoder._

## Late-training stability

![pending](plots/training_dynamics.png)

_TODO: does the CPC term still reverse the best→last degradation without the
encoder?_

## Protocol

We train one backbone per arm, single seed, on one RTX 4090. Each arm is the
**exact** #339/#341/#344 recipe — GRU patch-embedding, d_model 384 / 6 heads, a
6-layer full-width forecaster, the crossfade-triplet allt·0.8% data mix, qk-norm,
attention-output norm, the `xshh_allt` contrastive loss with positive-in-denominator
and floor subtraction, the encoder-side positive stop-gradient
(`--stopgrad-positive-h`), τ 0.10, batch 1024, 12,500 steps, seed 20260520 —
with the single change that the **encoder stack is removed**
(`--num-encoder-layers 0`). The two arms differ only in the loss:

- **base** — the contrastive loss alone.
- **+ CPC** — the same loss plus the #344 CPC InfoNCE auxiliary term
  (`--cpc-infonce-weight 1.0`).

To score a backbone we freeze it and train a fresh quantile forecasting head on
top, once with two transformer layers and once with six. We evaluate on
GIFT-Eval's 97 tasks at two checkpoints: the **best-loss** one (the lowest
smoothed contrastive loss) and the **last** one (step 12,500). The encoder'd
counterparts are the published same-arm numbers from #339/#341 (base) and #344
(+CPC); the analysis reuses their GM and paired-bootstrap code unchanged and
reproduces their GMs to three decimals.

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97
forecasting tasks, of a model's error divided by the seasonal-naive forecast's
error. Lower is better; 1.0 is seasonal-naive. Each pairwise Δ carries a
**paired-bootstrap** 90% interval — resample the 97-task list with repeats and
score both arms on each resample so per-task difficulty cancels.*

## The CPC term (unchanged from #344)

Alongside the contrastive loss the +CPC arm adds one CPC InfoNCE term (van den
Oord et al. 2018, Eq. 4, horizon k = 1). For each step *t* it predicts the next
embedding `e_{t+1}` from the forecaster's context `h_t` through a new learnable
log-bilinear score `f(e_j, h_t) = exp(e_jᵀ W₁ h_t)`:

```
L_cpc = − log(  exp(e_{t+1}ᵀ W₁ h_t)  /  Σ_{e_j ∈ C}  exp(e_jᵀ W₁ h_t)  )
```

The candidate set `C` is the true next embedding plus negatives (other sequences'
embeddings at the same step and the same sequence's embeddings at other steps).
The positive sits in the denominator, so the term is a normalized InfoNCE ≥ 0,
summed equal-weight, with no stop-gradient and no temperature (the unbounded `W₁`
carries the scale). Without the encoder, the embedding `e_t` it predicts is the
patch-embedding output.
