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

> **Side note on the CPC term (inherited from #344, unchanged here).** The CPC
> InfoNCE auxiliary follows a common *practical* CPC negative-sampling scheme —
> its negatives are other sequences' embeddings at the matched next step plus the
> same sequence's embeddings at other steps. This is **narrower** than van den
> Oord et al. (2018) Eq. 4, whose negatives are generic samples from the marginal
> proposal `p(x_{t+1})` (any sequence, *any* step, not only the matched one). The
> distinction concerns only the auxiliary term's negative set; it does not affect
> the comparison in this report, because #344 and #348 use the byte-identical
> term — so removing the encoder is the only thing that changes between the arms.

**Answer.** Removing the encoder does **not** improve transfer — it **reliably
degrades** the plain-contrastive arm (worse in all base cells: ~0.19–0.25 GM at
best-loss, ~0.05–0.08 at last, every 90% interval above zero). But **the CPC term
closes that gap entirely**: the no-encoder +CPC backbone **matches** its
encoder'd counterparts (best-loss cells all ns). The CPC term — a minor
late-training stabiliser when the encoder is present (#344, ~0.02 GM) — becomes a
**major peak-performance lever** without it, improving the no-encoder backbone's
best-loss GM by **0.20 (6L) to 0.26 (2L)**. In short, **the CPC auxiliary
substitutes for the encoder**: the encoder's contribution to transfer can be
recovered by the CPC term alone.

## Result

![Depth ladder of GM-Relative MASE across encoder depth {0 = no-encoder, 3, 6},
left for the base contrastive loss, right for + CPC, at 2L/6L heads × best/last.
Left: the no-encoder (blue) bars tower over enc3/enc6 in every cell. Right: the
no-encoder (green) bars sit level with enc3/enc6 — the CPC term recovers
parity.](plots/gm_summary.png)

GM-Relative MASE (GIFT-Eval full-97); encoder depth 0 = no-encoder (this work),
3/6 = #339/#341/#344. Lower is better.

| arm | 2L best / last | 6L best / last |
|---|--:|--:|
| base, **no-enc** | **1.425 / 1.264** | **1.353 / 1.239** |
| base, enc3 | 1.177 / 1.180 | 1.159 / 1.163 |
| base, enc6 | 1.180 / 1.213 | 1.161 / 1.193 |
| +CPC, **no-enc** | **1.168 / _TBD_** | **1.153 / _TBD_** |
| +CPC, enc3 | 1.185 / 1.153 | 1.158 / 1.144 |
| +CPC, enc6 | 1.179 / 1.180 | 1.158 / 1.162 |

**Removing the encoder reliably hurts the base arm.** Paired-bootstrap
Δ = GM(no-enc) − GM(encoder'd), 90% interval (positive ⇒ no-encoder worse):

| vs | 2L best | 6L best |
|---|--:|--:|
| enc3 | +0.249 (+0.197, +0.312) | +0.194 (+0.155, +0.242) |
| enc6 | +0.245 (+0.186, +0.314) | +0.192 (+0.144, +0.249) |

all reliably worse; the last-checkpoint gaps are smaller but also all above zero
(+0.05…+0.08).

**With the CPC term, removing the encoder is neutral.** Δ = GM(no-enc +CPC) −
GM(encoder'd +CPC), best-loss: vs enc3 −0.017 (2L) / −0.005 (6L); vs enc6 −0.011
(2L) / −0.004 (6L) — all four 90% intervals straddle zero.

**The CPC term is what closes the gap.** Adding CPC to the *no-encoder* backbone
(base → +CPC) improves best-loss GM by **−0.258 (−0.325, −0.200)** at 2L and
**−0.199 (−0.250, −0.156)** at 6L — both reliable, and an order of magnitude
larger than the ~0.02 the same term bought *with* the encoder in #344.

## Late-training stability

![Training dynamics, log-log. Solid = no-encoder (blue base / red +CPC), dashed =
enc6 reference (#344).](plots/training_dynamics.png)

_Last-checkpoint cpc cells completing; this section is filled once the two
no-encoder +CPC last evals land (does CPC reverse the base arm's best→last
behaviour without the encoder, as it did with it?)._

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
