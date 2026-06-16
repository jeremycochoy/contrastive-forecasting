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
| +CPC, **no-enc** | **1.168 / 1.165** | **1.153 / 1.160** |
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
GM(encoder'd +CPC): at best-loss all four 90% intervals straddle zero (vs enc3
−0.017/−0.005, vs enc6 −0.011/−0.004 for 2L/6L); at the last checkpoint three of
four straddle zero and the fourth is a marginal +0.016 vs enc3·6L. So the
no-encoder +CPC backbone sits within noise of its encoder'd counterparts at both
checkpoints.

**The CPC term is what closes the gap.** Adding CPC to the *no-encoder* backbone
(base → +CPC) improves GM **reliably at every checkpoint**: best-loss
**−0.258 (−0.325, −0.200)** at 2L and **−0.199 (−0.250, −0.156)** at 6L; last
−0.099 (2L) and −0.079 (6L), all 90% intervals below zero. The best-loss effect
is an order of magnitude larger than the ~0.02 the same term bought *with* the
encoder in #344.

## Why: without the encoder the CPC term no longer vanishes

![Training dynamics, log-log. Solid = no-encoder (blue base / red +CPC), dashed =
enc6 reference (#344). Panel 2 is the CPC term: the no-encoder +CPC curve (red)
stays elevated (~3) for the whole run, whereas with the encoder (orange dashed)
it collapses to ~0 by step ~1,000. Panels 1/6/8 (contrastive reference loss,
1−R², 1−AUC): the no-encoder +CPC arm (red) sits well below the no-encoder base
arm (blue) throughout.](plots/training_dynamics.png)

The mechanism is visible in the CPC term itself (panel 2). **With** the encoder
(#344) the unbounded bilinear `W₁` satisfied next-step prediction almost
immediately — the term fell below 10⁻³ by step ~1,000, contributed a vanishing
value, and only nudged the representation (hence its ~0.02 effect). **Without**
the encoder the same term stays around 3 for the entire run: predicting the next
patch-embedding from the forecaster context is genuinely harder, so the CPC
gradient keeps pressuring the representation end-to-end — and the no-encoder
+CPC arm holds a markedly lower contrastive reference loss, lower 1−R² and
near-perfect retrieval (1−AUC) than the no-encoder base arm throughout. That
sustained pressure is what lifts the no-encoder backbone back to encoder'd parity.

The checkpoints reflect this. The no-encoder **+CPC** arm is flat from best-loss
to last (2L 1.168→1.165, 6L 1.153→1.160) — the stability the term also showed
*with* the encoder. The no-encoder **base** arm behaves differently: its
best-loss checkpoint is *worse* than its last (2L 1.425→1.264, 6L 1.353→1.239),
so for this arm the contrastive-loss minimum does not coincide with best
transfer — but both checkpoints remain far behind every encoder'd and every
+CPC arm.

## What this means

Two takeaways. First, the #344 hypothesis — that removing the encoder denies a
positional shortcut and *improves* transfer — does not hold: the plain
contrastive backbone is reliably worse without the encoder, at every head and
checkpoint. The encoder contributes real transfer value to the contrastive
objective. Second, that value is **recoverable by the CPC auxiliary alone**: the
no-encoder +CPC backbone matches the encoder'd +CPC backbones within noise. The
CPC term and the encoder are, to first order, **substitutes** for one another on
this recipe — adding either one to the plain contrastive loss reaches the same
~1.15–1.18 GM band.

*Hypothesis (not tested here): because the CPC term stays active without the
encoder but vanishes with it (#344), the encoder + CPC combination is largely
redundant — and a CPC variant that remains active even with the encoder (e.g. a
bounded/temperatured score, or harder negatives per the side note above) might be
the lever that pushes past the current parity band.*

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
