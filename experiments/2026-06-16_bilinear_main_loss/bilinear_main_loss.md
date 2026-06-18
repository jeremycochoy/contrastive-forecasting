# Learnable bilinear W vs temperature in the main contrastive loss

The main contrastive loss scores a forecast against a candidate embedding with a
temperature-scaled dot product, `exp(uᵀv / τ)`, τ = 0.10 fixed. The CPC
auxiliary term added in #344 instead scores with a *learnable log-bilinear*,
`exp(eᵀ W₁ h)`, and no temperature — the matrix `W₁` carries the scale — and it
was what made the no-encoder backbone competitive in #348. This experiment asks
whether giving the **main** term the same treatment helps: replace `exp(uᵀv / τ)`
with `exp(uᵀ W v)`, drop τ, make `W` learnable. Everything else is the #348 best
no-encoder **+ CPC** arm, unchanged.

*A backbone here is patch-embedding (a GRU) → forecaster (a 6-layer causal
transformer; no encoder stack, `--num-encoder-layers 0`), trained by the
contrastive objective to predict the next token's embedding. To score it we
freeze it, train a fresh quantile forecasting head, and evaluate on GIFT-Eval.
**GM-Relative MASE** is the geometric mean, over GIFT-Eval's 97 tasks, of a
model's error divided by the seasonal-naive forecast's error; lower is better,
1.0 = seasonal-naive. `W` is an H×H matrix (H = 384) initialised to (1/τ₀)·I so
training starts exactly at the τ = 0.10 baseline.*

## Result

![Left: GIFT-Eval full-97 GM-Relative MASE for the τ-scaled dot product (#348
+ CPC baseline) and the learnable bilinear W, per head (2L, 6L) × checkpoint
(best-loss, last). Right: paired-bootstrap Δ = GM(bilinear) − GM(τ) with 90% CI
per cell.](plots/gm_summary.png)

GM-Relative MASE (GIFT-Eval full-97; lower is better). The τ baseline is #348's
saved + CPC no-encoder arm.

| main-loss score | 2L head, best / last | 6L head, best / last |
|---|--:|--:|
| `exp(uᵀv/τ)`, τ=0.10 (#348 + CPC) | 1.168 / 1.165 | 1.153 / 1.160 |
| `exp(uᵀ W v)`, W learnable (this work) | **[PENDING]** / **[PENDING]** | **[PENDING]** / **[PENDING]** |

Paired-bootstrap Δ = GM(bilinear) − GM(τ baseline), 90% interval (resampling the
97-task list with repeats; negative ⇒ the bilinear GM is lower / better):

| | 2L best | 6L best | 2L last | 6L last |
|---|--:|--:|--:|--:|
| bilinear − τ | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

**Verdict: [PENDING — match / improve / worse, per the CIs above].**

## What the learned W became

![Effective temperature τ_eff = 1/mean(diag W) and the off-diagonal energy
fraction ||offdiag||_F / ||W||_F across training, from the periodic
checkpoints.](plots/W_evolution.png)

**[PENDING:** did W stay ≈ (1/τ₀)·I — meaning a scalar temperature was already
near-optimal — or did it move its effective temperature and grow off-diagonal /
asymmetric structure a scalar τ cannot express?**]**

## Training curves

![Four log-log panels — contrastive reference loss (τ=0.07, comparable across
arms), CPC term value, 1−R²_naive, 1−retrieval-AUC — bilinear W (solid) vs the
τ baseline (dashed).](plots/training_dynamics.png)

`loss_tau_ref` is the CPC-free τ=0.07 contrastive reference, computed identically
for both arms, so it is directly comparable regardless of each arm's training
objective. **[PENDING: one sentence on whether the bilinear arm reaches a
comparable/lower reference loss, 1−R², 1−AUC.]**

## Protocol

One backbone, single seed (20260520), one RTX 4090. The recipe is the #348 best
no-encoder + CPC arm, byte-for-byte: GRU patch-embedding, d_model 384 / 6 heads,
a 6-layer full-width forecaster, no encoder stack, the crossfade-triplet
allt·0.8% data mix, qk-norm, attention-output norm, the `xshh_allt` contrastive
loss with positive-in-denominator and floor subtraction, the encoder-side
positive stop-gradient, the CPC InfoNCE auxiliary at weight 1.0, batch 1024,
12,500 steps. The **only** change is the main loss's similarity:
`exp(uᵀv / τ)` → `exp(uᵀ W v)`.

`W` is an H×H matrix (H = 384), one per run, initialised to (1/τ₀)·I with
τ₀ = 0.10 — so step 0 reproduces the τ baseline exactly — and excluded from
weight decay, like the fixed scalar temperature it replaces. To score a backbone
we freeze it and train a fresh quantile forecasting head, once with two
transformer layers and once with six, and evaluate on GIFT-Eval's 97 tasks at
the best-loss checkpoint (lowest smoothed contrastive loss) and the last
checkpoint (step 12,500) — the exact #348 head + eval protocol. The τ baseline
numbers come from #348's saved + CPC arm: the same code (this branch is cut from
#348), machine, seed, and recipe, scored by the same eval, so the comparison
changes only the main-loss similarity.

## The change

For two L2-normalised vectors `u` (a forecast) and `v` (a candidate embedding),
the main loss previously scored every pair with a temperature-scaled dot product
and assembled a normalized InfoNCE over the positive and the cross-channel,
cross-batch, within-series-all-time and cross-series-all-time negatives:

```
score(u, v) = uᵀv / τ           (τ = 0.10, fixed)
```

This experiment replaces it with a learnable log-bilinear, the same form the CPC
term uses, with no temperature:

```
score(u, v) = uᵀ W v            (W ∈ ℝ^{H×H}, learnable, no τ)
```

`W` multiplies the autoregressive forecast `f_t` — the anchor of the InfoNCE —
in the positive `s(f_t, h_{t+1})` and in its cross-batch negatives `s(f_t, h[b'])`
alike, so both score `s(f, h) = fᵀ Wᵀ h` with the same `W` on the same `f_t`
(the form the CPC term already uses, `fᵀ W₁ h`). Putting `W` on the target side
of the positive instead would make the numerator and denominator score different
maps once `W` is asymmetric, so the discrimination would not train. The
latent-uniformity negatives (h↔h, which carry no forecast) keep `W` on their
latent anchor. At `W = (1/τ)·I` every term equals the τ baseline, so the bilinear
is a strict generalisation of the temperature — it can rescale (a scalar
temperature) and additionally learn off-diagonal coupling between embedding
dimensions and an asymmetric forecast-vs-target metric. The CPC auxiliary term
keeps its own separate `W₁`, unchanged.
