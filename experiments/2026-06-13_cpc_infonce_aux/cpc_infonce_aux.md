# CPC for the forecaster: a late-training stabiliser, not a replacement for the contrastive loss

*Caveat: this is a partial CPC. Its InfoNCE negatives do not cover the full batch×time grid — each
anchor is contrasted against other sequences at the matched next step and the same sequence at
other steps, but not other sequences at other steps (see "The added term").*

**Question.** Our contrastive loss pairs the forecaster's context with the next encoder embedding
through a fixed-temperature cosine score; CPC (van den Oord et al. 2018) scores that same next-step
prediction with a *learnable* bilinear `W₁`. We compare three setups — the contrastive loss alone
(baseline), it **plus** a CPC term, and CPC **in place of** it (with a forecaster alignment) —
asking whether CPC helps as an addition, can stand alone, and changes late-training stability.

**Answer.** Added on top, CPC leaves the best-loss checkpoint untouched and reliably improves the
last, full-training one — a late-training stabiliser. In its place, training diverges and transfers
far worse. CPC is an addition, not a substitute.

## Result

![enc6 GM-Relative MASE for the three setups, per head × checkpoint. The CPC+align/no-main run
diverged, so its bars are capped at 1.25 with the true value labelled. The +CPC bars (green) sit at
or below the baseline, with the clearest gap at the last checkpoint.](plots/cpcalign_gm.png)

![Left: GM-Relative MASE, baseline vs +CPC, per arm × head × checkpoint — the best-loss bars are
level, every last-checkpoint +CPC bar sits below its baseline. Right: the +CPC − baseline
paired-bootstrap Δ with 90% interval — every best-loss cell straddles zero, every last-checkpoint
cell lies entirely below it.](plots/gm_summary.png)

GM-Relative MASE, all arms (bold = a reliable improvement over the same-cell baseline):

| arm · setup | 2L best | 2L last | 6L best | 6L last |
|---|--:|--:|--:|--:|
| enc3 · contrastive (baseline) | 1.177 | 1.180 | 1.159 | 1.163 |
| enc3 · + CPC | 1.185 | **1.153** | 1.158 | **1.144** |
| enc6 · contrastive (baseline) | 1.180 | 1.213 | 1.161 | 1.193 |
| enc6 · + CPC | 1.179 | **1.180** | 1.158 | **1.162** |
| enc6 · CPC + align, no contrastive | 1.99 | 1.38 | 1.43 | 1.21 |

*GM-Relative MASE: the geometric mean, over GIFT-Eval's 97 tasks, of a model's error divided by the
seasonal-naive forecast's — lower is better, 1.0 is seasonal-naive.*

## Training dynamics

![Training metrics, log-log (blues = no CPC, reds = main+CPC, green = CPC+align/no-main; enc3
solid, enc6 dashed). The added CPC term (top row, second panel) collapses toward zero within ~1,000
steps when the contrastive loss is present, but swings by orders of magnitude for the whole run
when it is absent — and that no-contrastive arm's reference loss, retrieval, and dimension-usage
curves stay poor and noisy throughout.](plots/training_dynamics.png)

So the added CPC term's *value* vanishes early, yet the +CPC arms hold better pretext diagnostics
than the baseline to the end — and that shows up in transfer only at the last checkpoint, not the
best-loss one.

## Protocol

One backbone per arm, single seed, on RTX 4090s. The baseline and +CPC arms reuse their same-arm
recipe from the stop-grad-capacity report (#341) **exactly** — GRU patch-embedding, d_model 384 / 6
heads, a 6-layer full-width forecaster, the crossfade-triplet allt·0.8% data mix, qk-norm,
attention-output norm, the `xshh_allt` contrastive loss (positive-in-denominator, floor
subtraction, encoder-side positive stop-gradient), τ 0.10, batch 1024, 12,500 steps, seed
20260520 — adding only `--cpc-infonce-weight 1.0` and differing in encoder depth (**enc3** = report
arm 2, **enc6** = report arm 3). The CPC+align/no-main arm drops the contrastive loss, keeping only
the CPC term and a BYOL-style alignment (forecaster output pulled toward the next encoder embedding,
that target stop-gradded); it ran 2-GPU (per-rank batch 512, global batch 1024 via gathered loss —
objective identical to the single-GPU baselines).

To score a backbone we freeze it, train a fresh quantile forecasting head on top (once with two
transformer layers, once with six), and evaluate on GIFT-Eval's 97 tasks at two checkpoints: the
**best-loss** one (lowest smoothed contrastive loss, within roughly the first quarter-to-half of
training) and the **last** one (step 12,500). Baselines are the published #341 numbers; the analysis
reuses that report's GM and paired-bootstrap code unchanged and reproduces its baseline GMs to three
decimals.

## The added term

The CPC term (van den Oord et al. 2018, Eq. 4, horizon k = 1) predicts the next encoder embedding
`e_{t+1}` from the autoregressive context `h_t` (the forecaster's output at *t*) through a new
learnable log-bilinear score:

```
L_cpc = − log(  exp(e_{t+1}ᵀ W₁ h_t)  /  Σ_{e_j ∈ C}  exp(e_jᵀ W₁ h_t)  )
```

The candidate set `C` is the true next embedding plus negatives. We draw negatives from two slices
only — other sequences' embeddings at the same next step, and the same sequence's embeddings at
other steps — not the full batch×time cross-product (the caveat above). The positive sits in the
denominator, so the term is a normalized InfoNCE, always ≥ 0. A new learnable `W₁` carries the
score (embeddings stay unit-normalized, so `W₁` alone sets the scale — no temperature divisor), and
there is **no stop-gradient**, matching the paper's joint training of encoder and autoregressive
model. As an addition, the total loss is the contrastive loss plus `L_cpc` at equal weight; the
BYOL alignment in the no-contrastive arm stop-grads its encoder target.

**Next:** #348 redoes these arms without the encoder, to deny a positional-embedding shortcut; #347
then tests whether this stabiliser rescues the bottleneck arm that collapsed at its last checkpoint
in #341.

## Annex — per-cell paired-bootstrap Δ (90% interval)

Δ = GM(arm) − GM(same-cell baseline); negative ⇒ the arm beats the baseline. The interval is a
**paired bootstrap** over the 97 tasks (resample with repeats, score both arms on each resample, so
per-task difficulty cancels) at a single training seed, as for the baselines — it reflects spread
across tasks, not across seeds.

| cell | +CPC Δ (90% CI) | CPC+align/no-main Δ (90% CI) |
|---|--:|--:|
| enc3 · 2L best | +0.008 (−0.007, +0.024) | — |
| enc3 · 2L last | **−0.027 (−0.043, −0.012)** | — |
| enc3 · 6L best | −0.000 (−0.015, +0.014) | — |
| enc3 · 6L last | **−0.019 (−0.033, −0.007)** | — |
| enc6 · 2L best | −0.001 (−0.016, +0.012) | +0.813 (+0.71, +0.93) |
| enc6 · 2L last | **−0.033 (−0.053, −0.014)** | +0.164 (+0.13, +0.21) |
| enc6 · 6L best | −0.003 (−0.011, +0.006) | +0.272 (+0.23, +0.33) |
| enc6 · 6L last | **−0.031 (−0.049, −0.014)** | +0.021 (−0.008, +0.052) |
