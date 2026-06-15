# CPC for the forecaster: a late-training stabiliser, not a replacement for the contrastive loss

**Question.** Our backbone trains on a contrastive loss that pairs the forecaster's context with
the *next* encoder embedding through a fixed-temperature cosine score. Contrastive Predictive
Coding (van den Oord et al. 2018) scores that same next-step prediction with a *learnable*
log-bilinear map `W₁` instead. We test how that idea fits our objective in three setups: the
contrastive loss alone (the baseline), the contrastive loss **plus** a CPC term, and CPC **in
place of** the contrastive loss (paired with a forecaster alignment). Does CPC help as an addition,
can it stand alone, and what does it do to late-training stability?

**Answer.** Added on top, CPC is a late-training stabiliser: it leaves the best-loss checkpoint
untouched and reliably improves the last, full-training checkpoint. Used instead of the contrastive
loss, it does not train — the run diverges and transfers far worse. CPC earns its place as an
addition to the contrastive objective, not a substitute for it.

## Result

![enc6 GM-Relative MASE for the three setups, per head × checkpoint. The CPC+align/no-main run
diverged, so its bars are capped at 1.25 with the true value labelled. The +CPC bars (green) sit at
or below the baseline, with the clearest gap at the last checkpoint.](plots/cpcalign_gm.png)

Two things stand out. **Dropping the contrastive loss** — training on CPC plus a forecaster
alignment alone — does not work: the run diverges and its frozen backbone forecasts well above
seasonal-naive. **Keeping the contrastive loss and adding CPC** leaves the best-loss checkpoint
unchanged but pulls the last checkpoint down, undoing the upward drift the baseline shows between
its best and last checkpoints.

![Left: GM-Relative MASE, baseline vs +CPC, for each arm × head × checkpoint. The best-loss bars
are level; every last-checkpoint +CPC bar sits below its baseline. Right: the +CPC − baseline
paired-bootstrap Δ with 90% interval — every best-loss cell straddles zero, every last-checkpoint
cell lies entirely below it.](plots/gm_summary.png)

The late-training gain is reliable across both encoder depths and both head sizes: at the best-loss
checkpoint the term is neutral, and at the last checkpoint every interval falls below zero.

GM-Relative MASE, all arms (bold = a reliable improvement over the same-cell baseline):

| arm · setup | 2L best | 2L last | 6L best | 6L last |
|---|--:|--:|--:|--:|
| enc3 · contrastive (baseline) | 1.177 | 1.180 | 1.159 | 1.163 |
| enc3 · + CPC | 1.185 | **1.153** | 1.158 | **1.144** |
| enc6 · contrastive (baseline) | 1.180 | 1.213 | 1.161 | 1.193 |
| enc6 · + CPC | 1.179 | **1.180** | 1.158 | **1.162** |
| enc6 · CPC + align, no contrastive | 1.99 | 1.38 | 1.43 | 1.21 |

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97 forecasting tasks,
of a model's error divided by the seasonal-naive forecast's error — lower is better, 1.0 is
seasonal-naive. Each comparison carries a **paired-bootstrap** 90% interval: resample the 97-task
list with repeats and score both arms on each resample, so per-task difficulty cancels. The
interval is over tasks at a **single training seed** (as for the baselines); seed-level replication
was not run, so "reliably" here means stable across tasks, not across seeds. Per-cell intervals are
in the annex.*

## Training dynamics

![Training metrics, log-log (blues = no CPC, reds = main+CPC, green = CPC+align/no-main; enc3
solid, enc6 dashed). The added CPC term (top row, second panel) collapses toward zero within the
first ~1,000 steps for the main+CPC arms but swings by orders of magnitude for the whole run when
the contrastive loss is absent.](plots/training_dynamics.png)

The dynamics track both halves of the result. With no contrastive loss, the green arm never
settles: its CPC term swings by orders of magnitude, and its reference loss, retrieval, and
dimension usage stay poor and noisy for the whole run. With the contrastive loss kept, the added
CPC term's value falls to near zero almost immediately; over the rest of training the +CPC arms
nonetheless show a lower contrastive reference loss, a smaller forecast gap, near-perfect retrieval,
and more time-wise dimensions in use than the baseline, held to the end. These pretext diagnostics
differ from the baseline while the best-loss transfer does not (table above); the only transfer
difference is the lower last checkpoint.

## Protocol

We train one backbone per arm, single seed, on RTX 4090s. The baseline and +CPC arms reuse their
same-arm recipe from the stop-grad-capacity report (#341) **exactly** — GRU patch-embedding,
d_model 384 / 6 heads, a 6-layer full-width forecaster, the crossfade-triplet allt·0.8% data mix,
qk-norm, attention-output norm, the `xshh_allt` contrastive loss with positive-in-denominator and
floor subtraction, the encoder-side positive stop-gradient, τ 0.10, batch 1024, 12,500 steps, seed
20260520 — differing only by the single addition `--cpc-infonce-weight 1.0` and the encoder depth
(**enc3** = report arm 2, **enc6** = report arm 3). The CPC+align/no-main arm keeps that recipe but
drops the contrastive loss, keeping only the CPC term and a BYOL-style alignment (the forecaster
output pulled toward the next encoder embedding, that target stop-gradded); it ran on two GPUs
(per-rank batch 512, global batch 1024 via gathered loss — identical objective to the single-GPU
baselines).

To score a backbone we freeze it and train a fresh quantile forecasting head on top, once with two
transformer layers and once with six, then evaluate on GIFT-Eval's 97 tasks at two checkpoints: the
**best-loss** one (the lowest smoothed contrastive loss, reached within roughly the first
quarter-to-half of training) and the **last** one (step 12,500). Baselines are the published
same-arm numbers from #341; the analysis reuses that report's GM and paired-bootstrap code
unchanged and reproduces its baseline GMs to three decimals.

## The added term

The CPC term (van den Oord et al. 2018, Eq. 4, horizon k = 1) predicts the next encoder embedding
`e_{t+1}` from the autoregressive context `h_t` (the forecaster's output at *t*) through a new
learnable log-bilinear score:

```
L_cpc = − log(  exp(e_{t+1}ᵀ W₁ h_t)  /  Σ_{e_j ∈ C}  exp(e_jᵀ W₁ h_t)  )
```

The candidate set `C` is the true next embedding plus negatives — other sequences' embeddings at
the same step, and the same sequence's embeddings at other steps. The positive sits in the
denominator, so the term is a normalized InfoNCE that is always ≥ 0. A new learnable `W₁` carries
the score (the embeddings stay unit-normalized, so `W₁` alone sets the scale — no temperature
divisor); there is **no stop-gradient**, matching the paper's joint training of the encoder and the
autoregressive model. When used as an addition, the total loss is the existing contrastive loss
plus `L_cpc`, equal weight; the BYOL alignment term in the no-contrastive arm stop-grads its
encoder target.

**Next:** #348 redoes these arms without the encoder, to deny a positional-embedding shortcut;
#347 then tests whether this late-training stabiliser rescues the bottleneck arm that collapsed at
its last checkpoint in #341.

## Annex — per-cell paired-bootstrap Δ (90% interval)

Δ = GM(arm) − GM(same-cell baseline); negative ⇒ the arm beats the baseline.

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
