<!-- DRAFT — method sections final; Result / Training-dynamics / Verdict filled after GIFT-Eval. -->
# Does a CPC InfoNCE auxiliary term help the stop-grad forecaster?

**Question.** The contrastive objective pairs the forecaster's context with the *next* encoder
embedding through a fixed-temperature cosine score. Contrastive Predictive Coding (van den Oord
et al. 2018) scores that same next-step prediction with a *learnable* log-bilinear map `W₁`
instead. Does adding CPC's InfoNCE term — summed equal-weight on top of the existing loss, with
its own learnable `W₁` and no stop-gradient — improve transfer for the two full-forecaster
stop-grad arms from the stop-grad-capacity report (enc3 and enc6), and does it change late-training
stability?

## The added term

Alongside the existing contrastive loss we add one CPC InfoNCE term (van den Oord et al. 2018,
Eq. 4, with horizon k = 1). For each step *t* it predicts the next encoder embedding `e_{t+1}`
from the autoregressive context `h_t` (the forecaster's output at *t*) through a new learnable
log-bilinear score `f(e_j, h_t) = exp(e_jᵀ W₁ h_t)`:

```
L_cpc = − log(  exp(e_{t+1}ᵀ W₁ h_t)  /  Σ_{e_j ∈ C}  exp(e_jᵀ W₁ h_t)  )
```

The candidate set `C` is the true next embedding `e_{t+1}` plus negatives — other sequences'
embeddings at the same step and the same sequence's embeddings at other steps. The positive sits
in the denominator, so the term is a normalized InfoNCE that is always ≥ 0. Three choices keep it
paper-faithful and orthogonal to the existing loss:

- **A new learnable `W₁`** (an `H×H` matrix) carries the score; the encoder embeddings stay
  unit-normalized, so `W₁` alone sets the scale. There is no temperature divisor, and the term's
  theoretical minimum is already 0 — the existing loss keeps its own temperature and floor.
- **No stop-gradient.** The paper trains the encoder and the autoregressive model jointly, so the
  gradient flows through both `h_t` and the targets `e_j`. The `+sg` in the arm names keeps
  governing only the existing contrastive term.
- **Equal weight.** Total loss = existing stop-grad contrastive loss **+** `L_cpc`.

## Result

<!-- RESULTS PENDING GIFT-Eval. Headline figure + inline GM table + verdict go here. -->

![Headline: GM-Relative MASE (baseline vs +CPC) and the CPC−baseline paired-bootstrap Δ with 90%
CI per arm × head × checkpoint.](plots/gm_summary.png)

| arm | 2-layer head, base / +CPC | 6-layer head, base / +CPC |
|---|--:|--:|
| enc3, full, sg — best-loss | _pending_ | _pending_ |
| enc3, full, sg — last | _pending_ | _pending_ |
| enc6, full, sg — best-loss | _pending_ | _pending_ |
| enc6, full, sg — last | _pending_ | _pending_ |

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97 forecasting
tasks, of a model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is
seasonal-naive. Each pairwise Δ carries a **paired-bootstrap** 90% interval — resample the 97-task
list with repeats and score both arms on each resample so per-task difficulty cancels.*

## Training dynamics

<!-- RESULTS PENDING. -->

![Training dynamics: contrastive reference loss, the CPC term, U_batch, and the forecast gap over
steps (CPC solid, baseline dashed).](plots/training_dynamics.png)

## Protocol

We train one backbone per arm, single seed, on one RTX 4090. Each arm reuses its same-arm baseline
**exactly** — the #339/#341 recipe (GRU patch-embedding, d_model 384 / 6 heads, a 6-layer
full-width forecaster, the crossfade-triplet allt·0.8% data mix, qk-norm, attention-output norm,
the `xshh_allt` contrastive loss with positive-in-denominator and floor subtraction, the
encoder-side positive stop-gradient `--stopgrad-positive-h`, τ 0.10, batch 1024, 12,500 steps,
seed 20260520) — with the single addition `--cpc-infonce-weight 1.0`. The two arms differ only in
encoder depth: **enc3** (3 layers, = report arm 2) and **enc6** (6 layers, = report arm 3).

To score a backbone we freeze it and train a fresh quantile forecasting head on top, once with two
transformer layers and once with six. We evaluate on GIFT-Eval's 97 tasks at two checkpoints: the
**best-loss** one (lowest smoothed contrastive loss, early in training) and the **last** one (step
12,500). The baselines are the published same-arm numbers from the stop-grad-capacity report; our
analysis reuses that report's GM and paired-bootstrap code unchanged, and reproduces its baseline
GMs to three decimals.
