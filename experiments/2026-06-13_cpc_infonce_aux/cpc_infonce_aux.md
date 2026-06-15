# A CPC InfoNCE auxiliary term stabilises late training

**Question.** The contrastive objective pairs the forecaster's context with the *next* encoder
embedding through a fixed-temperature cosine score. Contrastive Predictive Coding (van den Oord
et al. 2018) scores that same next-step prediction with a *learnable* log-bilinear map `W₁`
instead. Does adding CPC's InfoNCE term — summed equal-weight on top of the existing loss, with
its own learnable `W₁` and no stop-gradient — improve transfer for the two full-forecaster
stop-grad arms from the stop-grad-capacity report (enc3 and enc6), and does it change
late-training stability?

**Answer.** It does not move the best-loss checkpoint (4/4 cells neutral), but it **reliably
improves the last, full-training checkpoint** (4/4 cells better, every 90% interval below zero) —
reversing the late-training degradation the baselines show. The CPC term is a late-training
stabiliser, not a peak-performance lever. A further ablation (below) shows the CPC term is an
*addition* to the contrastive loss, not a replacement: training on CPC + a separate forecaster
alignment with the contrastive loss removed is reliably and substantially worse.

## Result

![Left: GM-Relative MASE, baseline (grey) vs + CPC (green), for each arm × head × checkpoint.
The best-loss bars are level; every last-checkpoint green bar sits clearly below its grey
baseline. Right: the CPC − baseline paired-bootstrap Δ with 90% interval per cell — all four
best-loss cells straddle zero (grey, ns), all four last-checkpoint cells lie entirely below zero
(green, reliably better).](plots/gm_summary.png)

The split is clean and holds across both encoder depths and both head sizes: at the **best-loss**
checkpoint the term is neutral; at the **last** checkpoint it reliably helps.

| arm | 2-layer head, base / +CPC | 6-layer head, base / +CPC |
|---|--:|--:|
| enc3, full, sg — best-loss | 1.177 / 1.185 | 1.159 / 1.158 |
| enc3, full, sg — **last** | 1.180 / **1.153** | 1.163 / **1.144** |
| enc6, full, sg — best-loss | 1.180 / 1.179 | 1.161 / 1.158 |
| enc6, full, sg — **last** | 1.213 / **1.180** | 1.193 / **1.162** |

Paired-bootstrap Δ = GM(+CPC) − GM(baseline), 90% interval (negative ⇒ CPC better):

| cell | best-loss Δ (CI) | last Δ (CI) |
|---|--:|--:|
| enc3 · 2L | +0.008 (−0.007, +0.024) · ns | **−0.027 (−0.043, −0.012)** |
| enc3 · 6L | −0.000 (−0.015, +0.014) · ns | **−0.019 (−0.033, −0.007)** |
| enc6 · 2L | −0.001 (−0.016, +0.012) · ns | **−0.033 (−0.053, −0.014)** |
| enc6 · 6L | −0.003 (−0.011, +0.006) · ns | **−0.031 (−0.049, −0.014)** |

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97 forecasting
tasks, of a model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is
seasonal-naive. Each pairwise Δ carries a **paired-bootstrap** 90% interval — resample the 97-task
list with repeats and score both arms on each resample so per-task difficulty cancels.*

The mechanism is visible in the baselines: their last checkpoint is *worse* than their best
(enc6·2L 1.180→1.213, enc6·6L 1.161→1.193; enc3 drifts up too). Adding the CPC term removes that
drift — the last checkpoint lands at or below the best — so the gain is concentrated entirely in
late training.

**Follow-up (#347):** the stabilisation should be most visible where late training is worst — the
enc6 + bottleneck + stop-grad arm (#341 arm 4) that *collapsed* at the last checkpoint
(GM ~2.2 vs ~1.18 at best-loss). #347 tests whether the CPC term rescues it.

## Training dynamics: the term vanishes, the representation does not

![Training metrics, log-log (blues = no CPC, reds = main + CPC, green = the CPC+align/no-main arm
below; enc3 solid, enc6 dashed). Panel 1 is the contrastive reference loss (normalized InfoNCE at
τ=0.07, logged for every arm). The CPC term (top row, 2nd panel) plunges from its early peak (~19)
to <10⁻³ within ~1,000 steps for the main+CPC arms; the CPC arms' reference loss, ratio-gap, 1−R²,
and 1−AUC all sit below the baselines, with higher time-wise dimension usage
(U_temporal).](plots/training_dynamics.png)

The unbounded bilinear `W₁` drives the CPC loss to ~0 almost immediately — by step ~1,000 it
contributes a vanishing *value*. But its early gradient reshapes the representation: the CPC arms
reach a markedly lower contrastive reference loss (≈10.2 vs ≈12.3–12.4), a lower ratio-gap
(≈0.40 vs 0.62–0.85, roughly halved for enc3), near-perfect retrieval (AUC→1.0) and higher
time-wise dimension usage than the baselines, and they hold those through to the end. So the
pretext task is learned better — but, per the table above, that pretext improvement does not raise
the best-loss transfer; it surfaces only as a steadier, better last checkpoint.

## Ablation: can CPC + a separate forecaster loss replace the contrastive loss? No

The auxiliary CPC term helps, so a natural question is whether CPC plus a *separate* forecaster
loss could stand in for the contrastive objective entirely. We trained one more enc6 arm on
**CPC + the BYOL align term (encoder target stop-gradded), with the main contrastive loss removed**
— same recipe otherwise. (To keep it comparable at a fraction of the wall-clock, this arm ran
2-GPU DDP at per-rank batch 512: the loss pools the gathered global batch of 1024, identical to the
single-GPU baselines.)

![enc6 GM-Relative MASE: baseline (contrastive, grey) vs main+CPC (green) vs CPC+align/no-main
(red), per head × checkpoint (bars clipped at 1.45, true value labelled). The CPC+align/no-main
bars tower over both — worst at the best-loss 2L cell (1.99).](plots/cpcalign_gm.png)

Removing the contrastive loss is **reliably and substantially worse**, not better:

| cell | baseline | + CPC | CPC+align, no main | Δ vs baseline (90% CI) |
|---|--:|--:|--:|--:|
| enc6 · 2L best | 1.180 | 1.179 | **1.993** | +0.813 (+0.71, +0.93) |
| enc6 · 2L last | 1.213 | 1.180 | **1.378** | +0.164 (+0.13, +0.21) |
| enc6 · 6L best | 1.161 | 1.158 | **1.432** | +0.272 (+0.23, +0.33) |
| enc6 · 6L last | 1.193 | 1.162 | **1.214** | +0.021 (−0.01, +0.05) · ns |

Three of four cells are reliably worse (the fourth only ties, and only because the baseline's own
6L-last is already degraded); against the main+CPC arm all four are reliably worse. The training
dynamics show why: without the contrastive loss anchoring the representation, the CPC term never
settles — it oscillates ~0.01↔10 for the whole run (green, top-row 2nd panel), and the green arm's
contrastive reference loss, 1−R², and 1−AUC stay elevated and noisy throughout. **The contrastive
objective carries the representation; CPC and a forecaster alignment are useful additions to it,
not a replacement for it.**

## Protocol

We train one backbone per arm, single seed, on one RTX 4090. Each arm reuses its same-arm
baseline **exactly** — the #339/#341 recipe (GRU patch-embedding, d_model 384 / 6 heads, a 6-layer
full-width forecaster, the crossfade-triplet allt·0.8% data mix, qk-norm, attention-output norm,
the `xshh_allt` contrastive loss with positive-in-denominator and floor subtraction, the
encoder-side positive stop-gradient `--stopgrad-positive-h`, τ 0.10, batch 1024, 12,500 steps,
seed 20260520) — with the single addition `--cpc-infonce-weight 1.0`. The two arms differ only in
encoder depth: **enc3** (3 layers, = report arm 2) and **enc6** (6 layers, = report arm 3).

To score a backbone we freeze it and train a fresh quantile forecasting head on top, once with two
transformer layers and once with six. We evaluate on GIFT-Eval's 97 tasks at two checkpoints: the
**best-loss** one (the lowest smoothed contrastive loss, reached within roughly the first
quarter-to-half of training) and the **last** one (step 12,500). Baselines are the published same-arm numbers from the stop-grad-capacity report; the
analysis reuses that report's GM and paired-bootstrap code unchanged and reproduces its baseline
GMs to three decimals.

## The added term

Alongside the existing contrastive loss we add one CPC InfoNCE term (van den Oord et al. 2018,
Eq. 4, horizon k = 1). For each step *t* it predicts the next encoder embedding `e_{t+1}` from the
autoregressive context `h_t` (the forecaster's output at *t*) through a new learnable log-bilinear
score `f(e_j, h_t) = exp(e_jᵀ W₁ h_t)`:

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
