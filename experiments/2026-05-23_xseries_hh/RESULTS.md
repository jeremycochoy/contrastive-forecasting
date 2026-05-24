# #318 — Deny the positional shortcut: cross-series, same-step h↔h negatives

> **Status: results landing.** Two backbones are being trained on elisa — the
> same-step arm and its all-time sibling — then each gets the q-head +
> GIFT-Eval matrix. Tables and the verdict below update as cells complete; the
> design, protocol, and loss implementations are final.

## The question

Our contrastive backbone is trained to make a forecast `f_t` look like the
*next* encoder state `h_{t+1}` and unlike a set of negatives. The strongest
recipe on this line, **β**, includes a within-series **cross-time** negative:
push `h_t` away from `h_l` for every other step `l`. That term has a cheap
escape hatch. A model can satisfy "`h_t` differs from `h_l`" by stamping each
step with a **positional fingerprint** — a code that says *"I am step 17"* —
that is **identical across all series**. Distinctness is achieved, but it costs
nothing in forecastable structure, and the encoder is then free to discard the
content that actually predicts the future.

Two standing observations on this line are consistent with that escape hatch
being taken: **bigger evaluation heads do not help**, and **more contrastive
training does not improve (often worsens) transfer**. Both are what you'd see
if the backbone were spending capacity on a content-free positional code.

**This card asks:** if we directly forbid the shortcut — repel, at *every*
step `l`, what *different* series share at that step (`cos(h_{b,l}, h_{b',l})`,
b ≠ b') — does the backbone move distinctness onto series-specific
(forecastable) content, and so transfer better than β?

## The idea, and a design fork

β has **no** cross-series h↔h term today (only a cross-series f↔h term). We add
one as a single, isolated edit on top of β — but *which* pairs to repel has a
fork, and we test **both arms**:

- **same-step** (`…_xshh`): `cos(h_{b,l}, h_{b',l})`, b ≠ b' — **same step on
  both sides**; repel only what *same-position* states of different series
  share. At a fixed step the only thing different series share is the positional
  component, so this targets the shortcut **and nothing else**. Cost is
  **B²·T** (each of the B·T states × B−1 same-step partners) — the size of β's
  existing cross-batch term, i.e. cheap.
- **all-time** (`…_xshh_allt`): `cos(h_{b,t}, h_{b',l})`, b ≠ b', **∀ l** — the
  cross-series analog of β's within-series all-time term. Same-step is its
  `l = t` slice, so this is the strict **superset**: it also repels states at
  *different* positions across series. Cost is **B²·T²** (each anchor sees all
  T steps of every other series) — the only expensive piece. Tests whether the
  broad cross-series repulsion beats the targeted one — or instead over-repels
  genuinely shared structure (e.g. same-frequency seasonal phase at different
  absolute steps).

Crucially, **both arms keep β's within-series all-time term**
`cos(h_{b,t}, h_{b,l})` ∀ `l` ≠ t — that is where the *different-`l`*
(cross-time) comparisons come from, *within* a series. So the fork changes
**only** the cross-series edge — same-step (`l = t`) vs all-`l`; the different-`l`
structure within a series is identical in both. Both arms also **remove** the
duplicated adjacent negative `cos(h_t, h_{t+1})` — at the single-channel
training config it is byte-for-byte the `l = t+1` slice already inside that
within-series all-time term, so dropping it de-duplicates rather than weakening
the objective. Everything else is byte-for-byte β.

The loss shapes are `cosine_similarity_batch_full_hh_negs_xshh` and
`…_xshh_allt` (`src/loss.py`); each exact term set is pinned against an
independent fp64 reference in `tests/test_loss.py` (`TestCrossSeriesSameStepHH`,
`TestCrossSeriesAllTimeHH`). The all-time edge's full `[B,B,T-1,T]` Gram is
≈ 264M entries ≈ **1 GB** at B=256, T=64. The implementation still computes it
in source-batch chunks with a gradient-checkpointed backward — a conservative
memory strategy that *would* matter at larger T but is not the binding
constraint here. The real cost is **compute**: the cross-series × cross-time
similarities are inherently `B²·T²` dot-products, measured at **~2.5× the
same-step step time** (6.3 → 2.5 sps).

### Exact negative set per arm

All three share **one positive** (the InfoNCE numerator) `cos(h_{t+1}, f_t)`.
The negatives differ only in the two edited families. Multiplicities are
**per anchor at C = 1** (the training config; the cross-channel families scale
with C in general — `xy = C`, `xx = C−1`, `zy = C`). The cross-series / f↔h
families are already summed over the `b' ≠ b` partners, and the InfoNCE
denominator is then pooled over the whole batch, so the realised pool size is
`N = B · Σ`. Counts verified by `src.loss._effective_negative_count`.

| negative family | repels (b, c fixed unless noted) | β | same-step | all-time |
|---|---|:--:|:--:|:--:|
| `xy` — adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — | — |
| `xx` — cross-channel h↔h | `cos(h_t^{c}, h_t^{c'})`, c≠c' | 0 | 0 | 0 |
| `zy` — forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 | 1 |
| `hh_all` — within-series, **all** l | `cos(h_t, h_l)`, l≠t | T−1 | T−1 | T−1 |
| `cross_fe` — cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})`, b'≠b | B−1 | B−1 | B−1 |
| `xshh` — cross-series, **same-step** h↔h | `cos(h_{b,t}, h_{b',t})`, b'≠b | — | B−1 | — |
| `xs_allt` — cross-series, **all-l** h↔h | `cos(h_{b,t}, h_{b',l})`, b'≠b, ∀l | — | — | (B−1)·T |
| **per-anchor Σ** | | **T+B** | **T+2B−2** | **T+(B−1)(T+1)** |
| **pooled N = B·Σ** (B=256, T=64) | | **81,920** | **146,944** (1.79×) | **4,259,584** (52×) |

Reading the table: the **same-step** arm swaps β's one adjacent `xy` term for
`B−1` same-step cross-series partners (net +B−2 per anchor, **1.79×** the pool);
the **all-time** arm instead adds `(B−1)·T` cross-series partners (**52×** the
pool). The **within-series all-time `hh_all` term (the "different-l"
comparisons) is identical in all three** — only the cross-series edge changes
across the fork. (T = 64 is the latent length: the HF streamer crops windows to
T_RAW=1024 regardless of `--t-raw`, and 1024 // W(16) = 64 — verified by a model
forward; this holds for β/v11c too, so all numbers are on the same footing.)

## Protocol

**Backbones.** Two — the same-step and all-time arms — each byte-identical to
the #309 **β** arm except `--loss-shape`: GRU patch-encoder → 6-layer causal
encoder → 1-layer forecaster with a d = 128 bottleneck, AdamW β2 = 0.98,
temperature τ = 0.10, dropkey 0.70 shared, fp16 body / fp32 residual +
patch-embedding, EWMA RevNorm span 128, seed 20260520, 50k steps, global batch
256, normalized-InfoNCE objective (`--pos-in-denominator`). Single 4090 each.

**Evaluation.** Each frozen backbone gets a fresh **30k quantile q-head**
(transformer, causal, forecast-len 16) and is scored on GIFT-Eval. To separate
*backbone* quality from *head* capacity we train **two** heads — a **2-layer**
(small) and a **6-layer** — exactly as in #315/#316. Protocol is byte-identical
to #309/#315 so the numbers compare directly.

**Metric.** *GM-Relative MASE* = geometric mean over GIFT-Eval configs of
(model MASE ÷ seasonal-naive MASE). **Lower is better; 1.0 = seasonal naive.**
Reported on **full-97** (all 97 configs, the trusted metric) and **triage-11**
(11-config fast subset, noisier). It measures point-forecast accuracy relative
to the seasonal-naive baseline; it does **not** measure calibration or
probabilistic sharpness.

**References.** β (#309): full-97 = **1.3272**, triage-11 = **1.4836** (2L head).
v11c (the encoder-forecaster champion): full-97 = **1.292**. Seasonal naive = 1.0.

## Results

### Headline — frozen-backbone GM-Relative MASE

_(landing)_

| Backbone | head | full-97 GM | triage-11 GM |
|----------|:----:|-----------:|-------------:|
| **β** (#309) | 2L | 1.3272 | 1.4836 |
| **β** (#309) | 6L | 1.4489 | 1.5271 |
| **xshh same-step** | 2L | **1.6194** | 1.9816 |
| **xshh same-step** | 6L | 1.6181 | 1.8762 |
| **xshh all-time** | 2L | **1.4143** | 1.6126 |
| **xshh all-time** | 6L | _(landing)_ | 1.7028 |
| v11c (ref) | 2L | 1.292 | — |

> **Same-step hurts, and a bigger head does not rescue it.** full-97 =
> **1.6194** (2L) / **1.6181** (6L) vs β 1.3272 / v11c 1.292 — +22% worse, and
> the 6L q-head is **no better than the 2L** (1.618 ≈ 1.619). So the regression
> lives in the **backbone**, not head capacity — directly consistent with the
> "bigger heads don't help" observation this card set out to probe. The
> cross-series same-step repulsion **does not deny a free positional shortcut;
> it removes broadly-useful content** (worse on all 7 domains, see below).
>
> **Both arms are worse than β, and the broad (all-time) repulsion hurts far
> *less* than the targeted (same-step) one** — the opposite of what the
> hypothesis predicted (it expected the *targeted* same-step edge to help most).
> full-97 2L: all-time **1.4143** (+6.6% vs β 1.3272) vs same-step **1.6194**
> (+22%); v11c 1.292. So adding cross-series h↔h negatives **hurts transfer**,
> and *concentrating* it on the same step hurts more, not less. (all-time 6L is
> re-running after an external job transiently filled GPU 1; β 6L full-97 and
> GM-vs-step still landing.)

![gm summary](plots/gm_summary.png)

### Does more contrastive training stop helping?

_(landing — full-97 GM-Relative MASE, 2L head, at 20k / 35k / 50k for xshh and β.)_

![gm vs step](plots/gm_vs_step.png)

### Per-domain transfer (2L head, full-97)

The same-step regression is **broad, not seasonal-specific** — arm A is worse
than β on **all 7 domains**, improving none (per-domain GM-Relative MASE):

| domain | same-step | β | Δ |
|--------|----------:|----:|----:|
| Econ/Fin | 2.734 | 2.018 | +0.716 |
| Web/CloudOps | 1.859 | 1.437 | +0.422 |
| Energy | 1.880 | 1.573 | +0.307 |
| Nature | 1.201 | 0.932 | +0.270 |
| Transport | 1.281 | 1.100 | +0.181 |
| Sales | 0.895 | 0.799 | +0.096 |
| Healthcare | 1.519 | 1.498 | +0.020 |

The issue anticipated *collateral damage to same-frequency seasonal phase* on
strongly-seasonal domains. The data does **not** match that: the most-seasonal
**Energy** domain is hurt *less* (+0.31) than the bursty **Web/CloudOps**
(+0.42) and **Econ/Fin** (+0.72). The degradation is general — the repulsion
removes broadly-useful content, not specifically a seasonal/positional code.
(2L head; arm B and the 6L head are still pending.)

![per-domain](plots/perdomain.png)

### Training dynamics

All arms share every hyperparameter, so the curves isolate the loss change.
**Raw losses are not comparable across arms**: the normalized-InfoNCE floor
`log(1 + N·e^(−1/τ))` grows with the negative count N, which differs ~52×
(β 82k, same-step 147k, all-time 4.26M → floors **1.55 / 2.04 / 5.27**). The
loss panel therefore plots each arm's loss **minus its own floor** (log-x,
symlog-y).

- Floor-subtracted, **all three converge to a similar small excess above their
  floors** — β **+0.58**, same-step **+0.64**, all-time **+0.57** at 50k. So
  the cross-series negatives (same-step or all-time) barely change the
  *contrastive* convergence once you account for the larger negative pool: the
  same-step arm's +0.55 *raw*-loss gap over β is almost entirely the higher
  floor (more negatives), not worse optimisation. The big differences are in
  **transfer**, not the contrastive task.
- AUC and Top-1 retrieval (shown as 1−AUC / 1−Top1, log-log) saturate for all
  arms within a few hundred steps.

![training curves](plots/training_curves.png)

## What we learned

_(verdict landing — will state plainly whether the cross-series same-step
repulsion beats β / v11c, whether the decoupling shrank, and any collateral
damage, with single-seed caveats flagged.)_
