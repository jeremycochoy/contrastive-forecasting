# #318 — Deny the positional shortcut: cross-series, same-step h↔h negatives

**Verdict — the hypothesis is falsified.** Adding a cross-series encoder-repulsion
term to deny a "positional shortcut" **hurts** GIFT-Eval transfer, and the more
precisely it targets the shortcut, the more it hurts. The **same-step** arm is
**+22%** worse than β (full-97 1.619 vs 1.327); the broader **all-time** arm
**+6.6%** (1.414). A 6-layer evaluation head rescues neither, so the regression
lives in the **backbone**, not readout capacity. What different series share at a
step is, on balance, **forecastably useful** — not a content-free code worth
destroying.

![full-97 GM summary](plots/gm_summary.png)
*Full GIFT-Eval (97 configs), GM-Relative MASE (lower better; 1.0 = seasonal
naive). Both cross-series h↔h arms sit above β and the v11c target at both head
sizes; same-step is worst and a 6L head does not move it.*

## Question

The backbone makes a forecast `f_t` resemble the next encoder state `h_{t+1}` and
differ from negatives. The strongest recipe, **β**, adds a within-series
**cross-time** negative — push `h_t` away from `h_l` for every other step `l`.
That term has an escape hatch: a model can satisfy "`h_t` ≠ `h_l`" by stamping
each step with a **positional fingerprint** that is *identical across all series*
— distinctness at zero cost in forecastable content. Two standing observations on
this line fit that story: bigger eval heads don't help, and more contrastive
training doesn't improve transfer.

**This card asks:** if we directly forbid the shortcut — repel, at every step,
what *different* series share there, `cos(h_{b,l}, h_{b',l})` (b ≠ b') — does the
backbone move distinctness onto series-specific (forecastable) content and
transfer better than β?

## Result

Each frozen backbone is scored with a fresh **2-layer** (small) and **6-layer**
quantile q-head, to separate backbone quality from readout capacity.

| backbone | head | full-97 GM | triage-11 GM |
|---|:--:|---:|---:|
| **β** (#309) | 2L | **1.3272** | 1.4836 |
| | 6L | 1.4489 | 1.5271 |
| **all-time** `xshh_allt` | 2L | 1.4143 | 1.6126 |
| | 6L | 1.4748 | 1.7028 |
| **same-step** `xshh` | 2L | 1.6194 | 1.9816 |
| | 6L | 1.6181 | 1.8762 |
| v11c (reference) | 2L | 1.292 | — |

*GM-Relative MASE = geometric mean over GIFT-Eval configs of (model MASE ÷
seasonal-naive MASE); 1.0 = seasonal-naive parity, lower is better. **full-97** =
all 97 configs (trusted); **triage-11** = an 11-config fast subset (noisier).*

1. **Both arms lose to β** — same-step +22%, all-time +6.6% (2L head). Adding
   *any* cross-series h↔h repulsion worsens transfer.
2. **Targeted hurts *more* than broad** — the reverse of the prediction. The
   same-step edge (the precise positional-code denier) is the **worst** arm;
   the broad all-time edge, which also repels across *different* steps, is
   *less* damaging.
3. **A bigger head does not rescue it** — same-step is flat 2L→6L
   (1.619 → 1.618), all-time worsens (1.414 → 1.475), and even β degrades at 6L.
   The regression is in the **backbone representation**, directly confirming the
   "bigger heads don't help" observation this card set out to probe.

### Per-domain — the damage is broad, not seasonal

![per-domain](plots/perdomain.png)
*Per-domain GM-Relative MASE (2L head, full-97). same-step is worse than β on
**all 7 domains**.*

| domain | β | same-step | all-time |
|---|---:|---:|---:|
| Econ/Fin | 2.018 | 2.734 | 3.514 |
| Web/CloudOps | 1.437 | 1.859 | 1.472 |
| Energy | 1.573 | 1.880 | 1.624 |
| Nature | 0.932 | 1.201 | 1.010 |
| Transport | 1.100 | 1.281 | 1.065 |
| Sales | 0.799 | 0.895 | 0.854 |
| Healthcare | 1.498 | 1.519 | 1.604 |

The issue anticipated collateral damage to *same-frequency seasonal phase*. There
is no seasonal signature: Energy is hurt *less* than the bursty Web/CloudOps and
Econ/Fin, so same-step's damage is **general**. all-time's gentler aggregate is
driven by the high-variance Web/CloudOps and Transport configs (1.86 → 1.47,
1.28 → 1.07), not a uniform gain — it is in fact *worse* than same-step on
Econ/Fin and Healthcare.

### Training dynamics — the contrastive task is learned; only transfer differs

![training curves](plots/training_curves.png)
*Loss minus each arm's InfoNCE floor (log–log). Floors differ — β 1.55 /
same-step 2.04 / all-time 5.27 — because the floor `log(1 + N·e^(−1/τ))` grows
with the negative-pool size N, which spans 52× across arms, so raw losses are not
comparable. Floor-subtracted, all three converge to a similar small excess
(β +0.58, same-step +0.64, all-time +0.57); retrieval AUC / Top-1 saturate within
a few hundred steps.*

Once the floor is accounted for, the contrastive objective converges cleanly and
near-identically for all three arms — yet transfer diverges by up to 22%. The
extra cross-series structure **is** learned; it is simply the **wrong** structure.
The "good contrastive metrics, indifferent transfer" decoupling is not shrunk by
this intervention — if anything it widens.

## Protocol

**Backbones.** same-step and all-time, each byte-identical to the #309 **β** arm
except `--loss-shape`: GRU patch-encoder → 6-layer causal encoder → 1-layer
forecaster (d = 128 bottleneck), AdamW β2 = 0.98, τ = 0.10, dropkey 0.70 shared,
fp16 body / fp32 residual + patch-embedding, EWMA RevNorm span 128, seed
20260520, 50k steps, batch 256, normalized-InfoNCE (`--pos-in-denominator`). One
RTX 4090 each. Loss shapes `cosine_similarity_batch_full_hh_negs_{xshh,xshh_allt}`
(`src/loss.py`), each pinned to an independent fp64 reference in
`tests/test_loss.py`.

**Evaluation.** Each frozen backbone → fresh 30k quantile q-head (transformer,
causal, forecast-len 16), trained at both 2 and 6 layers; GIFT-Eval strategy B4.
Byte-identical to #309/#315, so the numbers compare directly. References:
β full-97 1.3272; v11c (encoder-forecaster champion) 1.292; seasonal naive 1.0.

## The two arms, exactly

Both add **one** cross-series h↔h negative on top of β and both **remove** β's
adjacent `cos(h_t, h_{t+1})` (at single-channel training it is byte-for-byte the
`l = t+1` slice already inside β's within-series all-time term, so dropping it
de-duplicates). They differ only in the cross-series edge:

- **same-step** — `cos(h_{b,t}, h_{b',t})`, b ≠ b': repel only what
  *same-position* states of different series share — the shortcut and nothing
  else. Negative pool **1.79×** β.
- **all-time** — `cos(h_{b,t}, h_{b',l})`, b ≠ b', **∀ l**: the strict superset
  (same-step is its `l = t` slice); also repels different-position cross-series
  states. Pool **52×** β; ~2.5× the step time. Full multiplicities → [annex](#annex--exact-negative-set).

## Follow-ups (in flight)

- **6-layer forecaster** (1L → 6L *forecaster*, both arms): does a deeper
  predictor change the picture? — training/eval running; results append here.
- **Forked-continuation ARIMA** (data-side denial): inject **one** forked-ARIMA
  pair (2 samples) per 256-row batch — a pair sharing an exact prefix then
  diverging, so the same past maps to two plausible futures — while the other
  254 rows stay real, so transfer is **not** confounded by a synthetic mix.
  *(An earlier run mistakenly used a 50% synthetic mix; superseded by this
  minimal-injection re-run. Backbone re-training; 2L + 6L eval to follow.)*

## Caveats

Single seed per cell (prior work on this line: seed spread ≈ ±0.02, cf. #307; the
+6.6% / +22% margins are far outside that, though the all-time-vs-β ordering
deserves a second seed). A step-resolved GM-vs-step sweep was not run, so the
decoupling statement rests on the training curves plus the head-size result, not
a per-step transfer curve.

## Annex — exact negative set

All three arms share **one positive** (the InfoNCE numerator) `cos(h_{t+1}, f_t)`.
Multiplicities are **per anchor at C = 1** (the training config); the InfoNCE
denominator is then pooled over the batch, so the realised pool is `N = B · Σ`.
Counts verified by `src.loss._effective_negative_count`.

| negative family | repels | β | same-step | all-time |
|---|---|:--:|:--:|:--:|
| `xy` — adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — | — |
| `zy` — forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 | 1 |
| `hh_all` — within-series, all l | `cos(h_t, h_l)`, l≠t | T−1 | T−1 | T−1 |
| `cross_fe` — cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})`, b'≠b | B−1 | B−1 | B−1 |
| `xshh` — cross-series same-step h↔h | `cos(h_{b,t}, h_{b',t})`, b'≠b | — | B−1 | — |
| `xs_allt` — cross-series all-l h↔h | `cos(h_{b,t}, h_{b',l})`, b'≠b, ∀l | — | — | (B−1)·T |
| **per-anchor Σ** | | **T+B** | **T+2B−2** | **T+(B−1)(T+1)** |
| **pooled N** (B=256, T=64) | | **81,920** | **146,944** (1.79×) | **4,259,584** (52×) |

T = 64 is the **latent** length: the HF streamer crops windows to T_RAW = 1024
regardless of `--t-raw`, and 1024 ÷ patch-width(16) = 64 (verified by a model
forward). This holds for β/v11c too, so all numbers are on the same footing.
