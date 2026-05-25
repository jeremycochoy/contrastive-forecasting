# #318 — Deny the positional shortcut: loss-side vs data-side

**Verdict.** There are two ways to forbid the shortcut, and they behave oppositely.
Through the **loss** (cross-series h↔h repulsion) it **hurts** — same-step +22%,
all-time +6.6% vs β at the 2L head. Through the **data** (forked continuations:
the same past, two divergent futures) it is sharply **head-dependent**: the worst
arm with a small 2L head (1.637), but **the best backbone here with a 6L head**
(full-97 **1.4065**, beating β·6L 1.449 and all-time·6L 1.475). That forked number
is measured at a 50% synthetic mix, which entangles a data-distribution shift with
the fork itself; a minimal-injection re-run (one forked pair per 256-row batch) to
isolate the fork is in flight. Nothing yet beats β·2L (1.3272) overall.

![full-97 GM summary](plots/gm_summary.png)
*Full GIFT-Eval (97 configs), GM-Relative MASE (lower better; 1.0 = seasonal
naive). Loss-side arms (blue/green) sit above β at both heads. The data-side
forked arm (orange) swings from worst (2L) to best-of-6L (1.4065) — a head
dependence no other arm shows. The corrected 2/batch forked arm (brown) appears
here once it lands.*

## Question

The backbone makes a forecast `f_t` resemble the next encoder state `h_{t+1}` and
differ from negatives. The strongest recipe, **β**, adds a within-series
**cross-time** negative — push `h_t` from `h_l` for every other step `l`. That
term has an escape hatch: stamp each step with a **positional fingerprint**
*identical across all series*, and "`h_t` ≠ `h_l`" is satisfied at zero cost in
forecastable content. Two standing observations fit it: bigger eval heads don't
help, and more contrastive training doesn't improve transfer.

**Can we forbid the shortcut, and does it help?** We test denial through two
channels: the **loss** (repel what different series share at a step) and the
**data** (feed pairs whose identical past has divergent futures, so position
*cannot* encode the future).

## Result

Each frozen backbone is scored with a fresh **2-layer** (small) and **6-layer**
quantile q-head, to separate backbone quality from readout capacity.

| backbone | head | full-97 GM | triage-11 GM |
|---|:--:|---:|---:|
| **β** (#309) | 2L | **1.3272** | 1.4836 |
| | 6L | 1.4489 | 1.5271 |
| loss-side · **same-step** `xshh` | 2L | 1.6194 | 1.9816 |
| | 6L | 1.6181 | 1.8762 |
| loss-side · **all-time** `xshh_allt` | 2L | 1.4143 | 1.6126 |
| | 6L | 1.4748 | 1.7028 |
| data-side · **forked, 50% mix** | 2L | 1.6366 | 1.8824 |
| | 6L | **1.4065** | 1.5339 |
| data-side · **forked, 2/batch** | 2L · 6L | *in flight* | *in flight* |
| v11c (reference) | 2L | 1.292 | — |

*GM-Relative MASE = geometric mean over GIFT-Eval configs of (model MASE ÷
seasonal-naive MASE); 1.0 = seasonal-naive parity, lower better. full-97 = all 97
configs (trusted); triage-11 = an 11-config fast subset (noisier).*

**Loss-side denial hurts — and targeting it tighter hurts more.** Adding *any*
cross-series h↔h repulsion worsens transfer: same-step +22%, all-time +6.6% (2L).
The **same-step** edge — the precise positional-code denier the issue proposed —
is the **worst** arm; the broad **all-time** edge, which also repels across
*different* steps, is *less* damaging. A 6L head rescues neither (same-step flat
1.619→1.618, all-time worse 1.414→1.475; even β degrades, 1.327→1.449), so the
regression lives in the **backbone**, confirming the "bigger heads don't help"
observation this card probed. What different series share at a step is, on
balance, **forecastably useful** — not a code worth destroying.

**Data-side denial is the standout — but head-dependent and confounded.** The
forked backbone is the *worst* arm under a 2L head (1.637) yet the **best of any
backbone here under a 6L head** (1.4065 — below β·6L 1.449 and all-time·6L 1.475).
This 0.23 swing between heads is far larger than any other arm's (β 0.12,
all-time 0.06): the forked representation encodes something a small head cannot
read but a deeper one can. **Caveat — do not yet credit the fork:** this arm
trains on **50% synthetic ARIMA** while β/all-time use 100% real data, so the
data-distribution shift and the fork structure are entangled. The corrected
**2/batch** arm injects a *single* forked pair per 256-row batch (the rest real),
isolating the fork from the shift; it will say whether the 6L result survives.

### Per-domain (2L head, full-97)

![per-domain](plots/perdomain.png)
*The loss-side same-step arm is worse than β on all 7 domains. forked-50% is
mixed — it *beats* β on Econ/Fin and Healthcare but loses elsewhere.*

| domain | β | same-step | all-time | forked 50% |
|---|---:|---:|---:|---:|
| Econ/Fin | 2.018 | 2.734 | 3.514 | **1.937** |
| Energy | 1.573 | 1.880 | 1.624 | 1.968 |
| Healthcare | 1.498 | 1.519 | 1.604 | **1.376** |
| Nature | 0.932 | 1.201 | 1.010 | 1.175 |
| Sales | 0.799 | 0.895 | 0.854 | 0.948 |
| Transport | 1.100 | 1.281 | 1.065 | 1.390 |
| Web/CloudOps | 1.437 | 1.859 | 1.472 | 1.955 |

The issue anticipated collateral damage to *same-frequency seasonal phase*; there
is no such signature — Energy is hurt *less* than the bursty Web/CloudOps and
Econ/Fin, so the loss-side damage is **general**. forked-50% (even with its poor
2L aggregate) is the only arm that *improves* on β anywhere — exactly on the two
domains (Econ/Fin, Healthcare) the loss-side arms hurt most.

### Training dynamics

![training curves](plots/training_curves.png)
*Loss minus each arm's InfoNCE floor (log–log). Floors differ — β 1.55 /
same-step 2.04 / all-time 5.27 — because the floor `log(1 + N·e^(−1/τ))` grows
with the negative-pool size N (52× across arms); forked uses the all-time loss, so
its floor is 5.27 too. Floor-subtracted, every arm converges to a similar small
excess (≈ +0.54 to +0.64); retrieval AUC/Top-1 saturate within a few hundred
steps.*

The contrastive objective converges cleanly and near-identically once the floor
is accounted for, yet transfer spans 22%. The extra structure **is** learned; for
the loss-side arms it is the **wrong** structure. The "good contrastive metrics,
indifferent transfer" decoupling is not shrunk here.

## Protocol

**Backbones.** All four arms are byte-identical to the #309 **β** arm except the
denial edit: GRU patch-encoder → 6-layer causal encoder → 1-layer forecaster
(d = 128 bottleneck), AdamW β2 = 0.98, τ = 0.10, dropkey 0.70 shared, fp16 body /
fp32 residual + patch-embedding, EWMA RevNorm span 128, seed 20260520, 50k steps,
batch 256, normalized-InfoNCE (`--pos-in-denominator`). One RTX 4090 each.
Loss-side arms change only `--loss-shape`
(`cosine_similarity_batch_full_hh_negs_{xshh,xshh_allt}`, pinned to fp64
references in `tests/test_loss.py`); data-side arms use the all-time loss + a
forked-ARIMA data mix (`src/synthetic_forked_arma.py`). The all-time loss costs
**~2.2×** the same-step step time (6.4 → 2.9 sps).

**Evaluation.** Each frozen backbone → fresh 30k quantile q-head (transformer,
causal, forecast-len 16) at 2 and 6 layers; GIFT-Eval strategy B4. Byte-identical
to #309/#315 so the numbers compare directly. References: β full-97 1.3272; v11c
(encoder-forecaster champion) 1.292; seasonal naive 1.0.

## The arms, exactly

All four keep β's within-series all-time term `cos(h_t, h_l)` ∀l (the
"different-`l`" comparisons) and remove β's adjacent `cos(h_t, h_{t+1})` (at C = 1
it duplicates the `l = t+1` slice already inside that term). They differ in *how*
they deny the shortcut:

- **Loss-side**, one added cross-series h↔h negative:
  - **same-step** `cos(h_{b,t}, h_{b',t})`, b ≠ b' — repel only same-position
    cross-series states (the shortcut and nothing else). Pool 1.79× β.
  - **all-time** `cos(h_{b,t}, h_{b',l})`, b ≠ b', ∀l — the superset (same-step is
    its `l = t` slice); also repels different-position states. Pool 52× β.
- **Data-side**, the all-time loss + forked-ARIMA pairs: each pair shares an exact
  prefix `x_{1:l}` then diverges under perturbed coefficients, so the same past
  maps to two plausible futures. Injection fraction is the design axis: **50%** of
  the batch (confounds a synthetic-data shift with the fork) vs **2 samples** (one
  pair) per 256-row batch (isolates the fork; the rest stays real).

Full per-anchor multiplicities → [annex](#annex--exact-negative-set).

## What we learned

1. **Loss-side denial of the shortcut fails.** Both arms lose to β; the *targeted*
   same-step edge (+22%) is worse than the *broad* all-time one (+6.6%) — the
   reverse of the hypothesis. The shared same-step structure is forecastably
   useful, not a disposable positional code.
2. **A bigger head does not rescue the loss-side arms** — the regression is in the
   backbone representation, confirming the standing "bigger heads don't help"
   observation.
3. **Data-side denial is the one promising lead.** The forked backbone is
   best-of-6L (1.4065) with a striking 2L→6L swing — but at 50% synthetic mix it
   is confounded. The 2/batch isolation re-run is the deciding experiment;
   per-domain, forked is also the only arm that beats β anywhere.

## Caveats

Single seed per cell (prior work on this line: seed spread ≈ ±0.02, cf. #307; the
loss-side margins are far outside that, but the all-time-vs-β ordering and the
forked-6L result each deserve a second seed). The forked-50% number is confounded
by its synthetic-data fraction — the 2/batch arm is required to attribute anything
to the fork. A step-resolved GM-vs-step sweep was not run.

## Annex — exact negative set

All arms share **one positive** `cos(h_{t+1}, f_t)`. Multiplicities are per anchor
at C = 1; the denominator is pooled over the batch, so the realised pool is
`N = B · Σ`. Counts verified by `src.loss._effective_negative_count`. (Data-side
arms share the all-time column — the fork is in the *data*, not the loss.)

| negative family | repels | β | same-step | all-time / forked |
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
forward). Holds for β/v11c too, so all numbers are on the same footing.
