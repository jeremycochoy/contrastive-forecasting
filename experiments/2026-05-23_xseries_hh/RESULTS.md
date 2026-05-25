# #318 — Deny the positional shortcut: loss-side vs data-side

**Verdict.** Two ways to deny the shortcut, tested as separate arms. **Loss-side**
(cross-series h↔h repulsion — same-step, all-time): both **hurt**, +22% / +6.6% vs
β at the 2L head. **Data-side** (forked continuations — identical past, divergent
futures): **head-dependent** — worst arm at 2L (1.637), best backbone here at 6L
(full-97 **1.4065**), the only denial arm to beat β on any domain — but its 50%
synthetic mix confounds a data shift with the fork, so an isolating 2/batch re-run
is in flight. **β·2L (1.3272) is best overall.**

![full-97 GM summary](plots/gm_summary.png)
*GM-Relative MASE, lower = better; 1.0 = seasonal-naive. Every arm × {2L, 6L};
forked 2/batch (brown) appears once it lands.*

## Question

β denies within-series cross-time distinctness (`h_t` ≠ `h_l` ∀l), but a
**positional fingerprint** shared across all series satisfies that for free —
distinctness at no cost in forecastable content (consistent with the standing
observations: bigger eval heads don't help, more contrastive training doesn't
transfer). We forbid the shortcut two ways: through the **loss** (repel what
different series share at a step) and through the **data** (pairs whose identical
past has divergent futures, so position *cannot* encode the future).

## Result

Each frozen backbone is scored with a fresh 2-layer and 6-layer quantile q-head,
separating backbone quality from readout capacity.

| backbone | head | full-97 | triage-11 |
|---|:--:|---:|---:|
| **β** (#309) | 2L · 6L | **1.3272** · 1.4489 | 1.4836 · 1.5271 |
| loss-side **same-step** `xshh` | 2L · 6L | 1.6194 · 1.6181 | 1.9816 · 1.8762 |
| loss-side **all-time** `xshh_allt` | 2L · 6L | 1.4143 · 1.4748 | 1.6126 · 1.7028 |
| data-side **forked, 50% mix** | 2L · 6L | 1.6366 · **1.4065** | 1.8824 · 1.5339 |
| data-side **forked, 2/batch** | 2L · 6L | *in flight* | *in flight* |
| v11c (reference) | 2L | 1.292 | — |

*GM-Relative MASE = geometric mean over configs of model-MASE ÷ seasonal-naive-MASE
(1.0 = parity, lower better); full-97 = all 97 configs, triage-11 = noisy fast
subset. Single seed per cell (line spread ≈ ±0.02, #307).*

- **Loss-side hurts; tighter targeting hurts more.** same-step (the precise
  positional-code denier) is worst, broad all-time less so. A 6L head rescues
  neither (even β degrades 2L→6L), so the regression is in the **backbone** —
  what different series share at a step is forecastably useful, not a free code.
- **Data-side (forked) is head-dependent**: worst at 2L, best-of-6L at 6L — a 0.23
  swing vs β's 0.12, so the representation needs a deeper head to read out.
  **Confounded** by the 50% synthetic mix; the 2/batch arm isolates the fork.

### Per-domain (2L, full-97)

![per-domain radar](plots/perdomain.png)
*Log radial; dashed ring = seasonal-naive (1.0); innermost = best. Loss-side arms
bulge outward (same-step worse than β on all 7 domains, no seasonal signature);
forked-50% is the only arm inside β anywhere (Econ/Fin, Healthcare).*

### Training dynamics

![training curves](plots/training_curves.png)
*Loss − InfoNCE floor (log–log). Floors β 1.55 / same-step 2.04 / all-time 5.27
grow with the negative-pool size N (52× across arms; forked shares all-time's);
all converge to a similar excess (+0.54…+0.64) and AUC/Top-1 saturate early.* The
contrastive task is learned near-identically, yet transfer spans 22% — for the
loss-side arms, the structure is learned but wrong.

## Protocol

All arms are byte-identical to the #309 **β** recipe except the denial edit: GRU
patch-encoder → 6L causal encoder → 1L forecaster (d=128), AdamW β2=0.98, τ=0.10,
dropkey 0.70, fp16 body / fp32 residual, EWMA RevNorm span 128, seed 20260520,
50k steps, batch 256, `--pos-in-denominator`. **Loss-side** changes only
`--loss-shape …_{xshh,xshh_allt}` (fp64-pinned in `tests/test_loss.py`);
**data-side** uses the all-time loss + a forked-ARIMA mix
(`src/synthetic_forked_arma.py`), injection 50% vs 2/256. The all-time loss costs
**~2.2×** the step time (6.4→2.9 sps). Eval: fresh 30k q-head (2L/6L), GIFT-Eval
strategy B4, byte-identical to #309/#315. Refs: β 1.3272, v11c 1.292,
seasonal-naive 1.0.

## Annex — exact negatives (per anchor, C=1; pooled N = B·Σ)

| family | repels | β | same-step | all-time / forked |
|---|---|:--:|:--:|:--:|
| `xy` adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — | — |
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 | 1 |
| `hh_all` within-series ∀l | `cos(h_t, h_l)`, l≠t | T−1 | T−1 | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 | B−1 | B−1 |
| `xshh` cross-series same-step | `cos(h_{b,t}, h_{b',t})` | — | B−1 | — |
| `xs_allt` cross-series ∀l | `cos(h_{b,t}, h_{b',l})` | — | — | (B−1)·T |
| **pooled N** (B=256, T=64) | | **81,920** | **146,944** (1.79×) | **4,259,584** (52×) |

Latent T = 64 (HF crops to T_RAW=1024, ÷ patch-16; holds for β/v11c too). Data-side
arms share the all-time column — the fork is in the *data*, not the loss.

## Other artifacts (trained, not evaluated)

A 6L-forecaster variant of both loss-side arms (1L → 6L *forecaster*) was trained
— `bb_xshh_6Lf_50k`, `bb_xshh_allt_6Lf_50k` (50k each, on disk) — but downstream
eval was paused to prioritise the fork (one cell: same-step-6Lf 2L triage 1.5177).
See [`EXECUTION_LOG.md`](EXECUTION_LOG.md).
