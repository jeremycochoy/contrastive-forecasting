# #318 — Deny the positional shortcut: loss-side vs data-side

**Verdict.** Denying the shortcut through the **loss** (cross-series h↔h repulsion;
same-step, all-time) **hurts** — +22% / +6.6% vs β. Through the **data** (forked
continuations: identical past, divergent futures) the effect is **injection-
fraction- and loss-specific**: on the **all-time** loss it only hurts (every
fraction), and at **0.8%** on either loss it's neutral-to-worse (the 50%-mix 6L
"win" 1.4065 was a synthetic-fraction confound). But on **β at ≈10% injection it
robustly helps** — across **two seeds and both q-heads, β·10% beats β at all four
cells** (the only reproducing improvement on β here). It does **not**, however,
reliably reach v11c: the single-seed β·10%·6L = 1.2889 (≈ v11c 1.292) was a
favorable draw — at the second seed it is **1.3271** (above v11c), and β·10%·2L
likewise moved 1.3030 → 1.3805 (absolute GMs are markedly seed-variable, ≫ ±0.02).
**Net: the 10% β-fork is a real, reproducing gain over β, but lands *between* β and
v11c — not a v11c match.** (Details: §Second seed.)

![full-97 GM summary](plots/gm_summary.png)
*GM-Relative MASE, lower = better; 1.0 = seasonal-naive. Every arm × {2L, 6L}; the
v11c target (dashed) and β·2L sit left of all denial arms.*

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
separating backbone quality from readout capacity. Numbers are on the gm_summary
bars above and in the **[full scoreboard](#scoreboard--every-arm--head)** at the end.

*GM-Relative MASE = geometric mean over configs of model-MASE ÷ seasonal-naive-MASE
(1.0 = parity, lower better); full-97 = all 97 configs, triage-11 = noisy fast
subset. Single seed per cell (line spread ≈ ±0.02, #307).*

- **Loss-side hurts; tighter targeting hurts more.** same-step (the precise
  positional-code denier) is worst, broad all-time less so. A 6L head rescues
  neither (even β degrades 2L→6L), so the regression is in the **backbone** —
  what different series share at a step is forecastably useful, not a free code.
- **Data-side (forked): the gain is specific to β + ≈10% injection.** Sweeping the
  forked fraction is non-monotonic and loss-dependent. On **β** it peaks at 10%:
  0% (β) 1.3272 → 0.8% 1.5302 → **10% 1.3030 (2L) / 1.2889 (6L ≈ v11c)** — the best
  arm here. On the **all-time** loss the fork only hurts at every fraction
  (0.8% 1.4049/1.5100, 10% 1.6130/1.5304, 50% 1.6366/1.4065); the 50%·6L 1.4065 that
  first looked like a win was a synthetic-data-fraction confound (its 0.8% isolation
  gave 1.5100). A 50% **unforked-synthetic** control is still the clean test of
  whether β·10%'s gain is the fork structure or just ~10% synthetic data.

### Second seed (β·10%, seed 20260521)

A paired second seed (β·10% and a matched β, both via the forked launcher) re-tests
the fork's effect and its firmness.

| arm · head | seed 1 | seed 2 | β·10% − β (s1 / s2) |
|---|---:|---:|---:|
| **β·10% · 2L** | 1.3030 | 1.3805 | −0.024 / −0.079 |
| **β·10% · 6L** | 1.2889 | 1.3271 | −0.160 / −0.043 |
| β · 2L | 1.3272¹ | 1.4591 | |
| β · 6L | 1.4489¹ | 1.3702 | |

**β·10% beats β at all four (seed × head) cells** — the fork's gain on β is
**reproducible**. But absolute GMs are markedly seed-variable (β·10%·2L 1.30→1.38,
6L 1.29→1.33 — ≫ the ±0.02 we'd assumed), so the single-seed **v11c match (6L
1.2889) does not hold**: at seed 2 it is **1.3271**, above v11c (1.292). β·10% is a
**real but modest** gain over β, landing *between* β and v11c; a third seed would
tighten the magnitude.

¹ seed-1 β is the #309 reference backbone; seed-2 β is the matched forked-launcher
recipe (mix 0). The within-seed **paired gap** (β·10% − β) is the controlled
quantity — negative at both seeds.

### Per-domain (full-97, best q-head per arm)

![per-domain radar](plots/perdomain.png)
*Log radial; dashed ring = seasonal-naive (1.0); innermost = best. Each arm at its
best q-head (legend); curated to the headline comparison. The loss-side arms
(same-step 6L, all-time 2L) bulge outward — worse than β across domains, no
seasonal signature — while the data-side winner **forked β·10% (6L)** tracks β and
v11c on the inner ring.*

### Training dynamics

![training curves](plots/training_curves.png)
*All log–log, warm-up (step < 1000) skipped. **Loss − floor**: floors β 1.55 /
same-step 2.04 / all-time 5.27 grow with the 52×-spanning negative pool (forked
shares all-time's); all converge to a similar excess (+0.54…+0.64). **gap-ratio**
(1−ff)/(1−fp) (forecast↔future vs ↔present; 0 = perfect). **Dimension usage**
U = 1/(d·mean cos²) ∈ (0,1] (1 = full use, →0 = collapse), temporal and batch —
measured in-training on the live batch, distinct from the frozen-`h` eff-rank in
§Latent dimensionality.* The contrastive objective is learned near-identically
across arms, yet transfer spans 22% — for the loss-side arms the structure is
learned but wrong.

## Latent dimensionality

![latent dimensionality](plots/latent_dim.png)
*Frozen encoder latent `h` (d=384), one real-HF batch (B=128, T=64 → 8,192
positions, CPU). **Left**: normalised singular-value spectrum of mean-centred `h`
(log-y). **Right**: dimension-usage effective rank = `U·H`, `U = 1/(d·mean_{i≠j}
cos²)` (method from the 2026-05-08 init-u-sweep), batch and time axes. Colours
match the scoreboard.*

All arms are deeply collapsed — effective rank 1–8 of 384 (the post-training
collapse the init-u-sweep flagged) — except **forked β·2/b**, the clear outlier at
eff-rank ≈ 7.8 (PR 10.2), ~4–8× the rest. The fork's effect on rank is
**loss-specific**: on the β loss it lifts rank ~5× (β 1.45 → forked β·2/b 7.82),
but on the all-time loss it does not (all-time 1.93 → forked allt·2/b 1.08) — so
the boost is the β-loss × fork *interaction*, not the fork alone. And latent rank
does **not** order the arms by transfer: β (rank 1.45) transfers best, all-time
(rank 1.93) worse despite the higher rank, and the rank outlier forked β·2/b (7.82)
transfers *worse* than β at both heads (1.5302 / 1.4412) — the loss↔transfer
decoupling again.

| arm | eff-rank (batch / time) | PR |
|---|:--:|:--:|
| same-step | 1.26 / 1.60 | 2.78 |
| all-time | 1.93 / 1.95 | 7.53 |
| forked allt·50% | 1.01 / 1.69 | 4.14 |
| forked allt·2/b | 1.08 / 2.09 | 2.82 |
| **forked β·2/b** | **7.82 / 7.29** | **10.19** |
| β | 1.45 / 1.57 | 3.17 |

*eff-rank = `dim_usage · 384` over the batch / time axes; PR = participation ratio
of the squared spectrum. Script: `scripts/latent_dim.py`.*

## Protocol

All arms are byte-identical to the #309 **β** recipe except the denial edit: GRU
patch-encoder → 6L causal encoder → 1L forecaster (d=128), AdamW β2=0.98, τ=0.10,
dropkey 0.70, fp16 body / fp32 residual, EWMA RevNorm span 128, seed 20260520,
50k steps, batch 256, `--pos-in-denominator`. **Loss-side** changes only
`--loss-shape …_{xshh,xshh_allt}` (fp64-pinned in `tests/test_loss.py`);
**data-side** adds no loss term — it injects a forked-ARIMA mix
(`src/synthetic_forked_arma.py`): 50% or one pair/256 on the all-time loss, and
one pair/256 on the β loss. The all-time loss costs **~2.2×** the step time
(6.4→2.9 sps); the β-loss arms run at the β rate. Eval: fresh 30k q-head (2L/6L), GIFT-Eval
strategy B4, byte-identical to #309/#315. Refs: β 1.3272, v11c 1.292,
seasonal-naive 1.0.

## Scoreboard — every arm × head

| arm | head | full-97 | triage-11 | Δ full vs β·2L |
|---|:--:|---:|---:|---:|
| β (#309) | 2L | **1.3272** | 1.4836 | — |
| β (#309) | 6L | 1.4489 | 1.5271 | +9.2% |
| loss-side · same-step | 2L | 1.6194 | 1.9816 | +22.0% |
| loss-side · same-step | 6L | 1.6181 | 1.8762 | +21.9% |
| loss-side · all-time | 2L | 1.4143 | 1.6126 | +6.6% |
| loss-side · all-time | 6L | 1.4748 | 1.7028 | +11.1% |
| data-side · forked allt·50% | 2L | 1.6366 | 1.8824 | +23.3% |
| data-side · forked allt·50% | 6L | **1.4065** | 1.5339 | +6.0% |
| data-side · forked allt·2/b | 2L | 1.4049 | 1.6083 | +5.9% |
| data-side · forked allt·2/b | 6L | 1.5100 | 1.6348 | +13.8% |
| data-side · forked β·2/b | 2L | 1.5302 | 1.4376 | +15.3% |
| data-side · forked β·2/b | 6L | 1.4412 | 1.4027 | +8.6% |
| data-side · forked allt·10% | 2L | 1.6130 | 2.0115 | +21.5% |
| data-side · forked allt·10% | 6L | 1.5304 | 1.9293 | +15.3% |
| data-side · **forked β·10%** | 2L | **1.3030** | 1.4559 | **−1.8%** |
| data-side · **forked β·10%** | 6L | **1.2889** | 1.4747 | **−2.9%** |
| v11c (reference) | 2L | 1.292 | — | −2.7% |

*GM-Relative MASE, lower = better; Δ vs the best arm β·2L (1.3272).*

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
arms add no loss term — the fork is in the *data*; each shares its base loss's
negatives (all-time-forked → all-time column; β-forked → β column).

## Other artifacts (trained, not evaluated)

A 6L-forecaster variant of both loss-side arms (1L → 6L *forecaster*) was trained
— `bb_xshh_6Lf_50k`, `bb_xshh_allt_6Lf_50k` (50k each, on disk) — but not evaluated
downstream (one cell landed: same-step-6Lf 2L triage 1.5177). See
[`EXECUTION_LOG.md`](EXECUTION_LOG.md).
