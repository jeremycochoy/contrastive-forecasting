# #320 — Forked arms × 6-layer forecaster

**Verdict.** A deeper forecaster does **not** rescue #318's data-side fork. **No
forked-6Lf config beats β** (best 6Lf cell 1.4006, β = 1.3272, v11c = 1.292).
Paired-bootstrap on the same **97 configs** of GIFT-Eval's full set (so config
difficulty cancels — but this captures within-config-set spread only, not seed
variance: every cell is a **single backbone seed**) shows the 6Lf-vs-1L delta is
**reliably worse on 6 / 10 cells** (whole 90% CI > 0), reliably **better on 3**
(CI < 0), inconclusive on 1. The harm concentrates on the cells where the 1L fork
already scored best — β·10% on both heads (+0.263 / +0.394) and allt·0.8% on both
heads (+0.813 / +0.338) — and especially on the two 1L local optima (β·10%
overall, allt·0.8%·2L within the all-time arms). The modest improvements fall on
arms where the 1L fork was already weakest (β·0.8% on both heads, allt·50% on the
2L head). Read as one sentence: the 6L forecaster *un-finds* the 1L fork's local
optima, and only marginally improves the cells the 1L fork already missed.

![full-97 6Lf vs 1L, every arm × {2L, 6L}](plots/gm_summary.png)
*Full-97 GM-Relative MASE, lower = better. Light = 1L (#318), dark = 6Lf (this
card); blue = 2L head, orange = 6L head. β·2L (#309, 1.3272) and v11c (1.292)
sit left of every 6Lf cell. Whiskers = bootstrap 90% CI on each GM over its
97 per-config ratios.*

## Question

#318 (data-side, 1L forecaster) found exactly **one** arm beating β: the fork on
the **β loss at ≈10% injection**, reproducing on two seeds and both q-heads —
1.3030 / 1.2889 (full-97, 2L / 6L). Every other forked config — the all-time
loss at any fraction, and either loss at 0.8% — was neutral-to-worse. All those
runs used a **1-layer forecaster**.

This card asks whether deepening the forecaster (1 → 6 layers, with the 6L causal
*encoder* unchanged) changes that map — does any forked-6Lf config now beat **β =
1.3272**, or even reach **v11c = 1.292**?

The forecaster is the `--num-layers` transformer stack and it is not discarded at
eval: with `--reconstruction forecaster --head-train-input e_then_f`, the q-head
reads `[e_0..e_{T-1}, f_0..f_{T-1}]` — both the frozen encoder latents `e` *and*
the frozen forecaster latents `f`. So 1L → 6L changes (i) the contrastive
gradient that shapes the encoder during training and (ii) the `f` features the
q-head reads at eval. The GM numbers are the test; we do not attribute the
direction to a specific mechanism.

## Result

![paired Δ(6Lf − 1L), full-97 and triage-11](plots/forecaster_delta.png)
*Δ(6Lf − 1L) per (arm, head) on the full-97 left, triage-11 right. Whiskers =
**paired-bootstrap** 90% CI (same configs resampled jointly, so config difficulty
cancels — isolates the forecaster-depth effect). Green = whole 90% CI < 0
(6Lf reliably better); red = whole CI > 0 (reliably worse); grey = CI straddles 0.*

- **No 6Lf config beats β.** Best 6Lf full-97 cell is β·0.8% on the 6L head at
  **1.4006** — still **+0.073 over β (1.3272)** and +0.109 over v11c (1.292).
  The other "6Lf-better-than-1L" cells (β·0.8% 2L 1.4369; allt·50% 2L 1.4608)
  are also above β.
- **Where 1L was best, 6Lf hurts most.** β·10% (the only 1L arm beating β):
  6Lf worsens it by **+0.263 (2L)** and **+0.394 (6L)** — about 80% and 120% of
  the (β − seasonal-naive) gap (0.327), i.e. the 6L head puts β·10% on the
  *worse* side of seasonal-naive. allt·0.8% 2L (the best 1L *allt* cell at
  1.4049): 6Lf worsens it by **+0.813** — far past seasonal-naive.
- **Where 1L was weak, 6Lf helps modestly — but the gain is small.** β·0.8%
  recovers on both heads (−0.093 / −0.041) and allt·50% on the 2L head (−0.176);
  none of these lifts cross β. The remaining cells (allt·10% on both heads,
  allt·50% 6L) are inconclusive or worse.
- **Head-depth interaction.** Splitting the 10 cells by q-head: on the **2L
  head**, 6Lf helps 3 arms and hurts 2 (one inconclusive); on the **6L head**,
  6Lf reliably hurts **4 of 5 arms** (only β·0.8% improves, and only by 0.041).
  This is *consistent with* deeper forecaster latents reducing the marginal value
  of a deeper q-head, but we do not test that mechanism directly.

(Triage-11 mostly agreed in direction with full-97 but was noisy enough to
flip the sign on individual cells — e.g. allt·10% 2L triage said 6Lf was much
better (−0.378); full-97 said inconclusive (−0.016). The text above is the
full-97 read.)

## Scoreboard — every forked arm × {2L, 6L} q-head

*GM-Relative MASE = geometric mean over configs of model-MASE ÷
seasonal-naive-MASE (lower better; 1.0 = parity with seasonal-naive).
**full-97** = all 97 GIFT-Eval configs; **triage-11** = the noisy fast subset
used for early signal (carried for continuity with #309 / #315 / #318).
**1L** columns reused verbatim from #318 (not re-run); **6Lf** is this card.
Single backbone seed (20260520) per cell, paired 1L↔6Lf — so within-arm Δ on the
same 97 configs, not absolute standings, is the controlled quantity. 90% CIs on
Δ are paired bootstraps over the 97 shared configs — they capture within-config
variation but **not seed noise** (no second seed was run here). **Bold** =
reliable (whole CI on one side of 0).*

| arm | head | full-97 1L | full-97 6Lf | Δ full | 90% CI on Δ | triage-11 1L | triage-11 6Lf |
|---|:--:|---:|---:|---:|---|---:|---:|
| **β·10%**   | 2L | 1.3030 | 1.5662 | **+0.263** | (+0.210, +0.318) | 1.4559 | 1.6581 |
| **β·10%**   | 6L | 1.2889 | 1.6832 | **+0.394** | (+0.320, +0.472) | 1.4747 | 1.8429 |
| **β·0.8%**  | 2L | 1.5302 | 1.4369 | **−0.093** | (−0.146, −0.041) | 1.4376 | 1.6348 |
| **β·0.8%**  | 6L | 1.4412 | 1.4006 | **−0.041** | (−0.080, −0.0005) | 1.4027 | 1.5752 |
| **allt·50%**  | 2L | 1.6366 | 1.4608 | **−0.176** | (−0.231, −0.124) | 1.8824 | 1.7214 |
| **allt·50%**  | 6L | 1.4065 | 1.5387 | **+0.132** | (+0.085, +0.186) | 1.5339 | 1.7264 |
| **allt·10%**  | 2L | 1.6130 | 1.5973 | −0.016 | (−0.083, +0.050) | 2.0115 | 1.6333 |
| **allt·10%**  | 6L | 1.5304 | 1.6939 | **+0.163** | (+0.098, +0.228) | 1.9293 | 1.9596 |
| **allt·0.8%** | 2L | 1.4049 | 2.2180 | **+0.813** | (+0.697, +0.942) | 1.6083 | 2.2589 |
| **allt·0.8%** | 6L | 1.5100 | 1.8483 | **+0.338** | (+0.284, +0.398) | 1.6348 | 2.0528 |
| β (#309) | 2L | **1.3272** | — | — | — | 1.4836 | — |
| v11c | 2L | 1.292 | — | — | — | — | — |

## Protocol

All 5 arms are byte-identical to #318's data-side forked arms **except the
forecaster depth** (`--num-layers 1 → 6`; the 6L causal *encoder*
`--num-encoder-layers 6` is unchanged). Backbone: GRU patch-encoder → 6L causal
encoder → **6L** forecaster (d=128, h=4), AdamW β2=0.98, τ=0.10, dropkey 0.70
shared, fp16 body / fp32 residual+patch-emb, EWMA RevNorm span 128, seed
20260520, 50k steps, global batch 256, `--pos-in-denominator`,
`--synth-kind forked-arma --mix-ratio MIX`. The fork is in the **data** (no added
loss term); each arm carries its base loss's negatives (β-fork → β column,
all-time-fork → all-time column in the annex). Injection fractions: 0.8% = one
forked pair per 256-batch; 10% = 0.10; 50% = 0.5 (all-time only, the
#318-retained mis-specified arm).

Eval (byte-identical to #318 / #309 / #315): fresh 2L and 6L quantile q-head,
30k steps, transformer, causal, head-ffn-mult 4.0, dropout 0.1,
`--head-train-input e_then_f`, `--reconstruction forecaster`, forecast-len 16,
bs256, cosine LR, β2=0.98, `--amp-dtype none` (avoids the depthwise-conv
fp32/bf16 mismatch in `extract_forecaster_latents`). GIFT-Eval `--strategy B4`,
full-97 + triage-11. References: β 1.3272, v11c 1.292, seasonal-naive 1.0.

## Annex — exact negatives (per anchor, C=1; pooled N = B·Σ)

The fork adds no loss term, so each arm's negative pool is its base loss's
(unchanged from #318). For context (B=256, T=64 latent):

| family | repels | β loss | all-time loss |
|---|---|:--:|:--:|
| `xy` adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — |
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 |
| `hh_all` within-series ∀l | `cos(h_t, h_l)`, l≠t | T−1 | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 | B−1 |
| `xs_allt` cross-series ∀l | `cos(h_{b,t}, h_{b',l})` | — | (B−1)·T |
| **pooled N** | | **81,920** | **4,259,584** (52×) |

The all-time loss costs ~2× the step time (its B²·T² cross-series×cross-time Gram
is chunked + gradient-checkpointed). See #318 for the full derivation.
