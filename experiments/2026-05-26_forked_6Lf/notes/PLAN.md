# #320 — Forked arms × 6-layer forecaster (follow-up to #318)

## Question
#318 swept the **data-side fork** (forked-ARIMA continuations: identical past →
divergent futures) across two base losses (β, all-time) and three injection
fractions, all with a **1-layer forecaster**. Only one config beat β: the fork on
**β at ≈10% injection** (both seeds, both q-heads). Every other forked config — the
all-time loss at any fraction, and either loss at 0.8% — was neutral-to-worse.

Does deepening the **forecaster** (1L → 6L; the 6L causal *encoder* is unchanged)
move where the fork helps vs hurts — and can any forked-6Lf config beat
**β = 1.3272** or reach **v11c = 1.292** (full-97 GM-Relative MASE, lower better)?

## Why the forecaster depth could matter
The forecaster is the `--num-layers` transformer stack on top of the encoder. It is
not discarded at eval: with `--reconstruction forecaster --head-train-input
e_then_f`, the q-head reads **both** the frozen encoder latents `e` and the frozen
**forecaster latents `f`** (sequence `[e_0..e_{T-1}, f_0..f_{T-1}]`). So 1L → 6L
forecaster changes two things at once:
1. **Training** — a higher-capacity predictor changes the contrastive gradient that
   shapes the encoder.
2. **Eval** — the q-head reads deeper `f` latents.

The fork's purpose is to deny the predictor a positional / minimal-predictive-state
shortcut. A 1L forecaster may lack the capacity to convert that harder objective
into better representation; a 6L forecaster has more. Whether that *helps* the
transferable representation or merely *absorbs* the fork's difficulty in the
forecaster is the empirical question. (We avoid asserting a mechanism — the GM
numbers decide.)

## Arms (the 5 #318 forked configs, redone at `--num-layers 6`)
Backbone seed fixed at **20260520** for all, so each 1L↔6L comparison is paired on
seed and data — only forecaster depth differs.

| arm | base loss | mix-ratio (fork fraction) | #318 1L full-97 (2L / 6L) |
|---|---|---|---|
| **β·0.8%**  | `…hh_negs`           | 0.0078125 (2/256) | 1.5302 / 1.4412 |
| **β·10%**   | `…hh_negs`           | 0.10              | **1.3030 / 1.2889** |
| **allt·0.8%** | `…hh_negs_xshh_allt` | 0.0078125 (2/256) | 1.4049 / 1.5100 |
| **allt·10%**  | `…hh_negs_xshh_allt` | 0.10              | 1.6130 / 1.5304 |
| **allt·50%**  | `…hh_negs_xshh_allt` | 0.5               | 1.6366 / 1.4065 |

(1L scores reused from #318; not re-run.)

## Recipe (byte-identical to #318's forked arms except `--num-layers 1 → 6`)
GRU patch-enc → 6L causal encoder → **6L** forecaster (d=128, h=4), AdamW β2=0.98,
τ=0.10, dropkey 0.70 shared, fp16 body / fp32 residual+patch-emb, ewma span 128,
seed 20260520, 50k steps, global batch 256, `--pos-in-denominator`,
`--synth-kind forked-arma --mix-ratio MIX`.

## Eval protocol (identical to #318 / #309 / #315 so numbers are comparable)
Each frozen backbone scored with a fresh **2L and 6L** quantile q-head: 30k steps,
transformer, causal, head-ffn-mult 4.0, dropout 0.1, `--head-train-input e_then_f`,
`--reconstruction forecaster`, forecast-len 16, bs256, cosine LR, β2=0.98,
`--amp-dtype none` (avoids the depthwise-conv fp32/bf16 mismatch #318 hit on 6Lf).
GIFT-Eval `--strategy B4`, full-97 + triage-11.

## Targets
- full-97 & triage-11 **GM-Relative MASE** for every arm × {2L, 6L}, paired against
  the #318 1L scores and against **β = 1.3272**, **v11c = 1.292**, seasonal-naive 1.0.
- Where does 1L → 6L forecaster **help** vs **hurt** the fork? Does the β·10% win
  survive / strengthen? Does any forked-6Lf config beat β or reach v11c?

## Compute
Local elisa, 2× RTX 4090. The 6L forecaster needs ~18 GB, so only one backbone fits
per card: GPU 1 trains all 5 backbones sequentially; GPU 0 runs the lighter q-head +
eval downstream (≈10.5 GB) in a poller that overlaps with backbone training.
Idempotent drivers (skip finished cells). Backbone ≈2.4 h (β) / ≈5 h (all-time).
