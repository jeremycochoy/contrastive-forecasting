# #322 — Forked arms × 6-layer forecaster, retrained at batch 1024

<!-- WORKING DRAFT. Question / protocol / annex / the stability finding are final; the
downstream Verdict, Figures, and Scoreboard are filled from the runs as each (arm × head)
eval lands. -->

**Verdict (part 1 of 2 — final).** The one-knob change we set out to make is not one knob.
Pooling 4× more negatives (batch 256 → 1024) **destabilises #320's recipe**: the encoder's
self-attention output amplitude runs away (residual stream 38 → 6 400), fp16 then corrupts
the L2-normalised latents, and the model collapses directionally (cross-series cosine
0.001 → 0.57, the contrastive gap goes to zero) within ~6 k steps. Recovering a trainable
batch-1024 model required two standard numerical-stability additions the batch-256 recipe
never needed — **QK-norm** (RMSNorm on Q, K) to bound the attention logits and a
**Gemma2-style RMSNorm on the attention output** to bound the residual stream. Both are
required: either alone still diverges. With both, batch-1024 converges like batch-256
(amplitudes bounded — qk-logit ~10, residual ~40; gap ~1.03, cross-series cosine ~3e-4).

**Verdict (part 2 of 2 — final, 10/10 cells).** Once the collapse is fixed, the stabilised
batch-1024 recipe **beats #320's batch-256 on every one of the 10 (arm × head) cells, and
every win is reliable** — the whole paired-bootstrap 90 % CI on Δ sits below 0 in all 10
(Δ from −0.069 to −1.005 full-97 GM-Relative MASE). **allt·10% 6L = 1.191 is the best cell**;
**eight of ten beat v11c (1.292)** — the all-time arms sweep the top six, β·10% just behind,
only β·0.8% (1.32 / 1.33) sits above v11c. So the central question, "does 4× more pooled
negatives help?", answers **yes, decisively and uniformly**. The biggest gains land on the
arms that were *weakest* at batch 256 (allt·0.8% 2L 2.22 → 1.21, Δ −1.005; allt·10% 6L 1.69
→ 1.19) — the larger pool helps most where the smaller one left the most on the table; the
forked-ARIMA finding from #318/#320 (where forks mostly *failed* to beat β at batch 256) is
reversed at batch 1024. _The training-length ablation row (#10) lands when its two cells
finish. The batch ↔ norms confound (Stability + Protocol) still applies; a batch-256 + same-norms
control would isolate it._

## What we asked

#318 perturbed the contrastive training data with **forked-ARIMA continuations**
(identical past → divergent futures, so position alone cannot encode the future) and
found only one of five forked configurations beat the β baseline. #320 gave those same
five arms a deeper (6-layer) forecaster — still at **global batch 256** — and no cell
crossed β; the deepening mostly *un-found* the 1L fork's wins.

#322 changes exactly one knob: the contrastive **batch grows 256 → 1024**, with **all
1024 terms pooled together in the negatives** (no per-shard split). A larger
all-together negative pool is the lever contrastive learning is most expected to reward —
more negatives per anchor sharpens the InfoNCE objective. The question: does 4× more
pooled negatives change *where* the fork helps vs hurts, and does any arm now reach the
**β** baseline or **v11c**?

## What happened

### Stability — the 4× batch collapses, and the fix (the central finding)

The first batch-1024 run (the β·0.8% arm, otherwise #320's recipe) trained normally for
~5 k steps, then collapsed: the contrastive loss fell below its theoretical floor while
the **gap** (cos(forecast, future) − cos(forecast, present)) decayed to zero and the
**cross-series cosine** — which should sit near zero as different series repel — climbed
0.001 → 0.57. A model whose distinct series all point the same way has stopped encoding
anything; the loss "improves" only because the floor-subtracted denominator degenerates.

The amplitude diagnostic (`--log-attn-amplitude`, logging per-layer max-abs activations)
located the cause upstream of the loss: the **encoder self-attention output** amplitude
ran away, driving the residual stream from ~38 to ~6 400. At that magnitude the fp16
forward corrupts the subsequent L2-normalisation (the latents are scale-invariant by
construction, so the *only* thing the normalisation can preserve — direction — is exactly
what fp16 rounding destroys once the pre-norm vector is large). The collapse is therefore
**not** a learning-rate or data problem (LR 5e-4 and τ = 0.20 both still collapsed); it is
an activation-amplitude runaway that 4× more pooled negatives excite and that batch-256
never reached. Per the project rule, the fix is normalization, not gradient-clipping.

Two instabilities had to be closed, and **both** are necessary (verified — either alone
diverges):

1. **QK-norm** — RMSNorm applied to Q and K per head before the dot-product (PaLM/Gemma).
   Bounds the attention logits (observed max-abs 141 → 3.9 on the diverged trace), which
   keeps the softmax from saturating into a near-one-hot map that amplifies one value row.
2. **Attention-output RMSNorm** — a Gemma2-style sandwich norm on the attention output
   only (not the FFN — the FFN residual did not grow). Bounds the residual stream directly.

Implementation is a drop-in: QK-norm runs through `F.scaled_dot_product_attention` reusing
`nn.MultiheadAttention`'s own projection weights, so with the flags off the path is
**bit-identical** to the original MHA (verified diff 0.0) and there is no throughput
regression. Both are gated behind `--qk-norm` / `--attn-out-norm` training flags. With both
on, the β·0.8% arm clears the collapse zone and converges like batch-256 — loss − floor
13.3 → ~1.0, gap steady ~1.03, cross-series cosine flat ~3e-4, qk-logit ~10, residual ~40
(all bounded). The full debugging trace, plots, and the diverged-run CSVs are in
[`EXECUTION_LOG.md`](EXECUTION_LOG.md).

**Consequence for the comparison.** Because the norms are *required* at batch 1024 but
*absent* at batch 256 (#320), the headline b1024 ↔ b256 contrast confounds the batch with
the two norms. The norms are a stability mechanism, not a capacity lever, so the expectation
is that they are MASE-neutral — but that is an assumption until tested. The planned control
is **batch-256 + the same two norms** on ≥1 arm: if it matches #320's no-norm b256 score,
the norms are neutral and the b1024 ↔ b256 delta is cleanly the batch effect; if not, the
confound is real and reported as such.

### Downstream scores

With the recipe stabilised, the downstream result is clean and one-directional: **batch
1024 beats batch 256 on all ten (arm × head) cells, every Δ reliable** (whole 90 % CI < 0;
Scoreboard below, Figures 1–2). The effect is large — a median Δ ≈ −0.35 and up to −1.0 —
and it is *strongest on the arms #320 found weakest*: the all-time arms (allt·50/10/0.8 %),
which never beat β at batch 256, are the top six cells at batch 1024 and all clear v11c.
This reverses #318/#320's reading that forks mostly fail to beat the baseline — at a
sufficiently large all-together negative pool, the fork *helps*, most where the smaller
pool was leaving signal unused. The β arms improve too (β·10 % by ~0.32–0.41), just less
dramatically. No cell regresses.

![Figure 1 — full-97 GM-Relative MASE per arm × q-head, batch 256 (#320) vs batch 1024.
Whisker = bootstrap 90 % CI on the GM over its 97 configs. β shown as shaded 2-seed
ranges; v11c and seasonal-naive marked.](plots/gm_summary.png)

![Figure 2 — Δ(batch 1024 − batch 256) per (arm, q-head) on full-97; whisker =
paired-bootstrap 90 % CI over the 97 shared configs. Green = reliably better with the
larger pool, red = reliably worse, grey = inconclusive.](plots/batch_delta.png)

## Scoreboard

*Full-97 GM-Relative MASE = GM over GIFT-Eval's 97 configs of (model MASE) ÷
(seasonal-naive MASE). **Lower is better.** triage-11 is the noisy fast subset, kept for
continuity with #318/#320. **Δ = b1024 − b256**; the 90 % CI on Δ is a paired bootstrap
over the 97 shared configs (per-config difficulty cancels). b256 columns are #320's
measured 6Lf scores, reused verbatim. Single backbone seed per cell — the paired CI
captures config-set spread, not seed noise. **Bold** = reliable (whole CI one side of 0).*

Generated by `scripts/plots.py` → `results/gm_table.csv` (sorted by arm, #320 order).
**Every Δ is reliable — the whole 90 % CI sits below 0 (b1024 better) in all 10 cells.**

| Arm (mix) | Head | b256 | b1024 | Δ | 90 % CI on Δ | b256 tri | b1024 tri |
|---|---|--:|--:|--:|:--:|--:|--:|
| β·10% | 2L | 1.566 | **1.244** | **−0.322** | (−0.380, −0.262) | 1.658 | 1.380 |
| β·10% | 6L | 1.683 | **1.277** | **−0.406** | (−0.483, −0.330) | 1.843 | 1.420 |
| β·0.8% | 2L | 1.437 | **1.320** | **−0.117** | (−0.160, −0.079) | 1.635 | 1.444 |
| β·0.8% | 6L | 1.401 | **1.332** | **−0.069** | (−0.111, −0.031) | 1.575 | 1.471 |
| allt·50% | 2L | 1.461 | **1.218** | **−0.243** | (−0.284, −0.204) | 1.721 | 1.272 |
| allt·50% | 6L | 1.539 | **1.202** | **−0.337** | (−0.402, −0.279) | 1.726 | 1.242 |
| allt·10% | 2L | 1.597 | **1.222** | **−0.375** | (−0.438, −0.314) | 1.633 | 1.293 |
| **allt·10%** | **6L** | 1.694 | **1.191** | **−0.503** | (−0.585, −0.426) | 1.960 | 1.259 |
| allt·0.8% | 2L | 2.218 | **1.213** | **−1.005** | (−1.151, −0.871) | 2.259 | 1.261 |
| allt·0.8% | 6L | 1.848 | **1.198** | **−0.650** | (−0.748, −0.557) | 2.053 | 1.277 |

**allt·10% 6L = 1.191 is the best cell.** Eight of ten beat **v11c (1.292)** — the all-time
arms sweep the top six; only β·0.8% (1.32 / 1.33) sits just above v11c. The two largest
gains (allt·0.8% 2L −1.01, allt·10% 6L −0.50) are the arms that were *weakest* at batch 256,
so the larger pooled-negative batch helps most where the smaller pool underperformed.

*Training-length ablation (#10):* the best arm (allt·50%) re-trained only to its plateau
start (step 1000) and scored with both heads — _filled when the two ablation cells finish_
— tests whether the long tail of training (past the apparent gap-plateau) keeps improving
downstream MASE.

**References** (full-97 GM-Relative MASE, lower better): β · 2L = [1.3272, 1.4591]
(n = 2 seeds); β · 6L = [1.3702, 1.4489] (n = 2); v11c = 1.292; seasonal-naive = 1.0.

## Protocol

Each arm differs from its #320 counterpart in exactly three controlled ways:

1. **Contrastive batch 256 → 1024** (the knob of interest), all 1024 terms pooled in the
   negatives — no per-shard split.
2. **Budget 50k → 12.5k steps**, chosen to hold *total data seen* equal to #320
   (12.5k × 1024 = 50k × 256 = 12.8 M samples), so the negative-pool size is varied without
   varying the amount of data or (the backbone LR is constant) the schedule.
3. **+ QK-norm and + attention-output RMSNorm** — *not* a free choice: batch 1024 collapses
   without them (see Stability). #320 ran at batch 256 without either. This is the confound
   the planned batch-256 + same-norms control is designed to isolate; the norms are
   structurally a stability mechanism, expected MASE-neutral, but that is tested, not assumed.

Compute: a single process at global batch 1024 on one RTX 4090 — it pools all 1024 in the
negatives natively, no cross-device gather. To fit 1024 on one 24 GB card the GRU
patch-encoder is gradient-checkpointed; this is **byte-identical in the forward** (verified)
and so is a pure memory/throughput implementation detail, *not* a recipe change (unlike the
norms above). Recipe otherwise:
GRU patch-enc → 6L causal encoder → 6L forecaster (d = 128, h = 4), AdamW β2 = 0.98,
τ = 0.10, dropkey 0.70, fp16 body / fp32 residual + patch-emb, ewma span 128,
seed 20260520, `--pos-in-denominator`, `--qk-norm`, `--attn-out-norm`,
`--subtract-contrastive-floor` (gradient-neutral loss rebasing; logged loss − floor),
`--synth-kind forked-arma --mix-ratio MIX`.

Eval (byte-identical to #320 / #318 — same q-head recipe, same GIFT-Eval harness — so the
eval adds no confound and any b256 ↔ b1024 difference is attributable to the backbones):
each frozen backbone scored with a fresh **2L and 6L** quantile q-head — 30k steps, transformer,
causal, head-ffn-mult 4.0, dropout 0.1, `--head-train-input e_then_f`,
`--reconstruction forecaster`, forecast-len 16, **batch 256**, cosine LR, `--amp-dtype
none`. GIFT-Eval `--strategy B4`, full-97 + triage-11. The q-head batch stays 256 — #322
varies only the *backbone's* contrastive batch.

## Annex — exact negatives (per anchor, C = 1; pooled N = B·Σ at B = 1024, T = 64)

Forks add no loss term, so each arm's negatives are its base loss's (unchanged from
#318/#320). The batch is 4× #320's, so the pooled negative count grows super-linearly in
the cross-series terms — this is the quantity #322 enlarges:

| family | repels | β loss | all-time loss |
|---|---|:--:|:--:|
| `xy` adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — |
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 |
| `hh_all` within-series ∀ℓ | `cos(h_t, h_ℓ)`, ℓ≠t | T−1 | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 | B−1 |
| `xs_allt` cross-series ∀ℓ | `cos(h_{b,t}, h_{b',ℓ})` | — | (B−1)·T |
| **pooled N** | | **1,114,112** | **68,156,416** (61×) |

(At #320's B = 256 these were 81,920 and 4,259,584; the cross-series f↔h and all-time
terms scale with B, so the per-anchor negative pool is ≈13.6× larger for β and ≈16×
larger for all-time at B = 1024.)
