# Forked arms at contrastive batch 1024

**Verdict.** The five forked arms each deny the backbone a positional shortcut — the easy way to
satisfy its contrastive objective without learning to forecast. At batch 256 none beat the plain
baseline; quadruple the batch to 1024, all negatives pooled, and every arm improves on both
heads, all but one slipping below the strongest prior backbone, 1.29. The aggregate score puts
allt·10% first by a hair (1.19), but domain by domain the standout is **allt·0.8%**: it posts the
lowest error on more domains than any other arm. One caveat sits under the win — a checkpoint from
the temporary plateau, taken early in training, already forecasts within ~0.02 of the finished
model, a little better on two arms and a little worse on the third. The long tail of training buys
almost nothing.

*GM-Relative MASE = geometric mean, over GIFT-Eval's 97 forecasting tasks, of the model's MASE
divided by the seasonal-naive MASE. Lower is better; 1.0 is the seasonal-naive baseline.*

![Figure 1 — forecast error on GIFT-Eval for every arm and forecasting head, at the old batch
of 256 (light) and the new batch of 1024 (dark). Every dark bar is lower than its light bar;
the dashed line is the strongest prior backbone (1.29), the solid line the seasonal-naive
baseline (1.0).](plots/gm_summary.png)

Batch 1024 beats batch 256 on all ten (arm × head) cells. The arms that were *worst* at batch
256 — the all-time arms — gain the most, and all but one clear the prior backbone.

## What we asked

A contrastive backbone can satisfy its training loss with an **indexing shortcut**: a per-step
positional code, shared across all series, that makes latents distinct over time without
encoding anything you could forecast from. The five arms perturb the *training data* to deny
that shortcut — each pairs every sample with a **forked continuation** (identical past,
divergent future), so position alone can no longer predict the future. They span two loss
families (**β**, the base contrastive loss; **all-time**, which additionally pushes apart every
pair of different series at every lag) crossed with how much forked data is mixed in (the
·10% / ·0.8% / ·50% labels).

A contrastive loss rewards more negatives per anchor, so the one lever pulled here is the
**contrastive batch size, 256 → 1024**, with all 1024 samples pooled as a single negative set.
The question: does the larger negative pool let any forked arm beat the baseline, and where?

## What happened

![Figure 2 — change in forecast error when the contrastive batch is quadrupled, per arm and
head. Bars left of zero mean the larger batch is better; whiskers are a paired 90% bootstrap
interval over the 97 shared tasks. Every bar is reliably left of zero.](plots/batch_delta.png)

Every cell improves, and every interval clears zero. The effect is largest exactly where batch
256 was weakest (allt·0.8% on the 2L head moves from 2.22 to 1.21), and smallest on the arms
that were already near the baseline.

![Figure 3 — forecast error broken out by data domain. Each arm is one colour; its dashed line
is batch 256, its solid line batch 1024 (each at its better head). The solid line sits inside
the dashed almost everywhere — the gain is broad across domains, not a few easy ones. Inner is
better; the black ring is seasonal-naive.](plots/perdomain.png)

The improvement holds across domains rather than coming from a handful — for every arm the
batch-1024 profile sits inside its batch-256 profile on most domains. On this per-domain view
the best model appears to be **allt·0.8% (red)**: it has the lowest error on the most domains
(3 of 7). The scoreboard's task-weighted geomean does not surface this — it weights tasks, not
domains, and places allt·10% marginally first.

## Training past the plateau

Each arm's contrastive loss keeps falling long after its forecasting score has settled. Three
checks make the gap concrete — and they do not all point the same way.

For **allt·50%**, a checkpoint at step 1 000 — where the loss is still 1.27, far above its
final 0.89 — already forecasts as well as the finished model: 1.206 / 1.185 (2L / 6L) versus
1.218 / 1.202 at the end. The loss fell by a third over the remaining 11 500 steps and bought
nothing the forecasting head could use.

For the **best scorer, allt·10%**, the loss has an outright **temporary plateau** — it stalls
and even rises, from 1.19 up to 1.22 over steps ~1 500–3 000, before resuming its descent. A
checkpoint from the middle of that plateau (step 2 500, loss ~1.21) matches the finished model —
1.209 / 1.186 (2L / 6L) versus 1.222 / 1.191 — even though the loss goes on to fall to 0.85.

The third arm, **allt·0.8%** (the per-domain winner), runs the *other* way: its mid-plateau
checkpoint is slightly **worse** than the final on both heads (1.224 / 1.208 versus 1.213 / 1.198),
so the long tail of training is not uniformly wasted — the effect's sign depends on the arm.

| Arm | head | early / mid-plateau checkpoint | fully trained | longer training helped? |
|---|:--:|--:|--:|:--:|
| allt·50% | 2L | 1.206 (step 1 000, loss 1.27) | 1.218 | no |
| allt·50% | 6L | 1.185 (step 1 000, loss 1.27) | 1.202 | no |
| allt·10% | 2L | 1.209 (step 2 500, loss 1.21) | 1.222 | no |
| allt·10% | 6L | 1.186 (step 2 500, loss 1.21) | 1.191 | no |
| allt·0.8% | 2L | 1.224 (step 2 500, loss 1.41) | 1.213 | yes |
| allt·0.8% | 6L | 1.208 (step 2 500, loss 1.41) | 1.198 | yes |

![Figure 4 — the early / mid-plateau checkpoint (hollow circle) versus the fully-trained model
(filled circle). On allt·50% and allt·10% the early checkpoint is as good or slightly better; on
allt·0.8% it is slightly worse — all within ~0.02, while the contrastive loss keeps falling far
past it (left panel).](plots/plateau.png)

The reading: across the three arms the early-vs-final gap stays under ~0.02 and changes sign —
the contrastive loss keeps falling without the forecasting head extracting anything consistently
better, so most of the forecastable signal is in place by the plateau and the long tail is at
best a wash. Whether that residual descent is harmless refinement or the shortcut the fork was
built against (loss falling without forecastable content), this single-seed test can't say — an
open thread.

## Scoreboard

*Full-97 GM-Relative MASE; lower is better. triage-11 is a noisy fast subset, kept for
continuity. Δ = batch 1024 − batch 256, with a paired 90% bootstrap interval over the 97 shared
tasks. Batch-256 columns are the prior run's measured scores. Single backbone seed per cell —
the interval is over tasks, not seeds. **Bold** marks the best score in each batch column.*

| Arm | head | batch 256 | batch 1024 | Δ | 90% interval | b256 triage | b1024 triage |
|---|:--:|--:|--:|--:|:--:|--:|--:|
| β·10% | 2L | 1.566 | 1.244 | −0.322 | (−0.380, −0.262) | 1.658 | 1.380 |
| β·10% | 6L | 1.683 | 1.277 | −0.406 | (−0.483, −0.330) | 1.843 | 1.420 |
| β·0.8% | 2L | 1.437 | 1.320 | −0.117 | (−0.160, −0.079) | 1.635 | 1.444 |
| β·0.8% | 6L | **1.401** | 1.332 | −0.069 | (−0.111, −0.031) | 1.575 | 1.471 |
| allt·50% | 2L | 1.461 | 1.218 | −0.243 | (−0.284, −0.204) | 1.721 | 1.272 |
| allt·50% | 6L | 1.539 | 1.202 | −0.337 | (−0.402, −0.279) | 1.726 | 1.242 |
| allt·10% | 2L | 1.597 | 1.222 | −0.375 | (−0.438, −0.314) | 1.633 | 1.293 |
| allt·10% | 6L | 1.694 | **1.191** | −0.503 | (−0.585, −0.426) | 1.960 | 1.259 |
| allt·0.8% | 2L | 2.218 | 1.213 | −1.005 | (−1.151, −0.871) | 2.259 | 1.261 |
| allt·0.8% | 6L | 1.848 | 1.198 | −0.650 | (−0.748, −0.557) | 2.053 | 1.277 |

![Figure 5 — the same scores pooled to one number per forecasting head, batch 256 vs batch
1024. Both heads improve by a similar margin; the larger batch helps the 2L and 6L heads about
equally.](plots/per_head.png)

The best single cell is **allt·10% on the 6L head, 1.191**; eight of the ten cells beat the
prior backbone (1.29), the two exceptions being β·0.8%.

## Protocol

Five arms, each scored with a fresh **2-layer and 6-layer** quantile forecasting head (30k
steps) on **GIFT-Eval** (97-task full set + 11-task triage). The backbones differ from the
batch-256 run in three controlled ways: the **contrastive batch (256 → 1024)**, the **step
budget (50k → 12.5k**, set so each run sees the same 12.8M samples), and **two normalisations
added inside the encoder's attention** that batch 1024 needs to train at all (Annex A). The
norms are a stability fix rather than a capacity change, but they are a genuine difference from
the batch-256 run, so the cleanest reading of the table is "the stabilised batch-1024 recipe vs
the batch-256 recipe", not "batch alone". AdamW, lr 1e-3, τ 0.10, fp16 attention / fp32
residual, seed 20260520; one process holds all 1024 samples on a single GPU.

## Annex A — batch 1024 needs two normalisations to train

At batch 1024 the original recipe collapses by ~6k steps: the encoder's self-attention output
amplitude runs away (the residual stream grows from ~38 to ~6400), fp16 then corrupts the
direction of the L2-normalised latents, and the model sends every series the same way (the
cross-series cosine climbs from ~0 to 0.57 while the forecast-vs-future gap falls to zero). Two
standard additions fix it, and both are needed — either alone still diverges: **QK-norm**
(RMSNorm on the attention queries and keys, bounding the attention logits) and an
**attention-output RMSNorm** (Gemma2-style, bounding the residual stream). With the flags off
the code path is numerically identical to ordinary attention (max difference < 1e-4). This is
the only recipe change batch 256 did not need.

![Per-layer maximum activation amplitude through training. Without the two norms the
attention output and residual stream run past 10³ within ~5k steps (the collapse); with both
(black) all three traces stay bounded and the run converges.](plots/activation_amplitudes.png)

## Annex B — exact negatives per anchor

Forks add no loss term, so each arm's negatives are its base loss's. At batch B = 1024 with
T = 64 latent positions, the pooled negative count is the quantity this experiment enlarges:

| family | repels | β loss | all-time loss |
|---|---|:--:|:--:|
| `xy` adjacent h↔h | `cos(h_t, h_{t+1})` | 1 | — |
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 | 1 |
| `hh_all` within-series ∀ℓ | `cos(h_t, h_ℓ)`, ℓ≠t | T−1 | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 | B−1 |
| `xs_allt` cross-series ∀ℓ | `cos(h_{b,t}, h_{b',ℓ})` | — | (B−1)·T |
| **pooled total** | | **1,114,112** | **68,156,416** |

At batch 256 these totals were 81,920 and 4,259,584; the cross-series terms grow with batch, so
the pooled negative count is ≈13–16× larger at batch 1024.
