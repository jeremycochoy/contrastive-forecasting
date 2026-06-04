# allt·0.8% forked arm at contrastive batch 2048

**Verdict.** #322 found that quadrupling the contrastive batch (256 → 1024), all of it pooled as
negatives, rescued the allt·0.8% arm from a collapsed 2L 2.22 to a strong 1.21. #327 takes the next
step on that curve — double the batch again, 1024 → 2048 (step budget halved to hold total data
seen fixed; everything else identical) — and the lever **reverses**: full-97 GM-Relative MASE rises
from b1024's **1.198** to **2.172** on the 6-layer head (Δ +0.97, 90% bootstrap interval (0.67,
1.37) — entirely above zero, a reliable regression), and identically on the 2-layer head (1.213 →
2.189, Δ +0.98). The damage is **entirely in the medium- and long-horizon tasks** (5.4 / 5.2 vs
b1024's 1.40 / 1.36); short horizons are essentially level (1.10 vs 1.08). A mid-plateau checkpoint
scores the same as the fully-trained one, so the extra training bought nothing. What *is* directly
visible during training is a **measurably weaker one-step forecaster** at batch 2048; the most
likely cause of the long-horizon blow-out is that this per-step error **compounds over the
multi-step rollout** long horizons require (a one-seed reading, not proven here).

*GM-Relative MASE = geometric mean, over GIFT-Eval's 97 forecasting tasks, of the model's MASE
divided by the seasonal-naive MASE. Lower is better; 1.0 is the seasonal-naive baseline.*

![Figure 1 — forecast error split by forecast horizon (6L head), at contrastive batch 256 / 1024
/ 2048. b2048 (dark) matches b1024 on short horizons and blows out on medium and long. The dotted
line is seasonal-naive (1.0).](plots/horizon_split_6L.png)

Batch 2048 is level with batch 1024 on the 55 short-horizon tasks (1.10 vs 1.08) and 3.8–3.9×
worse on the 42 medium/long tasks. The full-97 geomean (2.17) is the long-horizon blow-out
bleeding into the aggregate; the **triage** subset — GIFT-Eval's 11-task fast-feedback set, all
short-horizon — misses it (b2048 triage 1.32 vs b1024 1.28), which is why triage alone would have
read "roughly level."

## What we asked

allt·0.8% pairs the **all-time** contrastive loss (which repels every pair of different series at
every lag, on top of the base loss) with a **0.8% forked-ARMA injection** (a small fraction of
samples carry an identical past and a divergent future, denying the backbone a per-step positional
shortcut). A contrastive loss rewards more negatives per anchor, and #322 showed the contrastive
batch size is a live lever for this arm (256 → 1024 turned its worst score into its best). #327
asks the obvious follow-up:

**Double the contrastive batch once more, 1024 → 2048 (all 2048 pooled as negatives), every other
knob fixed — does allt·0.8% keep improving, or has the batch-size lever run out?**

## What happened — the batch-size lever reverses

![Figure 2 — full-97 forecast error for allt·0.8% across the contrastive-batch ladder 256 → 1024
→ 2048, per head. Whiskers are a bootstrap 90% interval over the 97 tasks. The 256 → 1024 step is
a large drop; the 1024 → 2048 step fully reverses it — on the 6L head b2048 overshoots even
b256.](plots/gm_ladder.png)

The 256 → 1024 → 2048 sequence is **down then up**: the gain from the first doubling-pair is
largely given back by the second. On the 6L head the paired Δ(2048 − 1024) is **+0.97**, interval
(0.67, 1.37) — reliably worse, not noise. This is the cleanest available isolation of the pool
lever: the recipe is byte-identical to #322 except the contrastive batch (1024 → 2048) and the
step budget (12 500 → 6 250, set so both runs see the same 12.8 M samples), so the comparison
varies the negative-pool size and holds total data seen fixed.

## Why — a weaker per-step forecaster, compounded over the rollout

![Figure 3 — training curves, step ≥ 100, batch 1024 vs 2048. The quotient gap (1−ff)/(1−fp)
measures how far the forecaster is from perfectly predicting the next encoder latent (→ 0 is
perfect); R²_random / R²_naive score the same prediction against a random and a persistence
baseline. b2048 (dark) settles measurably short of b1024 on all three.](plots/train_curves_loglog.png)

The downstream failure is foreshadowed in the backbone's own forecasting signal. Define the
**quotient gap** as (1 − ff)/(1 − fp), where ff is the cosine alignment between the forecaster's
output and the true next latent and fp the alignment with the past latent; it goes to 0 as the
forecaster perfectly predicts one step ahead. At batch 1024 the quotient gap reaches **0.011**
(ff ≈ 0.99); at batch 2048 it stalls at **0.069** (best 0.049 before the plateau pushes it back
up) — ~6× further from perfect. R²_random (0.988 → 0.936) and R²_naive (0.989 → 0.931) agree: the
b2048 forecaster is a worse one-step predictor.

A worse *per-step* forecaster is nearly invisible on short horizons (a few rollout steps, little
to compound) and ruinous on long ones (many steps, error compounding) — exactly the short-OK /
long-broken split of Figure 1. The training-internal quotient gap and the downstream horizon
collapse are the same fact seen at two scales. *(This compounding reading is the natural
explanation for the horizon split; with one seed it is a strong hypothesis, not a proof.)*

## The training tail buys nothing

*Plateau test — a fresh 2L and 6L head on the step-2500 checkpoint (the floor-subtracted loss's
local maximum, mid-training) vs on the fully-trained step-6250 backbone. Full-97 GM-Relative MASE.*

| head | mid-plateau (step 2500) | fully trained (step 6250) |
|---|--:|--:|
| 2L | 2.179 | 2.189 |
| 6L | 2.138 | 2.172 |

The fully-trained 6L backbone (2.172) is, if anything, a hair *worse* than the mid-plateau
checkpoint (2.138). Unlike #322's b1024 plateau test — where the long training tail also bought
little but the model was already excellent — here the tail neither helps nor rescues: the backbone
is stuck on a high, bumpy loss plateau (it never re-descends below ~2, vs b1024's ~0.9) and its
forecasting quality is set early and stays poor.

## Scoreboard

*Full-97 GM-Relative MASE; lower is better. Δ = batch 2048 − batch 1024, paired 90% bootstrap over
the 97 shared tasks. b256 / b1024 columns are the #320 / #322 measured scores. Single backbone
seed per cell — the interval is over tasks, not seeds.*

| head | b256 | b1024 | **b2048** | Δ(2048−1024) | 90% interval | b2048 triage |
|---|--:|--:|--:|--:|:--:|--:|
| 2L | 2.218 | 1.213 | **2.189** | +0.976 | (0.671, 1.376) | 1.313 |
| 6L | 1.848 | 1.198 | **2.172** | +0.974 | (0.667, 1.372) | 1.319 |

Eight of #322's ten (arm × head) cells beat the prior backbone 1.29; b2048 allt·0.8% does not —
it lands above seasonal-naive-relative 2.0, and on the 6L head (2.172) even above the batch-256
backbone it started from (1.848): the second doubling more than gives back the first's gain.

## Protocol

One backbone, allt·0.8%, trained at contrastive batch **2048** with all 2048 samples pooled as a
single negative set. Recipe byte-identical to #322's stabilised allt·0.8% run except the batch
(1024 → 2048) and, to hold total data seen equal, the step budget (12 500 → **6 250**; 6 250 ×
2048 = 12 500 × 1024 = 12.8 M samples). `--qk-norm --attn-out-norm`, `--subtract-contrastive-floor
--pos-in-denominator`, the all-time loss, forked-ARMA `--mix-ratio 0.0078125`, AdamW lr 1e-3,
τ 0.10, mixup p 0.3, EWMA RevIN, fp16 attention / fp32 residual, seed 20260520. One process holds
all 2048 on a single GPU; the backbone transformer is gradient-checkpointed to fit (Annex A).

Each frozen backbone is scored with a fresh **2L and 6L** quantile forecasting head (30k steps,
batch 256, transformer, causal, head-ffn-mult 4.0, dropout 0.1, `--head-train-input e_then_f`,
`--reconstruction forecaster`, forecast-len 16, cosine LR, β2 0.98, `--amp-dtype none`) on
**GIFT-Eval** (`--strategy B4`, 97-task full set + 11-task triage). Identical to #322, so b1024 ↔
b2048 is a clean paired comparison; the q-head batch stays 256 — only the backbone's contrastive
batch changed.

## Annex A — fitting batch 2048 on one card

Global batch 2048 does not fit the backbone-transformer forward on one 24 GB card under this
recipe. The fix is one env-gated flag, `BACKBONE_CKPT`, that gradient-checkpoints the backbone
transformer's non-fp32 layers (mirroring the GRU's existing `PATCH_ENC_CKPT`). Checkpointing trades
stored activations for recompute and is **byte-identical (forward and backward)**: a matched 8-step
run with the flag off vs on gives bit-identical loss and gap at every step — identical multi-step
trajectories imply identical gradients — so the trained backbone equals the recipe run
un-checkpointed. Measured peak ~20.5 GB;
~14 s/step on one RTX 4090.

## Annex B — exact negatives per anchor

Forks add no loss term, so the negatives are the all-time loss's. At batch B = 2048 with T = 64
latent positions, the pooled negative count is the quantity this experiment enlarges:

| family | repels | all-time loss |
|---|---|:--:|
| `zy` forecaster f↔f | `cos(f_{t+1}, f_t)` | 1 |
| `hh_all` within-series ∀ℓ | `cos(h_t, h_ℓ)`, ℓ≠t | T−1 |
| `cross_fe` cross-series f↔h | `cos(f_{b,t}, h_{b',t+1})` | B−1 |
| `xs_allt` cross-series ∀ℓ | `cos(h_{b,t}, h_{b',ℓ})` | (B−1)·T |
| **pooled total** | | **272 627 712** |

At batch 1024 this total was 68 156 416; the cross-series terms grow with batch, so doubling the
batch ≈ quadruples the pooled negative count (and the (B·T)² cross-series Gram, hence the ~14 s/step).
