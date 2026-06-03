# #325 — Regime-crossfade synthetic on top of allt·10%

## Question
#322's best cell is **allt·10%** (90% real + 10% forked-ARMA, all-time loss, batch
1024): full-97 GM-Relative MASE 2L = 1.222, **6L = 1.191**. #325 keeps that arm fixed
and adds a **third** synthetic stream — a **regime crossfade** — to ~10% of the batch.
Does a hard-negative built from two *real* windows improve the downstream forecaster, or
is the all-time fork already saturating what the contrastive objective can use?

## The crossfade primitive (new: `src/synthetic_crossfade.py`)
For each crossfade row, two distinct real windows A, B from the same step's real
sub-batch are z-normalised per series and blended by a monotone weight s(t):

    C(t) = (1 − s(t))·A(t) + s(t)·B(t)
    s(t) = 0  (t ≤ l);  (t−l)/(l′−l)  (l < t < l′);  1  (t ≥ l′)

with, per sample (T = window length = 1024 at runtime):
    midpoint  m  ~ U(0, T)
    width     w  ~ LogUniform(T/128, T)     (sharp → gradual)
    l = m − w/2, l′ = m + w/2, clipped to [0, T]
One s(t) per sample, shared across channels. Because A and B remain in the batch as
their own rows, C shares **A's past** and **B's future** with batch-mates → hard
negatives the contrastive loss must separate by content, not position.

## The single change vs allt·10% (#322)
Batch composition **80% real / 10% forked-ARMA / 10% crossfade** (was 90/10/0). The
crossfade rows are blended from the 80% real sub-batch (they consume no extra HF rows).
Everything else is **byte-identical** to #322's allt·10%:
single-GPU batch 1024 (GRU patch-encoder gradient-checkpointed), 12.5k steps, AdamW
lr 1e-3 / β1 0.9 / β2 0.98 / wd 0.1, τ 0.10, **qk-norm + attn-out-norm** (the b1024
collapse fix), dropkey 0.70, depthwise-conv 3, mixup-p 0.3, ewma span 128, freq/seas-emb
3/3, fp16 attn/ffn/conv + fp32 residual/patch-emb, `--subtract-contrastive-floor`,
loss `cosine_similarity_batch_full_hh_negs_xshh_allt`, seed 20260520.

## Eval protocol — identical to #322 (clean paired comparison)
Frozen backbone scored with a fresh **2L and 6L** quantile q-head (30k steps,
transformer, causal, head-ffn-mult 4.0, dropout 0.1, `--head-train-input e_then_f`,
`--reconstruction forecaster`, forecast-len 16, batch 256, cosine LR, `--amp-dtype none`).
GIFT-Eval `--strategy B4`, full-97 + triage-11 (local data on elisa).

## Targets
full-97 & triage-11 GM-Relative MASE per {2L, 6L}; paired-bootstrap Δ vs #322's
allt·10% (2L = 1.222, 6L = 1.191) over the 97 shared configs; vs **v11c = 1.292** and
seasonal-naive = 1.0. Does the crossfade move the best cell below #322's 1.191?

## Compute & orchestration (elisa, GPU 1)
GPU 0 is held by foreign Jupyter kernels (~17 GB); GPU 1 is free. Single process @1024
on GPU 1 (~12–21 GB, GRU-checkpointed) — pools all 1024 in the negatives natively.
Backbone ~12 h; the two q-heads + 4 evals ~6–8 h. A laptop-side sync_loop pulls the
backbone/optimizer/losses/log from elisa every 15 min (offsite backup + local analysis);
elisa's own `--save-every 2500` is the resume net if the box reboots.
