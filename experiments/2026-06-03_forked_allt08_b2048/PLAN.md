# #327 — allt·0.8% forked arm retrained at batch 2048 (follow-up to #322)

## Question
#322 retrained the five forked arms at contrastive batch **1024** (4× #320's 256),
all 1024 samples pooled in the negatives, behind a two-norm stability fix. The
per-domain standout was **allt·0.8%** (all-time negatives + 0.8% forked-arma
injection): 2L **1.213**, 6L **1.198** GM-Relative MASE on GIFT-Eval full-97, up
from a catastrophic 2L 2.218 / 6L 1.848 at batch 256.

#327 asks the next question on that curve: **double the negative pool again, 1024 →
2048 (all pooled), and does allt·0.8% improve further, or has the pool-size lever
saturated?**

## The one lever
Contrastive batch **1024 → 2048**, all 2048 samples pooled as a single negative set
(`--shard-loss-on-batch` off). Everything else is #322's stabilised allt·0.8% recipe,
byte-for-byte:
`--loss-shape cosine_similarity_batch_full_hh_negs_xshh_allt` (all-time negatives),
`--synth-kind forked-arma --mix-ratio 0.0078125` (0.8% fork),
`--qk-norm --attn-out-norm` (the collapse fix the 4× pool needed at 1024; the 8× pool
needs it at least as much), `--subtract-contrastive-floor --pos-in-denominator`,
AdamW lr 1e-3, τ 0.10, fp16 attention / fp32 residual, seed 20260520.

## Step budget — same data as #322 (decided, not asked)
#320: 50k × 256 = 12.8 M samples. #322: 12.5k × 1024 = 12.8 M. #327: **6 250 × 2048 =
12.8 M**. Holding total data seen equal isolates the negative-pool size (1024 → 2048)
from training length — #322's clean-isolation logic, extended one doubling. The
backbone LR is constant (no schedule), so fewer steps is simply less data, no confound.

## How batch 2048 fits on one card (the "how to fit" the card leaves open)
A single process @2048 OOMs the **backbone-transformer forward** at ~22 GB on a 24 GB
card (the GRU-encoder path's main transformer is not gradient-checkpointed). 2-GPU DDP
@1024/rank with all-gather (the #322-built mechanism, `tests/test_dist_gather.py`)
fits the forward but its 2048-pooled loss needs ~19 GB/rank — which OOMs on this box's
GPU 0 (permanently ~4.5 GB-held by foreign tenants, leaving 19.6 GB), exactly #322's
"GPU 0 is occupied → pivot to single-GPU" situation.

So, continuing #322's pivot: **single GPU @2048 on the free GPU 1**, with the backbone
transformer's non-fp32 layers gradient-checkpointed (`BACKBONE_CKPT=1`, new env-gated
flag in `src/blocks.py`, mirroring the existing `PATCH_ENC_CKPT` for the GRU). The GRU
patch-encoder is also checkpointed+chunked as in #322. Checkpointing is **byte-identical**
— recompute trades stored activations for compute; a matched 8-step run with the flag
off vs on gives bit-identical loss and gap, so the trained backbone equals the recipe
at 2× batch. This pools all 2048 in the negatives natively (one batch, no gather).

## Eval protocol — identical to #322 (clean paired b1024 ↔ b2048)
Frozen backbone scored with a fresh **2L and 6L** quantile q-head: 30k steps,
transformer, causal, head-ffn-mult 4.0, dropout 0.1, `--head-train-input e_then_f`,
`--reconstruction forecaster`, forecast-len 16, **bs 256**, cosine LR, β2 0.98,
`--amp-dtype none`. GIFT-Eval `--strategy B4`, full-97 + triage-11, **GM-Relative MASE**.
The q-head batch stays 256 — only the backbone's contrastive batch changed.

## Targets
full-97 & triage-11 GM-Relative MASE per {2L, 6L}; paired-bootstrap Δ vs #322's
**b1024** allt·0.8% (2L 1.213, 6L 1.198) on the 97 shared configs; context lines at
**b256** (2L 2.218, 6L 1.848), the strongest prior backbone **1.29**, and seasonal-naive
**1.0**. Verdict: does the 1024 → 2048 doubling still buy a reliable gain, or has the
negative-pool lever flattened?
