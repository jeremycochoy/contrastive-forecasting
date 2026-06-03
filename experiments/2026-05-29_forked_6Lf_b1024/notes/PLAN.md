# #322 — Forked arms × 6L forecaster, retrained at batch 1024

## Question
#320 trained #318's five **data-side forked** arms (forked-ARIMA continuations:
identical past → divergent futures) with a 6-layer forecaster at **global batch 256**.
No 6Lf cell crossed β. #322 asks: does enlarging the contrastive **negative pool 4×**
— global batch **1024**, with **all 1024 terms together in the negatives** (not sharded
per-GPU) — change where the fork helps vs hurts, and does any arm now reach β / v11c?

A bigger all-together negative pool is the one knob that changes here. Everything else
is byte-identical to #320.

## The five arms (each retrained at batch 1024)
| arm | base loss | mix-ratio (fork fraction) | #320 b256 full-97 (2L / 6L) |
|---|---|---|---|
| **β·0.8%**   | `…full_hh_negs`            | 0.0078125 | 1.4369 / 1.4006 |
| **β·10%**    | `…full_hh_negs`            | 0.10      | 1.5662 / 1.6832 |
| **allt·0.8%**| `…full_hh_negs_xshh_allt` | 0.0078125 | 2.2180 / 1.8483 |
| **allt·10%** | `…full_hh_negs_xshh_allt` | 0.10      | 1.5973 / 1.6939 |
| **allt·50%** | `…full_hh_negs_xshh_allt` | 0.5       | 1.4608 / 1.5387 |
(b256 columns are #320's measured 6Lf scores — the paired reference for #322's Δ.)

## How batch 1024 with "all negatives together" is realised
2-GPU **DDP**: `--batch-size 512` per rank, `torchrun --nproc_per_node=2` → global 1024.
`--shard-loss-on-batch` is **OFF** (default), so `train.py` all-gathers per-rank latents
(`DifferentiableAllGather`) and the contrastive loss pools its negatives over the full
1024-batch — provably equal to a single-process @1024 run
(`tests/test_dist_gather.py`). This is the only configuration that puts **all 1024 in
the negative pool**; `--shard-loss-on-batch` (per-rank negatives) is explicitly *not*
used, as the card requires all terms together.

Why DDP and not 1 GPU: measured — a single card OOMs in the GRU patch-encoder forward
at batch 1024 (the GRU op alone needs ~17.9 GB). DDP splits the forward to 512 seqs/rank
(~22.5 GB, the memory wall), then gathers the small latents for the full-1024 loss.
See EXECUTION_LOG for the memory/sps measurements.

## Step budget — **same data as #320 (12.5k steps)**, decided, not asked
#320 ran 50k steps × batch 256 = 12.8 M samples. At batch 1024 the same data is
**12.5k steps**. We hold *total data seen* equal to #320 and vary only the
negative-pool size — the clean isolation of #322's question. Two further reasons:
1. **Feasibility.** The all-time arms' cross-series Gram is (B·T)², ~16× the b256 cost
   per step; 50k steps would be days/arm on a shared 2-GPU box. 12.5k is ~10 h/arm.
2. **No schedule confound.** The backbone LR is constant (no warmup/decay), so fewer
   steps simply means less data — nothing else changes.
All five arms use the same 12.5k budget for within-#322 consistency. (Reported plainly;
the alternative — 50k steps = 4× more data than #320 — would confound pool size with
training length and is infeasible for all-time.)

## Eval protocol — **identical to #320** (so b256↔b1024 is a clean paired comparison)
Each frozen backbone scored with a fresh **2L and 6L** quantile q-head: 30k steps,
transformer, causal, head-ffn-mult 4.0, dropout 0.1, `--head-train-input e_then_f`,
`--reconstruction forecaster`, forecast-len 16, **bs 256**, cosine LR, β2 0.98,
`--amp-dtype none`. GIFT-Eval `--strategy B4`, full-97 + triage-11. The q-head batch
stays 256 — only the *backbone's* contrastive batch changed, which is what #322 varies.

## Targets
full-97 & triage-11 **GM-Relative MASE** per arm × {2L, 6L}; paired-bootstrap Δ vs
#320's b256 6Lf scores on the 97 shared configs; vs **β** (2-seed range), **v11c =
1.292**, seasonal-naive = 1.0. Where does 4× more all-together negatives help vs hurt
the fork? Does any b1024 cell beat β or reach v11c?

## Compute & orchestration (elisa, 2× RTX 4090)
Backbone DDP needs **both** cards near-empty (~22.5 GB/rank). A GPU-gated orchestrator
launches the 5 backbones sequentially the moment both GPUs are free (β arms first), is
idempotent (skips FINAL-complete arms) and resumable (periodic checkpoints + `--resume`).
Downstream q-heads/evals are light (bs 256, ~10.5 GB) → one per GPU, no DDP, can run on
whichever card is free. A 1 h wake-up loop supervises the multi-day run.
