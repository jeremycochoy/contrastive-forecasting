# Bottleneck + full-fh-negs (normalized InfoNCE) + 2-GPU DDP — 50k

> Note: this is the original 2L-forecaster/bf16 plan (Arm 1, which
> diverged). The run pivoted to **1L forecaster + fp16** (Arm 2, stable);
> see [`RESULTS.md`](RESULTS.md) for the executed config and outcome.

Authoritative spec for this run. Autonomous execution; on divergence the run
is stopped and the user notified (no auto-restart).

## Question
On the v13-style forecaster-bottleneck backbone, does the new
all-(f_t, h_l)-negatives loss under the **normalized InfoNCE** objective
(positive in numerator *and* denominator) train stably and well at
dropkey 0.70, with a 2-layer (instead of 1-layer) forecaster, on 2 GPUs
(global batch 256, full cross-rank negatives)?

## Exact recipe

Code: worktree `/home/jupyter/cf-wt-bottleneck-fullfh` @ `origin/experiments`
(6fdfe89) + PR #294 (`feat/conv-dtype-independent-knob`, commit dcabeb3).
Entrypoint `experiments/2026-04-27_freq-embedding/scripts/train.py`.

| Group | Setting | Flag |
|---|---|---|
| Bottleneck arch (v13) | enc 6L @ d384/6h, **fcst 2L @ d128/4h** Linear bottleneck | `--num-encoder-layers 6 --num-layers 2 --d-model 384 --n-heads 6 --forecaster-d-model 128 --forecaster-n-heads 4` |
| Dropkey | 0.70, shared heads + layers (v11c/v13/v16 family) | `--encoder-dropkey 0.70 --encoder-dropkey-share-heads --encoder-dropkey-share-layers` |
| Optimizer | AdamW, wd 0.1, β1 0.9, **β2 0.95**, lr 1e-3 | `--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --lr 1e-3` |
| Loss | full (f_t,h_l) negatives ∀ l≠t+1, **normalized InfoNCE** (pos in num+denom, ≥0) | `--loss-shape cosine_similarity_batch_full_fh_negs --pos-in-denominator` |
| Negatives | full **cross-rank** global pool (NOT sharded) | *(no `--shard-loss-on-batch`)* |
| Precision | residual **fp32**; attn/ffn/**conv** **bf16**; patch-emb fp32 | `--residual-dtype fp32 --attn-dtype bf16 --ffn-dtype bf16 --conv-dtype bf16 --patch-emb-dtype fp32` |
| Instrumentation | attention + residual amplitude every 200 steps | `--log-attn-amplitude --log-attn-amplitude-every 200` |
| Budget / batch | 50k steps; **128/GPU × 2 GPU = 256 global** | `--total-steps 50000 --batch-size 128` + `torchrun --nproc_per_node=2` |
| Data | gift-pretrain-full-4096 / small_v1 | `--hf-repo jeremycochoy/gift-pretrain-full-4096 --hf-path small_v1 --t-raw 4096 --n-channels 1` |
| Family constants | τ=0.10, ewma span128, new conv, GRU, mixup 0.3 | `--tau 0.10 --rev-norm-kind ewma --rev-norm-span 128 --depthwise-conv 3 --deprecated-depthwise-conv 0 --encoder-type gru --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3 --seasonality-emb-dim 3` |
| Seed | 20260517 (fresh; from-scratch family used 20260516) | `--seed 20260517` |
| Checkpoints | every 5000 (+ `_optimizer.pth` companions) | `--save-every 5000` |

Run name: `enc_fcst_bneck128_dk07_fullfh_norminfonce_ddp_50k`

## Decisions made autonomously (auditable)

1. **conv bf16 + residual fp32 was not expressible** in `origin/experiments`
   (conv was hard-tied to `--residual-dtype`). The user's own documented,
   load-bearing rule is *fp32 residual is the divergence anchor; never
   compromise it* (RESULTS.md / EXPERIMENT_LOG fp16 study). Resolution:
   added an independent `--conv-dtype` knob (PR #294, byte-identical
   default) so the user's full spec — residual fp32 **and** conv bf16 —
   is honored exactly. Per user instruction during the session.
2. **patch-emb dtype = fp32** (user did not specify; spec named only
   attn/conv/ffn). The GRU patch-emb in bf16 is the riskiest axis in the
   fp16 study; fp32 is consistent with the "residual fp32" safety intent
   and every stable family run. Easy to revisit.
3. **dropkey share-heads + share-layers**: the "bottleneck architecture of
   the previous experiment" is v13, which used both; kept for consistency.
4. **Full cross-rank negatives** (no `--shard-loss-on-batch`): "cover all
   negatives" + global batch 256 ⇒ the gathered global pool, byte-loss-
   equivalent to single-GPU @ 256.

## Divergence rule (watcher: `scripts/watch_divergence.sh`)
Stop + notify (no restart) on ANY of:
- NaN/Inf in the `loss` column of `<run>_losses.csv`;
- an `<run>_EMERGENCY_*.pth` checkpoint appears;
- post-warmup (step ≥ 3000) `loss` > 2.5 × post-warmup running-min **and**
  still rising over the last window;
- torchrun process group dies before `<run>_final.pth` (crash).
Clean stop: `<run>_final.pth` written / last step ≥ 50000 → report.

## Layout
- code: `/home/jupyter/cf-wt-bottleneck-fullfh` (worktree, PR #294)
- artifacts (main checkout, survives worktree teardown):
  `experiments/2026-05-17_bottleneck_fullfh_ddp/{runs/ (ckpts+CSVs, *.pth gitignored), results/ (logs), plots/}`
