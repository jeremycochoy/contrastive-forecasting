# #316 execution log (operational; the science is in RESULTS.md)

## Setup
- New experiment branch `experiment/2026-05-23-cpc-multistep-linear` off
  `origin/experiments` (826dfdd; includes β/#309 and #315).
- Implementation: `forecaster_kind="linear_cpc"` + `--cpc-k-steps` on
  `ConfigurableModel` / `TransformerBlock` (K linear heads W_k: H→H replace
  the transformer forecaster), a `cpc_multistep` multi-step InfoNCE in
  `src/loss.py`, the `forward_step` multi-step path in the freq-embedding
  `train.py`, and CPC auto-detection in the q-head trainer + GIFT-Eval
  (reads `transformer.cpc_heads.*`). All gated behind defaults that keep the
  legacy transformer path byte-identical.
- CPU sanity tests (`scripts/test_cpc.py`): shapes, loss ≥0 with gradients to
  heads AND encoder, overfit 2.70→0.001, state_dict round-trip via the
  auto-detect, downstream extraction contract, legacy-path regression — all pass.

## Per-arm training recipes (reference — to retrain a variation)

All arms share one base command (`experiments/2026-04-27_freq-embedding/scripts/train.py`):
`--batch-size 256 --lr 1e-3 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
--total-steps 50000 --seed <S> --hf-repo jeremycochoy/gift-pretrain-full-4096
--hf-path small_v1 --t-raw 4096 --n-channels 1 --d-model 384 --n-heads 6
--num-encoder-layers 6 --num-layers 1 --encoder-dropkey 0.70
--encoder-dropkey-share-heads --encoder-dropkey-share-layers --depthwise-conv 3
--deprecated-depthwise-conv 0 --pos-in-denominator --tau 0.10 --rev-norm-kind ewma
--rev-norm-span 128 --encoder-type gru --mixup-p 0.3 --mix-ratio 0.0 --freq-emb-dim 3
--seasonality-emb-dim 3` + fp32 dtypes (`--residual/attn/ffn/conv/patch-emb-dtype fp32`).

Only the forecaster + loss flags differ per arm:

| arm (head / negatives / k) | `--forecaster-kind` | `--cpc-k-steps` | `--loss-shape` | run-name | launcher command |
|---|---|---|---|---|---|
| transformer-1L / β-neg / k=1  (= β) | transformer (d=128 / 4-head bottleneck) | — | cosine_similarity_batch_full_hh_negs | `bb_beta_50k` | β recipe, trained in `2026-05-20_bottleneck_beta2_confound` |
| transformer-1L / β-neg / k=12 | cpc  (`--forecaster-d-model 128 --forecaster-n-heads 4`) | 12 | cpc_multistep | `bb_cpctrf_k12_s<S>_fp32_50k` | `elisa_run.sh <S> <gpu> fp32 12` |
| linear / β-neg / k=1 | linear_cpc | 1 | cpc_multistep | `bb_linbn_k1_s<S>_fp32_50k` | `elisa_run_linear.sh <S> <gpu> 1 beta` |
| linear / β-neg / k=12 | linear_cpc | 12 | cpc_multistep | `bb_linbn_k12_s<S>_fp32_50k` | `elisa_run_linear.sh <S> <gpu> 12 beta` |
| linear / CPC-neg / k=1 | linear_cpc | 1 | cpc_multistep_cpcnegs | `bb_lincn_k1_s<S>_fp32_50k` | `elisa_run_linear.sh <S> <gpu> 1 cpcneg` |
| linear / CPC-neg / k=12 | linear_cpc | 12 | cpc_multistep_cpcnegs | `bb_cpc_k12_s<S>_fp32_50k`  (original run; a fresh run via the launcher would be named `bb_lincn_k12_...`) | `elisa_run_linear.sh <S> <gpu> 12 cpcneg` |

Seeds: `20260520` for every arm; `20260523` only for the *linear / CPC-neg / k=12* second seed (`bb_cpc_k12_s20260523`, via the `r2` resume — see below).

Downstream (identical for every arm): `downstream.sh <runs>/<run-name>_FINAL.pth <2|6> <gpu> both`
→ quantile head 30k + GIFT-Eval triage(11) + full(97); the CPC backbone is auto-detected from the checkpoint.

**To retrain a variation:** change the relevant flag (e.g. `--cpc-k-steps`, `--forecaster-kind`, `--loss-shape`, or any base hyperparameter) and rerun the launcher with a **fresh `--run-name`/seed** so nothing is clobbered.

## 2026-05-23 ~00:08 — first launch (fp16, β's precision) DIVERGES
Two seeds launched 1-GPU bs256, fp16 body (β's exact precision), lr 1e-3.
Both diverged identically: raw loss bottomed at step ~300 (≈4.60) then climbed
monotonically (≈5.56 by step 900); ff/fp collapsed together (0.96→0.08), AUC
0.95→0.77, dim-usage stuck ~0.003. QK^T logits only ~2664 (far below the fp16
65504 ceiling), so not a hard overflow — an fp16 × high-lr optimisation
instability. (Mirrors #309's fp16 no-bottleneck divergence class.)

## 2026-05-23 ~00:20 — precision/lr probe (single-variable)
Killed both; relaunched two single-variable changes in parallel:
- **fp32 @ lr1e-3** (only precision changed): STABLE — loss 5.88→0.735 by
  step 700, ff→0.929, AUC/Top1→1.0, dim-usage rising.
- **fp16 @ lr3e-4** (only lr changed): STABLE — loss→0.63 by step 1300, AUC 1.0.
Conclusion: the divergence is an fp16 × high-lr interaction; raising precision
OR lowering lr each fixes it independently. Adopted **fp32 @ lr1e-3** for all
CPC runs — it keeps β's lr (only precision differs from β) and matches the
precision of the v11c champion (also fp32), so the comparison stays clean.

## Runs of record (fp32, lr1e-3, β2=0.98, τ=0.10, dropkey 0.70, 50k, bs256)
- seed 20260520 (β's seed) — GPU1.
- seed 20260523 — GPU1, serial after seed 20260520.
Downstream (GPU0, as checkpoints land): q-head 30k (small 2L + 6L), GIFT-Eval
triage(11) + full(97); CPC backbone auto-detected. Steps-curve: small-head
triage on periodic checkpoints.

## 2026-05-23 — seed A trained; headline evals; disk-full incident
- seed A (20260520) trained cleanly to 50k in 3.7h (best_loss@49.2k,
  best_gap@45.6k — both near convergence, no early best_gap≪best_loss
  decoupling). Final contrastive metrics healthy: AUC/Top1=1.0, R²_naive≈0.96,
  gap≈0.75, dim-usage rising 0.013→~0.15.
- **Disk-full incident.** The shared elisa disk (1.8T, ~97% full from other
  projects) briefly hit 0 free; seed A's small-head q-head crashed at step 21k
  with `OSError: No space left on device` (so did the 40k steps-curve q-head).
  Disk later recovered to 100+G free (external churn). Re-ran the small head
  manually on GPU0; serialized seed B's small head behind it (gpu0_queue.sh) to
  avoid double-booking GPU0. The 6L head + steps-curve(10/20/30k) were
  unaffected. (A bulk cleanup of optimizer files was attempted but declined;
  not needed once disk recovered.)

## Results so far (full-97 GM-Relative MASE; refs β=1.3272, v11c=1.292)
- seed A 6L head: **1.4722** — well above β (worse). 
- steps-curve (small-head triage, 11 cfg): 10k=1.659, 20k=1.663, 30k=1.660 —
  flat; β triage=1.484. Downstream transfer does not improve with training and
  sits above β throughout.
- Pending: seed A small-head full; seed B (both heads) for the variance check.

## 2026-05-23 — DESIGN CORRECTION after PR #317 review (jeremycochoy)
The first implementation had **two confounds** vs the question "does k=12 improve
β"; both fixed:
1. **Negatives.** The loss used a CPC-canonical negative pool, not β's. Rewrote
   `cpc_multistep` to reuse β's exact `cosine_similarity_batch_full_hh_negs`
   negatives (xy/xx/zy/encoder-all-time-hh/cross-batch, batch-pooled) and change
   ONLY the positive to the multi-step average over k. Unit test: at k=1 the loss
   equals β's `full_hh_negs` exactly (7.468332, C=1).
2. **Forecaster head.** The first run REPLACED β's transformer forecaster with
   linear heads — confounding forecaster-type with k, and not CPC-faithful (CPC
   keeps the autoregressive context = β's transformer forecaster). Now the
   forecaster is **K transformer-1L heads, each architecturally identical to β's
   forecaster** (d=128/4-head bottleneck), head k → h_{t+k}; K=1 ≡ β.

The earlier numbers (6L 1.4722 / small 1.5240) were the confounded
(linear + CPC-negs) family — they become study arm **#3 (k=12)**, not the headline.

## Study plan (priority order, per review)
- **#1** (priority): k=12 transformer-head (β-arch, β-negs) vs β (1.3272).
  fp32 bs256 fits at 20.9/24.6 GB. Training (seed 20260520).
- **#2**: linear-head k=1 & k=12, β-negs.
- **#3**: k=1 baseline for the original linear-head + CPC-negs family (k=12 done).
Goal: check the k-trend is consistent across families.

## Compute
All on elisa (free), 2× RTX 4090. No vast spend.
