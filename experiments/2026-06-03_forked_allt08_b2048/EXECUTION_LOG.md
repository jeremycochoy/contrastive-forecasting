# #327 — execution log

Operational journey for this card (infra, decisions, corrections). The science is in
[`RESULTS.md`](RESULTS.md); the design rationale is in [`PLAN.md`](PLAN.md).

## Base branch
Branched from `experiments` (28bf980, after #326). All the code #327 needs — the forked-ARMA
synth, the all-time loss `cosine_similarity_batch_full_hh_negs_xshh_allt`, qk-norm /
attn-out-norm, `--subtract-contrastive-floor`, the GRU checkpoint env gates, the DDP
all-gather — is already merged via #322 (PR #324). The only new code is the backbone-transformer
checkpoint (below).

## Compute layout (elisa, 2× RTX 4090, 24 GB each)
- GPU 1: free (~22.6 GB) apart from one ~1.5 GB foreign tenant. **Training card.**
- GPU 0: ~4.5 GB permanently held by 5 foreign `rnd` Jupyter kernels (idle, multi-day). Per the
  shared-machine rule, never touched. Used only for the light downstream q-heads later.

## Fitting batch 2048 — three measured attempts
1. **Single-GPU @2048, no extra checkpointing → OOM.** The backbone *transformer* forward OOMs at
   ~22 GB (`blocks.py` FFN); the GRU-encoder path's main transformer is not gradient-checkpointed,
   and there is no CLI flag for it.
2. **2-GPU DDP @1024/rank (the #322-built all-gather) → OOM on GPU 0.** The forward fits at
   1024/rank, but the gathered 2048-pooled all-time loss needs ~19.1 GB/rank, which exceeds GPU 0's
   ~19.6 GB free (foreign-squeezed) by ~64 MB. GPU 1 alone would fit, but DDP needs both cards.
   This is #322's "GPU 0 is occupied → single-GPU pivot", one batch-doubling on.
3. **Single-GPU @2048 + backbone checkpoint → fits.** Added `BACKBONE_CKPT` (env-gated, training-only)
   in `src/blocks.py`: gradient-checkpoints the backbone transformer's non-fp32 encoder/forecaster
   layers, mirroring the existing `PATCH_ENC_CKPT` for the GRU. Masks are built outside the
   checkpoint (no RNG dependence); the fp32 last-layer boundary is left un-checkpointed. Measured on
   GPU 1: **~20.5 GB peak**, fwd 6.4 s + bwd 7.7 s = **~14 s/step → ~24.5 h / 6250 steps**.

## Byte-identity of the backbone checkpoint
A matched 8-step run with `BACKBONE_CKPT=0` vs `=1` (same seed, batch 256, full recipe) gives
**bit-identical** loss and gap at every step (13.9947 / 13.6120 / 11.9569 / 9.6115 / 8.4499 / 7.9823).
Identical trajectories over multiple optimiser steps prove the forward *and* backward are byte-identical
— checkpointing only trades stored activations for recompute. So the trained backbone equals the recipe
run un-checkpointed; the flag is a pure memory device, reported as infra, not a recipe change.

## Notes
- `--log-attn-amplitude` is OFF for this run: under checkpointing the amplitude side-effect logger
  would double-count recomputed layers. Collapse is instead read from train.py's standard signals
  (forecast-vs-future gap, cross-batch cosine, retrieval AUC/Top1, R²) — the same signals #322 used
  to characterise its b1024 collapse.
- `XSHH_ALLT_CHUNK=4`, `PATCH_ENC_CHUNK=8` (2048/8 == 1024/4 seqs/chunk, matching #322's GRU chunk
  size). Both are numerically exact memory/speed knobs.
- Step budget 6250 (= 12.8 M samples, #322's data) chosen to isolate the negative-pool size from
  training length. Periodic checkpoints every 1250 steps; `--resume` on relaunch (crash-safe).

## FINAL checkpoint selection — corrected (best_loss was an early checkpoint)
The training wrapper copied `best_loss.pth` to FINAL, as #322's did. That works when the
floor-subtracted loss descends monotonically (it did at b1024). At b2048 the loss is
**non-monotonic** — lowest at **step ~1089** (1.78), then it rises onto the bumpy plateau and never
returns below ~2.2. So `best_loss` was a heavily *under-trained* step-1089 checkpoint, not the
trained backbone. Caught at the downstream hand-off (the first final q-heads had already started on
it); repointed FINAL to `_final.pth` (the step-6250 end-of-training weights — the honest "fully
trained" backbone and the comparator to #322's b1024 end-of-budget eval), re-ran the final cells,
and changed the wrapper's selection to prefer the end-of-training checkpoint over `best_loss`. The
plateau-peak cells were unaffected (they load the explicit step-2500 `_2k.pth`).

## Plateau test (added)
A second set of cells trains a fresh 2L/6L head on the **step-2500 plateau-peak** checkpoint (the
floor-subtracted loss's local maximum, `_2k.pth`) alongside the step-6250 cells — #322's plateau
test, repeated at b2048: does the training tail past the plateau buy forecasting skill? (It does
not — see RESULTS.)

## Downstream compute — shared-box contention
GPU 1 freed when the backbone finished, then was immediately taken by a foreign job
(`/tmp/cf-328`, a batch-1024 ×12500 run) for the whole downstream phase — untouchable per the
shared-machine rule. So the four q-head + GIFT-Eval cells ran **serially on GPU 0** (gated so a
q-head never co-runs with another cell or the foreign tenants), the plateau cells while the backbone
still owned GPU 1, the final cells after. GIFT-Eval full-97 is slow here (~3 h/cell — a handful of
long-horizon configs dominate), so the four cells spanned the day. Nothing about the contention
touches the science; it only set the wall-clock.
