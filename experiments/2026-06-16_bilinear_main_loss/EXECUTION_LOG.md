# Execution log — #350 learnable bilinear W in the main contrastive loss

Operational notes (kept out of the report, per REPORT_STANDARD "science, not journey").

## Design decisions
- **One trained arm.** Only the bilinear-W backbone is trained here. The τ
  baseline is #348's saved + CPC no-encoder arm (same code — this branch is cut
  from #348's `experiment/2026-06-15-no-encoder-redo` — same machine, same seed
  20260520, same recipe). Its saved GIFT-Eval per-task results
  (`.../2026-06-15_no_encoder_redo/results/gift_eval_full_*_cpc*`) reproduce the
  #348 report exactly (2L 1.1678/1.1646, 6L 1.1532/1.1600) and drive the paired
  bootstrap, so no baseline retrain is needed.
- **W placement (run-1, the canonical form).** The positive scores
  `s(f_t, h_{t+1}) = (W h_{t+1}) · f_t = f_tᵀ W h_{t+1}` (W on the target
  `h_{t+1}`). The cross-batch f↔h negative scores `(W f_t) · h'_{t+1} = f_tᵀ Wᵀ h'_{t+1}`
  (W on the forecast). The two terms use bilinear forms transposed of each
  other — equivalent up to a transpose for a free learnable W; at init
  `W = (1/τ₀)·I` is symmetric so both forms coincide. The h↔h uniformity
  terms keep `W` on their latent anchor `h_t`. The xs_allt cross-series Gram
  pre-projects its anchor and runs with τ=1 so the existing chunked/fused
  autograd backward is untouched.
- **Inference applies Wᵀ to the forecaster output, per step.** Under this
  placement the predicted next encoder latent is `Wᵀ f_t` (the maximum-
  correlation direction of `h` against `f` under `(W h)·f`). Implemented in
  `extract_forecaster_latents` (covers the sliding-window value-rollout and
  the head trainer's default path) and inside `rollout_latent` (per step,
  applied to the new token before both `generated` and the recurrent
  feed-back into `seq`). New gate `test_inference_bilinear_W.py` verifies
  byte-equality with the baseline at W=I and a visible change at W≠I.
- **Run history.** Run-1 (W on target, this form) ran ~9.3k steps then was
  swapped at 2026-06-18 08:19 by a prior agent who misread the cross-side
  use of W as incoherent and put W on the forecast in both terms (run-2).
  Run-2 was started fresh and reached ~3.9k steps before being stopped
  (2026-06-18 12:43, no clear sign of training-time instability, but the
  formulation change was unnecessary). Reverted to run-1's loss code,
  added the Wᵀ-per-step inference correction (the missing piece from
  before), archived run-2 under `_run2_aborted/`, relaunched from scratch.
- **W init (1/τ₀)·I = 10·I** so step 0 is the τ=0.10 baseline byte-for-byte; W
  then learns freely. **Excluded from weight decay** (it is the loss's
  temperature/scale analog; the scalar τ it replaces is fixed and not decayed —
  decay would shrink it ~70% over 12.5k steps and confound the comparison).
- Memory knob `XSHH_ALLT_CHUNK=16` (vs #348's 1): byte-identical to the loss
  (memory↔kernel launches), ~3.6 s/step vs ~5 s/step.

## Correctness gates (all passed before launch)
- `test_main_bilinear_W.py`: W=(1/τ)I == τ baseline (Δ=0, fp64) on both the
  default checkpoint path and the fused autograd Function; grad on W matches
  finite differences; non-identity W changes the loss.
- Full repo test suite: 198 passed (incl. the allt-Gram speedup tests).
- Smoke: bilinear arm trains (W moves off 10·I by gradient only, off-diagonal
  grows, weight-decay group correct); baseline arm unchanged (no main_w in
  state_dict; param count 11,579,196 vs 11,726,652 = +147,456 = 384² for main_w).
- Resume verified with the 2-param-group optimizer.

## Runs
- Backbone (bilinear): GPU1, batch 1024, 12,500 steps, seed 20260520.
  Launched 2026-06-17 22:06. Output:
  `/home/jupyter/workspaces/contrastive-forecasting/experiments/2026-06-16_bilinear_main_loss/`.
- Downstream: 2L + 6L quantile heads at the best-loss and last backbone, then
  GIFT-Eval full-97 — exact #348/#344 head + eval recipe.

## Repo note
The goal references `contrastive-forecasting#350`. There is an unrelated
`jeremycochoy/rnd#350` (a production diversity report, already merged as rnd
PR #352) — a coincidental issue-number collision in a different repository.
