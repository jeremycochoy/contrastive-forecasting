# #313 execution log (operational; science lives in ../align_floor_loss_B.md)

## Question
Does adding the two opt-in loss features from PR #312 to the **(B)** recipe
close the full-97 GM-MASE gap to **v11c** (1.292)?
- **(B)** = `2026-05-19_crossed_loss_ablation` `cl_hh_50k`: bneck d=128/h=4,
  fp16 body, τ=0.10, β2=0.95, dropkey 0.70 shared, hh_negs, pos-in-denom,
  50k. full-97 1.3572 / triage-11 1.4461.
- **v11c** = `2026-05-11_exp_encoder_forecaster` v11c. full-97 1.2920 / triage 1.3878.

## What actually changes vs (B) (from code, src/loss.py)
- `--align-loss-weight 1.0` → adds `L_align = λ·(2 − 2·cos(f_t, sg(h_{t+1})))`
  to the loss (stop-grad on the encoder target `h_{t+1}`). **Affects gradients.**
- `--subtract-contrastive-floor` → subtracts the constant `log(1+N·e^(−1/τ))`
  (a Python float; `N=B·(T+B)` for hh_negs at C=1). **Gradient-neutral** —
  re-bases only the *logged* loss so it reads ~0 at the InfoNCE uniformity floor.
- ⇒ the only training change is **L_align**; the floor flag is cosmetic
  (makes the loss curve interpretable). One new model to train.

## Protocol (matches #309 downstream so numbers are comparable)
1. Train `bb_alignfloor_50k`: (B) recipe + the two flags, 50k, seed 20260520.
2. q-head: 2L causal transformer quantile head, 30k, `--reconstruction forecaster`.
3. GIFT-Eval triage (11) + full (97), strategy B4, forecast-len 16.
4. Plots: `loss.png`, `perdomain_star.png` (radar vs (B) vs v11c),
   `gm_summary.png` (per-arm rectangles vs v11c line).

## Compute
Local elisa, free. No vast, no remote sync (outputs land directly in the
MAIN checkout). Code runs from worktree `/home/jupyter/cf-wt-align-floor`.

## Timeline
- 2026-05-22 ~11:10 BST — picked up #313. Both elisa GPUs busy with another
  session's `bb_bbase_tau08_50k` (#309) 2-GPU DDP run (started 11:09, 50k).
  Did NOT disturb it. Scaffolded dir + scripts while waiting; validated all
  three plot scripts against the (B)/v11c baselines (new arm skipped cleanly).
- 13:08 — that run freed GPU1; backbone training auto-launched (1-GPU bs256).
  Healthy throughout: at step 8900 loss_tau_ref 0.235 vs (B) 0.244, AUC/top1
  1.0, no NaN. fp16+bottleneck stable (as expected — L_align gradient is
  bounded). 50k done 15:25. (InfoNCE floor = log(1+N·e^(−1/τ)) ≈ 1.94 at
  T=256/B=256/C=1; logged loss is floor-subtracted. New arm's converged
  contrastive loss is ~0.35 ABOVE (B)'s on that common baseline; L_align≈0.003
  at convergence — near-redundant, the gap is contrastive not align.)
- 15:25 — q-head 30k started on GPU1. To get the eval numbers faster (user
  request), the full-97 eval is SHARDED across GPU0∥GPU1 via
  `parallel_downstream.sh` (≈halves the ~1.5h single-GPU eval). Mechanics:
  the eval `--config-filter` matches `f"{ds_name}/{term}"` (freq only for
  multi-freq datasets), NOT the freq-bearing output name — shard regexes are
  built on the filter strings (49/48 split, verified disjoint+complete=97).
  Merge recomputes the GM over the union (validated to reproduce (B)'s 1.3572
  exactly). Triage(11) runs separately with the #309 filter. Gated on GPU0
  free-mem ≥10GB (else single-GPU fallback); merge verifies 97 (else fallback).
  A vast box would not help: q-head is single-GPU-bound everywhere; the eval
  shard on elisa's free 2nd GPU is the cheaper/faster lever.
- 16:37 — q-head done (30k); QF promoted (best-ema fell at step 30k, so
  best≡final). 16:37–17:42 — full-97 sharded (GPU0 free=18.4GB) ∥ GPU1; merged
  to 97 configs. 17:45 — triage(11) done. Results: full-97 **1.4308**, triage-11
  **1.6154** — worse than (B) (1.3572/1.4461) and v11c (1.292/1.3878). Verdict:
  L_align(+floor) does not match v11c; it hurts. Science → ../align_floor_loss_B.md.
- 17:50 — wrote ../align_floor_loss_B.md; sub-agent review pass; addressed 2 required fixes
  (panel-1 caption floor/plateau wording; loss_tau_ref "indistinguishable" →
  "marginally lower", which the data show and which sharpens the finding).
