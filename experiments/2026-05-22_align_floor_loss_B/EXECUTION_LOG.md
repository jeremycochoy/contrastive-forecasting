# #313 execution log (operational; science lives in RESULTS.md)

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
