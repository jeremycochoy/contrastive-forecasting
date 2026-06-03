# Experiment log — joint ARMA + correlation, channel-mixing on top

Operational notes that don't belong in `../contrastive-arma-correlation.md`. Kept for traceability.

## Loss bug found mid-experiment

The first run (V6) used `cosine_similarity_batch_no_time_neg` as written. Audit
showed it was missing `neg_xy_hat`, the same-time cross-channel negative on
the forecaster: `cos(h^{b,t,c1}, h_hat^{b,t,c2})` for `c1 ≠ c2`. The variant
correctly drops the cross-time negatives but had silently dropped this same-
time term too. Without it the only cross-channel pressure in the loss is on
`h × h` (encoder spread), which doesn't gradient through `h_hat`.

Fix: `src/loss.py:133–172`, commit `7443a77`. We re-ran the full pipeline
(backbone + both heads) as V7 to verify the fix produced the intended
mechanical change. It did — `CC(h, h_hat)` flipped past zero from +0.017
(V6) to −0.013 (V7), see `plots/v6_v7_ratios.png`. Backbone gap and ARMA
recovery were essentially the same, so V7's checkpoint is what
`../contrastive-arma-correlation.md` discusses; V6's plots are kept under `plots/` only as the
side-by-side overlay (`v6_v7_compare.png`, `v6_v7_ratios.png`).

## Optimisation note

V7 had a delayed phase transition vs V6 (cross-batch differentiation
kicked in at step ~14k instead of ~4k, then caught up by ~step 100k).
The added cross-channel constraint slows the early ramp; the final
state is comparable. See `plots/v6_v7_compare.png`.

## Head architecture realisation

The first correlation head (`JointCorrelationHead`) wraps the V5
`GRUCorrelationHead` with a `Linear(C·H → H)` projection so the V5
shape matches our `[B, T, C, H]`. That linear projection is sample-
independent and cannot compute second-order cross-channel statistics
(`h^{b,c1} · h^{b,c2}`), which is where per-sample correlation lives.
We added `JointCorrelationHeadDirect` that feeds `[B, T, C·H]` to the
GRU directly so the recurrent gates' nonlinearity can pick up
cross-channel quadratic structure over time.

## Files / commits

- Loss fix: `src/loss.py:133–172`, commit `7443a77`.
- V6 → V7 rerun: commit `7837f48`.
- Relative-gap analysis: commit `f8cc6c8`, script `analyze_ratios.py`.
- Direct correlation head class + CLI flag: in `train_correlation_head.py`.
