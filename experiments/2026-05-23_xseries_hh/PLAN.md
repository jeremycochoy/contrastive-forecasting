# #318 — β + cross-series same-step h↔h negatives (deny the positional shortcut)

## Question
Does repelling, at **every** step `l`, what *different* series share at that
step — `cos(h_{b,l}, h_{b',l})`, b ≠ b' — push the backbone to encode
forecastable **content** instead of a content-free **positional** code, and so
improve transfer over **β**?

## Hypothesis
β's within-series cross-time negatives `cos(h_t, h_l)` can be satisfied by a
per-step positional fingerprint that is *shared across all series* — a
content-free code that costs no forecastable structure. At a fixed step the
only thing different series share is that positional component; a cross-series,
same-step encoder repulsion penalises it directly, moving distinctness onto
(series-specific, forecastable) content.

## The change (one clean edit on top of β)
New loss shape `cosine_similarity_batch_full_hh_negs_xshh` = β
(`cosine_similarity_batch_full_hh_negs`) with exactly two edits:
1. **ADD** `cos(h_{b,t}, h_{b',t})` ∀ b' ≠ b, anchored at h_t (every step l = t)
   — cross-series, same-step encoder repulsion.
2. **REMOVE** the adjacent `log_neg_xy` = `cos(h_t, h_{t+1})`; at C = 1 it is
   byte-for-byte the l = t+1 slice already inside β's all-time `cos(h_t, h_l)`
   term, so dropping it de-duplicates.
Everything else (positive, xx, zy, all-time h↔h, cross-batch f↔h) byte-for-β.
Pinned by `tests/test_loss.py::TestCrossSeriesSameStepHH` (fp64 reference).

## Recipe (byte-identical to the #309 β arm except `--loss-shape`)
GRU patch-enc → 6L causal encoder → 1L forecaster, bottleneck d=128/h=4,
AdamW β2=0.98, τ=0.10, dropkey 0.70 shared, fp16 body / fp32 residual+patch-emb,
ewma span128, seed 20260520, 50k, global batch 256, `--pos-in-denominator`.

## Targets
- full-97 & triage-11 **GM-Relative MASE** vs **β = 1.3272** and **v11c = 1.292**
  (lower is better; 1.0 = seasonal naive). Each frozen backbone eval'd with
  **both** the 2L (small) and 6L q-head (#315/#316).
- **GM-MASE vs training step** — does the "more training stops helping"
  decoupling shrink? (full-97, 2L head, at {20k, 35k, 50k} for mine and β.)
- Per-domain breakdown — flag collateral damage to genuinely shared structure
  (same-frequency seasonal phase) on strongly seasonal domains.

## Eval protocol (identical to #309 / #315 so numbers are comparable)
q-head: 30k, transformer, causal, head-ffn-mult 4.0, dropout 0.1,
`--head-train-input e_then_f`, `--reconstruction forecaster`, forecast-len 16,
bs256, cosine LR, β2=0.98. Eval: `eval_gift_eval_official.py --strategy B4`.

## Compute
Local elisa. Backbone: 1×4090 bs256 ~4 h. Eval matrix grinds on whichever GPU
frees first (idempotent driver, skips completed cells).
