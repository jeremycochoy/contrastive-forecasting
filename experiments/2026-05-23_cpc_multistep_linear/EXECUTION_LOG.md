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

## Compute
All on elisa (free), 2× RTX 4090: GPU1 = training chain, GPU0 = eval factory.
No vast spend.
