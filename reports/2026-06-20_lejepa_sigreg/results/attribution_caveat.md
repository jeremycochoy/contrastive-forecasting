# Attribution caveat (#356 review P1.1, P1.3)

## Multi-axis confound is intrinsic to the issue's one-arm spec

The issue asks: *"is SIGReg + B=512 a viable replacement when B=1024 budget is unavailable?"* The single-arm design (`--sigreg-embedding`, `--sigreg-encoding`, `--batch-size 512`) is the answer to that exact joint question. Three axes change versus the #353 EMA-target enc3+CPC reference (B=1024, no SIGReg): SIGReg on `e_t`, SIGReg on `h_t`, batch 1024 → 512. The gm_table head-to-head therefore measures the **joint** perturbation, not any one axis on its own.

Clean isolation of either axis would require additional arms beyond the issue spec — at minimum a no-SIGReg B=512 control (isolates batch) and/or a SIGReg B=1024 control (isolates SIGReg, would need the OOM fix that motivated the half-batch arm in the first place). The issue explicitly asks for a single B=512 arm; those controls are out of scope here and were not run. Any per-axis decomposition of the observed deltas is unsupported by this artefact set.

## Single-seed; deltas at ~0.005 MASE are below typical seed noise

Every cell of `gm_table.csv` is a single-seed measurement (seed `20260520` for this arm, single seeds for #344 / #353 references too). The head-to-head deltas the runner comment cites against the #353 EMA-target reference are:

| head/ckpt | Δ vs #353 EMA |
|---|---:|
| 2L / best | −0.0004 |
| 2L / last | −0.0059 |
| 6L / best | −0.0033 |
| 6L / last | −0.0041 |

Prior arms in this codebase (#338 and adjacent) flagged cross-arm deltas of this magnitude at this regime (12.5k steps, single seed, GIFT-Eval full-97) as comparable to seed variation — i.e. **the observed deltas are not separable from seed noise** without per-arm multi-seed replicates. Treat the four numeric deltas as descriptive, not as a verdict.

## What this means for the table

The four-cell GM-Rel MASE table is the head-matched fact under the issue's spec. It does **not** support a causal read on SIGReg alone, on batch size alone, or on either SIGReg term individually; and the per-cell deltas vs #353 sit at or below the single-seed noise floor for this regime.
