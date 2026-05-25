# #318 — execution log

Operational journey for this card. The science is in [`RESULTS.md`](RESULTS.md);
this file holds the rest (artifacts, corrections, infra) so the report stays
fact/protocol/science.

## 6L-forecaster variant (trained, evaluation paused)

A user-requested follow-up: redo both loss-side arms with a **6-layer forecaster**
(the 1L → 6L forecaster change; encoder stays 6L). Both backbones trained to 50k:

- `runs/bb_xshh_6Lf_50k_FINAL.pth` (same-step)
- `runs/bb_xshh_allt_6Lf_50k_FINAL.pth` (all-time)

Downstream evaluation was **paused** (user: "cancel the 6L-forecaster for now",
2026-05-25) to prioritise the fork. Only one cell landed before pausing:
same-step-6Lf, 2L head, triage-11 = **1.5177**. The all-time-6Lf q-head eval had
first crashed under `--amp-dtype bf16` (a depthwise-conv dtype mismatch in
`extract_forecaster_latents`: input cast to bf16, weight stayed fp32); the driver
`downstream_6Lf.sh` was fixed to `--amp-dtype none`. Backbones are on disk and the
follow-up is resumable from them without retraining.

## Forked-continuation injection fraction (corrected)

The data-side fork was first run at **`--mix-ratio 0.5`** (128 synthetic rows per
256-batch) — ~64× too much. The intended design injects a **single forked pair
(2 samples) per batch** (`--mix-ratio 2/256`), keeping the other 254 rows real so
transfer is not confounded by a synthetic-distribution shift. `train_backbone_forked.sh`
was corrected (run-name `bb_xshh_allt_forked2_50k`); the 50%-mix run
(`bb_xshh_allt_forked_50k`) is retained and reported as the `forked, 50% mix` arm.
Both are first-class arms differing only in injection fraction; the 2/batch arm is
the clean isolation of the fork. Generator integrates (ARIMA d=1) deterministically
— `cumsum` of the shared prefix stays identical, preserving the fork (kept as-is
per user).

## Compute / infra

Local elisa, 2× RTX 4090, shared with a concurrent agent's job — GPU picked at
runtime from whichever was free; never disrupted the other agent's work. Each
backbone ~4–5 h on one 4090; the all-time loss runs ~2.2× the same-step step time
(6.4 → 2.9 sps; the B²·T² cross-series×cross-time Gram is chunked + gradient-
checkpointed, so the cost is compute, not the ~1 GB tensor). The eval matrix is an
idempotent driver (skips completed cells); plots regenerate incrementally as cells
land (`scripts/plots.py`).
