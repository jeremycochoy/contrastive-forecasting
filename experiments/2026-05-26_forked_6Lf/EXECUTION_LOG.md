# #320 — execution log

Operational journey for this card. The science is in [`RESULTS.md`](RESULTS.md);
this file holds the rest (recipe provenance, infra, corrections) so the report
stays goal / protocol / science.

## Recipe provenance
The five arms are #318's data-side forked arms run verbatim except the forecaster
depth: `--num-layers 1 → 6` (the 6L causal *encoder* `--num-encoder-layers 6` is
unchanged). The training driver `scripts/train_backbone_forked_6Lf.sh` unifies
#318's `train_backbone_beta_forked.sh` (β loss) and `train_backbone_forked.sh`
(all-time loss); every other flag is copied byte-for-byte. The fork itself
(`--synth-kind forked-arma --mix-ratio MIX`) and the loss code
(`cosine_similarity_batch_full_hh_negs[_xshh_allt]`, `src/synthetic_forked_arma.py`)
are #318's, used unchanged via `PYTHONPATH` to this card's worktree (branched off
#318). The 1L scores are reused from #318, not re-run.

Downstream (`scripts/downstream_6Lf.sh`) is #318's 6Lf downstream verbatim, only the
OUT/WT paths repointed: `--num-layers 6` in both q-head training and eval (so the
q-head reads the 6L forecaster's latents), and `--amp-dtype none` (the fix for the
depthwise-conv fp32/bf16 mismatch in `extract_forecaster_latents` that crashed
#318's first 6Lf eval under bf16).

## Compute layout
Local elisa, 2× RTX 4090 (24 GB), GPU 0 sharing ~7.6 GB with an unrelated process.

Measured at batch 256: a 6L-forecaster **backbone** needs ~18 GB — two will not
co-reside on one 24 GB card, and a second cannot fit on GPU 0 beside the 7.6 GB
tenant. A **q-head** (6L, worst case) needs ~10.5 GB and *does* fit on GPU 0
(18.1 GB total, no OOM, ~4 sps → ~2.1 h / 30k head). So the layout is:

- **GPU 1** (free): all 5 backbones, sequential — `scripts/backbones.sh`.
- **GPU 0** (shared): a downstream poller — `scripts/downstream_poll.sh` — waits for
  each backbone's `_FINAL.pth`, then trains its 2L + 6L q-heads and runs triage-11 +
  full-97. Overlaps the heavy q-head/eval work with backbone training.

Both drivers are idempotent (skip finished cells), so a crash is resumable by
re-launching. Backbone ≈2.4 h (β arms) / ≈5 h (all-time arms, the chunked +
gradient-checkpointed B²·T² cross-series×cross-time Gram costs ~2× step time).

## Notes / corrections
(— none yet —)
