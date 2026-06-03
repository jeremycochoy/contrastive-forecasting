# #325 execution log (operational — not part of RESULTS.md)

Records the operational journey (GPU contention, infra incidents, recovery) so the
report stays "science, not journey." Nothing here changes the result; if the
experiment were re-run on a clear machine none of it would recur.

## Code + preflight (2026-06-01)
- `src/synthetic_crossfade.py` + 3-way `MixedForkedArmaLoader` + `--crossfade-ratio`;
  16 unit tests green (convex-blend, shared-across-channels s(t), z-norm, monotone
  weight, determinism, 3-way loader, guards); forked-arma regressions green.
- Built on the **#322 branch** (`experiment/2026-05-29-forked-6Lf-b1024`), since the
  forked-arma generator + qk-norm/attn-out-norm it depends on are not yet on
  `experiments`. Worktree on elisa from the pushed branch.
- On-GPU smoke (batch 64): `MIX 80% HF + 10% synth (forked-arma) + 10% crossfade,
  hf_bs=52, synth_bs=6, cross_bs=6`, loss finite & descending, rc=0.

## Compute pivot — single-GPU GPU0 (shared box)
Both elisa GPUs were held by other concurrent agent sessions (GPU0 ~17 GB, GPU1
~21 GB). The orchestrator gated politely (`wait_for_gpu`) rather than OOM. After
~3.5 h GPU0's foreign 12.4 GB job ended (→18 GB free, stable); GPU1's foreign job
partially freed later. Took the stable **18 GB GPU0 slot** (gate 15 GB), single
process @ batch 1024 with the GRU patch-encoder gradient-checkpointed
(`PATCH_ENC_CKPT=1 PATCH_ENC_CHUNK=4`, `XSHH_ALLT_CHUNK=2`) — fits ~18 GB, pools all
1024 in the negatives natively. The near-full card ran ~0.2 sps (≈17 h wall) vs a
clear card's faster rate; checkpointing is byte-identical so the trained backbone is
unaffected. tmux for disconnect-resilience; a laptop-side `sync_pull.sh` mirrored
checkpoints every 15 min; `--save-every 2500` was the resume net.

Backbone converged healthily (no collapse): gap rose to ~1.18, loss−floor ~1.0,
cross-series cosine flat ~3e-3 (qk-norm + attn-out-norm holding under the crossfade).

## Downstream — parallelised, then a transient HF crash on 6L
After the backbone freed the GPUs, ran the two q-heads concurrently (2L→GPU0,
6L→GPU1) instead of serially, halving wall-clock.

**Incident:** the 6L q-head crashed at step ~20 000 on a transient
`cas-bridge.xethub.hf.co` connect-timeout (HF xet CDN, 10 s connect timeout) while
streaming a training shard — a network blip, not a code/recipe fault. 2L was
unaffected (it had finished training and was in eval on local GIFT-Eval data).

**Recovery (`recover_6L.sh`):** resumed from the step-20 000 checkpoint (model +
optimizer + step + HF data position all restored → **0 steps lost**), with a retry
loop and `HF_HUB_DOWNLOAD_TIMEOUT=60` for further flakiness. Reached 30 k on the first
retry; chained the idempotent triage + full GIFT-Eval. Per the post-mortem rule, the
fix lives in the failure path (resume+retry script), not a runbook note.

## Eval
GIFT-Eval `--strategy B4`, full-97 + triage-11, identical to #322 (clean paired
comparison). Full-97 is ~4–5 h/cell; the two cells' full evals ran in parallel
(2L GPU0, 6L GPU1).
