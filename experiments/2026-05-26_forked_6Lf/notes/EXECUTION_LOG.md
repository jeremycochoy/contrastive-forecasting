# #320 — execution log

Operational journey for this card. The science is in [`RESULTS.md`](../forked_6Lf.md);
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

**Disk-full crash (2026-05-26 23:50).** The shared root filesystem (1.8 TB, normally
~98% full from other projects/worktrees) hit 0 bytes free while the β·10% backbone
was at step 22300/50k. `train.py` died with `OSError: No space left on device` mid
CSV-flush; the backbone driver then wrote a truncated 8 KB `_FINAL.pth`, and the
downstream poller failed loading it (`Invalid argument`). Cause was external (not
this card's footprint — it needs only ~3 GB). **Recovery:** freed 23 GB of
`~/.cache/pip` (reproducible) → ~59 GB free; restarted 2026-05-27 12:09. **Guards
added (then tuned):** the watcher exits on `free < 2 GB` (lowered from 8 GB after the
normal Shinka oscillation troughs to ~4 GB without harm); backbones and q-heads now
prune intermediate + optimizer checkpoints after their FINAL (keep footprint
≈ FINAL-only). **Backbones now resume** from the latest periodic checkpoint via
`--resume` on restart (trajectory-identical to from-scratch); never wipe a partial
run. None of #318's / rnd's / other worktrees' data was touched.

**Network/HF uplink degradation (2026-05-28 from 00:29, ongoing).** A transient DNS
failure (~00:29) was followed by a sustained ~1 MB/s uplink — slow not just to HF
but to Cloudflare (~1.2 MB/s) and Fastly/PyPI (~1.0 MB/s), with the interface idle
otherwise (~2.7 KB/s RX/TX, no local bandwidth hog). So this is the machine's
upstream link, not HF-specific (a mirror would not help) and not local contention.
Effect: any HF-streaming training (backbones / q-heads) is data-starved (GPU
util ≈ 0%, ≈0.5 sps vs the normal 3–6 sps) — the symptom CLAUDE.md flags for
unauthenticated streams, here with the token present. GIFT-Eval (local data) is
unaffected and finished β·10% before being paused.

After β·10% completed (all 4 cells), the downstream poller was paused cleanly
(allt·0.8% had also been paused after its failed network-blip start) to stop
wasting GPU on a starved stream and to stop hammering a degraded link. A
`results/.wait_hf` sentinel switches the watcher into HF-wait mode: it probes the
HF CDN every ~3 min and exits `EVENT=hf_recovered` once sustained throughput
≥ 5 MB/s; on that event both lanes resume.

**Vast.ai offload for the q-heads (user request, 2026-05-28).** With elisa still
slow, a 1× RTX 4090 prosumer instance (`vastrun-provision 38189788`, label
`320-qheads`, $1.33/h, 0.997 reliability) trained the β·0.8% and allt·50%
q-heads (4 q-heads, $13.72 total — within the original $14.77 credit). Vast HF
throughput was steady ~7.5 MB/s vs elisa's degraded ~1 MB/s, giving the q-heads
real compute (≈3.2 sps). FINALs were pulled back via `scp` to elisa for the
GIFT-Eval step (local data, no HF needed); the instance was destroyed as soon as
q-head 4 finished. The elisa downstream then ran in parallel: β·0.8% evals on
GPU 0, allt·10% / allt·0.8% q-heads + evals on GPU 1. When GPU 0 freed I
launched a one-off `downstream_6Lf.sh` on it for the allt·0.8% **6L head only**
(disjoint from GPU 1's 2L head), halving allt·0.8%'s wall time.

**Backbone-script trap, fixed.** The original `train_backbone_forked_6Lf.sh`
copied `best_loss → FINAL` on **any** exit, so a network-blip crash near step
~3 600 produced a "DONE" FINAL with a barely-trained backbone. Now FINAL is only
created on `rc == 0`; on crash the script keeps the periodic checkpoints so the
next launch resumes from the latest. `--save-every` was lowered 5 000 → 2 000 so a
typical inter-blip gap (~1 h at 0.5 sps) reliably yields at least one resumable
checkpoint. Watcher failure patterns were also tightened — generic `Traceback` /
`RuntimeError` matched stale HF-retry tracebacks the training had survived;
the watcher now fires only on `No space left`, `CUDA out of memory`,
`FAILED no checkpoint`, `QH FAILED`. Both fixes were validated on allt·0.8%'s
clean 50 k completion (~05:30 → 09:30 wall on a recovered network).
