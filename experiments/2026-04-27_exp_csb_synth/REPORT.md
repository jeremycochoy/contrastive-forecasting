# 2026-04-27_exp_csb_synth — cosine_similarity_batch on the span=512 best arm

**Status: complete (single seed).**

## Why

After the synth span sweep landed `span=512` as the best arm
(`2026-04-27_exp_span_sweep_synth`, GM-MASE 0.848), the queued FOLLOWUP-1 from
`../freq-embedding/FOLLOWUP.md` was: re-introduce the within-series,
within-channel time negative `h[b, t-1, c] vs h[b, t, c]` that was
removed during ARMA-era tuning. On periodic data, adjacent latents
walk a non-trivial manifold; pushing them apart should sharpen the
representation.

The paper-matching loss `cosine_similarity_batch` already includes
this term (alongside cross-channel time negatives). This experiment
flips ONLY the loss flag on the otherwise-frozen best arm.

## Setup

| Knob | Value |
|---|---|
| Steps | 30k bb + 30k qhead |
| Mix ratio | 1.0 (synth-only) |
| Freq emb | dim=3, mixup=0.3 |
| Reversible norm | RevEWMNorm span=512 |
| Loss | `cosine_similarity_batch` (re-includes cross-time negatives) |
| Backbone selector | `best_loss` |
| Eval | 1024 held-out synth samples (same protocol/seed as
   `2026-04-27_exp_span_sweep_synth`) |

Single-axis change vs the existing `fe+mu @ 30k span=512` baseline,
which used `cosine_similarity_batch_no_time_neg`.

## Results (single seed)

| Arm | GM-MASE | GM-WQL | MASE skill | WQL skill |
|---|---:|---:|---:|---:|
| `fe+mu @ 30k span=512` (no_time_neg, baseline) | 0.848 | 0.413 | −71% | −20% |
| **`fe+mu @ 30k span=512 +cosine_similarity_batch` (this)** | **0.886** | **0.434** | **−78%** | **−26%** |
| Seasonal Naive | 0.497 | 0.344 | 0% | 0% |

The new arm is **4.5% worse on MASE and 5.1% worse on WQL** than the
baseline.

## What was measured (no interpretation)

- Adding the `cosine_similarity_batch` loss to the otherwise-best
  span=512 setup did not improve forecasting metrics on the held-out
  synth eval at this training scale.
- The training-time gap was higher with the new loss (~0.86 vs ~0.84
  for the no_time_neg baseline at peak). Higher gap did not translate
  to better forecast quality on this single-seed run.
- Backbone training was interrupted by remote-instance failures three
  times during this experiment (the vast.ai instance kept stopping
  mid-run); each time the run resumed cleanly from the latest
  periodic snapshot via `--resume`. The final 30k-step checkpoint is
  the result of: 8k steps from a fresh start → resume from 8k →
  trained to ~24k → resume from 24k → trained to 30k.

## Speculation (single seed, not validated)

These are guesses for why the paper-matching loss didn't help:

1. **Resume-from-snapshot RNG state**: each `--resume` re-seeds RNG
   from the optimizer-state RNG. With the same seed across all our
   runs, this should be deterministic, but there's still the
   possibility of CUDA non-determinism interacting with the resumed
   state in ways that make the final weights slightly different from
   a single-shot 30k run. Hard to disentangle from the loss change
   itself.

2. **Single seed**: the 4-5% gap is within the variance band we'd
   expect from seed sensitivity at this scale. A second seed of
   either arm could flip the ranking.

3. **The cross-channel time negatives in `cosine_similarity_batch`
   may be detrimental** for synth where each channel was independent
   periodic. The loss includes `neg_xy = sum_c2 sim(hx[c1], hy[c2])`
   which sums over all c2 — including c2≠c1 (cross-channel
   different-period). On synth with different per-channel periods,
   this might be pushing representations apart in an unhelpful way.
   A more focused variant ("within-channel time negative only") could
   isolate the effect. (We had such a variant earlier in the session;
   it was removed in favour of using the paper-matching loss.)

4. **The model may be saturated** at this scale. Span=512 was already
   close to SN on WQL (-20% skill); single-knob improvements get
   harder as the model saturates. The remaining gap to SN may need
   architectural changes other than loss tweaks.

## Caveats

- Single seed.
- Three remote-instance failures during training. The final checkpoint
  is from a multi-resume run, not a clean single-shot 30k.
- Loss-shape value isn't directly comparable to the baseline because
  the negatives are different — only the downstream synth-eval metrics
  are.

## Open questions

- Does a clean single-shot run (no resume) produce the same number?
  Useful to disambiguate "loss change hurt" from "resume corrupted
  the run".
- What happens with a focused within-channel-only time negative
  (drops the cross-channel time terms)? Would need to re-introduce
  the deprecated `cosine_similarity_batch_with_within_time_neg`
  variant.
- What about across multiple seeds — is the 4-5% gap real or noise?

## Provenance

- `run.sh` is a copy of `/tmp/run_wtn_v2.sh` from the remote vast.ai
  instance at the time the run was launched.
- `run_v1.sh` is the earlier variant (`run_within_time_neg.sh` at
  repo root, `--loss-shape cosine_similarity_batch_with_within_time_neg`)
  that was superseded by v2 after the user clarified they meant the
  paper-matching `cosine_similarity_batch`. Preserved for provenance.

## Artefacts

- Backbone: `checkpoints/tiny_femu_span512_synth30k_csb_FINAL.pth`
  (~80 MB, not tracked in git; available on remote and locally in
  `sync_csb/checkpoints/`).
- Qhead: `checkpoints/R1q_femu_span512_synth30k_csb_FINAL.pth` (~2.5 MB).
- Eval CSV row: "fe+mu @ 30k span=512 +cosine_similarity_batch" in
  `../2026-04-27__aggregate/results/synth_eval.csv`.
- Training loss CSVs: `tiny_femu_span512_synth30k_csb_losses.csv`,
  `R1q_femu_span512_synth30k_csb_losses.csv`.
