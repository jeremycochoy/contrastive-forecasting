# Follow-up experiments — proposed, not yet run

Captured for later. Not part of the current synth-only / span-sweep / patch-stats round.

## 1. Re-introduce the "future-against-past" negative in the contrastive loss

### Background

The current contrastive loss in `src/loss.py` only uses *cross-batch* negatives:
`f[b, t]` is pulled toward `e[b, t+1]` (positive) and pushed away from `e[b', t']`
for `b' != b` (cross-batch negatives). The original recipe also included a
**within-batch, within-time** negative term: `h[b, t-1, c]` vs `h[b, t, c]` —
push the current step's latent away from the previous step's latent of the
same series.

That term was removed when the codebase was tuned for ARMA training data:
ARMA(1, p, q) has very high autocorrelation at lag 1, so adjacent latents
genuinely *should* be similar. Treating them as negatives degraded training.

### Why it's worth revisiting now

We're no longer training on ARMA. Our data is the periodic synth (and
mix=0.5 / mix=0.0 with HF base-bundles). On periodic data, adjacent
timesteps' latents are NOT trivially similar — they walk along the
manifold of the cycle. The within-time negative *should* sharpen the
representation by forcing each step to be discriminable from its neighbour.

This becomes especially relevant once we've picked a best arm (best span
+ best architecture choice between fe+mu and fe+mu+pstats and the
RevIN/EWMA family). Adding the term back is a single-axis ablation on the
otherwise-frozen best config.

### Plan (when we get to it)

1. Pick the best arm from the current sweep (synth-only patch-stats result
   + real-data span winner).
2. Add a flag in `src/loss.py::contrastive_latent_loss` to re-enable the
   within-time negative. Likely `loss_shape='cosine_similarity_batch_with_time_neg'`
   alongside the current `cosine_similarity_batch_no_time_neg`.
3. Train a single backbone with this flag at the same step count as the
   chosen baseline (probably 30k-60k synth-only).
4. Compare:
   - Contrastive gap and loss curves
   - Synth-eval (1024-sample held-out): GM-MASE, GM-WQL, skill scores
   - Synth grid plot for visual inspection

### Open design questions

- **Weighting**: should the within-time negative carry the same weight as
  the cross-batch negatives, or be down-weighted? On periodic data with
  P=8 minimum, adjacent steps within a single period are about 1/8 of a
  cycle apart — measurably different but not arbitrary distractors.
  Probably start at weight=1, ablate later.
- **Time delay**: only `t vs t-1`, or also `t vs t-k` for k > 1? The
  longer-range version was the basis of `contrastive_latent_delay` in
  `LOSS_SPEC` — currently set to 0. We could sweep delay ∈ {0, 1, 2, 4}
  but that's a follow-up of a follow-up.
- **Interaction with patch-stats / RevIN**: if patch-stats turns out to
  matter, the within-time negative may interact with the per-patch dmean
  feature — adjacent patches now have a constant feature (dmean ~ const
  within a smooth signal), so the negative may be even cleaner.

### Status

**Not implemented. Not running. Note only.** Pick this up after the current
synth-only + real-data span sweep + RevIN-synth comparison lands.
