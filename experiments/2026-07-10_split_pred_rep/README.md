# Split the main loss into L_pred + L_rep (#374)

Un-mix the champion (arm C, #366) contrastive loss into two independent
terms sharing a single positive:

- **L_pred** — normalized InfoNCE with the f-anchored (prediction)
  families in the denominator: cross-batch `f_t ↔ h'_{t+1}` and adjacent
  `f_{t+1} ↔ f_t`. Same τ=0.10, same batch pooling, same teacher-side
  positive as arm C.
- **L_rep** — pooled logsumexp of the h-anchored (repulsion) families,
  no positive: cross-channel `h_t ↔ h_t`, within-series all-time
  `h_t ↔ h_l`, cross-series all-time `h_t ↔ h_{b',l}`. Same τ=0.10,
  same batch pooling as arm C.
- **L** = L_pred + L_rep, equal weight.

Everything else stays at the champion recipe: batch B=512, 12,500 steps,
seed 20260520, CPC/SIGReg/EMA-teacher unchanged. `--pos-in-denominator`
and `--subtract-contrastive-floor` are dropped (both are derived for the
combined shape and are gradient-neutral).

## Loss shape

The new loss shape is `cosine_similarity_batch_split_pred_rep`
(`src/loss.py`). It uses the same negative families as arm C's
`cosine_similarity_batch_full_hh_negs_xshh_allt`, but assembles them into
two independent terms. Guards match arm C: EMA-teacher positive and
`stopgrad_positive_h` are supported; `include_positive_in_denominator`
and `subtract_contrastive_floor` raise NotImplementedError (L_pred is
normalized by construction, the combined-shape floor is not defined for
the split).

## Question

Do the h-anchored (repulsion) and f-anchored (prediction) objectives
interfere in the shared denominator? On periodic series the h↔h logits
saturate at seasonal lags (cos ≈ 1, scaled by 1/τ = 10), dominate the
pooled denominator, and starve the prediction negatives of gradient —
exactly on the task cluster that carries the entire GM deficit (28 tasks
≥1.25 rel: solar / electricity / ett1 / m4_hourly / bizitobs). Does
letting each term live its own life improve transfer?

## Success criteria

Rank on full-97 GM-Rel MASE (paired bootstrap vs arm C; single-seed
noise ±0.02). Secondary: periodic cluster subset, medium/long horizons.
Report per-family logit magnitudes (mean/max) at a few checkpoints so
the interference story is measured, not assumed.

Follow-ups if it wins (separate cards, not this one): λ_rep weighting;
dropping L_rep entirely and letting SIGReg carry uniformity;
seasonal-lag false-negative masking inside L_rep.

## Links

- Parent: —
- Previous: #366 (champion arm C recipe)
- Related: #303 / #307 (crossed-loss family history), #355 (SIGReg),
  #373 (re-entry; composable later)
