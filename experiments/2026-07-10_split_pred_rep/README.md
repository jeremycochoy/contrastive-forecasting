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

## Per-family logit-magnitude measurement

Offline post-run analyzer, not part of training. Runs once after the
backbone finishes, before writing up the report; never on a live
training loop.

- **Script.** `scripts/measure_logit_magnitudes.py` (added when the
  backbone lands) loads a list of checkpoints and reports per-family
  logit-tensor mean/max at τ=0.10. Consumes: the run's `*_best_loss.pth`
  + periodic `*_{2500,5000,7500,10000,12500}k.pth` from `runs/`, and one
  fixed validation batch (same seed + shape as training: B=512, T=4096,
  synth-kind forked-arma). Emits `results/logit_magnitudes.csv` with
  columns `checkpoint, step, family, mean, max, share_of_denominator`.
- **Families.** The five negative families that split across L_pred /
  L_rep in `cosine_similarity_batch_split_pred_rep`:
  - L_pred side (f-anchored): `f_t ↔ h'_{t+1}` (cross-batch pred),
    `f_{t+1} ↔ f_t` (adjacent f).
  - L_rep side (h-anchored): `h_t ↔ h_t` (cross-channel), within-series
    `h_t ↔ h_l` (all-time), cross-series `h_t ↔ h_{b',l}` (all-time).
- **How.** For each checkpoint: load into the same encoder used at
  training, run the fixed val batch through student + EMA teacher to
  get `f`, `h`, `h'`, assemble each family's logit tensor exactly as
  `cosine_similarity_batch_split_pred_rep` does (τ=0.10, same masks
  and pooling), record its unreduced mean/max, and its share of the
  pooled logsumexp denominator on the side it belongs to (L_pred's
  softmax denominator for the two pred families; L_rep's pooled
  logsumexp for the three rep families). No gradients, `.eval()`,
  single forward per checkpoint.
- **When.** Runs once against the final backbone checkpoints on elisa,
  before the report is written. Not part of `launch_arms.sh` — invoked
  manually from the experiment directory.
- **Read-out.** The interference story is confirmed if the h↔h family
  dominates the combined-shape denominator (share ≫ pred-family shares)
  in the arm C reference run; the split is expected to keep pred-family
  shares stable across steps.

## Links

- Parent: —
- Previous: #366 (champion arm C recipe)
- Related: #303 / #307 (crossed-loss family history), #355 (SIGReg),
  #373 (re-entry; composable later)
