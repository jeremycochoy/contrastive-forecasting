# Split the main loss into L_pred + L_rep (#374)

Report: [`reports/2026-07-10_split_pred_rep/split_pred_rep.md`](../../reports/2026-07-10_split_pred_rep/split_pred_rep.md).

## Loss shape

`cosine_similarity_batch_split_pred_rep` in `src/loss.py` un-mixes the
champion (arm C) contrastive loss into two independent terms sharing a
single positive:

- **L_pred** — normalized InfoNCE with the f-anchored (prediction)
  families in the denominator: cross-batch `f_t ↔ h'_{t+1}` and
  adjacent `f_{t+1} ↔ f_t`.
- **L_rep** — pooled logsumexp of the h-anchored (repulsion) families,
  no positive: cross-channel `h_t ↔ h_t`, within-series all-time
  `h_t ↔ h_l`, cross-series all-time `h_t ↔ h_{b',l}`.
- **L** = L_pred + L_rep, equal weight.

## Runs

Same champion recipe (12,500 steps, B = 512, T = 4096, seed 20260520,
SIGReg λ_e = λ_h = 1, EMA teacher τ = 0.90, contrastive τ = 0.10, CPC
auxiliary) for each arm; `--loss-shape` and `--moco-negatives` are the
only training-flag differences. The full arm table, the paired-bootstrap
CI table, the denominator-share measurement and every scored evaluation
live in the report linked above.

Denominator-share is measured by `scripts/gradient_share_measurement.py`
(no `logit_magnitudes.py`; the earlier plan file used a different name);
the CSV lands at `results/gradient_share_measurement.csv`.
