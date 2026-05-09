# exp_init_u_sweep — init scheme sweep for encoder-latent dimension usage

## Goal

In training the contrastive-forecasting backbone we observe `U_b ≈ 0.003` on
the encoder latent `o_lat` after just 100 steps — i.e. the encoder's outputs
are nearly collinear across the batch axis right out of the gate. Backbone-beta
eventually converges to `U_b = 0.0762` at step 167k, but the journey starts
from a nearly-degenerate point. This experiment asks whether a different
parameter init (or encoder choice, patch width, or batch size for measurement)
can lift `U_b(o_lat)` at step 0 (no training), so the backbone starts from a
less-collapsed point.

## Layout

- `RESULTS.md` — consolidated findings across all sub-experiments.
- `scripts/` — measurement scripts (`measure_init_u*.py`, `u_audit.py`,
  `u_per_stage*.py`, `u_per_encoder.py`).
- `results/` — CSV outputs per sub-experiment.

## Sub-experiments

1. **8-init scheme sweep** — `scripts/measure_init_u.py` →
   `results/init_u_sweep.csv`, `results/init_u_sweep_raw.csv`.
2. **W-sweep (patch width / per-token input dim)** —
   `scripts/measure_init_u_w_sweep.py` → `results/init_u_w_sweep.csv`.
3. **B-sweep (batch size)** — `scripts/measure_init_u_b_sweep.py` →
   `results/init_u_b_sweep.csv`.
4. **U-metric audit** — `scripts/u_audit.py` → `results/u_audit.csv`.
5. **Per-stage U_b inside GRUEncoder** — `scripts/u_per_stage.py` →
   `results/u_per_stage_default.csv`, `results/u_per_stage_ortho.csv`.
6. **Five patch encoders compared** — `scripts/u_per_encoder.py` and
   `scripts/u_per_stage_residual_silu.py` →
   `results/init_u_per_encoder.csv`, `results/u_per_stage_residual_silu.csv`.
7. **Orthogonal-subspace inits** —
   `scripts/u_per_stage_orthogonal_subspaces.py` →
   `results/init_u_orthogonal_subspaces.csv`.

Total runtime across the sweep: a few minutes on CPU per script.

See `RESULTS.md` for findings.
