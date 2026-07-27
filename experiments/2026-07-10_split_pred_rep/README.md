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
auxiliary) for each arm; arms differ in `--loss-shape`,
`--moco-negatives`, `--align-loss-weight`, and
`--align-moco-loss-weight` (arms 5 and 6 use `L_align` / `L_align_moco`
respectively). The full arm table, the paired-bootstrap
CI table, the denominator-share measurement and every scored evaluation
live in the report linked above.

Denominator-share is measured by `scripts/gradient_share_measurement.py`
(no `logit_magnitudes.py`; the earlier plan file used a different name);
the CSV lands at `results/gradient_share_measurement.csv` (original probe at each arm's best-cell step) and `results/gradient_share_measurement_step12500.csv` (matched-step re-probe at every arm's step-12,500 backbone, bimoco included — the one the report uses).

## Result directories

Every scored cell is the `Aggregate GM-Relative MASE (97 configs)` line of
`summary.txt` under `<dir>/gift_eval_full_<arm base name>[_suffix]_<2L|6L>/`;
suffixes `_2k` / `_25k` / `_50k` / `_last`, `best` has no suffix.

- arm 1 — `results/`, base name `…_split_pred_rep_xftrip_…`
- arm 3 — `results/`, base name `…_split_pred_rep_moco_xftrip_…`
- arm 4 — `results_arm4/`
- arm 5 — `results_arm5/`
- arm 6 — `results_arm6_v2/`
- bimoco — `results_bimoco_v2/`
- arm C — `results_armC_seed2/gift_eval_full_armC_seed2_step{12500,25000,50000}_{2L,6L}/`

`results_arm6/` and `results_bimoco/` hold superseded wrong-implementation
runs and are not used by the report. Seasonal-naive reference:
`results/seasonal_naive_all_results.csv`.

Arm C's original seed-20260520 per-task files were never committed with
`2026-06-28_sigreg_lambda_tau_cross`; only the aggregate row in that
experiment's `results/gm_table.csv` survives. The seed-2 retrain comes from
`2026-07-07_b512_armC_seed2_traj` (branch feature/contrastive-forecasting-371).

## Checkpoints

Per arm: `bb_<run>_FINAL.pth` is a copy of `_best_loss.pth` (the `best`-cell
backbone); `bb_<run>_final.pth` is the end-of-training step-12,500 checkpoint
(the `last`-cell backbone). Arm 1 recorded no best-loss save, so its two files
are byte-identical (`results/backbone_step_verification.log`).

## Loss-curve sources (alignment figures in the report)

The loss and `1 − ff` curves concatenate, per arm:

- arm 1 — `runs/…_losses_full.csv` + `…_r2_losses.csv` + `…_r3_losses.csv`
- arm 3 — `runs/…_moco_…_losses.csv` + `…_ext25k_losses.csv` + `…_r3_losses.csv`;
  the two resume runs re-index their step counter from 1 and are offset by +12,500
- arms 4, 5, 6 — `runs_arm4/`, `runs_arm5/`, `runs_arm6_v2/`, each base + `_r2` + `_r3`
- bimoco — `runs_bimoco_v2/` base + `_r2`; no 25k–50k segment

`auc` / `top1` retrieval diagnostics come from the same CSVs.

## CI panels

- `results/pairwise_bootstrap_ci.csv` — arm-1/3/4 pairwise (n_boot 20,000)
- `results/pairwise_bootstrap_ci_{arm5,arm6,bimoco}_nboot200k.csv` — vs arm 5 / 6 / bimoco references
- `results/pairwise_bootstrap_ci_vs_armC.csv` — vs seed-2 arm C at matching steps (n_boot 200,000)
- `results/pairwise_bootstrap_ci_periodic.csv` — 37-config periodic subset
- clustered / medium-long / short variants: `results/pairwise_bootstrap_ci_*.csv`
