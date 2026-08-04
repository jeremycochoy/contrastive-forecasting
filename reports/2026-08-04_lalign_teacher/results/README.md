# Data behind this report

#390 retrained the ten `L_align` cells (arm 5 and arm 6 v2 × the five #379
settings) with `L_align` targeting the EMA teacher instead of the student.
The twenty cells that carry no `L_align` term — arm 1, arm 3, arm 4, bimoco —
cannot move under that change and were not retrained; their measurements here
are #379's, copied verbatim. `seasonal_naive_all_results.csv` is byte-identical
between the two reports, so all 30 arms sit on one scale.

| Path | Contents |
|------|----------|
| `gm_relative_mase.csv` | one row per measured cell: `arm_slug`, `variant`, `bb_steps`, `head_steps`, `cell`, `gm_rel_mase`, `n_configs`, `source`. 68 cells over 30 arms. `source` is `#390` for a retrained cell and `#379` otherwise. |
| `eval_gm_mase/<cell>_summary.txt` | one line, the aggregate GM-Relative MASE and the config count for that cell |
| `eval_gm_mase/<cell>/all_results.csv` | per-config GIFT-Eval output for that cell — one row per dataset/config, with MASE, the seasonal-naive reference and the relative ratio. This is what the aggregate is the geometric mean of. Every one of the 68 cells has all 97 configs. |
| `training_curves/<run>_losses.csv` | per-step backbone training metrics: `loss`, `gap`, `ff`, `u_batchtime`, `u_batchtime_e`, `sigreg_e`, `sigreg_h`, `cpc_aux`, `auc`, `top1`, `top3` and more (27 columns for the #390 runs, which add `ema_tau`). **Downsampled**: every step up to 500, then every 200th step, by `scripts/downsample_curve.py`. |
| `attn_amplitude/<run>_attn_amplitude.csv` | per-layer attention amplitude diagnostics (`qk_logit_maxabs`, `sa_in_maxabs`, `sa_out_maxabs`, `resid_post_sa_maxabs`, `resid_post_ffn_maxabs`), logged every 200 steps. |
| `latent_drift/<run>_latent_drift.csv` | #390 only. In-training drift of the student and the teacher latents against an earlier step: `drift_cos`, `drift_cos_aligned`, `rot_gap`, `cka`, for `student_h` and `teacher_h`. Written by the trainer, so no checkpoint reload is involved. |
| `latent_movement_pairs.csv` | one row per arm per adjacent-checkpoint pair: `step_later`, `drift_h`, `drift_e`. 250 pairs over all 30 arms. |
| `seasonal_naive_all_results.csv` | the seasonal-naive baseline, one row per config. The denominator of every GM-Relative MASE in the report — `MASE(model) / MASE(seasonal_naive)`, geometric mean over configs. |
| `checkpoint_manifest.csv` | every checkpoint the #390 waves left on elisa: run name, step, size, mtime, sha256 prefix. The `.pth` files themselves are too large for the repository. |
| `logs/` | the trainer, eval, orchestrator, watchdog and pipeline logs of the three #390 waves, plus the per-wave state JSONs. |

Cell naming is `<arm><variant>_bb<backbone step>k_hd<head steps>s`, e.g.
`arm6_v2_combab_bb100k_hd30000s`.

A run that was resumed writes a fresh `_r<N>` file rather than appending, so a
full trajectory is the concatenation of the `_losses.csv` files sharing a run
name, ordered by their first `step`. #390's waves are `_` (0 → 40k),
`_r2` (40k → 100k) and `_r3` (100k → 200k).

## Two things to know when reading across the 30 arms

**arm 5 and arm 6 v2 mean something different here than in #379.** In this
directory they are the teacher-target retrain. The pre-teacher measurements of
the same ten cells are still readable at
`reports/2026-07-21_split_pred_rep_small/results/`, under the identical cell
names — that pair is the before/after of #390.

**`latent_movement_pairs.csv` mixes two measurement runs, on one scale.** The
twenty un-retrained arms are #379's rows; the ten retrained arms were measured
here by `scripts/make_latent_movement_390.py`. Both use the same fixed
held-out batch (`reports/2026-07-21_split_pred_rep_small/plots/_latent_movement_batch.pt`,
B=8/T=4096/C=1) and the same `mean_one_minus_cos` from
`src.eval_latent_movement`. Re-running #379's own script with `--arms arm1`
against that batch reproduces its committed rows to the last digit, which is
what pins the two halves together. Arms with 7 pairs stop at 100k; arms with
11 pairs ran to 200k.

Full-resolution (per-step, ~27 MB per run) loss curves and all backbone / head
checkpoints stay on elisa under
`experiments/2026-08-01_lalign_teacher/{runs,eval_gm_mase}/`; they are too
large for the repository. `checkpoint_manifest.csv` indexes them.

## How to rebuild this directory

```bash
WT=/home/jupyter/wt-cf-390-train REPO=<repo> \
  bash experiments/2026-08-01_lalign_teacher/scripts/collect_artefacts.sh
REPO=<repo> bash experiments/2026-08-01_lalign_teacher/scripts/merge_379_cells.sh
python3 experiments/2026-08-01_lalign_teacher/scripts/make_latent_movement_390.py \
  --runs-dir $WT/experiments/2026-08-01_lalign_teacher/runs \
  --batch reports/2026-07-21_split_pred_rep_small/plots/_latent_movement_batch.pt \
  --out /tmp/latent_movement_390.csv
python3 experiments/2026-08-01_lalign_teacher/scripts/merge_latent_movement.py \
  reports/2026-07-21_split_pred_rep_small/results/latent_movement_pairs.csv \
  /tmp/latent_movement_390.csv \
  reports/2026-08-04_lalign_teacher/results/latent_movement_pairs.csv
python3 experiments/2026-08-01_lalign_teacher/scripts/make_gm_table.py \
  reports/2026-08-04_lalign_teacher/results \
  reports/2026-08-04_lalign_teacher/results/gm_relative_mase.csv
```
