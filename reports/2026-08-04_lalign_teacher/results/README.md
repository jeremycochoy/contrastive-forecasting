# Data behind this report

#390 retrained the ten `L_align` cells (arm 5 and arm 6 v2 × the five #379
settings) with `L_align` targeting the EMA teacher instead of the student.
The twenty cells that carry no `L_align` term — arm 1, arm 3, arm 4, bimoco —
cannot move under that change and were not retrained; their measurements here
are #379's, copied verbatim. `seasonal_naive_all_results.csv` is byte-identical
between the two reports, so all 30 arms sit on one scale.

| Path | Contents |
|------|----------|
| `gm_relative_mase.csv` | one row per measured cell: `arm_slug`, `variant`, `align_target`, `bb_steps`, `head_steps`, `bb_seed`, `head_seed`, `cell`, `gm_rel_mase`, `n_configs`, `source`. 93 cells over 30 arms. `align_target` is `teacher` / `student` / `none` (the twenty arms with no `L_align` term). `source` is `#390` for a cell measured here and `#379` for a copied one. Every teacher cell has its student counterpart at the same arm and backbone step, so the report's central delta rebuilds from this file alone. |
| `eval_bootstrap_ci.csv` | 95% intervals on the teacher/student ratio per arm per backbone step, from a **dataset-level** paired bootstrap over the per-config log differences, with the config-level interval beside it for contrast. `ci_excludes_1` is the per-cell verdict. |
| `eval_paired_tests.csv` | the same comparison across the ten arms at a fixed backbone step: counts, median ratio, sign-test and Wilcoxon p. |
| `anomaly_inspection.csv` | one row per backbone: non-finite count, rise from the run's own loss minimum, largest step-to-step jump in units of the run's IQR, peak attention logit magnitude and its trend. All 40 backbones, so "unusual" is measured against neighbours. |
| `anomaly_windows.csv` | the same runs cut at the wave boundaries (0–40k, 40k–100k, 100k–200k). A whole-run summary hides a wave; the flagged cells' behaviour is inside one. |
| `probe_inertness_ab.txt` | arm5's command line, 200 steps, latent-drift probe on vs off, same seed, back to back on one 4090. |
| `eval_gm_mase/<cell>_summary.txt` | one line, the aggregate GM-Relative MASE and the config count for that cell |
| `eval_gm_mase/<cell>/all_results.csv` | per-config GIFT-Eval output for that cell — one row per dataset/config, with MASE, the seasonal-naive reference and the relative ratio. This is what the aggregate is the geometric mean of. Every cell has all 97 configs. |
| `training_curves/<run>_losses.csv` | per-step backbone training metrics: `loss`, `gap`, `ff`, `u_batchtime`, `u_batchtime_e`, `sigreg_e`, `sigreg_h`, `cpc_aux`, `auc`, `top1`, `top3` and more (27 columns for the #390 runs, which add `ema_tau`). **Downsampled**: every step up to 500, then every 200th step, by `scripts/downsample_curve.py`. |
| `attn_amplitude/<run>_attn_amplitude.csv` | per-layer attention amplitude diagnostics (`qk_logit_maxabs`, `sa_in_maxabs`, `sa_out_maxabs`, `resid_post_sa_maxabs`, `resid_post_ffn_maxabs`), logged every 200 steps. |
| `latent_drift/<run>_latent_drift.csv` | #390 only. In-training drift of the student and the teacher latents against an earlier step: `drift_cos`, `drift_cos_aligned`, `rot_gap`, `cka`, for `student_h` and `teacher_h`. Written by the trainer, so no checkpoint reload is involved. |
| `latent_movement_pairs.csv` | one row per arm per adjacent-checkpoint pair: `step_later`, `drift_h`, `drift_e`. 250 pairs over all 30 arms. |
| `seasonal_naive_all_results.csv` | the seasonal-naive baseline, one row per config. The denominator of every GM-Relative MASE in the report — `MASE(model) / MASE(seasonal_naive)`, geometric mean over configs. |
| `checkpoint_manifest.csv` | every checkpoint the #390 waves left on elisa: run name, step, size, mtime, sha256 prefix. The `.pth` files themselves are too large for the repository. |
| `logs/` | the trainer, eval, orchestrator, watchdog and pipeline logs of the three #390 waves, plus the per-wave state JSONs. |

Cell naming is `<arm><variant><tags>_bb<backbone step>k_hd<head steps>s`, e.g.
`arm6_v2_combab_bb100k_hd30000s`. Three tags appear:

| Tag | Meaning |
|-----|---------|
| *(none)* | the wave measurement — teacher target, head seed 20260722 |
| `_alignstudent379` | the student counterpart, copied from #379's sweep |
| `_alignstudent` | the student control, same target measured on this branch |
| `_s<seed>` | a head-seed replicate on the same frozen backbone |

The `379` is load-bearing: `arm5_alignstudent379_bb40k_hd15000s` is #379's
number and `arm5_alignstudent_bb40k_hd15000s` is this branch's re-run of it.
They are the two sides of the code-boundary check and must not share a name.

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

## What the numbers support

**Across the ten arms the teacher target has no direction.** At a fixed
backbone step the ten teacher/student ratios are 6/10 below 1.0 at 40k and
4/10 at 100k, median ratio 0.9896 and 1.0026. Sign test p = 0.75 at both
steps; Wilcoxon p = 0.63 and p = 0.43 (`eval_paired_tests.csv`). Six of the
twenty per-cell intervals cover 1.0.

**Resample datasets, not configs.** The 97 configs are (dataset, frequency,
term) triples over 28 base datasets — `electricity` alone contributes 8 — so
a config-level bootstrap counts correlated rows as independent. Clustering
on the dataset widens the intervals by 15% on average and changes the
verdict of one cell in twenty (`arm5_tr1` at 100k).

**The latent-drift probe does not perturb training.** It runs in every #390
wave and did not exist when #379's sweep ran: #379's logs print no
`Latent-drift CSV` line and its `runs/` holds no drift CSV. So the teacher
cells and the copied student cells differ by the probe as well as by
`--align-target`, which the flag-for-flag launcher test cannot see. On GPU
the two legs are identical at step 200 in every training field — loss,
`ema_loss`, `gap`, `ema_gap`, `mixup`, `cpc`, R², AUC, Top1 — differing only
in wall-clock sps (`probe_inertness_ab.txt`). The probe holds a fixed ARMA
batch drawn from a private numpy generator, runs `no_grad` in `eval()` mode,
and dropkey is gated on `self.training`. All three rows of the earlier
200-step A/B ran in this checkout with the probe on, so it is equally
present in all of them.

**The three flagged cells are three different things**
(`anomaly_inspection.csv`, `anomaly_windows.csv`; no non-finite loss or gap
anywhere in the 40 backbones):

| Cell | What the curves show |
|------|----------------------|
| `arm1_combab` 3.1251 at 40k (copied) | never trained. Loss sits at ~19–20 the whole run and is *higher* at 40k (19.82) than at step 0 (19.03) — the largest rise-from-minimum of all 40 runs. Every other arm1 variant fell by 3.1 to 7.1 over the same window. Attention amplitude is the second lowest in the set, so this is a flat objective, not a blow-up. |
| `arm5_nse` 1.8887 at 200k | clean loss, extreme attention. Loss falls 19.80 → 14.11 → 13.95 and flattens; rise-from-minimum is at the 18th percentile. Mean attention logit magnitude grows 7.63 → 14.28 → 35.09 across the three waves, peak 259.94 — the largest of all 40. |
| `arm6_v2` base 1.4322 → 1.9057 (40k → 100k) | no backbone signature at all. Loss 12.98 → 4.92 in wave 1, flat 4.98 → 4.92 in wave 2. Attention bounded and near the bottom of the set. Nothing in the backbone moved across the window where the eval number nearly doubled. |

## Caveats the report has to carry

* **The 200k row is doubly selected.** Teacher cells were promoted to wave 3
  by the teacher trajectory and student cells by their own, so no
  like-for-like 200k claim exists. `eval_paired_tests.csv` deliberately
  stops at 100k for that reason.
* **Head budget moves with backbone step** — 15 000 steps at 40k, 30 000
  later. Any within-arm statement about the 40k → 100k trajectory is
  confounded by that.
* **Every measured cell is above 1.0**, so every arm in this report is worse
  than seasonal naive.
* **The head trainer uses `--grad-clip 1.0`**, against the standing project
  rule. Kept because #379's heads used it and the comparison is to #379.

## How to rebuild this directory

```bash
WT=/home/jupyter/wt-cf-390-train REPO=<repo> \
  bash experiments/2026-08-01_lalign_teacher/scripts/collect_artefacts.sh
REPO=<repo> bash experiments/2026-08-01_lalign_teacher/scripts/merge_379_cells.sh
python3 experiments/2026-08-01_lalign_teacher/scripts/eval_bootstrap.py \
  --teacher-results reports/2026-08-04_lalign_teacher/results \
  --student-results reports/2026-07-21_split_pred_rep_small/results \
  --out-ci reports/2026-08-04_lalign_teacher/results/eval_bootstrap_ci.csv \
  --out-tests reports/2026-08-04_lalign_teacher/results/eval_paired_tests.csv
python3 experiments/2026-08-01_lalign_teacher/scripts/inspect_anomalies.py \
  --results reports/2026-08-04_lalign_teacher/results \
  --extra-results reports/2026-07-21_split_pred_rep_small/results \
  --out reports/2026-08-04_lalign_teacher/results/anomaly_inspection.csv \
  --out-windows reports/2026-08-04_lalign_teacher/results/anomaly_windows.csv
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
