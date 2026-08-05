# Data behind this report

This retrain covers the ten `L_align` cells (arm 5 and arm 6 v2 × the five
settings of the earlier small-model sweep) with `L_align` targeting the EMA teacher instead of the student.
The twenty cells that carry no `L_align` term — arm 1, arm 3, arm 4, bimoco —
cannot move under that change and were not retrained; their measurements here
are the earlier sweep's, copied verbatim. `seasonal_naive_all_results.csv` is byte-identical
between the two reports, so all 30 arms sit on one scale.

| Path | Contents |
|------|----------|
| `gm_relative_mase.csv` | one row per measured cell: `arm_slug`, `variant`, `align_target`, `code_snapshot`, `bb_steps`, `head_steps`, `bb_seed`, `head_seed`, `cell`, `gm_rel_mase`, `n_configs`, `source`. `align_target` is `teacher` / `student` / `none` (the twenty arms with no `L_align` term). **`code_snapshot` is `#379-sweep` (the earlier sweep's code) or `#390-branch` (this branch's code)** — which code produced the number, see below. `source` records the same split from the artefact's side. A cell with a same-branch student control keeps BOTH student rows; neither replaces the other. |
| `controlled_delta_40k.csv` | **the headline table.** Per arm at backbone 40k: `gm_teacher_390`, `gm_student_390`, `gm_student_379`, the controlled delta `gm_teacher_390 - gm_student_390` with a dataset-level paired bootstrap interval, the cross-experiment delta the wave tables report, and `code_snapshot_shift` = `gm_student_390 - gm_student_379`. Both sides of the controlled delta ran on this branch under launchers that differ by `--align-target` and nothing else, so it is the only column in this directory attributable to the flag. **`delta_controlled` is head seed 20260722 on all ten arms**, so it is one comparable column. `n_head_seeds`, `head_seeds` and `head_seed_spread_measured` say how many head seeds that arm carries; `delta_per_seed`, `delta_seed_mean`, `delta_seed_min/max/range`, `delta_sign_stable`, `teacher_seed_range` and `student_seed_range` are filled on the two replicated arms and **left empty on the other eight**, which ran one head seed and have no measured spread. An empty spread column is not a spread of zero. |
| `controlled_paired_tests_40k.csv` | the comparisons across the ten arms at 40k — controlled at head seed 20260722, the same controlled comparison with the three-seed mean substituted on the two replicated arms, cross-experiment, and the snapshot shift on its own — with counts, mean and median delta, sign-test and Wilcoxon p. `head_seed_basis` names which head seed each row is built on. The gap between the controlled and the cross-experiment row is the size of the contamination; the gap between the two controlled rows is what the head seed is worth to the aggregate. |
| `snapshot_reproduction_40k.csv` | `code_snapshot_shift` at six decimals, because at four a difference of 0.000028 and an exact reproduction print the same thing. Per arm: the earlier sweep's student number, this branch's re-run, the difference, and `reproduces` (within 0.0002). `n_configs_identical` and `max_abs_rel_config_diff` go under the aggregate — an aggregate can agree while the configs behind it do not. |
| `eval_bootstrap_ci.csv` | 95% intervals on the teacher/student ratio per arm per backbone step, from a **dataset-level** paired bootstrap over the per-config log differences, with the config-level interval beside it for contrast. `ci_excludes_1` is the per-cell verdict. **Cross-experiment**: the student side is the earlier sweep, so these intervals cover the flag and the snapshot together. |
| `eval_paired_tests.csv` | the same comparison across the ten arms at a fixed backbone step: counts, median ratio, sign-test and Wilcoxon p. Cross-experiment, same caveat. |
| `replicate_provenance_40k.csv` | which backbone replicate the earlier sweep's published 40k row was evaluated on, per arm, read off its `eval.log`; the step span of every replicate it left; and whether this branch's re-run repeats one of them step for step. Separates a code-snapshot shift from a replicate mismatch. |
| `seed_spread.csv` | per cell with replicate head seeds: the seeds, min/max/mean, and the range the cell moves under nothing but the head seed. **One row is one (arm, backbone step, align target, code snapshot)**, so a 40k teacher arm and a 40k student arm never merge into one row — at 40k they are two different backbones. The bar a claimed gap has to clear, and it is that cell's bar only: the eight rows span 0.0018 to 0.0908, and the four at 40k alone span a factor of 42. |
| `anomaly_inspection.csv` | one row per backbone: non-finite count, rise from the run's own loss minimum, largest step-to-step jump in units of the run's IQR, peak attention logit magnitude and its trend. All 40 backbones, so "unusual" is measured against neighbours. |
| `anomaly_windows.csv` | the same runs cut at the wave boundaries (0–40k, 40k–100k, 100k–200k). A whole-run summary hides a wave; the flagged cells' behaviour is inside one. |
| `probe_inertness_ab.txt` | arm5's command line, 200 steps, latent-drift probe on vs off, same seed, back to back on one 4090. |
| `eval_gm_mase/<cell>_summary.txt` | one line, the aggregate GM-Relative MASE and the config count for that cell |
| `eval_gm_mase/<cell>/all_results.csv` | per-config GIFT-Eval output for that cell — one row per dataset/config, with MASE, the seasonal-naive reference and the relative ratio. This is what the aggregate is the geometric mean of. Every cell has all 97 configs. |
| `training_curves/<run>_losses.csv` | per-step backbone training metrics: `loss`, `gap`, `ff`, `u_batchtime`, `u_batchtime_e`, `sigreg_e`, `sigreg_h`, `cpc_aux`, `auc`, `top1`, `top3` and more (27 columns for the teacher-target runs, which add `ema_tau`). **Downsampled**: every step up to 500, then every 200th step, by `scripts/downsample_curve.py`. |
| `attn_amplitude/<run>_attn_amplitude.csv` | per-layer attention amplitude diagnostics (`qk_logit_maxabs`, `sa_in_maxabs`, `sa_out_maxabs`, `resid_post_sa_maxabs`, `resid_post_ffn_maxabs`), logged every 200 steps. |
| `latent_drift/<run>_latent_drift.csv` | teacher-target runs only. In-training drift of the student and the teacher latents against an earlier step: `drift_cos`, `drift_cos_aligned`, `rot_gap`, `cka`, for `student_h` and `teacher_h`. Written by the trainer, so no checkpoint reload is involved. |
| `latent_drift_setting_vs_base.csv` | per setting: how many of the six loss recipes move `h_t` (and `e_t`) less than their own base setting, which recipes those are, and the two-sided exact binomial p on the count. Written from `latent_movement_pairs.csv` by `scripts/make_report_tables.py`, so the report's section-6 table is rebuildable. |
| `latent_movement_pairs.csv` | one row per arm per adjacent-checkpoint pair: `step_later`, `drift_h`, `drift_e`. 250 pairs over all 30 arms. |
| `seasonal_naive_all_results.csv` | the seasonal-naive baseline, one row per config. The denominator of every GM-Relative MASE in the report — `MASE(model) / MASE(seasonal_naive)`, geometric mean over configs. |
| `checkpoint_manifest.csv` | every checkpoint the waves left on elisa: run name, step, size, mtime, sha256 prefix. The `.pth` files themselves are too large for the repository. |
| `logs/` | the trainer, eval, orchestrator, watchdog and pipeline logs of the three waves, plus the per-wave state JSONs. |

Cell naming is `<arm><variant><tags>_bb<backbone step>k_hd<head steps>s`, e.g.
`arm6_v2_combab_bb100k_hd30000s`. Three tags appear:

| Tag | Meaning |
|-----|---------|
| *(none)* | the wave measurement — teacher target, head seed 20260722 |
| `_alignstudent379` | the student counterpart, copied from the earlier sweep |
| `_alignstudent` | the student control, same target measured on this branch |
| `_s<seed>` | a head-seed replicate on the same frozen backbone |

The `379` suffix is load-bearing: `arm5_alignstudent379_bb40k_hd15000s` is the
earlier sweep's number and `arm5_alignstudent_bb40k_hd15000s` is this branch's re-run of it.
They are the two sides of the code-boundary check and must not share a name.

## The earlier sweep's numbers, and what they do to every delta

arm5, backbone step 40 000, `--align-target student`, seed 20260520, the
same command line, all 97 configs:

| | GM-Relative MASE |
|---|---|
| student target, the earlier sweep's published row | 1.5478 |
| student target, this branch (the control) | 1.4501 |
| teacher target, this branch | 1.3515 |

The first two rows are 0.0977 apart, so the teacher-vs-earlier-sweep delta of
-0.1963 is not a measurement of the flag. On this one cell, both sides from
this branch, the flag moves -0.0986; over the ten controlled cells the mean is
-0.0192 with sign test p = 0.75, so the arm5 number is a cell, not a value of
the flag.

Nor is -0.0986 a stable number for that cell. Re-heading both sides under
seeds 20260723 and 20260724 puts the same difference at -0.0482 and +0.0174:
mean -0.0431, range 0.1160, and the sign changes. `arm5 combab`, re-headed
the same way, stays negative at -0.0140 / -0.0133 / -0.0230 with a range of
0.0097. Substituting those two three-seed means into the aggregate moves it
from -0.0192 to -0.0140 and leaves both tests where they were (sign p = 0.75,
Wilcoxon 0.70 -> 0.77).

All ten cells carry that check. Nine reproduce the earlier sweep within
0.0002 — five of them bit-identical across all 97 configs, four agreeing to
at worst 0.000157 in the aggregate. arm5 base is the only one that does not,
and it moves 0.09773. `snapshot_reproduction_40k.csv` holds the ten rows.

**That 0.0977 is two backbones, not one path run twice.**
`replicate_provenance_40k.csv` settles it from artefacts already on disk.
The earlier sweep's launcher resumes a crashed arm under a fresh `_r<N>` run
name, so an arm can leave more than one `_40k.pth`, each written by a process
that entered the HF stream at a different point; the eval takes the newest by
mtime. arm5's published row was evaluated on `..._r3_40k.pth`, written by a
run resumed at step 25 001. The other nine published rows used the base run.
This branch's re-run repeats the base run of all ten arms step for step —
40 000 of 40 000 steps identical on `loss`, `ff`, `gap` and
`hf_rows_consumed` — so arm5's training path is reproducible, and the one row
that misses is the one row published off a resumed backbone. Under the same
flags and the same seed, the base run and the resumed run disagree on 14 999
of their 15 000 shared steps.

So a delta in this directory means one of two different things, and
`code_snapshot` is how a reader tells them apart:

* **controlled** — both sides `#390-branch`, this branch's code. `controlled_delta_40k.csv`.
  Attributable to `--align-target`. Exists at backbone 40k only.
* **cross-experiment** — teacher `#390-branch`, student `#379-sweep`.
  `eval_bootstrap_ci.csv`, `eval_paired_tests.csv`, and every 100k and 200k
  row anywhere in this directory. Carries the flag and the snapshot shift
  together and cannot separate them.

The 100k and 200k rows have no same-branch student control and cannot get
one without retraining those cells to 100k/200k with the student target.
They stay cross-experiment and are labelled that way.

A run that was resumed writes a fresh `_r<N>` file rather than appending, so a
full trajectory is the concatenation of the `_losses.csv` files sharing a run
name, ordered by their first `step`. This experiment's waves are `_` (0 → 40k),
`_r2` (40k → 100k) and `_r3` (100k → 200k).

## Two things to know when reading across the 30 arms

**arm 5 and arm 6 v2 mean something different here than in the earlier sweep.** In this
directory they are the teacher-target retrain. The pre-teacher measurements of
the same ten cells are still readable at
`reports/2026-07-21_split_pred_rep_small/results/`, under the identical cell
names — that pair is the before/after of this retrain.

**`latent_movement_pairs.csv` mixes two measurement runs, on one scale.** The
twenty un-retrained arms are the earlier sweep's rows; the ten retrained arms were measured
here by `scripts/make_latent_movement_390.py`. Both use the same fixed
held-out batch (`reports/2026-07-21_split_pred_rep_small/plots/_latent_movement_batch.pt`,
B=8/T=4096/C=1) and the same `mean_one_minus_cos` from
`src.eval_latent_movement`. Re-running the earlier sweep's own script with `--arms arm1`
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

**The latent-drift probe does not perturb training.** It runs in every wave here and did not
exist when the earlier sweep ran: that sweep's logs print no
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
| `arm1_combab` 3.1251 at 40k (copied) | never trained. Loss is 19.03 at step 0, 19.82 at 40k, 19.54 at 100k and 19.75 at 200k (min 15.30, max 24.76), the largest rise-from-minimum of all 40 runs at 4.4648. Every other arm1 variant fell by 3.1 to 7.1 over the same window. Peak attention logit magnitude, 6.51, is the lowest in the set, so this is a flat objective, not a blow-up. |
| `arm5_nse` 1.8887 at 200k | clean loss, extreme attention. Loss falls 19.80 → 14.11 → 13.95 and flattens; rise-from-minimum is 0.0688, the 7th lowest of the 40 runs. Mean attention logit magnitude grows 7.63 → 14.28 → 35.09 across the three waves, peak 259.94 — the largest of all 40. |
| `arm6_v2` base 1.4322 → 1.9057 (40k → 100k) | no backbone signature at all. Loss 12.98 → 4.92 in wave 1, flat 4.98 → 4.92 in wave 2. Attention bounded and near the bottom of the set. Nothing in the backbone moved across the window where the eval number nearly doubled. |

## Caveats the report has to carry

* **Only the 40k comparison is controlled.** Everything else in this
  directory compares a this-branch number against an earlier-sweep one across a code
  boundary worth 0.0977 on the one cell where both sides were measured.
  The 100k and 200k rows cannot be decontaminated without retraining, so
  they stay cross-experiment.
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
  rule. Kept because the earlier sweep's heads used it and the comparison is to that sweep.

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
python3 experiments/2026-08-01_lalign_teacher/scripts/controlled_delta.py \
  --results reports/2026-08-04_lalign_teacher/results \
  --student-379-results reports/2026-07-21_split_pred_rep_small/results \
  --out-delta reports/2026-08-04_lalign_teacher/results/controlled_delta_40k.csv \
  --out-tests reports/2026-08-04_lalign_teacher/results/controlled_paired_tests_40k.csv
python3 experiments/2026-08-01_lalign_teacher/scripts/snapshot_reproduction.py \
  --results reports/2026-08-04_lalign_teacher/results \
  --student-379-results reports/2026-07-21_split_pred_rep_small/results \
  --out reports/2026-08-04_lalign_teacher/results/snapshot_reproduction_40k.csv
python3 experiments/2026-08-01_lalign_teacher/scripts/replicate_provenance.py \
  --runs-379 experiments/2026-07-21_split_pred_rep_small/runs \
  --eval-379 experiments/2026-07-21_split_pred_rep_small/eval_gm_mase \
  --runs-390 experiments/2026-08-01_lalign_teacher/runs \
  --snapshot reports/2026-08-04_lalign_teacher/results/snapshot_reproduction_40k.csv \
  --out reports/2026-08-04_lalign_teacher/results/replicate_provenance_40k.csv
python3 experiments/2026-08-01_lalign_teacher/scripts/seed_spread.py \
  --table reports/2026-08-04_lalign_teacher/results/gm_relative_mase.csv \
  --controlled reports/2026-08-04_lalign_teacher/results/controlled_delta_40k.csv \
  --out reports/2026-08-04_lalign_teacher/results/seed_spread.csv
```

The twelve 40k head-seed measurements are gated before they enter any of
those tables:

```bash
python3 experiments/2026-08-01_lalign_teacher/scripts/verify_head_seeds_40k.py \
  --results reports/2026-08-04_lalign_teacher/results/eval_gm_mase \
  --naive reports/2026-08-04_lalign_teacher/results/seasonal_naive_all_results.csv
```

It re-reads each `all_results.csv`, checks the 97 config rows, checks that
all twelve cover the same 97 configs and that every one is in the
seasonal-naive denominator, and recomputes `exp(mean log(MASE / naive))`
against the committed `_summary.txt`. Non-zero exit on any mismatch.

The nine remaining student controls themselves come from
`scripts/run_student_control_batch.sh` on elisa, which runs each cell's
earlier-sweep command line with `--align-target student` at seed 20260520 to step
40 000, then the 15 000-step head and the full 97-config GIFT-Eval.

The head-seed replicates come from `scripts/run_head_seeds.sh`, eight cells
under seeds 20260723 and 20260724 on the same frozen backbones. With the
waves' own 20260722 that is three seeds each, and those eight are every cell
in this directory that has more than one. Four of them are the 40k
comparison itself — `arm5 base` and `arm5 combab`, teacher side and student
side — run with `SIDES="teacher student"`, so the bar for the controlled
delta is measured where the delta lives instead of carried in from 100k or
200k. The other four are teacher-side only: `arm6_v2 base` at 100k, and
`arm5 nse`, `arm6_v2 ncpc` and `arm6_v2 combab` at 200k.

Then the report's tables and figures:

```bash
python3 experiments/2026-08-01_lalign_teacher/scripts/make_report_tables.py \
  reports/2026-08-04_lalign_teacher/results /tmp/tables.md
for f in reports/2026-08-04_lalign_teacher/plots/_make_*.py; do python3 "$f"; done
```
