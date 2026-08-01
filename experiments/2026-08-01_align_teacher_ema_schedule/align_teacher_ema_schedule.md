# Pointing L_align at the EMA teacher removes the latent drift, and both align arms end on one direction

**TL;DR.** With the target fixed to the EMA teacher's latent, early `drift_cos`
falls from 1.0715 to 0.0531. Both align arms end at the dimension-usage floor,
one live direction out of 64, so that low drift is measured on a representation
with almost nothing left to rotate.

## The target fix

![L_align with the teacher as target](plots/align_fix.png)

*`drift_cos` of the student `h_t` between checkpoints 5000 steps apart, fixed
ARMA probe batch, α = 0.9 constant. Purple: the earlier `align` arm, whose
target was the student's own `sg(h_{t+1})`. Blue: `align_teacher`, target = EMA
teacher `h_{t+1}`. The orange band is the EMA-teacher latent of the same run;
it lies under the student curve. Mean over 10k-25k: 1.0715 against 0.0531.*

## Why that number does not mean the representation settled

![Dimension usage](plots/supporting/dim_usage.png)

*`U = 1/(64 · mean cos²)` over the time axis, from the training logs. 1 is
isotropic, 1/64 = 0.015625 means every timestep points the same way. At 100k:
`align_teacher` 0.0167 with α = 0.9 and 0.0182 with the schedule, the earlier
`align` arm exactly 0.015625, `rep_moco` 0.5800 and `rep_moco_sched` 0.5609,
`pred_moco` 0.0443 and `pred_moco_sched` 0.0510. Wide pale curves are constant
α, dashed are the schedule; on the MoCo arms the two coincide.*

`dim_usage` L2-normalises before measuring, so it reports direction only. The
rotation-invariant columns of `results/drift.csv` say the same thing about the
earlier `align` arm: raw adjacent `drift_cos` spans 0.0133 to 1.3226, while
`drift_cos_aligned` peaks at 0.0464 and sits at or below 0.0001 from 40k
onward, and `cka` against the 5k checkpoint is 0.0000 at 100k. Removing the
global feature-axis rotation removes nearly all of the raw drift.

## All arms, same probe

![Drift between checkpoints 5000 steps apart](plots/drift_headline.png)

*One panel per loss term, student `h_t` in blue, EMA-teacher `h_t` as the orange
band beneath it. Shaded panels have a teacher. Solid is α = 0.9 constant,
dashed is α: 0.9 → 1.0.*

![Cumulative drift away from the 5k checkpoint](plots/cumulative_drift.png)

*Same probe, each checkpoint compared to the run's 5k checkpoint instead of to
its predecessor. `rep_moco` ends close to orthogonal at 100k: 0.9870 with
constant α, 0.9719 with the schedule.*

![Drift between probes 500 steps apart](plots/drift_500.png)

*The in-training probe, 500-step spacing, four runs of this card only. All three
arms drop below 0.1 between 1000 and 2000 steps and stay there.*

## The α schedule

![EMA momentum against training step](plots/alpha_schedule.png)

*α is the weight the teacher keeps on itself in `θ_teacher ← α·θ_teacher +
(1 − α)·θ_student`, applied after every optimizer step. The three scheduled runs
share one line; `align_teacher_a09` is the flat line at 0.9.*

Stated rotation-invariantly, on `drift_cos_aligned` against the 5k checkpoint at
100k, the schedule moves `rep_moco` from 0.7836 to 0.7789 and `pred_moco` from
0.4619 to 0.4588. On `align_teacher` the late-window raw `drift_cos` reads
0.0123 with constant α and 0.0120 with the schedule; both sit at the floor, and
a metric at its floor cannot move, so this arm carries no information about the
schedule. One seed per arm and one probe batch, so every ratio here is a single
measurement without a spread.

No downstream forecasting evaluation was run in this experiment.

## What this measures

`drift_cos` is one minus the mean cosine similarity between two sets of encoder
latents `h_t` on the same fixed ARMA probe batch, rows L2-normalised. 0 means
the representation did not move, 1 means orthogonal.

`drift_cos_aligned` is the same quantity after a Procrustes rotation removes the
best global feature-axis rotation: the movement a downstream linear head cannot
absorb. `rot_gap` is the difference, the part that is pure rotation. `cka` is
linear centered CKA, rotation- and scale-invariant, in [0, 1].

## Runs

Four runs, same backbone, seed, dataset and 100k-step budget as the loss-term
isolation experiment
([report](../2026-07-28_loss_term_isolation/loss_term_isolation.md)), which
provides the constant-α halves of `pred_moco` and `rep_moco` and the earlier
`align` arm. `pred`, `rep`, `sigreg_e`, `sigreg_h` and `cpc` have no teacher and
were not re-run.

| Run | Loss term | α |
|---|---|---|
| `align_teacher_a09` | `L_align`, target = EMA teacher's `h_{t+1}` | 0.9 constant |
| `align_teacher_sched` | same | 0.9 → 1.0 linear |
| `pred_moco_sched` | `L_pred` + MoCo negatives | 0.9 → 1.0 linear |
| `rep_moco_sched` | `L_rep` + MoCo keys | 0.9 → 1.0 linear |

## Drift per arm and latent

`drift_cos` of the adjacent-checkpoint probe, averaged over the two windows.
Slope is per decade of training step, from `results/summary.csv`.

| Run | Latent | α | mean 10k-25k | mean 80k-100k | slope / decade |
|---|---|---|---|---|---|
| `pred` | student | none | 0.5672 | 0.7837 | +0.2778 |
| `rep` | student | none | 0.8281 | 0.4090 | −0.3678 |
| `align` | student | none | 1.0715 | 0.2653 | −0.9308 |
| `sigreg_e` | student | none | 0.0959 | 0.0367 | −0.0861 |
| `sigreg_h` | student | none | 0.5171 | 0.3415 | −0.3007 |
| `cpc` | student | none | 0.6645 | 0.6393 | +0.0107 |
| `pred_moco` | student | const 0.9 | 0.4619 | 0.4198 | −0.0722 |
| `pred_moco` | teacher | const 0.9 | 0.4584 | 0.4091 | −0.0769 |
| `rep_moco` | student | const 0.9 | 0.4028 | 0.2531 | −0.2277 |
| `rep_moco` | teacher | const 0.9 | 0.4046 | 0.2470 | −0.2422 |
| `align_teacher_a09` | student | const 0.9 | 0.0531 | 0.0123 | −0.0608 |
| `align_teacher_a09` | teacher | const 0.9 | 0.0530 | 0.0122 | −0.0602 |
| `align_teacher_sched` | student | 0.9 → 1.0 | 0.0402 | 0.0120 | −0.0472 |
| `align_teacher_sched` | teacher | 0.9 → 1.0 | 0.0406 | 0.0111 | −0.0488 |
| `pred_moco_sched` | student | 0.9 → 1.0 | 0.3958 | 0.1531 | −0.3239 |
| `pred_moco_sched` | teacher | 0.9 → 1.0 | 0.3851 | 0.1409 | −0.3269 |
| `rep_moco_sched` | student | 0.9 → 1.0 | 0.3701 | 0.0326 | −0.4608 |
| `rep_moco_sched` | teacher | 0.9 → 1.0 | 0.3669 | 0.0257 | −0.4656 |

## Reproducing

```bash
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/smoke.sh 0
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/orchestrate.sh 100000
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/analyse.sh 0
```

`orchestrate.sh` runs the four arms two per GPU on elisa and writes checkpoints
to `/home/jupyter/checkpoints_backup/cf-388/`, outside the worktree.
`monitor.sh` ticks every 15 minutes: it copies the training CSVs into `sync/`
(a local mirror, git-ignored) and watches for NaN and for a trainer that died
before the budget. `analyse.sh` runs the post-hoc probe over every 5000-step
checkpoint of both experiments, reduces the training CSVs, and writes the tables
and the figures.

The full-resolution `<run>_losses.csv` (~30 MB each) stay in the run directory;
`results/` carries the every-100-steps reduction the figures read.
