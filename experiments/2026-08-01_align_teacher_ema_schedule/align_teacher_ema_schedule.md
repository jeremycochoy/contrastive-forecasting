# L_align against the EMA teacher, and a 0.9 → 1.0 EMA-momentum schedule

*Runs in flight. Figures, table and verdict land when the four 100k-step runs finish.*

## What this measures

`drift_cos` is one minus the cosine similarity between two sets of encoder
latents `h_t`, computed on the same fixed ARMA probe batch. 0 means the
representation did not move; 1 means the two are orthogonal.

α is the EMA momentum: the weight the teacher keeps on its own previous value
in `θ_teacher ← α·θ_teacher + (1 − α)·θ_student`, applied after every optimizer
step. Higher α is a slower teacher; α = 1 freezes it.

## Runs

Four runs, same backbone, seed, dataset and 100k-step budget as
[#382](../2026-07-28_loss_term_isolation/loss_term_isolation.md).

| Run | Loss term | α |
|---|---|---|
| `align_teacher_a09` | `L_align`, target = EMA teacher's `h_{t+1}` | 0.9 constant |
| `align_teacher_sched` | same | 0.9 → 1.0 linear |
| `pred_moco_sched` | `L_pred` + MoCo negatives | 0.9 → 1.0 linear |
| `rep_moco_sched` | `L_rep` + MoCo keys | 0.9 → 1.0 linear |

The constant-α halves of `pred_moco` and `rep_moco` come from #382. `pred`,
`rep`, `sigreg_e`, `sigreg_h` and `cpc` have no teacher and are not re-run.

## Reproducing

```bash
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/smoke.sh 0
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/orchestrate.sh 100000
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/analyse.sh 0
```

`orchestrate.sh` runs the four arms two per GPU on elisa and writes checkpoints
to `/home/jupyter/checkpoints_backup/cf-388/`, outside the worktree.
`monitor.sh` ticks every 15 minutes: it copies the training CSVs into `sync/`
and watches for NaN and for a trainer that died before the budget.
`analyse.sh` runs the post-hoc probe over every 5000-step checkpoint of both
experiments, reduces the training CSVs, and writes the table and the figures.

The full-resolution `<run>_losses.csv` (~30 MB each) stay in the run directory;
`results/` carries the every-100-steps reduction the figures read.
