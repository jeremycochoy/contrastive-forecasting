# A per-loss-term map of latent movement, measured on the student and the EMA teacher latent

**TL;DR.** Across nine loss terms, the exponential-moving-average (EMA) teacher latent tracks the student latent. The nine terms split into four movement regimes.

![Drift between checkpoints 5000 steps apart](plots/drift_headline.png)

*Solid = student, dashed = EMA teacher; shaded panels have a teacher, blue/orange = α 0.9, green/pink = schedule.*

Four regimes: near-zero drift for `align_teacher` and `sigreg_e`; falling drift for `align`, `rep`, `sigreg_h` and `rep_moco`; no fall for `pred` and `cpc`; `pred_moco` flat under α = 0.9 and falling only under the schedule.

![L_align with the teacher as target](plots/align_fix.png)

*`align_teacher` at α = 0.9, target = EMA teacher `h_{t+1}`.*

![Dimension usage](plots/supporting/dim_usage.png)

*Both align arms end near the 1/64 time-axis floor, so their low drift is measured on a representation with almost nothing left to rotate.*

One seed per arm, one probe batch, no downstream forecasting evaluation, so every number is a single measurement without a spread.

## Other windows on the same probe

![Cumulative drift away from the 5k checkpoint](plots/cumulative_drift.png)

*Slow per-window drift still accumulates: `align_teacher_a09` reaches 0.8529 against its 5k checkpoint.*

![Drift between probes 500 steps apart](plots/drift_500.png)

## The α schedule

![EMA momentum against training step](plots/alpha_schedule.png)

*α is the weight the teacher keeps on itself in `θ_teacher ← α·θ_teacher + (1 − α)·θ_student`, applied after every optimizer step.*

## What this measures

| Quantity | Definition |
|---|---|
| probe batch | one ARMA draw, 64 series, 1024 raw timesteps, seed 20260722, shared by every checkpoint of every run |
| `drift_cos` | 1 − mean cosine similarity between two sets of encoder latents `h_t`; 0 = no movement, 1 = orthogonal |
| `drift_cos_aligned` | `drift_cos` after removing the best orthogonal (Procrustes) map; the movement a linear head cannot absorb |
| `cka` | linear centered kernel alignment, rotation- and scale-invariant, in [0, 1]; reads 0 when a latent is collinear across time, as `align` is at 100k |
| `U` | dimension usage `1/(d · mean cos²)`, `d = 64`; 1 = isotropic, 1/64 = 0.015625 = every vector aligned |
| adjacent probe | checkpoints 5000 steps apart; `vs_initial` compares each checkpoint to the run's 5k checkpoint |

## Runs

| Setup | Value |
|---|---|
| new runs | 4 |
| shared with [loss-term isolation](../2026-07-28_loss_term_isolation/loss_term_isolation.md) | backbone, seed, dataset, 100k-step budget |
| taken from that experiment | `align`, and the constant-α halves of `pred_moco` and `rep_moco` |

| Run | Loss term | α |
|---|---|---|
| `align_teacher_a09` | `L_align`, target = EMA teacher's `h_{t+1}` | 0.9 constant |
| `align_teacher_sched` | same | 0.9 → 1.0 linear |
| `pred_moco_sched` | `L_pred` + MoCo negatives | 0.9 → 1.0 linear |
| `rep_moco_sched` | `L_rep` + MoCo keys | 0.9 → 1.0 linear |
| `align` | `L_align`, target = student `sg(h_{t+1})` | no teacher |
| `pred`, `rep`, `sigreg_e`, `sigreg_h`, `cpc` | as in loss-term isolation | no teacher |

## Late-window adjacent drift, student `h_t`, mean over 80k–100k, from `results/drift.csv`

| Run | α | mean drift_cos 80k–100k | mean drift_cos_aligned 80k–100k |
|---|---|---|---|
| `align` | none | 0.2653 | 0.0000 |
| `align_teacher_a09` | const 0.9 | 0.0123 | 0.0013 |
| `align_teacher_sched` | 0.9 → 1.0 | 0.0120 | 0.0038 |
| `pred_moco` | const 0.9 | 0.4198 | 0.2942 |
| `pred_moco_sched` | 0.9 → 1.0 | 0.1531 | 0.1111 |
| `rep_moco` | const 0.9 | 0.2531 | 0.1240 |
| `rep_moco_sched` | 0.9 → 1.0 | 0.0326 | 0.0258 |

## Drift against the 5k checkpoint at 100k, student `h_t`, from `results/drift.csv` (`kind=vs_initial`)

| Run | α | drift_cos vs 5k | drift_cos_aligned vs 5k | cka vs 5k |
|---|---|---|---|---|
| `align` | none | 1.237727 | 0.000035 | 0.000000 |
| `align_teacher_a09` | const 0.9 | 0.852939 | 0.011372 | 0.368196 |
| `align_teacher_sched` | 0.9 → 1.0 | 0.697010 | 0.029031 | 0.288741 |
| `pred_moco` | const 0.9 | 0.899133 | 0.461897 | 0.158591 |
| `pred_moco_sched` | 0.9 → 1.0 | 0.924765 | 0.458751 | 0.171612 |
| `rep_moco` | const 0.9 | 0.986973 | 0.783621 | 0.084590 |
| `rep_moco_sched` | 0.9 → 1.0 | 0.971899 | 0.778909 | 0.084094 |

## Dimension usage at 100k, from `results/loss_curve.csv`, floor 1/64 = 0.015625 on both axes

| Run | U, time axis | U, batch axis |
|---|---|---|
| `align` | 0.015625 | 0.015625 |
| `align_teacher_a09` | 0.016709 | 0.057068 |
| `align_teacher_sched` | 0.018166 | 0.074540 |
| `pred_moco` | 0.044333 | 0.559589 |
| `pred_moco_sched` | 0.050975 | 0.597697 |
| `rep_moco` | 0.580033 | 0.794081 |
| `rep_moco_sched` | 0.560926 | 0.827109 |

## Adjacent-checkpoint drift per arm and latent, mean raw `drift_cos` per window, slope per decade of training step, from `results/summary.csv`

| Run | Latent | α | mean drift_cos 10k-25k | mean drift_cos 80k-100k | slope / decade |
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
