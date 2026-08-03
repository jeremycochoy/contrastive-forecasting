# A per-loss-term map of latent movement, measured on the student and the EMA teacher latent

Across nine loss terms, the exponential-moving-average (EMA) teacher latent tracks the student latent. The nine terms split into four movement regimes.

![Drift between checkpoints 5000 steps apart](plots/drift_headline.png)

Four regimes:

- Near-zero drift: `align_teacher`, `sigreg_e`.
- Falling drift: `align`, `rep`, `sigreg_h`, `rep_moco`.
- No fall: `pred`, `cpc`.
- `pred_moco`: flat under α = 0.9, falling only under the schedule.

![L_align with the teacher as target](plots/align_fix.png)

![Dimension usage](plots/supporting/dim_usage.png)

One seed per arm, one probe batch, no downstream forecasting evaluation, so every number is a single measurement without a spread.

## Drift measured over other step intervals

![Cumulative drift away from the 5k checkpoint](plots/cumulative_drift.png)

![Drift between probes 500 steps apart](plots/drift_500.png)

## What this measures

| Quantity | Definition |
|---|---|
| `h_t` | the encoder latent at time t, 64 dimensions, the quantity every drift number is computed on |
| α | the weight the EMA teacher keeps on itself in `θ_teacher ← α·θ_teacher + (1 − α)·θ_student`, applied after every optimizer step |
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

## Adjacent-checkpoint drift per arm and latent, mean raw `drift_cos` per window, slope per decade of training step, from `results/summary.csv` and `results/drift.csv`

| Run | Latent | α | mean drift_cos 10k-25k | mean drift_cos 80k-100k | slope / decade | mean drift_cos_aligned 80k-100k |
|---|---|---|---|---|---|---|
| `pred` | student | none | 0.5672 | 0.7837 | +0.2778 | 0.6296 |
| `rep` | student | none | 0.8281 | 0.4090 | −0.3678 | 0.3566 |
| `align` | student | none | 1.0715 | 0.2653 | −0.9308 | 0.0000 |
| `sigreg_e` | student | none | 0.0959 | 0.0367 | −0.0861 | 0.0338 |
| `sigreg_h` | student | none | 0.5171 | 0.3415 | −0.3007 | 0.3035 |
| `cpc` | student | none | 0.6645 | 0.6393 | +0.0107 | 0.3198 |
| `pred_moco` | student | const 0.9 | 0.4619 | 0.4198 | −0.0722 | 0.2942 |
| `pred_moco` | teacher | const 0.9 | 0.4584 | 0.4091 | −0.0769 | 0.2910 |
| `rep_moco` | student | const 0.9 | 0.4028 | 0.2531 | −0.2277 | 0.1240 |
| `rep_moco` | teacher | const 0.9 | 0.4046 | 0.2470 | −0.2422 | 0.1184 |
| `align_teacher_a09` | student | const 0.9 | 0.0531 | 0.0123 | −0.0608 | 0.0013 |
| `align_teacher_a09` | teacher | const 0.9 | 0.0530 | 0.0122 | −0.0602 | 0.0013 |
| `align_teacher_sched` | student | 0.9 → 1.0 | 0.0402 | 0.0120 | −0.0472 | 0.0038 |
| `align_teacher_sched` | teacher | 0.9 → 1.0 | 0.0406 | 0.0111 | −0.0488 | 0.0034 |
| `pred_moco_sched` | student | 0.9 → 1.0 | 0.3958 | 0.1531 | −0.3239 | 0.1111 |
| `pred_moco_sched` | teacher | 0.9 → 1.0 | 0.3851 | 0.1409 | −0.3269 | 0.1033 |
| `rep_moco_sched` | student | 0.9 → 1.0 | 0.3701 | 0.0326 | −0.4608 | 0.0258 |
| `rep_moco_sched` | teacher | 0.9 → 1.0 | 0.3669 | 0.0257 | −0.4656 | 0.0195 |

## Reproducing

```bash
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/smoke.sh 0
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/orchestrate.sh 100000
WT=$PWD bash experiments/2026-08-01_align_teacher_ema_schedule/scripts/analyse.sh 0
```
