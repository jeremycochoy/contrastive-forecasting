# Training the forecaster on its own output: the one clean cell improves 9.8%, and the training machine moves scores by more

`--train-rollout-depth K` duplicates every f-bearing loss term at depth
1..K, so training composes the operator that eval composes.

B1 at `k = 3` scores 1.0850 against its own `k = 0` at 1.2025, which is 9.8%
better on the student head and 8.8% on the teacher, and it is unreplicated at
one seed and one stop. It is the study's only comparison with both sides
trained on one machine and with a `k = 0` that reproduces its published value
exactly.

![depth response](plots/depth_response.png)

*Every trained depth against the same arm's own retrained `k = 0`, bb40k,
97 GIFT-Eval configs.*

## What ran

| | |
|---|---|
| trained | A3, B1, B5, B9 — 4 of the card's 14 cells |
| never ran | A1, A2, A4, B2, B3, B4, B6, B7, B8, B10 |
| stops reached | bb40k only; no bb100k, no bb200k, so the card's extend rule never fired |
| card's primary criterion (med+long ≥ 5% better, short losing < 2%) | met by B1, both sides on elisa; met by B9, whose two sides trained on two machines |
| card's secondary criterion (full 97 beats `k = 0` by more than the head-seed band ±0.0384) | same two cells, same caveat on B9; its bb100k and bb200k half is out of reach |
| heads missing | `B5·s3`'s teacher head (annex) |

## The machine moved the baselines; the seed did not

![reproduction](plots/reproduction.png)

*Each retrained `k = 0` against the value its parent report published, student
head, bb40k.*

Backbone runs on the rented vast.ai boxes do not reproduce the published
numbers, and runs on elisa do.

![B5 backbones](plots/b5_backbones.png)

*B5 trained three times on one recipe, one code snapshot, one head seed and
one eval.*

| contrast | what changes | Δ at `k = 0` | 95% CI |
|---|---|---|---|
| B5·s1 against B5·s3 | **the machine**, at one seed | **−0.1166** | [−0.1885, −0.0645] |
| B5·s2 against B5·s3 | the seed, on one machine | +0.0035 | [−0.0183, +0.0230] |
| *what both intervals bound* | *paired dataset-cluster bootstrap over one run pair each: the eval sample, not run-to-run variance* | | |

B9's −17.9%, A3's `k = 1` ladder and A3's ×4 re-weighting control each cross
a machine boundary, so read them as direction and not as magnitude.

## Where the change lands

![horizon split, student head](plots/horizon_split_student.png)

*GM-Relative MASE by horizon term, depth against the same arm's `k = 0`,
student head. Teacher head:
[`horizon_split_teacher.png`](plots/horizon_split_teacher.png).*

![ladder](plots/ladder.png)

*This study's bb40k points on each cell's own published `k = 0` trajectory.*

![domain radar, student head](plots/domain_radar_student.png)

*Per-domain GM-Relative MASE, each arm against its own `k = 0`, student head.
Teacher head: [`domain_radar_teacher.png`](plots/domain_radar_teacher.png).*

## A3: the depth, or the weight it carries?

![A3 depth against weight](plots/a3_depth.png)

*A3's depth ladder against the re-weighting control, both heads, bb40k.*

The ×4 re-weighting costs 0.0401 of A3's 0.1429, so the depth carries the
other 72%.

| A3, student head | `k = 0` | `k = 1` | `k = 0`, `L_align` ×4 | `k = 3` |
|---|---|---|---|---|
| GM-Relative MASE | 1.2189 | 1.1995 | 1.2590 | 1.3618 |
| Δ against `k = 0` | — | −1.6%, CI [−0.0537, +0.0148] | +3.3% | +11.7%, CI [+0.0893, +0.2122] |

## The composed operator does get more faithful

![rollout fidelity](plots/rollout_fidelity.png)

*`cos` between the rolled latent and the true `h_{t+d}` for `d = 1..16`, no
head in the way, on a fixed diagnostic batch.*

Every arm improves at `k = 3` at all 16 rollout depths, by +0.002 to +0.545,
including the two arms that score worse.

![per-depth forecast error](plots/cos_err_depth.png)

*`1 − cos(f^(j)_t, h_{t+1+j})` during training, one line per depth, against
the `k = 0` run's single line.*

| depth-0 error, `k = 3` run minus `k = 0` run | last 50% | last 25% | last 10% | final step | one sign over all four windows |
|---|---|---|---|---|---|
| B9 | −0.0707 | −0.0893 | −0.0938 | −0.0817 | yes |
| B1 | −0.0968 | −0.0915 | −0.1122 | −0.0807 | yes |
| A3 | −0.0469 | −0.0004 | +0.0489 | +0.0623 | **no** |
| B5·s2 | +0.0121 | +0.0061 | +0.0023 | −0.0129 | **no** |

Training loss is flat on every depth run.

*Per-arm windows and diagnostics:
[`results/depth0_gap.csv`](results/depth0_gap.csv),
[`per_run_loss.png`](plots/per_run_loss.png),
[`latent_movement.png`](plots/latent_movement.png),
[`dim_usage_per_arm.png`](plots/dim_usage_per_arm.png),
[`cos_error_per_arm.png`](plots/cos_error_per_arm.png).*

![encoder delta](plots/encoder_delta.png)

*Teacher-encoder head minus student-encoder head, per arm per depth. Every
value is inside ±0.0198.*

## Tables

<!-- TABLES:BEGIN -->

### Coverage

The card names 14 cells. This study trained **4 of them**: A3, B1, B5, B9. It never ran **10**: A1, A2, A4, B2, B3, B4, B6, B7, B8, B10.

| cell | f-bearing term | EMA α | depths trained |
|---|---|---|---|
| A1 | — | — | **never ran** |
| A2 | — | — | **never ran** |
| A3 | rep_only + L_align | scheduled 0.9 -> 1.0 | k = 0, k = 1, k = 3 |
| A4 | — | — | **never ran** |
| B1 | rep_only + L_align | fixed 0.9 | k = 0, k = 3 |
| B2 | — | — | **never ran** |
| B3 | — | — | **never ran** |
| B4 | — | — | **never ran** |
| B5 | pooled xshh_allt | fixed 0.9 | k = 0, k = 3 |
| B6 | — | — | **never ran** |
| B7 | — | — | **never ran** |
| B8 | — | — | **never ran** |
| B9 | split L_pred | fixed 0.9 | k = 0, k = 3 |
| B10 | — | — | **never ran** |

Every trained stop is bb40k. No cell reached bb100k or bb200k, so the card's extend rule never fired and this study publishes one stop.

### Reproduction of the published k = 0

Same cell, same recipe, same head seed 20260722, same 97-config B4 eval, student head. The rows are sorted by the machine, because that is what the check separates on: every retrain on elisa lands on its published value and neither retrain on a rented box does.

Two gates, because the rows ask two questions. A retrain at the parents' own backbone seed 20260520 is repeating the published run, and takes the card's 0.0002. A retrain at another seed is drawing a new run, and takes the seed band.

| backbone | seed | machine | published k = 0 | retrained k = 0 | \|Δ\| | gate | verdict |
|---|---|---|---|---|---|---|---|
| B1 | 20260520 | elisa | 1.2025 | 1.2025 | 0.0000 | 0.0002, the card | PASS |
| B5·s3 | 20260520 | elisa | 1.2748 | 1.2751 | 0.0003 | 0.0002, the card | at printed precision |
| B9 | 20260520 | elisa | 1.5579 | 1.5583 | 0.0004 | 0.0002, the card | at printed precision |
| B5·s2 | 20260521 | elisa | 1.2748 | 1.2716 | 0.0032 | 0.0230, the seed band | inside the seed band |
| A3 | 20260520 | vast box d | 1.1895 | 1.2189 | 0.0294 | 0.0002, the card | FAIL |
| B5·s1 ✗ | 20260520 | vast box d | 1.2748 | 1.3917 | 0.1169 | 0.0002, the card | FAIL |
| B5·pub | 20260520 | the parent report's box | 1.2748 | 1.2751 | 0.0003 | 0.0002, the card | at printed precision |

The parents print four decimals, so a difference below 0.0005 is the smallest the published table can resolve. The card's gate of 0.0002 is stricter than that.

The seed band is 0.0230, the far end of the 95% interval on this study's one measurement of a seed change: `B5·s2` against `B5·s3`, one machine, one recipe, +0.0035 [-0.0183, +0.0230]. It is one run pair, and the interval is over that pair's eval sample rather than over seeds, so the band is a floor on what a seed can move and not a bound on it. B5·s2 is the only row it gates; every other row here carries the parents' own seed.

`B5·pub` is not a training: it takes the parent report's own published B5 checkpoint and puts this study's head and eval on it, so its row bounds the head and the eval rather than the trainer. `B5·s3` is a training, at the protocol seed, on elisa.

### Depth response, against each arm's own k = 0

| arm | seed | machine held | head | k | k = 0 | this k | Δ | all | short | med+long | criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B9 | 20260520 | no, elisa → vast box c | student | 3 | 1.5583 | 1.2791 | -0.2791 | -17.9% | -12.6% | -24.4% | **MET** |
| B9 | 20260520 | no, elisa → vast box c | teacher | 3 | 1.5599 | 1.2728 | -0.2871 | -18.4% | -12.8% | -25.2% | **MET** |
| B1 | 20260520 | yes, elisa | student | 3 | 1.2025 | 1.0850 | -0.1175 | -9.8% | -5.4% | -15.2% | **MET** |
| B1 | 20260520 | yes, elisa | teacher | 3 | 1.2001 | 1.0948 | -0.1053 | -8.8% | -5.1% | -13.4% | **MET** |
| B5·s1 ✗ | 20260520 | no, vast box d → vast box a | student | 3 | 1.3917 | 1.3204 | -0.0713 | -5.1% | -6.4% | -3.4% | not met |
| B5·s1 ✗ | 20260520 | no, vast box d → vast box a | teacher | 3 | 1.3719 | 1.3216 | -0.0503 | -3.7% | -4.4% | -2.6% | not met |
| B5·s2 | 20260521 | yes, elisa | student | 3 | 1.2716 | 1.3292 | +0.0575 | +4.5% | +7.0% | +1.4% | not met |
| B5·s2 | 20260521 | yes, elisa | teacher | 3 | 1.2661 | 1.3260 | +0.0599 | +4.7% | +8.1% | +0.5% | not met |
| A3 | 20260520 | no, vast box d → elisa | student | 1 | 1.2189 | 1.1995 | -0.0195 | -1.6% | -2.6% | -0.2% | not met |
| A3 | 20260520 | no, vast box d → vast box b | student | 3 | 1.2189 | 1.3618 | +0.1429 | +11.7% | +17.1% | +5.1% | not met |
| A3 | 20260520 | no, vast box d → elisa | teacher | 1 | 1.2184 | 1.2063 | -0.0121 | -1.0% | -1.5% | -0.4% | not met |
| A3 | 20260520 | no, vast box d → vast box b | teacher | 3 | 1.2184 | 1.3521 | +0.1337 | +11.0% | +15.8% | +4.9% | not met |

Criterion, from the card: medium+long (42 configs) at least 5% better, short (55 configs) losing less than 2%.

`machine held` = did the two sides train on the same box. A `no` row carries a machine change as well as a depth change. The B5 table below measures the machine alone, at one seed, at 0.1166, so a `no` row carries a term larger than most of the deltas in this table. Only the `yes` rows report the depth and nothing else.

✗ marks a retracted row: B5·s1's `k = 0` trained on a rented box and misses its published value by 0.1169; `B5·s3` retrains it at the same seed on elisa and lands 0.0003 away, so the baseline the -5.1% rests on is a rented-box artefact and the delta is retracted.

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone. It does not bound the machine.

### Paired dataset-cluster bootstrap, per horizon subset

The resampling unit is the dataset: `<ds>/short`, `/medium` and `/long` are three configs of one series and are not independent draws. 95% percentile interval over 10,000 resamples. Each interval is over one run pair's 97 configs, so it bounds the eval sample and not run-to-run variance.

| arm | head | k | subset | n | Δ | 95% CI | resamples improved |
|---|---|---|---|---|---|---|---|
| B9 | student | 3 | all | 97 | -0.2791 | [-0.3548, -0.1980] | 100.0% |
| B9 | student | 3 | short | 55 | -0.1736 | [-0.2470, -0.1038] | 100.0% |
| B9 | student | 3 | medium_long | 42 | -0.4472 | [-0.5655, -0.3382] | 100.0% |
| B9 | teacher | 3 | all | 97 | -0.2871 | [-0.3644, -0.2032] | 100.0% |
| B9 | teacher | 3 | short | 55 | -0.1751 | [-0.2501, -0.1018] | 100.0% |
| B9 | teacher | 3 | medium_long | 42 | -0.4670 | [-0.5952, -0.3523] | 100.0% |
| B1 | student | 3 | all | 97 | -0.1175 | [-0.1801, -0.0615] | 100.0% |
| B1 | student | 3 | short | 55 | -0.0556 | [-0.1017, -0.0184] | 99.9% |
| B1 | student | 3 | medium_long | 42 | -0.2244 | [-0.3504, -0.1243] | 100.0% |
| B1 | teacher | 3 | all | 97 | -0.1053 | [-0.1661, -0.0515] | 100.0% |
| B1 | teacher | 3 | short | 55 | -0.0523 | [-0.0980, -0.0146] | 99.7% |
| B1 | teacher | 3 | medium_long | 42 | -0.1963 | [-0.3129, -0.1047] | 100.0% |
| B5·s1 ✗ | student | 3 | all | 97 | -0.0713 | [-0.1327, -0.0267] | 100.0% |
| B5·s1 ✗ | student | 3 | short | 55 | -0.0848 | [-0.1732, -0.0068] | 98.3% |
| B5·s1 ✗ | student | 3 | medium_long | 42 | -0.0504 | [-0.0972, -0.0137] | 99.7% |
| B5·s1 ✗ | teacher | 3 | all | 97 | -0.0503 | [-0.0965, -0.0108] | 99.4% |
| B5·s1 ✗ | teacher | 3 | short | 55 | -0.0571 | [-0.1215, +0.0086] | 95.8% |
| B5·s1 ✗ | teacher | 3 | medium_long | 42 | -0.0395 | [-0.0882, +0.0028] | 96.6% |
| B5·s2 | student | 3 | all | 97 | +0.0575 | [+0.0173, +0.1094] | 0.2% |
| B5·s2 | student | 3 | short | 55 | +0.0809 | [+0.0231, +0.1549] | 0.2% |
| B5·s2 | student | 3 | medium_long | 42 | +0.0199 | [-0.0215, +0.0745] | 19.0% |
| B5·s2 | teacher | 3 | all | 97 | +0.0599 | [+0.0214, +0.1105] | 0.1% |
| B5·s2 | teacher | 3 | short | 55 | +0.0925 | [+0.0345, +0.1701] | 0.0% |
| B5·s2 | teacher | 3 | medium_long | 42 | +0.0074 | [-0.0268, +0.0543] | 38.7% |
| A3 | student | 1 | all | 97 | -0.0195 | [-0.0537, +0.0148] | 86.9% |
| A3 | student | 1 | short | 55 | -0.0294 | [-0.0652, +0.0007] | 97.1% |
| A3 | student | 1 | medium_long | 42 | -0.0029 | [-0.0565, +0.0628] | 55.8% |
| A3 | student | 3 | all | 97 | +0.1429 | [+0.0893, +0.2122] | 0.0% |
| A3 | student | 3 | short | 55 | +0.1899 | [+0.1254, +0.2739] | 0.0% |
| A3 | student | 3 | medium_long | 42 | +0.0698 | [+0.0226, +0.1415] | 0.0% |
| A3 | teacher | 1 | all | 97 | -0.0121 | [-0.0479, +0.0261] | 74.0% |
| A3 | teacher | 1 | short | 55 | -0.0163 | [-0.0596, +0.0275] | 77.9% |
| A3 | teacher | 1 | medium_long | 42 | -0.0052 | [-0.0572, +0.0602] | 59.5% |
| A3 | teacher | 3 | all | 97 | +0.1337 | [+0.0839, +0.2004] | 0.0% |
| A3 | teacher | 3 | short | 55 | +0.1760 | [+0.1177, +0.2537] | 0.0% |
| A3 | teacher | 3 | medium_long | 42 | +0.0673 | [+0.0197, +0.1414] | 0.0% |

### One cell, three backbones

B5 (`arm4_combab_fix09`) trained three times on one recipe, one code snapshot, one head seed and one eval. They differ by backbone seed and by machine, and each contrast below names which of the two it changes. The machine moves the score and the seed does not.

| backbone | seed | machine | k = 0 | k = 3 | k = 3 − k = 0 |
|---|---|---|---|---|---|
| B5·s1 ✗ | 20260520 | a rented box | 1.3917 | 1.3204 | -0.0713 |
| B5·s2 | 20260521 | elisa | 1.2716 | 1.3292 | +0.0575 |
| B5·s3 | 20260520 | elisa | 1.2751 | — | — |

| contrast | what changes | k | Δ | 95% CI |
|---|---|---|---|---|
| B5·s1 against B5·s3 | the machine, at one seed | 0 | -0.1166 | [-0.1885, -0.0645] |
| B5·s2 against B5·s3 | the seed, on one machine | 0 | +0.0035 | [-0.0183, +0.0230] |
| B5·s1 against B5·s2 | the seed AND the machine | 0 | -0.1200 | [-0.1825, -0.0742] |
| B5·s1 against B5·s2 | the seed AND the machine | 3 | +0.0088 | [-0.0306, +0.0520] |

Student head, 97 configs. `B5·s3` holds `B5·s1`'s seed and `B5·s2`'s machine, so the first two rows separate what the third confounds: the machine moves `k = 0` by 0.1166 and the seed by 0.0035.

Every interval here is a paired dataset-cluster bootstrap over the 97 eval configs of ONE run pair. It bounds the eval sample: how far the difference between these two runs could move if the datasets had been drawn again. It does not bound run-to-run variance, and neither contrast has a replicate to bound it with. No two of B5's three backbones share both a seed and a machine.

A retrain at a fixed seed is a machine test only if the seed pins the data order. It does. `mixup` counts the examples the mixer touched in the 200-step window, so two runs that see the same batches print the same count. `B5·s1` and `B5·s3` carry one seed and print one count at every step: they saw the same batches in the same order, on two machines, and the losses beside the counts still part.

| step | B5·s1<br>seed 20260520, a rented box | B5·s2<br>seed 20260521, elisa | B5·s3<br>seed 20260520, elisa |
|---|---|---|---|
| 200 | 5.5767  `61/200` | 5.6595  `53/200` | 5.5610  `61/200` |
| 400 | 5.1220  `58/200` | 5.3568  `62/200` | 5.2078  `58/200` |
| 600 | 4.9019  `65/200` | 5.0143  `51/200` | 4.9412  `65/200` |
| 800 | 4.9475  `65/200` | 5.1256  `65/200` | 5.1249  `65/200` |

### One loss shape, two EMA regimes

B1 and A3 train the same f-bearing term, `rep_only` + `L_align`, on the same `arm6_v2 combab` arm. They differ in the EMA schedule — and, since A3's two depths trained on two boxes, in the machine as well.

| arm | EMA α | machine held | head | k = 0 | k = 3 | Δ | Δ% |
|---|---|---|---|---|---|---|---|
| B1 | fixed 0.9 | yes, elisa | student | 1.2025 | 1.0850 | -0.1175 | -9.8% |
| B1 | fixed 0.9 | yes, elisa | teacher | 1.2001 | 1.0948 | -0.1053 | -8.8% |
| A3 | scheduled 0.9 -> 1.0 | no | student | 1.2189 | 1.3618 | +0.1429 | +11.7% |
| A3 | scheduled 0.9 -> 1.0 | no | teacher | 1.2184 | 1.3521 | +0.1337 | +11.0% |

### A3: is the damage the depth, or the weight?

Summing the depths multiplies `L_align`'s weight against the f-free terms by k + 1. The `L_align x4` row applies that re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 | share of the k = 3 damage the re-weighting explains |
|---|---|---|---|---|---|
| student | 1.2189 | 1.2590 | 1.1995 | 1.3618 | 28% |
| teacher | 1.2184 | 1.2558 | 1.2063 | 1.3521 | 28% |

Every column trained on a different box from at least one other. A3_k0: vast box d · G3_A3_k0_aw4: elisa · G3_A3_k1: elisa · A3_k3: vast box b. The machine alone is worth 0.1166 on this study's one controlled measurement of it, which is more than either control's own size, so read the two controls as direction and not as magnitude.

### What the depth costs

Median `fwd + bwd` per step, from each run's own trainer log. A median is a cost of the depth only where the run had the card to itself, so the table says which did. `run_provenance.py` reads that off the driver logs and [`results/steptime_solo.csv`](results/steptime_solo.csv) carries it per run.

| arm | f-bearing term | k | machine | card | fwd+bwd | alone? |
|---|---|---|---|---|---|---|
| B9 | split L_pred | 0 | elisa | RTX 4090 | 212.6 ms, shared | no — another backbone for 96% of the run; head training for 4% of it |
| B9 | split L_pred | 3 | vast box c | RTX 4090 | 425.2 ms | yes |
| B1 | rep_only + L_align | 0 | elisa | RTX 4090 | 178.6 ms, shared | no — another backbone for 100% of the run; head training for 100% of it |
| B1 | rep_only + L_align | 3 | elisa | RTX 4090 | 235.1 ms, shared | no — another backbone for 68% of the run; head training for 100% of it |
| B5·s1 | pooled xshh_allt | 0 | vast box d | RTX 5090 | 117.6 ms | yes |
| B5·s1 | pooled xshh_allt | 3 | vast box a | RTX 5090 | 301.9 ms | yes |
| B5·s2 | pooled xshh_allt | 0 | elisa | RTX 4090 | 201.1 ms, shared | no — another backbone for 100% of the run; head training for 98% of it |
| B5·s2 | pooled xshh_allt | 3 | elisa | RTX 4090 | 500.9 ms, shared | no — another backbone for 43% of the run; head training for 100% of it |
| A3 | rep_only + L_align | 0 | vast box d | RTX 5090 | 115.9 ms | yes |
| A3 | rep_only + L_align | 1 | elisa | RTX 4090 | 214.7 ms, shared | no — another backbone for 72% of the run; head training for 100% of it |
| A3 | rep_only + L_align | 3 | vast box b | RTX 5090 | 131.5 ms | yes |

The ratios that survive that test:

| arm | f-bearing term | k = 0 | k = 3 | change | both sides |
|---|---|---|---|---|---|
| B5·s1 | pooled xshh_allt | 117.6 ms | 301.9 ms | +157% | vast box d → vast box a |
| A3 | rep_only + L_align | 115.9 ms | 131.5 ms | +13% | vast box d → vast box b |

No ✗ in this table. The retraction is of B5·s1's depth delta, which rests on a `k = 0` the parents do not recognise; its wall clock is unaffected.


<!-- TABLES:END -->

*Full paired dataset-cluster bootstraps, including the per-domain splits:
[`results/bootstrap.csv`](results/bootstrap.csv). Every table above comes
from `scripts/tables.py`, which writes
[`results/scores.md`](results/scores.md) in the same pass.*

## Protocol

Backbone `d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3,
batch_size=64`, seed 20260520 (B5's second training uses 20260521); dataset
`gift-pretrain-full-4096 / small_v1`; `--ema-embedding --ema-encoder`. Group
B holds EMA α at 0.9; group A raises it linearly from 0.9 to 1.0 by step
100k. Every cell starts fresh at step 0. Two heads per checkpoint, student
and teacher, trained separately on their own encoder, 15,000 steps, head seed
20260722, `--grad-clip 1.0` on the head for comparability with the parents.
97 GIFT-Eval configs, official B4 strategy, forecast horizon 16, one shared
seasonal-naive denominator file. The parent reports are
[`split_pred_rep_small`](../2026-07-21_split_pred_rep_small/small_long.md),
[`lalign_teacher`](../2026-08-04_lalign_teacher/lalign_teacher.md) and
[`ema_sched_ladder`](../2026-08-04_ema_sched_ladder/ema_sched_ladder.md).

**Deviation from the card.** The card's default is to compute the h-anchored
negative families once and reuse them unshifted at every depth. This
implementation takes the card's stated alternative and **shifts them with the
depth**, so a depth-`j` copy is a literal copy of the depth-0 objective under
one rule: every `h` index moves by `j`. It touches exactly one of the 14
cells. B5 (`arm4`, pooled `xshh_allt`) is the only cell whose f-bearing
denominator holds h-anchored families; B9's `L_pred` denominator is
f-anchored only, and the other twelve cells' f-bearing term is `L_align`,
which has no denominator.

**Rebuild.** `bash scripts/make_report_assets.sh` rebuilds every figure and
table here from the committed tree: the eval outputs and trainer logs under
[`results/`](results/), and every backbone's losses CSV under
[`sync/<box>/`](sync/) for the runs pulled off a rented box and
[`curves/elisa/`](curves/) for the runs that trained on elisa. The curves are
downsampled to every step below 1000 and every 20th after it. Every figure
and table reads one run registry ([`scripts/runs.py`](scripts/runs.py)); no
consumer parses a run tag or a checkpoint name by hand.

**Two figures need more than the repository holds.** `rollout_fidelity.png`
and `latent_movement.png` load backbone checkpoints, which are 80 MB each and
stay out of git; their `results/*.csv` are committed, so the numbers are
auditable and only the re-derivation needs the checkpoint store.

## Annex

**`B5·s3` has no teacher-head number.** Its teacher head waited 4 hours for
VRAM on elisa and aborted: other projects held both cards, GPU 1 had
4916 MiB free and the head needs 6000
([`results/stops.log`](results/stops.log),
[`results/eval/G7_B5_k0_e_bb40k_teacher/stop.log`](results/eval/G7_B5_k0_e_bb40k_teacher/stop.log)).
The group-B parent reports publish the student-encoder head only, so the
student number is the comparison the reproduction check needs, and the
encoder-delta figure measures the encoder choice at 0.52 of the head-seed
band.

**The fidelity batch is not held out.** It is the parent report's committed
`_latent_movement_batch.pt`, the same batch the two parent reports'
latent-movement figures use. Nothing here establishes it is disjoint from
`gift-pretrain-full-4096 / small_v1`, which is what these backbones trained
on. It holds every curve on one scale, and that is what it is for.

**The step-time table publishes solo medians.** A median over a run's timing
windows is a cost of the depth only where the run had the card to itself.
`steptime_provenance.py` reads each run's contention off the driver logs and
[`results/steptime_solo.csv`](results/steptime_solo.csv) carries it per run.
Only 2 of the 11 depth-ladder runs pass that test. The one measurement that
holds the card fixed is a controlled probe: B5 alternating `k = 0` and
`k = 3` on elisa's GPU 1, 3 reps of 600 steps, 190.2 ms against 509.9 ms,
+168%. That card carried another session's job throughout (8946 MiB at the
start, 44% mean utilisation), so the probe alternates on a shared card rather
than owning one.

**Operational events** are in
[`results/execution_log.md`](results/execution_log.md).
