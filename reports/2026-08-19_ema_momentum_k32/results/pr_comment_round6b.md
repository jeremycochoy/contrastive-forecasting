«Agent ExperimentRunner claude-opus-5 writing»

The experiment directory is `reports/2026-08-19_ema_momentum_k32`.

## What ran this round

Round 6 trained three backbones to 40,000 steps and started a head on each.
Two heads landed. The third, `w3_s08`, was at about 26,000 steps of 30,000
when the round 6 driver destroyed the box on a MISSING checkpoint it had
logged as a warning. The backbone survived.

This round recovered that:

| leg | where | what it cost |
|---|---|---|
| `r100_09` eval, 97 configs | elisa CPUs, 4 shards | no GPU, no box |
| `r100_08` eval, 97 configs | elisa CPUs, 4 shards | no GPU, no box |
| `w3_s08` head, 30,000 steps | one RTX 5090, 31 min | $0.30 |
| `w3_s08` eval, 97 configs | elisa CPUs, 4 shards | no GPU, no box |

The head carries round 1's protocol, flag for flag: `--encoder-source student
--quantile-head --forecast-len 16 --batch-size 256 --lr 1e-3 --total-steps
30000 --seed 20260722 --head-arch transformer`.

`scripts/recover_w3_head.sh` asks ONE question before it destroys a box: is
the head on elisa's disk, by name and above 400,000 bytes? The head reached
elisa at 23:48:30 and the destroy call went out in that same second. A missing final
checkpoint is a stop, not a warning.

## The 9 scored arms

| arm | EMA momentum | holds at 40k | L_align weight | backbone seed | GM-Relative MASE | vs k = 3 at bb40k |
|---|---|---|---|---|---|---|
| r100_09 | 0.90, to 1.0 at 100k | 0.940 | 1 | 20260520 | 1.1507 | +0.0645 |
| s08 | 0.80, to 1.0 at 200k | 0.840 | 1 | 20260520 | 1.1782 | +0.0920 |
| s09 | 0.90, to 1.0 at 200k | 0.920 | 1 | 20260520 | 1.1784 | +0.0922 |
| a09 | 0.90, fixed | 0.900 | 1 | 20260520 | 1.1819 | +0.0957 |
| a095 | 0.95, fixed | 0.950 | 1 | 20260520 | 1.1907 | +0.1045 |
| w3_s08 | 0.80, to 1.0 at 200k | 0.840 | 3 | 20260520 | 1.2060 | +0.1198 |
| r100_08 | 0.80, to 1.0 at 100k | 0.880 | 1 | 20260520 | 1.2235 | +0.1373 |
| a08 | 0.80, fixed | 0.800 | 1 | 20260520 | 1.2309 | +0.1447 |
| s08b | 0.80, to 1.0 at 200k | 0.840 | 1 | 20260521 | 1.5459 | +0.4597 |

`holds at 40k` is the momentum the backbone trains against at the stop. A ramp arm does not hold the value it starts at.

`L_align weight` is `--align-loss-weight`. The rollout depth duplicates the align term and not the repel term, and the reduction is a mean, so this flag sets the balance between one h-anchored repel term and the mean of k + 1 f-anchored pull terms.

## The repeat family, seed by seed

**The s08 arm at 2 backbone seeds.** Alpha 0.8 rising to 1.0 at 200000, k = 32, mean reduction, align target teacher, 40000 backbone steps, 30,000 head steps, head seed 20260722, the 97-config eval.

| arm | backbone seed | AUC at 40,000 | GM-Relative MASE | verdict |
|---|---|---|---|---|
| `s08` | 20260520 | 0.957 | 1.1782 | stable |
| `s08b` | 20260521 | 0.575 | 1.5459 | **collapsed** |

**1 of 2 collapsed**, by the AUC at 40000 steps against a line at 0.8. The stable arms of this card hold 0.93 to 0.98 and the collapsed one holds 0.57, so any line inside that band gives the same count.

Fewer than two seeds survived, so this round measures no spread.

## Backbones that trained but carry no score

| arm | EMA momentum | backbone seed | contrastive AUC at 40k | verdict |
|---|---|---|---|---|
| s08c | 0.80, to 1.0 at 200k | 20260522 | 0.9776 | healthy |
| s08d | 0.80, to 1.0 at 200k | 20260523 | 0.9746 | healthy |

These arms trained no head and ran no eval, so they have no GM-Relative MASE. Their AUC still says whether the backbone lived.

## Does the spread separate 0.90 fixed from 0.95 fixed?

The card cannot answer this yet: it has fewer than two stable seeds of one arm, or one of the two arms has no score.

## The verdict

`r100_09` wins at **1.1507**, EMA momentum 0.90, to 1.0 at 100k. It sits +0.0645 from the k = 3 score at the same 40,000 steps, 1.0862, so it does NOT go below that score.

`r100_09` is the FIRST arm of this card to go below the k = 0 parent of this
cell, 1.1600 at the same 40,000 steps. It sits 0.0093 under it. Every other
arm of the card sits above that line.

The ramp LENGTH moved the score more than the momentum VALUE did. `r100_09`
and `s09` start at the same 0.9 and differ in the ramp alone, 100,000 steps
against 200,000, and they sit 0.0277 apart. The whole fixed-momentum row, 0.8
to 0.95, spans 0.0490.

The L_align weight went the wrong way. `w3_s08` is `s08` with
`--align-loss-weight 3.0` and nothing else moved, same momentum, same ramp,
same seed. It scores 1.2060 against 1.1782, so weight 3.0 costs 0.0278. Its
contrastive AUC also falls, 0.936 against 0.957. One arm at one weight does
not map the axis, but it does not point up.

Figures: `plots/momentum_at_stop.png`, `plots/backbone_health.png`,
`plots/domain_radar.png`, `plots/loss_curves.png`.

Runs completed this round: 3.
Cost: $0.30 of box time for the head this round, credit $6.91 to $6.61.
