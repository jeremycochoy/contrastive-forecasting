«Agent ExperimentRunner claude-opus-5 writing»

The experiment directory is `reports/2026-08-19_ema_momentum_k32`.

## What round 7 ran

Round 7 trained TWO backbones to 40,000 steps on ONE card of ONE box, then a
head on each, then the two 97-config GIFT-Evals on elisa's CPUs.

| arm | EMA momentum | holds at 40k | L_align weight | GM-Relative MASE |
|---|---|---|---|---|
| `r60_09` | 0.90, to 1.0 at 60k | 0.967 | 1 | 1.1873 |
| `r100_095` | 0.95, to 1.0 at 100k | 0.970 | 1 | 1.2130 |

Both carry backbone seed 20260520 and head seed 20260722. Every other flag is
round 1's. Both backbones stayed healthy: contrastive AUC 0.976 and 0.978 at
the stop, against the 0.93 to 0.98 band of the other stable arms.

The box was one RTX 5090 at $0.3611/h, instance 48255116, datacenter,
reliability at or above 0.99. Two lanes held 2.7 and 2.6 steps each second and
took 11,363 MiB of 32,607 MiB.

## The 11 scored arms

| arm | EMA momentum | holds at 40k | L_align weight | backbone seed | GM-Relative MASE | vs k = 3 at bb40k |
|---|---|---|---|---|---|---|
| r100_09 | 0.90, to 1.0 at 100k | 0.940 | 1 | 20260520 | 1.1507 | +0.0645 |
| s08 | 0.80, to 1.0 at 200k | 0.840 | 1 | 20260520 | 1.1782 | +0.0920 |
| s09 | 0.90, to 1.0 at 200k | 0.920 | 1 | 20260520 | 1.1784 | +0.0922 |
| a09 | 0.90, fixed | 0.900 | 1 | 20260520 | 1.1819 | +0.0957 |
| r60_09 | 0.90, to 1.0 at 60k | 0.967 | 1 | 20260520 | 1.1873 | +0.1011 |
| a095 | 0.95, fixed | 0.950 | 1 | 20260520 | 1.1907 | +0.1045 |
| w3_s08 | 0.80, to 1.0 at 200k | 0.840 | 3 | 20260520 | 1.2060 | +0.1198 |
| r100_095 | 0.95, to 1.0 at 100k | 0.970 | 1 | 20260520 | 1.2130 | +0.1268 |
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

## What the two new arms say

The card asked whether the START value or the momentum AT THE STOP sets the
score. The pair separates the two, and both move it.

**The stop value moves it more.** `r100_09` and `r60_09` share the start value
0.9 and differ in the ramp alone. They hold 0.940 and 0.967 and they score
1.1507 and 1.1873, which is 0.0366 apart.

**The start value moves it too.** `r60_09` and `r100_095` hold 0.967 and
0.970, which is 0.003 apart, and they start at 0.9 and 0.95. They score 0.0257
apart, and the LOWER start wins.

**0.940 is a turn, not an edge.** At start 0.9 the card now holds four points:
0.900 gives 1.1819, 0.920 gives 1.1784, 0.940 gives 1.1507 and 0.967 gives
1.1873. The score falls to 0.940 and rises after it.

So this round does NOT lower the card's best score. It bounds it from the
other side: past 0.940 the score climbs again, and `r100_09` stays the only
arm under the k = 0 parent.

HOW FAR THESE NUMBERS CARRY. This card measures no repeat spread of its own:
one of its two seeds of `s08` collapsed. #373 measured 0.6 % to 1.3 % on this
protocol, which is 0.007 to 0.015 at a score of 1.15. Both gaps above, 0.0366
and 0.0257, are larger than that upper bound.

## The verdict

`r100_09` wins at **1.1507**, EMA momentum 0.90, to 1.0 at 100k. It sits +0.0645 from the k = 3 score at the same 40,000 steps, 1.0862, so it does NOT go below that score.

1 arm(s) go below the k = 0 parent of this cell, 1.1600 at the same 40,000 steps:

- `r100_09`, 1.1507, 0.0093 under it. It holds 0.940 at the stop.

Runs completed this round: 2.
Cost: $2.35 of box time, credit $6.58 to $4.23.
