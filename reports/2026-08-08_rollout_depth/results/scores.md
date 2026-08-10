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

Same cell, same recipe, same head seed 20260722, same 97-config B4 eval. The only thing that differs from the parent report is the code snapshot.

| arm | published k = 0 | retrained k = 0 | \|Δ\| | verdict (threshold 0.0002) |
|---|---|---|---|---|
| B9 | 1.5579 | 1.5583 | 0.0004 | at printed precision |
| B1 | 1.2025 | 1.2025 | 0.0000 | PASS |
| B5·s1 | 1.2748 | 1.3917 | 0.1169 | FAIL |
| B5·s2 | 1.2748 | 1.2716 | 0.0032 | FAIL |
| A3 | 1.1895 | 1.2189 | 0.0294 | FAIL |

The parents print four decimals, so a difference below 0.0005 is the smallest the published table can resolve. The card's gate of 0.0002 is stricter than that.


And one control that changes the backbone instead of the code: #379's own published B5 backbone, re-headed and re-scored by this study.

| backbone | head + eval | GM-Relative MASE |
|---|---|---|
| #379's published B5 bb40k | this study | 1.2751 |
| #379's published B5 bb40k | as published | 1.2748 |

### Depth response, against each arm's own k = 0

| arm | EMA α | f-bearing term | head | k | k = 0 | this k | Δ | all | short | med+long | criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B9 | fixed 0.9 | split L_pred | student | 3 | 1.5583 | 1.2791 | -0.2792 | -17.9% | -12.6% | -24.4% | **MET** |
| B9 | fixed 0.9 | split L_pred | teacher | 3 | 1.5599 | 1.2728 | -0.2871 | -18.4% | -12.8% | -25.2% | **MET** |
| B1 | fixed 0.9 | rep_only + L_align | student | 3 | 1.2025 | 1.0850 | -0.1175 | -9.8% | -5.4% | -15.2% | **MET** |
| B1 | fixed 0.9 | rep_only + L_align | teacher | 3 | 1.2001 | 1.0948 | -0.1053 | -8.8% | -5.1% | -13.4% | **MET** |
| B5·s1 | fixed 0.9 | pooled xshh_allt | student | 3 | 1.3917 | 1.3204 | -0.0713 | -5.1% | -6.4% | -3.4% | not met |
| B5·s1 | fixed 0.9 | pooled xshh_allt | teacher | 3 | 1.3719 | 1.3216 | -0.0503 | -3.7% | -4.4% | -2.6% | not met |
| B5·s2 | fixed 0.9 | pooled xshh_allt | student | 3 | 1.2716 | 1.3292 | +0.0576 | +4.5% | +7.0% | +1.4% | not met |
| B5·s2 | fixed 0.9 | pooled xshh_allt | teacher | 3 | 1.2661 | 1.3260 | +0.0599 | +4.7% | +8.1% | +0.5% | not met |
| A3 | scheduled 0.9 -> 1.0 | rep_only + L_align | student | 1 | 1.2189 | 1.1995 | -0.0194 | -1.6% | -2.6% | -0.2% | not met |
| A3 | scheduled 0.9 -> 1.0 | rep_only + L_align | student | 3 | 1.2189 | 1.3618 | +0.1429 | +11.7% | +17.1% | +5.1% | not met |
| A3 | scheduled 0.9 -> 1.0 | rep_only + L_align | teacher | 1 | 1.2184 | 1.2063 | -0.0121 | -1.0% | -1.5% | -0.4% | not met |
| A3 | scheduled 0.9 -> 1.0 | rep_only + L_align | teacher | 3 | 1.2184 | 1.3521 | +0.1337 | +11.0% | +15.8% | +4.9% | not met |

Criterion, from the card: medium+long (42 configs) at least 5% better, short (55 configs) losing less than 2%.

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone. The backbone-seed table below measures the backbone seed, which is larger.

### Two backbone seeds of one cell

B5 (`arm4_combab_fix09`) trained twice. Same code, same recipe, same head seed, same eval; the backbone seed is the only difference.

| head | k | seed 20260520 | seed 20260521 | seed spread |
|---|---|---|---|---|
| student | 0 | 1.3917 | 1.2716 | -0.1201 |
| student | 3 | 1.3204 | 1.3292 | +0.0088 |
| teacher | 0 | 1.3719 | 1.2661 | -0.1058 |
| teacher | 3 | 1.3216 | 1.3260 | +0.0044 |

| head | seed | k = 0 | k = 3 | k = 3 − k = 0 |
|---|---|---|---|---|
| student | 20260520 | 1.3917 | 1.3204 | -0.0713 |
| student | 20260521 | 1.2716 | 1.3292 | +0.0576 |
| teacher | 20260520 | 1.3719 | 1.3216 | -0.0503 |
| teacher | 20260521 | 1.2661 | 1.3260 | +0.0599 |

### One loss shape, two EMA regimes

B1 and A3 train the same f-bearing term, `rep_only` + `L_align`, on the same `arm6_v2 combab` arm. They differ in the EMA schedule.

| arm | EMA α | head | k = 0 | k = 3 | Δ | Δ% |
|---|---|---|---|---|---|---|
| B1 | fixed 0.9 | student | 1.2025 | 1.0850 | -0.1175 | -9.8% |
| B1 | fixed 0.9 | teacher | 1.2001 | 1.0948 | -0.1053 | -8.8% |
| A3 | scheduled 0.9 -> 1.0 | student | 1.2189 | 1.3618 | +0.1429 | +11.7% |
| A3 | scheduled 0.9 -> 1.0 | teacher | 1.2184 | 1.3521 | +0.1337 | +11.0% |

### A3: is the damage the depth, or the weight?

Summing the depths multiplies `L_align`'s weight against the f-free terms by k + 1. The `L_align x4` row applies that re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 | share of the k = 3 damage the re-weighting explains |
|---|---|---|---|---|---|
| student | 1.2189 | 1.2590 | 1.1995 | 1.3618 | 28% |
| teacher | 1.2184 | 1.2558 | 1.2063 | 1.3521 | 28% |

