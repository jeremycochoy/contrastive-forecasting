### GM-Relative MASE at bb40k, 97 configs

| cell | head | published k = 0 | this study k = 0 | k = 3 | k = 3 − k = 0 |
|---|---|---|---|---|---|
| A3 | student | 1.1895 | 1.2189 | 1.3618 | +0.1429 |
| A3 | teacher | 1.1793 | 1.2184 | 1.3521 | +0.1337 |
| B5 | student | 1.2748 | 1.3917 | 1.3204 | -0.0713 |
| B5 | teacher | — | 1.3719 | 1.3216 | -0.0503 |
| B9 | student | 1.5579 | — | 1.2791 | -0.2788 |
| B9 | teacher | — | — | 1.2728 | — |

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone; the backbone-training spread is not measured.

### Baseline validity gate

| group | cell | head | published | retrained k = 0 | |Δ| | verdict (threshold 0.0002) |
|---|---|---|---|---|---|---|
| A | A3 | student | 1.1895 | 1.2189 | 0.0294 | FAIL |
| A | A3 | teacher | 1.1793 | 1.2184 | 0.0391 | FAIL |
| B | B5 | student | 1.2748 | 1.3917 | 0.1169 | FAIL |

### Horizon split

| cell | head | short k=0 | short k=3 | short Δ% | med+long k=0 | med+long k=3 | med+long Δ% | criterion |
|---|---|---|---|---|---|---|---|---|
| A3 | student | 1.1128 | 1.3027 | +17.1% | 1.3734 | 1.4432 | +5.1% | not met |
| A3 | teacher | 1.1107 | 1.2868 | +15.8% | 1.3754 | 1.4427 | +4.9% | not met |
| B5 | student | 1.3154 | 1.2306 | -6.4% | 1.4983 | 1.4479 | -3.4% | not met |
| B5 | teacher | 1.2868 | 1.2297 | -4.4% | 1.4919 | 1.4524 | -2.6% | not met |

Criterion, from the card: medium+long (42 configs) at least 5% better, short (55 configs) losing less than 2%.

