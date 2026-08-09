### GM-Relative MASE at bb40k, 97 configs

| cell | head | published k = 0 | this study k = 0 | k = 3 | k = 3 − k = 0 |
|---|---|---|---|---|---|
| A3 | student | 1.1895 | 1.2189 | 1.3618 | +0.1429 |
| A3 | teacher | 1.1793 | — | 1.3521 | +0.1728 |
| B5 | student | 1.2748 | 1.3917 | — | — |
| B5 | teacher | — | 1.3719 | — | — |

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone; the backbone-training spread is not measured.

### Baseline validity gate

| group | cell | head | published | retrained k = 0 | |Δ| | verdict (threshold 0.0002) |
|---|---|---|---|---|---|---|
| A | A3 | student | 1.1895 | 1.2189 | 0.0294 | FAIL |
| B | B5 | student | 1.2748 | 1.3917 | 0.1169 | FAIL |

### Horizon split

| cell | head | short k=0 | short k=3 | short Δ% | med+long k=0 | med+long k=3 | med+long Δ% | criterion |
|---|---|---|---|---|---|---|---|---|
| A3 | student | 1.1128 | 1.3027 | +17.1% | 1.3734 | 1.4432 | +5.1% | not met |

Criterion, from the card: medium+long (42 configs) at least 5% better, short (55 configs) losing less than 2%.

