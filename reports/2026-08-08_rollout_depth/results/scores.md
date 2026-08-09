### GM-Relative MASE at bb40k, 97 configs

| cell | head | published k = 0 | this study k = 0 | k = 3 | k = 3 − k = 0 |
|---|---|---|---|---|---|
| A3 | student | 1.1895 | — | 1.3618 | +0.1723 |
| A3 | teacher | 1.1793 | — | 1.3521 | +0.1728 |

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone; the backbone-training spread is not measured.

### Baseline validity gate

| group | cell | head | published | retrained k = 0 | |Δ| | verdict (threshold 0.0002) |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | no gate has finished |

### Horizon split

| cell | head | short k=0 | short k=3 | short Δ% | med+long k=0 | med+long k=3 | med+long Δ% | criterion |
|---|---|---|---|---|---|---|---|---|

Criterion, from the card: medium+long (42 configs) at least 5% better, short (55 configs) losing less than 2%.

