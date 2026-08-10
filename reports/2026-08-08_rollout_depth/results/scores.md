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

Same cell, same recipe, same head seed 20260722, same 97-config B4 eval, student head. The rows are sorted by the machine, because that is what the check separates on.

| backbone | seed | machine | published k = 0 | retrained k = 0 | \|Δ\| | verdict (threshold 0.0002) |
|---|---|---|---|---|---|---|
| B1 | 20260520 | elisa | 1.2025 | 1.2025 | 0.0000 | PASS |
| B9 | 20260520 | elisa | 1.5579 | 1.5583 | 0.0004 | at printed precision |
| B5·s2 | 20260521 | elisa | 1.2748 | 1.2716 | 0.0032 | FAIL |
| A3 | 20260520 | vast box d | 1.1895 | 1.2189 | 0.0294 | FAIL |
| B5·s1 ✗ | 20260520 | vast box d | 1.2748 | 1.3917 | 0.1169 | FAIL |
| B5·pub | 20260520 | #379's box | 1.2748 | 1.2751 | 0.0003 | at printed precision |

The parents print four decimals, so a difference below 0.0005 is the smallest the published table can resolve. The card's gate of 0.0002 is stricter than that.

`B5·pub` is not a training: it takes #379's own published B5 checkpoint and puts this study's head and eval on it, so its row bounds the head and the eval rather than the trainer. `B5·s3` is a training, at the protocol seed, on elisa.

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

`machine held` = did the two sides train on the same box. A `no` row carries a machine change as well as a depth change, and the reproduction table separates on the machine at up to 0.1169.

✗ marks a retracted row: B5·s1's `k = 0` misses its published value by 0.1169 and trained on a rented box; its depth delta is retracted.

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone. It does not bound a retraining, which the B5 table below measures at 0.1200.

### Paired dataset-cluster bootstrap, per horizon subset

The resampling unit is the dataset: `<ds>/short`, `/medium` and `/long` are three configs of one series and are not independent draws. 95% percentile interval over 10,000 resamples.

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

B5 (`arm4_combab_fix09`) trained three times on one recipe, one code snapshot, one head seed and one eval. They differ by backbone seed and by machine, and each contrast below names which of the two it changes.

| backbone | seed | machine | k = 0 | k = 3 | k = 3 − k = 0 |
|---|---|---|---|---|---|
| B5·s1 ✗ | 20260520 | a rented box | 1.3917 | 1.3204 | -0.0713 |
| B5·s2 | 20260521 | elisa | 1.2716 | 1.3292 | +0.0575 |

| contrast | what changes | k | Δ | 95% CI |
|---|---|---|---|---|
| B5·s1 against B5·s2 | the seed AND the machine | 0 | -0.1200 | [-0.1825, -0.0742] |
| B5·s1 against B5·s2 | the seed AND the machine | 3 | +0.0088 | [-0.0306, +0.0520] |

Student head, 97 configs. `B5·s3` is this study's answer to the third row: it holds `B5·s1`'s seed and `B5·s2`'s machine.

The seed pins the data order, so a retrain at a fixed seed is a machine test and not a second draw of the batches. `mixup=n/200` counts the examples the mixer touched in the window, and two runs that see the same batches print the same count.

| backbone | seed | machine | step | loss | mixup |
|---|---|---|---|---|---|
| B5·s1 | 20260520 | vast box d | 200 | 5.5767 | `61/200` |
| B5·s1 | 20260520 | vast box d | 400 | 5.1220 | `58/200` |
| B5·s2 | 20260521 | elisa | 200 | 5.6595 | `53/200` |
| B5·s2 | 20260521 | elisa | 400 | 5.3568 | `62/200` |

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

Every column trained on a different box from at least one other. A3_k0: vast box d · G3_A3_k0_aw4: elisa · G3_A3_k1: elisa · A3_k3: vast box b.

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

