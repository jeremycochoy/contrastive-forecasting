### Coverage

The card names 14 cells. This study scored **14 of them**: A1, A2, A3, A4, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10. Every cell carries a number.

| cell | f-bearing term | EMA α | depths trained | stops scored |
|---|---|---|---|---|
| A1 | L_align only | scheduled | k = 3 | bb40k, bb100k |
| A2 | L_align + CPC auxiliary | scheduled | k = 3 | bb40k, bb100k, bb200k |
| A3 | L_align only | scheduled | k = 0, k = 1, k = 3 | bb40k, bb100k, bb200k |
| A4 | L_align only | scheduled | k = 3 | bb40k, bb100k, bb200k |
| B1 | L_align only | fixed 0.9 | k = 0, k = 3 | bb40k, bb100k, bb200k |
| B2 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |
| B3 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k |
| B4 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |
| B5 | pooled xshh_allt, floor subtracted | fixed 0.9 | k = 0, k = 3 | bb40k, bb100k |
| B6 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |
| B7 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k |
| B8 | L_align + CPC auxiliary | fixed 0.9 | k = 3 | bb40k, bb100k |
| B9 | split L_pred + CPC auxiliary | fixed 0.9 | k = 0, k = 3 | bb40k, bb100k |
| B10 | L_align + CPC auxiliary | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |

Stops scored: bb40k, bb100k, bb200k. The card's extend rule reads a cell's bb40k number against its bb100k number, so it fires only where both are in hand.

### This study's k = 3 against the published k = 0

GM-Relative MASE over the same 97 GIFT-Eval configs, strategy B4, horizon 16. Δ is this study minus the published number, so negative is a gain. A verdict reads Δ against the ±0.0384 head-seed band: closer than that is `flat`.

A dash is a number no parent published. Group B's two parents print one head per row, the student, so group B has no published teacher to meet.

At bb100k, the stop every one of the 14 cells reached. The count is over distinct MODELS. ‡ marks the one student two cells share, so 14 cells hold 13 student models and the shared one counts once. Student head: 13 distinct models, **8 better, 3 flat, 2 worse**. Teacher head, group A only: 4 distinct models, **3 better, 0 flat, 1 worse**.

Read the verdict column as a screen and not as a test. It compares against a baseline this study did not retrain on its own machine, and the ±0.0384 band it thresholds on bounds the HEAD seed alone. The card's own criterion is the per-horizon one, and the depth-response table below is where it is applied.

The second line of a verdict cell is its 95% paired dataset-cluster interval, on 23 of the deltas. Two of the three parents committed their per-config CSVs, so the pairing against them is recoverable: same 97 configs, same seasonal-naive denominator, same resampling unit as every other interval here. `published_bootstrap.py` takes a parent CSV only after it reproduces that parent's own printed aggregate, and all 23 did. Group A's parent committed no per-config CSV, so its rows carry no interval. The interval bounds the eval sample. It does not bound the machine, which separates the two sides of every one of these deltas.

| cell | head | 40k k=3 | 40k pub | Δ | | 100k k=3 | 100k pub | Δ | | 200k k=3 | 200k pub | Δ | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | student | 1.1305 | 1.2596 | -0.1291 | better | 1.1676 | 1.2102 | -0.0426 | better | — | 1.1910 | — | — |
| A1 | teacher | 1.1318 | 1.2347 | -0.1029 | better | 1.1565 | 1.2407 | -0.0842 | better | — | — | — | — |
| A2 | student | 1.2735 | 1.4238 | -0.1503 | better | 1.2479 | 1.3913 | -0.1434 | better | 1.2507 | 1.3586 | -0.1079 | better |
| A2 | teacher | 1.2753 | 1.4177 | -0.1424 | better | 1.2514 | 1.3746 | -0.1232 | better | 1.2500 | 1.3459 | -0.0959 | better |
| A3 | student | 1.3618 | 1.1895 | +0.1723 | worse | 1.3010 | 1.1921 | +0.1089 | worse | 1.3998 | — | — | — |
| A3 | teacher | 1.3521 | 1.1793 | +0.1728 | worse | 1.3151 | 1.1963 | +0.1188 | worse | 1.2913 | — | — | — |
| A4 | student | 1.0862 | 1.1603 | -0.0741 | better | 1.0801 | 1.1945 | -0.1144 | better | 1.0660 | — | — | — |
| A4 | teacher | 1.0855 | 1.1544 | -0.0689 | better | 1.0874 | 1.1837 | -0.0963 | better | 1.0828 | — | — | — |
| B1 | student | 1.0850 | 1.2025 | -0.1175 | better<br>[-0.1801, -0.0615] | 1.0881 | 1.1616 | -0.0735 | better<br>[-0.1287, -0.0255] | 1.1009 | 1.1652 | -0.0643 | better<br>[-0.1230, -0.0130] |
| B1 | teacher | 1.0948 | — | — | — | 1.0897 | — | — | — | 1.1001 | — | — | — |
| B2 | student | 1.3976 | 1.2765 | +0.1211 | worse<br>[+0.0690, +0.1889] | 1.3443 | 1.2514 | +0.0929 | worse<br>[+0.0541, +0.1415] | 1.2904 | 1.1850 | +0.1054 | worse<br>[+0.0609, +0.1621] |
| B2 | teacher | 1.4041 | — | — | — | 1.3117 | — | — | — | 1.2825 | — | — | — |
| B3 | student ‡ | 1.1305 | 1.2868 | -0.1563 | better<br>[-0.2263, -0.0966] | 1.1676 | 1.2456 | -0.0780 | better<br>[-0.1265, -0.0365] | — | 1.2034 | — | — |
| B3 | teacher | 1.1343 | — | — | — | 1.1618 | — | — | — | — | — | — | — |
| B4 | student | 1.3334 | 1.2728 | +0.0606 | worse<br>[+0.0166, +0.1147] | 1.2804 | 1.3678 | -0.0874 | better<br>[-0.1607, -0.0155] | 1.3182 | — | — | — |
| B4 | teacher | 1.3339 | — | — | — | 1.2748 | — | — | — | 1.3202 | — | — | — |
| B5 | student | 1.3204 | 1.2748 | +0.0456 | worse<br>[+0.0145, +0.0846] | 1.3383 | 1.3219 | +0.0164 | flat<br>[-0.0256, +0.0634] | — | — | — | — |
| B5 | teacher | 1.3216 | — | — | — | 1.3428 | — | — | — | — | — | — | — |
| B6 | student | 1.2297 | 1.3623 | -0.1326 | better<br>[-0.1998, -0.0742] | 1.2151 | 1.2978 | -0.0827 | better<br>[-0.1356, -0.0321] | 1.2207 | 1.3011 | -0.0804 | better<br>[-0.1287, -0.0340] |
| B6 | teacher | 1.2184 | — | — | — | 1.2110 | — | — | — | 1.2339 | — | — | — |
| B7 | student | 1.2617 | 1.3159 | -0.0542 | better<br>[-0.1016, -0.0147] | 1.3205 | 1.3012 | +0.0193 | flat<br>[-0.0166, +0.0601] | — | 1.3325 | — | — |
| B7 | teacher | 1.2444 | — | — | — | 1.2780 | — | — | — | — | — | — | — |
| B8 | student | 1.2857 | 1.3074 | -0.0217 | flat<br>[-0.0565, +0.0140] | 1.3157 | 1.3368 | -0.0211 | flat<br>[-0.0674, +0.0292] | — | — | — | — |
| B8 | teacher | 1.2865 | — | — | — | 1.3239 | — | — | — | — | — | — | — |
| B9 | student | 1.2791 | 1.5579 | -0.2788 | better<br>[-0.3543, -0.1978] | 1.3299 | 1.4548 | -0.1249 | better<br>[-0.1982, -0.0383] | — | 1.3308 | — | — |
| B9 | teacher | 1.2728 | — | — | — | 1.3094 | — | — | — | — | — | — | — |
| B10 | student | 1.2669 | 1.3791 | -0.1122 | better<br>[-0.1996, -0.0340] | 1.2403 | 1.3914 | -0.1511 | better<br>[-0.2239, -0.0908] | 1.2624 | — | — | — |
| B10 | teacher | 1.2730 | — | — | — | 1.2499 | — | — | — | 1.2440 | — | — | — |

### The stop ladder: what the second 100,000 steps buys

Δ is bb200k minus bb100k, so a negative number is an improvement: GM-Relative MASE is a ratio against seasonal-naive and lower is better. Of the 16 extended measurements in hand, **7 improved** at bb200k and 9 got worse. The largest gain is B2 student, -0.0539. Over all 16: mean +0.0079, median +0.0042. The ±0.0384 head-seed band covers 13 of them.

The interval is a 95% paired dataset-cluster bootstrap over the pair's 97 configs. It bounds the eval sample, not run-to-run variance. The head-seed band is ±0.0384.

| cell | head | bb40k | bb100k | bb200k | Δ | 95% CI | % | note |
|---|---|---|---|---|---|---|---|---|
| A1 | student | 1.1305 | 1.1676 | — | — | — | — | the extend rule held this cell at 100k |
| A1 | teacher | 1.1318 | 1.1565 | — | — | — | — | the extend rule held this cell at 100k |
| A2 | student | 1.2735 | 1.2479 | 1.2507 | +0.0028 | [-0.0103, +0.0190] | +0.2% |  |
| A2 | teacher | 1.2753 | 1.2514 | 1.2500 | -0.0014 | [-0.0145, +0.0122] | -0.1% |  |
| A3 | student | 1.3618 | 1.3010 | 1.3998 | +0.0988 | [+0.0602, +0.1509] | +7.6% |  |
| A3 | teacher | 1.3521 | 1.3151 | 1.2913 | -0.0238 | [-0.0646, +0.0067] | -1.8% |  |
| A4 | student | 1.0862 | 1.0801 | 1.0660 | -0.0141 | [-0.0265, -0.0024] | -1.3% |  |
| A4 | teacher | 1.0855 | 1.0874 | 1.0828 | -0.0046 | [-0.0199, +0.0123] | -0.4% | extended by hand; the rule's move is inside the band |
| B1 | student | 1.0850 | 1.0881 | 1.1009 | +0.0128 | [+0.0001, +0.0284] | +1.2% | bb40k written by round 1 as `G6_B1_…`; same checkpoint, same head budget |
| B1 | teacher | 1.0948 | 1.0897 | 1.1001 | +0.0104 | [-0.0037, +0.0280] | +1.0% | bb40k written by round 1 as `G6_B1_…`; same checkpoint, same head budget |
| B2 | student | 1.3976 | 1.3443 | 1.2904 | -0.0539 | [-0.0935, -0.0197] | -4.0% |  |
| B2 | teacher | 1.4041 | 1.3117 | 1.2825 | -0.0292 | [-0.0604, -0.0016] | -2.2% |  |
| B3 | student | 1.1305 | 1.1676 | — | — | — | — | the extend rule held this cell at 100k |
| B3 | teacher | 1.1343 | 1.1618 | — | — | — | — | the extend rule held this cell at 100k |
| B4 | student | 1.3334 | 1.2804 | 1.3182 | +0.0378 | [+0.0089, +0.0742] | +3.0% |  |
| B4 | teacher | 1.3339 | 1.2748 | 1.3202 | +0.0454 | [+0.0181, +0.0807] | +3.6% |  |
| B5 | student | 1.3204 | 1.3383 | — | — | — | — | the extend rule held this cell at 100k |
| B5 | teacher | 1.3216 | 1.3428 | — | — | — | — | the extend rule held this cell at 100k |
| B6 | student | 1.2297 | 1.2151 | 1.2207 | +0.0056 | [-0.0101, +0.0212] | +0.5% |  |
| B6 | teacher | 1.2184 | 1.2110 | 1.2339 | +0.0229 | [+0.0032, +0.0440] | +1.9% |  |
| B7 | student | 1.2617 | 1.3205 | — | — | — | — | the extend rule held this cell at 100k |
| B7 | teacher | 1.2444 | 1.2780 | — | — | — | — | the extend rule held this cell at 100k |
| B8 | student | 1.2857 | 1.3157 | — | — | — | — | trained from 0 this round; queued to 100k only |
| B8 | teacher | 1.2865 | 1.3239 | — | — | — | — | trained from 0 this round; queued to 100k only |
| B9 | student | 1.2791 | 1.3299 | — | — | — | — | the extend rule held this cell at 100k |
| B9 | teacher | 1.2728 | 1.3094 | — | — | — | — | the extend rule held this cell at 100k |
| B10 | student | 1.2669 | 1.2403 | 1.2624 | +0.0221 | [+0.0032, +0.0481] | +1.8% |  |
| B10 | teacher | 1.2730 | 1.2499 | 1.2440 | -0.0059 | [-0.0220, +0.0105] | -0.5% |  |

### Stop reasons: what the extend rule read at each cell

The rule reads one cell's bb40k number against its bb100k number, per head. A head that moved down earns the second 100,000 steps; a head that moved up stops. Both columns are bb100k minus bb40k, so negative is an improvement. It held 6 cells at 100k.

| cell | 40k→100k student | 40k→100k teacher | decision | why |
|---|---|---|---|---|
| A1 | +0.0371 | +0.0248 | **stop at 100k** | both heads moved up |
| A2 | -0.0256 | -0.0239 | **extend both heads** | both heads moved down |
| A3 | -0.0608 | -0.0370 | **extend both heads** | both heads moved down |
| A4 | -0.0061 | +0.0019 | **extend both heads** | the student head moved down; the teacher head moved +0.0019, 5% of the ±0.0384 head-seed band, so the rule decides nothing there. Extended by hand, on free hardware |
| B1 | +0.0031 | -0.0051 | **extend both heads** | the card's call: both moves sit inside the ±0.0384 head-seed band, so the rule decides nothing |
| B2 | -0.0533 | -0.0924 | **extend both heads** | both heads moved down |
| B3 | +0.0371 | +0.0276 | **stop at 100k** | both heads moved up |
| B4 | -0.0530 | -0.0591 | **extend both heads** | both heads moved down |
| B5 | +0.0179 | +0.0212 | **stop at 100k** | both heads moved up |
| B6 | -0.0146 | -0.0074 | **extend both heads** | both heads moved down |
| B7 | +0.0587 | +0.0336 | **stop at 100k** | both heads moved up |
| B8 | +0.0300 | +0.0374 | **stop at 100k** | both heads moved up |
| B9 | +0.0508 | +0.0365 | **stop at 100k** | both heads moved up |
| B10 | -0.0266 | -0.0231 | **extend both heads** | both heads moved down |

**The rule selects the panel, and it selects it on an improving first leg.** Three properties of that, stated plainly:

1. It reads the one contrast this study calls not head-matched. A bb40k head trains 15,000 steps and a bb100k head 30,000, so part of every move in the two columns above is the head's own extra 15,000 steps. The Protocol section says so for the depth verdict. It is equally true of the rule.
2. It fires inside its own noise band. 4 of the 6 stopped cells (A1, B3, B5, B8) moved less than ±0.0384 on BOTH heads. The verdict table above calls a move of that size `flat`.
3. The manual overrides go one way. A4 and B1 were extended by hand because the rule decides nothing inside the band. That reasoning applies with the same force to the cells in point 2, and none of them was extended.

So the 8 extended cells are enriched for cells that happened to improve from bb40k to bb100k, and regression to the mean is the expected null at bb200k. This study runs no control for it. **Read the 200k verdict as conditional on a panel selected for having improved.**

### The four same-arm pairs: two models, or one

Each pair runs ONE arm under the two EMA regimes, group A's schedule against group B's fixed 0.9. Every tensor of both backbones is compared, split into the student side the student head reads and the `teacher_*` side the teacher head reads.

Each entry is the count of tensors that agree exactly, out of the count compared. A head's file md5 differs between two cells even when every weight agrees, so the comparison is tensor by tensor and never by md5.

| pair | arm | stop | student | teacher | student head | teacher head |
|---|---|---|---|---|---|---|
| A1/B3 | `arm5_combab_alignS` | bb40k | 110/110 | 0/52 | 28/28 | 0/28 |
| A1/B3 | `arm5_combab_alignS` | bb100k | 110/110 | 0/52 | 28/28 | 0/28 |
| A4/B1 | `arm6_v2_combab_alignS` | bb40k | 4/110 | 0/52 | — | — |
| A4/B1 | `arm6_v2_combab_alignS` | bb100k | 4/110 | 0/52 | 0/28 | 0/28 |
| A4/B1 | `arm6_v2_combab_alignS` | bb200k | 4/110 | 0/52 | 0/28 | 0/28 |
| A3/B2 | `arm6_v2_combab_alignT` | bb40k | 4/110 | 0/52 | — | — |
| A3/B2 | `arm6_v2_combab_alignT` | bb100k | 4/110 | 0/52 | 0/28 | 0/28 |
| A3/B2 | `arm6_v2_combab_alignT` | bb200k | 4/110 | 0/52 | 0/28 | 0/28 |
| A2/B8 | `arm6_v2_nse_alignT` | bb40k | 4/111 | 0/52 | 0/28 | 0/28 |
| A2/B8 | `arm6_v2_nse_alignT` | bb100k | 4/111 | 0/52 | 0/28 | 0/28 |

Full table, with the largest absolute difference on each side: [`results/pair_identity.tsv`](results/pair_identity.tsv).

**A1/B3 hold one student, not two.** `arm5_combab` aligns to the student and carries no `--moco-rep-keys`, so no loss term reads the EMA encoder and the regime sends no gradient into the student. One student number for both cells is the right answer, and it is ONE measurement: the student row of one of them is not a replication of the other. The teacher side differs at every stop, and the teacher numbers do too.

**A2/B8, A3/B2, A4/B1 hold two students.** Their arms carry `--moco-rep-keys`, whose keys come from the EMA encoder, or align to the teacher. Either path reaches the student's gradient, so the regime moves it.

### The A1/B3 duplicate, re-run end to end

Each row trains a fresh student head from the checkpoint its own cell names, seed 20260722, and runs the 97 configs into `results/eval/<cell>rep_…`, a directory no other cell writes. A path that ignored the cell would land the re-run on the other cell's number.

| cell | stop | backbone md5 | first pass | re-run | Δ |
|---|---|---|---|---|---|
| A1 | bb40k | `f99fa42c` | 1.1305 | 1.1447 | +0.0142 |
| A1 | bb100k | `dbd23cbe` | 1.1676 | 1.1610 | -0.0066 |
| B3 | bb40k | `b3a51f06` | 1.1305 | 1.1447 | +0.0142 |
| B3 | bb100k | `0efbb813` | 1.1676 | 1.1610 | -0.0066 |

The largest re-run move is 0.0142. The two cells carry different backbone md5s and reproduce their own first-pass numbers, so the head and the eval read the file each cell names. The duplicate is the student weights, not the path.

### Reproduction of the published k = 0

Same cell, same recipe, same head seed 20260722, same 97-config B4 eval, student head. Rows are grouped by machine.

A row at the parents' own backbone seed 20260520 takes the card's 0.0002; a row at any other seed takes the seed band.

| backbone | seed | machine | published k = 0 | retrained k = 0 | \|Δ\| | gate | verdict |
|---|---|---|---|---|---|---|---|
| B1 | 20260520 | elisa | 1.2025 | 1.2025 | 0.0000 | 0.0002, the card | PASS |
| B5·s3 | 20260520 | elisa | 1.2748 | 1.2751 | 0.0003 | 0.0002, the card | at the re-run floor |
| B9 | 20260520 | elisa | 1.5579 | 1.5583 | 0.0004 | 0.0002, the card | at the re-run floor |
| B5·s2 | 20260521 | elisa | 1.2748 | 1.2716 | 0.0032 | 0.0230, the seed band | inside the seed band |
| A3 | 20260520 | vast box d | 1.1895 | 1.2189 | 0.0294 | 0.0002, the card | FAIL |
| B5·s1 ✗ | 20260520 | vast box d | 1.2748 | 1.3917 | 0.1169 | 0.0002, the card | FAIL |
| B5·pub | 20260520 | the parent report's box | 1.2748 | 1.2751 | 0.0003 | 0.0002, the card | at the re-run floor |

Two things this comparison cannot resolve, added: 0.0003 for the head and the eval, which is what `B5·pub` moves the score by while training nothing, and 0.0001 for the parents' four printed decimals. A |Δ| at or below 0.0004 is a run this pipeline cannot separate from the published one. The card's gate of 0.0002 is stricter than that.

The seed band is 0.0230, the far end of the 95% interval on this study's one measurement of a seed change: `B5·s2` against `B5·s3`, one machine, one recipe, +0.0035 [-0.0183, +0.0230]. It is one run pair, and the interval is over that pair's eval sample rather than over seeds, so the band is a floor on what a seed can move and not a bound on it. B5·s2 is the only row it gates; every other row here carries the parents' own seed.

`B5·pub` is not a training: it takes the parent report's own published B5 checkpoint and puts this study's head and eval on it, so its row bounds the head and the eval rather than the trainer. `B5·s3` is a training, at the protocol seed, on elisa, and its 97-config eval output is byte-identical to `B5·pub`'s (`results/eval/G7_B5_k0_e_bb40k_student/all_results.csv` against `results/eval/G1_B5pub_bb40k_student/all_results.csv`): the elisa retrain reproduced the parent's backbone exactly, and the 0.0003 both rows carry is the head and the eval.

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

**This table is the only place the card's criterion is applied, and it is answered for 2 machine-held arms (B1, B5·s2) at one stop, bb40k.** The 14-cell verdict above answers a different question with a different rule.

`machine held` = did the two sides train on the same box. A `no` row carries a machine change as well as a depth change. The B5 table below measures the machine alone, at one seed, at 0.1166, so a `no` row carries a term larger than most of the deltas in this table. Only the `yes` rows report the depth and nothing else.

✗ marks a retracted row: B5·s1's `k = 0` trained on a rented box and misses its published value by 0.1169; `B5·s3` retrains it at the same seed on elisa and lands 0.0003 away, so the baseline the -5.1% rests on is a rented-box artefact and the delta is retracted.

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone. It does not bound the machine, and it does not bound the BACKBONE seed: this study holds one backbone seed in 14 cells and one replicate of it (B5·s2 against B5·s3, at k = 0, at bb40k), so backbone-seed variance is unmeasured. Every better / flat / worse verdict in this report rests on a band that bounds one of the two seeds in play.

The depths trained are k = 1, k = 3, and only k = 3 ran on the 14 cells. The one ladder that holds more than a single depth is A3's, the cell where k = 3 does the most damage, and its k = 1 row is machine-crossed and covers zero. So this study supports **depth 3 moves the score**. It does NOT support *depth 3 is the right depth*: no cell measures a second depth against a machine-held k = 0.

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

Student head, 97 configs. `B5·s3` holds `B5·s1`'s seed and `B5·s2`'s machine.

Every interval here is a paired dataset-cluster bootstrap over the 97 eval configs of ONE run pair. It bounds the eval sample: how far the difference between these two runs could move if the datasets had been drawn again. It does not bound run-to-run variance, and neither contrast has a replicate to bound it with. No two of B5's three backbones share both a seed and a machine.

`mixup` counts the examples the mixer touched in the 200-step window, so one count at every step is one data order. `B5·s1` and `B5·s3` carry one seed, print one count at every step, and their losses still part.

| step | B5·s1<br>seed 20260520, a rented box | B5·s2<br>seed 20260521, elisa | B5·s3<br>seed 20260520, elisa |
|---|---|---|---|
| 200 | 5.5767  `61/200` | 5.6595  `53/200` | 5.5610  `61/200` |
| 400 | 5.1220  `58/200` | 5.3568  `62/200` | 5.2078  `58/200` |
| 600 | 4.9019  `65/200` | 5.0143  `51/200` | 4.9412  `65/200` |
| 800 | 4.9475  `65/200` | 5.1256  `65/200` | 5.1249  `65/200` |

### A3: is the damage the depth, or the weight?

Summing the depths multiplies `L_align`'s weight against the f-free terms by k + 1. The `L_align x4` row applies that re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 |
|---|---|---|---|---|
| student | 1.2189 | 1.2590<br>+0.0401 [+0.0116, +0.0767] | 1.1995<br>-0.0195 [-0.0537, +0.0148] | 1.3618<br>+0.1429 [+0.0893, +0.2122] |
| teacher | 1.2184 | 1.2558<br>+0.0374 [+0.0058, +0.0756] | 1.2063<br>-0.0121 [-0.0479, +0.0261] | 1.3521<br>+0.1337 [+0.0839, +0.2004] |

Second line of each cell: the difference against `k = 0` and its 95% paired dataset-cluster interval.

Every column trained on a different box from at least one other. A3_k0: vast box d · G3_A3_k0_aw4: elisa · G3_A3_k1: elisa · A3_k3: vast box b. The machine alone is worth 0.1166 on this study's one controlled measurement of it, which is more than either control's own size, so read the two controls as direction and not as magnitude. This table therefore does not divide one column by another.

### What the depth costs

Median `fwd + bwd` per step, from each run's own trainer log. A median is a cost of the depth only where the run had the card to itself, so the table says which did. `run_provenance.py` reads that off the driver logs and [`results/steptime_solo.csv`](results/steptime_solo.csv) carries it per run. A3's `k = 3` shared vast box b with a clone of itself up to step 14,800, and its 131.5 ms is the median over the 127 windows after that.

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

The ratios both of whose sides are solo:

| arm | f-bearing term | k = 0 | k = 3 | change | both sides | read as |
|---|---|---|---|---|---|---|
| B5·s1 | pooled xshh_allt | 117.6 ms | 301.9 ms | +157% | vast box d → vast box a | the depth, plus the box |
| A3 | rep_only + L_align | 115.9 ms | 131.5 ms | +13% | vast box d → vast box b | **not comparable** — its `k = 3` median covers 127 of 273 windows |

Two probes of the same quantity agree and one does not. B5·s1 reads +157% with both sides solo throughout, and the controlled alternating probe on one elisa card reads +168% (190.2 ms against 509.9 ms, 3 reps of 600 steps, [`results/steptime_B5_solo_card.csv`](results/steptime_B5_solo_card.csv)). A3 reads +13%, an order of magnitude below both, off a median over the tail of its run and across a box change. This study does not know why. **Carry +157% to +168%, the two probes that agree, and do not carry the low row.** No cell of the 14 has a same-card k = 0 / k = 3 pair, which is what would settle it.

### The depth-0 forecast error, deeper run minus its own k = 0

`1 - cos(f_t, h_{t+1})` during training: the same quantity on both runs, unlike the loss. Negative means the deeper run forecasts one step ahead better. Four end-of-run windows, because a gap that changes sign between them is not a result.

| arm | k | last 50% | last 25% | last 10% | final step | one sign over all four |
|---|---|---|---|---|---|---|
| B9 | 3 | -0.0707 | -0.0893 | -0.0938 | -0.0817 | yes |
| B1 | 3 | -0.0968 | -0.0915 | -0.1122 | -0.0807 | yes |
| B5·s1 ✗ | 3 | +0.0157 | +0.0102 | +0.0121 | +0.0150 | yes |
| B5·s2 | 3 | +0.0121 | +0.0061 | +0.0023 | -0.0129 | **no** |
| A3 | 1 | +0.0871 | +0.0902 | +0.1159 | +0.0401 | yes |
| A3 | 3 | -0.0469 | -0.0004 | +0.0489 | +0.0623 | **no** |

### Glossary

| term | what it means here |
|---|---|
| the card | the issue this study answers, and the 14 cells, stops and criteria it names |
| cell | one of those 14 recipes, `A1`..`A4` and `B1`..`B10` |
| arm | a (cell, backbone seed, machine) triple. B5 trained three, so the cell is not the unit a delta lives in |
| bb40k | backbone step 40,000, the one stop every run here reached |
| GM-Relative MASE | geometric mean over the 97 GIFT-Eval configs of each config's MASE divided by the seasonal-naive MASE. Lower is better; 1.0 is seasonal-naive parity |
| B4 eval strategy | GIFT-Eval's official evaluation strategy, the one the parent reports use |
| student / teacher head | the quantile head is trained twice per backbone, once on the student encoder and once on its EMA copy, the teacher. The two are separate measurements of one backbone |
| f-bearing term | the loss term that the forecast operator `f` enters. `--train-rollout-depth K` duplicates it at depth 1..K |
| `rep_only` | the representation loss with no forecast term |
| `L_align` | the term that aligns `f`'s output with the future latent |
| `L_pred` | the predictive contrastive term, split from the representation term |
| `xshh_allt` | negatives pooled across the batch and across channels, taken over every time index |
| `u_batchtime` | dimension usage of a latent over the pooled (batch × time) sample axis: `1 / (H · mean off-diagonal squared cosine)`, capped at 1. 1.0 is all `H` dimensions in use and a value near `1/H` is one direction. `h_t` is the encoder latent, `e_t` the embedding it reads |
| collapse | the latent falling onto few directions, so `u_batchtime` runs toward zero. The card watches for it because a model can win the deeper f-bearing terms by flattening `f` |
| `arm4`, `arm6_v2 combab` | the launcher recipes the cells run; the Coverage table gives each cell's |
| head-seed band ±0.0384 | how far the head seed alone moved a score in `ema_sched_ladder.md`, pooled. It bounds the head seed and nothing else |
| `mixup` | the count of examples the batch mixer touched in a 200-step window. Two runs on one data order print one count |

