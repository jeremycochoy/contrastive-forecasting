### Per cell, per stop, per head

| cell | stop | S k=3 | S k=0 | S Δ | T k=3 | T k=0 | T Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 40k | 1.1305 | 1.2596 | -0.1291 | 1.1318 | 1.2347 | -0.1029 |
| A1 | 100k | 1.1676 | 1.2102 | -0.0426 | 1.1565 | 1.2407 | -0.0842 |
| A2 | 40k | 1.2735 | 1.4238 | -0.1503 | 1.2753 | 1.4177 | -0.1424 |
| A2 | 100k | 1.2479 | 1.3913 | -0.1434 | 1.2514 | 1.3746 | -0.1232 |
| A3 | 40k | 1.3618 | 1.1895 | +0.1723 | 1.3521 | 1.1793 | +0.1728 |
| A3 | 100k | 1.3010 | 1.1921 | +0.1089 | 1.3151 | 1.1963 | +0.1188 |
| A4 | 40k | 1.0862 | 1.1603 | -0.0741 | 1.0855 | 1.1544 | -0.0689 |
| A4 | 100k | 1.0801 | 1.1945 | -0.1144 | 1.0874 | 1.1837 | -0.0963 |
| B1 | 40k | 1.0850 | 1.2025 | -0.1175 | 1.0948 | — | — |
| B1 | 100k | 1.0881 | 1.1616 | -0.0735 | 1.0897 | — | — |
| B2 | 40k | 1.3976 | 1.2765 | +0.1211 | 1.4041 | — | — |
| B2 | 100k | 1.3443 | 1.2514 | +0.0929 | 1.3117 | — | — |
| B3 | 40k | 1.1305 | 1.2868 | -0.1563 | 1.1343 | — | — |
| B3 | 100k | 1.1676 | 1.2456 | -0.0780 | 1.1618 | — | — |
| B4 | 40k | 1.3334 | 1.2728 | +0.0606 | 1.3339 | — | — |
| B4 | 100k | 1.2804 | 1.3678 | -0.0874 | 1.2748 | — | — |
| B5 | 40k | 1.3204 | 1.2748 | +0.0456 | 1.3216 | — | — |
| B5 | 100k | 1.3383 | 1.3219 | +0.0164 | 1.3428 | — | — |
| B6 | 40k | 1.2297 | 1.3623 | -0.1326 | 1.2184 | — | — |
| B6 | 100k | 1.2151 | 1.2978 | -0.0827 | 1.2110 | — | — |
| B7 | 40k | 1.2617 | 1.3159 | -0.0542 | 1.2444 | — | — |
| B7 | 100k | 1.3205 | 1.3012 | +0.0193 | 1.2780 | — | — |
| B9 | 40k | 1.2791 | 1.5579 | -0.2788 | 1.2728 | — | — |
| B9 | 100k | 1.3299 | 1.4548 | -0.1249 | 1.3094 | — | — |
| B10 | 40k | 1.2669 | 1.3791 | -0.1122 | 1.2730 | — | — |
| B10 | 100k | 1.2403 | 1.3914 | -0.1511 | 1.2499 | — | — |

26 of 42 (cell, stop) pairs measured. `S` is the student-encoder head, `T` the teacher-encoder head. A `—` in a `k = 0` column means the parent report published no such number: group B's two parents publish the student head only, so a group-B teacher row carries a value and no delta.

### Stop reasons

| cell | stop | extend | heads kept | reason |
|---|---:|---|---|---|
| A1 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| A1 | 100k | no | — | neither head down (S +0.0371, T +0.0247) — stop |
| A2 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| A2 | 100k | yes | student, teacher | both heads down (S -0.0256, T -0.0239) — extend, keep both |
| A3 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| A3 | 100k | yes | student, teacher | both heads down (S -0.0608, T -0.0370) — extend, keep both |
| A4 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| A4 | 100k | yes | student | student down (S -0.0061, T +0.0019) — extend, keep student |
| B1 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B1 | 100k | yes | teacher | teacher down (S +0.0031, T -0.0051) — extend, keep teacher |
| B2 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B2 | 100k | yes | student, teacher | both heads down (S -0.0533, T -0.0924) — extend, keep both |
| B3 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B3 | 100k | no | — | neither head down (S +0.0371, T +0.0275) — stop |
| B4 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B4 | 100k | yes | student, teacher | both heads down (S -0.0530, T -0.0591) — extend, keep both |
| B5 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B5 | 100k | no | — | neither head down (S +0.0179, T +0.0212) — stop |
| B6 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B6 | 100k | yes | student, teacher | both heads down (S -0.0146, T -0.0074) — extend, keep both |
| B7 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B7 | 100k | no | — | neither head down (S +0.0588, T +0.0336) — stop |
| B9 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B9 | 100k | no | — | neither head down (S +0.0508, T +0.0366) — stop |
| B10 | 40k | yes | student, teacher | 40k and 100k are unconditional |
| B10 | 100k | yes | student, teacher | both heads down (S -0.0266, T -0.0231) — extend, keep both |

The rule is the card's: per head against its own previous stop, both heads down extends and keeps both, one head down extends and keeps that head, neither down stops. 40k and 100k run unconditionally, so 40k decides nothing. Down means lower GM-Relative MASE.

### What each cell is

| cell | arm | `L_align` | EMA | f-bearing term the depth copies |
|---|---|---|---|---|
| A1 | `arm5 combab` | student | scheduled | L_align only |
| A2 | `arm6_v2 nse` | teacher | scheduled | L_align + CPC auxiliary |
| A3 | `arm6_v2 combab` | teacher | scheduled | L_align only |
| A4 | `arm6_v2 combab` | student | scheduled | L_align only |
| B1 | `arm6_v2 combab` | student | fixed 0.9 | L_align only |
| B2 | `arm6_v2 combab` | teacher | fixed 0.9 | L_align only |
| B3 | `arm5 combab` | student | fixed 0.9 | L_align only |
| B4 | `arm5 combab` | teacher | fixed 0.9 | L_align only |
| B5 | `arm4 combab` | none | fixed 0.9 | pooled xshh_allt, floor subtracted |
| B6 | `arm6_v2 ncpc` | student | fixed 0.9 | L_align only |
| B7 | `arm6_v2 ncpc` | teacher | fixed 0.9 | L_align only |
| B8 | `arm6_v2 nse` | teacher | fixed 0.9 | L_align + CPC auxiliary |
| B9 | `arm1 nse` | none | fixed 0.9 | split L_pred + CPC auxiliary |
| B10 | `arm6_v2 nse` | student | fixed 0.9 | L_align + CPC auxiliary |

Rule 2 of the card — `f` in the numerator and in every denominator — is exercised by B5, B9 and the CPC auxiliary of A2, B8 and B10. In the other nine cells the flag touches `L_align`, which has no denominator.
