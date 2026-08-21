| arm | EMA momentum | holds at 40k | L_align weight | backbone seed | AUC at the stop | GM-Relative MASE | seed range | vs k = 3 at bb40k |
|---|---|---|---|---|---|---|---|---|
| r100_09b | 0.9, to 1.0 at 100k | 0.940 | 1 | 20260524 | 0.974 | 1.1491 | 0.0016 | +0.0629 |
| r100_09 | 0.9, to 1.0 at 100k | 0.940 | 1 | 20260520 | 0.978 | 1.1507 | 0.0016 | +0.0645 |
| s08 | 0.8, to 1.0 at 200k | 0.840 | 1 | 20260520 | 0.957 | 1.1782 | 0.1432 | +0.0920 |
| s09 | 0.9, to 1.0 at 200k | 0.920 | 1 | 20260520 | 0.972 | 1.1784 | one seed | +0.0922 |
| a09 | 0.9, fixed | 0.900 | 1 | 20260520 | 0.979 | 1.1819 | one seed | +0.0957 |
| r60_09 | 0.9, to 1.0 at 60k | 0.967 | 1 | 20260520 | 0.976 | 1.1873 | one seed | +0.1011 |
| a095 | 0.95, fixed | 0.950 | 1 | 20260520 | 0.982 | 1.1907 | one seed | +0.1045 |
| w3_s08 | 0.8, to 1.0 at 200k | 0.840 | 3 | 20260520 | 0.936 | 1.2060 | one seed | +0.1198 |
| r100_095 | 0.95, to 1.0 at 100k | 0.970 | 1 | 20260520 | 0.978 | 1.2130 | one seed | +0.1268 |
| r100_08 | 0.8, to 1.0 at 100k | 0.880 | 1 | 20260520 | 0.954 | 1.2235 | one seed | +0.1373 |
| a08 | 0.8, fixed | 0.800 | 1 | 20260520 | 0.927 | 1.2309 | one seed | +0.1447 |
| s08d | 0.8, to 1.0 at 200k | 0.840 | 1 | 20260523 | 0.975 | 1.2893 | 0.1432 | +0.2031 |
| s08c | 0.8, to 1.0 at 200k | 0.840 | 1 | 20260522 | 0.978 | 1.3214 | 0.1432 | +0.2352 |
| s08b | 0.8, to 1.0 at 200k | 0.840 | 1 | 20260521 | 0.575 (collapsed) | 1.5459 | not counted | +0.4597 |

| reference | GM-Relative MASE |
|---|---|
| k = 3, bb200k, the best score of the project | 1.0660 |
| k = 3, bb40k | 1.0862 |
| k = 32, mean, student, bb200k | 1.1637 |
| k = 32, mean, student, bb40k | 1.2082 |
| the same backbone with no rollout (k = 0), at 40,000 steps | 1.1600 |

`r100_09b` wins, at 1.1491. Its momentum starts at 0.9 (to 1.0 at 100k) and holds 0.940 at 40,000 steps. It sits +0.0629 from the k = 3 score at bb40k, 1.0862, so it does not go below that score.

`s08`, `s08c`, `s08d` are one arm at 3 backbone seeds that did not collapse. They span 0.1432 (12.2%), which is the widest repeat this card measures. The best cell holds 2 seeds of its own, 1.1491 to 1.1507, a span of 0.0016. Its worst seed sits 0.0275 from the best seed of every other arm, 1.1782. `s09`, `a09`, `r60_09`, `a095`, `w3_s08`, `r100_095`, `r100_08`, `a08` carry one seed each, and this card does not separate them from each other.

`a09` 1.1819 and `a095` 1.1907 are 0.0088 apart. The repeat spread is 0.1432. The gap is SMALLER than the spread, so this card does NOT separate the two arms.
