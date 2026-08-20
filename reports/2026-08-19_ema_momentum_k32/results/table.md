| arm | EMA momentum | holds at 40k | L_align weight | backbone seed | AUC at the stop | GM-Relative MASE | vs k = 3 at bb40k |
|---|---|---|---|---|---|---|---|
| s08 | 0.8, to 1.0 at 200k | 0.840 | 1 | 20260520 | 0.957 | 1.1782 | +0.0920 |
| s09 | 0.9, to 1.0 at 200k | 0.920 | 1 | 20260520 | 0.972 | 1.1784 | +0.0922 |
| a09 | 0.9, fixed | 0.900 | 1 | 20260520 | 0.979 | 1.1819 | +0.0957 |
| a095 | 0.95, fixed | 0.950 | 1 | 20260520 | 0.982 | 1.1907 | +0.1045 |
| a08 | 0.8, fixed | 0.800 | 1 | 20260520 | 0.927 | 1.2309 | +0.1447 |
| s08b | 0.8, to 1.0 at 200k | 0.840 | 1 | 20260521 | 0.575 (collapsed) | 1.5459 | +0.4597 |

| reference | GM-Relative MASE |
|---|---|
| k = 3, bb200k, the best score of the project | 1.0660 |
| k = 3, bb40k | 1.0862 |
| k = 32, mean, student, bb200k | 1.1637 |
| k = 32, mean, student, bb40k | 1.2082 |
| the k = 0 parent of this cell, bb40k | 1.1600 |

`s08` wins, at 1.1782. Its momentum starts at 0.8 (to 1.0 at 200k) and holds 0.840 at 40,000 steps. It sits +0.0920 from the k = 3 score at bb40k, 1.0862, so it does not go below that score.
