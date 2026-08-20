| arm | EMA momentum | backbone seed | AUC at the stop | GM-Relative MASE | vs k = 3 at bb40k |
|---|---|---|---|---|---|
| s08 | 0.8, to 1.0 at 200k | 20260520 | 0.957 | 1.1782 | +0.0920 |
| s09 | 0.9, to 1.0 at 200k | 20260520 | 0.972 | 1.1784 | +0.0922 |
| a09 | 0.9, fixed | 20260520 | 0.979 | 1.1819 | +0.0957 |
| a095 | 0.95, fixed | 20260520 | 0.982 | 1.1907 | +0.1045 |
| a08 | 0.8, fixed | 20260520 | 0.927 | 1.2309 | +0.1447 |
| s08b | 0.8, to 1.0 at 200k | 20260521 | 0.575 (collapsed) | 1.5459 | +0.4597 |

| reference | GM-Relative MASE |
|---|---|
| k = 3, bb200k, the best score of the project | 1.0660 |
| k = 3, bb40k | 1.0862 |
| k = 32, mean, student, bb200k | 1.1637 |
| k = 32, mean, student, bb40k | 1.2082 |
| the k = 0 parent of this cell, bb40k | 1.1600 |

The EMA momentum 0.8 (to 1.0 at 200k) wins, at 1.1782. It sits +0.0920 from the k = 3 score at bb40k, 1.0862, so it does not go below that score.
