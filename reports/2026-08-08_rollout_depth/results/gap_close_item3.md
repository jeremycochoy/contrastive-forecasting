| head | k = 0 | k = 0, `L_align` x4 | k = 3 | the re-weighting<br>k = 0 → x4 | the depth<br>x4 → k = 3 | share |
|---|---|---|---|---|---|---|
| student | 1.2025 | 1.1513 | 1.0850 | -0.0512 | -0.0663 | 44% |
| teacher | 1.2001 | 1.1482 | 1.0948 | -0.0519 | -0.0534 | 49% |

Intervals, 95% paired dataset-cluster over the 97 eval configs:

- student: re-weighting [-0.1001, -0.0023], depth [-0.1070, -0.0331], total [-0.1801, -0.0615]
- teacher: re-weighting [-0.0987, -0.0066], depth [-0.0874, -0.0237], total [-0.1661, -0.0515]

**Both pay.** The re-weighting carries 44% of the student's -0.1175 and the extra horizons carry the rest. Neither alone accounts for the win.

Every column trained on elisa at backbone seed 20260520 on the same head budget: 15,000 head steps at seed 20260722, then 97 GIFT-Eval configs. This is the study's one machine-held, seed-held, head-budget-matched set, so it may divide one column by another. The two cards are both RTX 4090s of the one box.

What it cannot separate: `k = 3` puts its four copies of `L_align` on four horizons and `k = 0` x4 puts all four on t+1. So the depth column is the extra HORIZONS at a held total weight, not depth net of everything else.
