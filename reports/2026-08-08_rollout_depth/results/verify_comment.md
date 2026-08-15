«Agent ExperimentRunner claude-opus-5 writing»

## Both gap runs are checked against the branch, and the round is closed

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(`results/`, `plots/`, `scripts/`, `sync/`)

This session trained nothing, evaluated nothing and rented nothing. It read
the close of the previous session and checked it. The full close, with the
ten items and both new runs, is the comment above.

### Runs completed, and what each one reads

```
item 3  B1 k = 0                1.2025 student   1.2001 teacher
item 3  B1 k = 0, L_align x4    1.1513 student   1.1482 teacher
item 3  B1 k = 3                1.0850 student   1.0948 teacher
item 6  A3 bb200k student, draw 1 seed 20260722   1.3998
item 6  A3 bb200k student, draw 2 seed 20260723   1.4098
item 6  A3 bb200k teacher,      seed 20260722     1.2913
```

**Item 3 splits the win about half and half.** The re-weighting carries 44%
of the student's -0.1175 and the extra horizons carry the rest. Neither
alone accounts for it. Both segments exclude zero on both heads.

**Item 6 reproduces the outlier.** The redraw reads 1.4098 against 1.3998,
0.0100 apart, 26% of the ±0.0384 head-seed band, and it moves away from the
teacher's 1.2913. So "one bad draw" is not the explanation, and the two
lines the review put at risk both stand.

### What the check read

| what | how it reads |
|---|---|
| the control's flags | `K=0 SEED=20260520 GAP_ARGS='--align-loss-weight 4.0' TARGET_STEPS=40000` in `results/gap3_preflight.txt`. The control moves the weight and leaves the depth at 0 |
| the head budget | all six heads of the three B1 columns log `steps=15000 seed=20260722`, so one column may be divided by another |
| the evals | 98 lines in each `all_results.csv`, so 97 configs. 99 score files tracked on the branch |
| the intervals | each one in the item-3 table has its own row in `results/bootstrap.csv` |
| the posted comment | it differs from the committed `results/gap_close_comment.md` by one trailing blank line |

### Spend

`vastrun-status` returns "No running instances found." No `#373` process
holds a card, and the queue is empty, so no card sits idle on unfinished
work. Credit $11.45 against the $5.50 floor, unchanged. This session spent
$0.00.
