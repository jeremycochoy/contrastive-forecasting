«Agent ExperimentRunner claude-opus-5 writing»

## Round 3 closed — every deliverable is scored

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
- results `results/`, plots `plots/`, scripts `scripts/`
- run tree `/home/jupyter/cf373_r3`, 96 score files on the branch

### Coverage — 14 cells x 3 stops x 2 heads

| cell | 40k S | 40k T | 100k S | 100k T | 200k S | 200k T |
|---|---|---|---|---|---|---|
| A1 | 1.1305 | 1.1318 | 1.1676 | 1.1565 | stop | stop |
| A2 | 1.2735 | 1.2753 | 1.2479 | 1.2514 | 1.2507 | 1.2500 |
| A3 | 1.3618 | 1.3521 | 1.3010 | 1.3151 | 1.3998 | 1.2913 |
| A4 | 1.0862 | 1.0855 | 1.0801 | 1.0874 | 1.0660 | 1.0828 |
| B1 | 1.0850 | 1.0948 | 1.0881 | 1.0897 | 1.1009 | 1.1001 |
| B2 | 1.3976 | 1.4041 | 1.3443 | 1.3117 | 1.2904 | 1.2825 |
| B3 | 1.1305 | 1.1343 | 1.1676 | 1.1618 | stop | stop |
| B4 | 1.3334 | 1.3339 | 1.2804 | 1.2748 | 1.3182 | 1.3202 |
| B5 | 1.3204 | 1.3216 | 1.3383 | 1.3428 | stop | stop |
| B6 | 1.2297 | 1.2184 | 1.2151 | 1.2110 | 1.2207 | 1.2339 |
| B7 | 1.2617 | 1.2444 | 1.3205 | 1.2780 | stop | stop |
| B8 | 1.2857 | 1.2865 | 1.3157 | 1.3239 | stop | stop |
| B9 | 1.2791 | 1.2728 | 1.3299 | 1.3094 | stop | stop |
| B10 | 1.2669 | 1.2730 | 1.2403 | 1.2499 | 1.2624 | 1.2440 |

deliverables 72   done 72   running 0   queued 0   NOT STARTED 0   (+12 stops, not deliverables)
done=number in hand  run=own head/eval running  bb-run=backbone training now  plan=queued, not started  MISS-e=eval not run  MISS-h=head not trained  MISS-t=backbone not trained  stop=not a deliverable this round

### This study's k = 3 against the published k = 0

GM-Relative MASE over the same 97 GIFT-Eval configs, strategy B4, horizon 16. Δ is this study minus the published number, so negative is a gain. A verdict reads Δ against the ±0.0384 head-seed band: closer than that is `flat`.

A dash is a number no parent published. Group B's two parents print one head per row, the student, so group B has no published teacher to meet.

At bb100k, the stop every one of the 14 cells reached. Student head: 14 cells, **9 better, 3 flat, 2 worse**. Teacher head, group A only: 4 cells, **3 better, 0 flat, 1 worse**.

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
| B1 | student | 1.0850 | 1.2025 | -0.1175 | better | 1.0881 | 1.1616 | -0.0735 | better | 1.1009 | 1.1652 | -0.0643 | better |
| B1 | teacher | 1.0948 | — | — | — | 1.0897 | — | — | — | 1.1001 | — | — | — |
| B2 | student | 1.3976 | 1.2765 | +0.1211 | worse | 1.3443 | 1.2514 | +0.0929 | worse | 1.2904 | 1.1850 | +0.1054 | worse |
| B2 | teacher | 1.4041 | — | — | — | 1.3117 | — | — | — | 1.2825 | — | — | — |
| B3 | student | 1.1305 | 1.2868 | -0.1563 | better | 1.1676 | 1.2456 | -0.0780 | better | — | 1.2034 | — | — |
| B3 | teacher | 1.1343 | — | — | — | 1.1618 | — | — | — | — | — | — | — |
| B4 | student | 1.3334 | 1.2728 | +0.0606 | worse | 1.2804 | 1.3678 | -0.0874 | better | 1.3182 | — | — | — |
| B4 | teacher | 1.3339 | — | — | — | 1.2748 | — | — | — | 1.3202 | — | — | — |
| B5 | student | 1.3204 | 1.2748 | +0.0456 | worse | 1.3383 | 1.3219 | +0.0164 | flat | — | — | — | — |
| B5 | teacher | 1.3216 | — | — | — | 1.3428 | — | — | — | — | — | — | — |
| B6 | student | 1.2297 | 1.3623 | -0.1326 | better | 1.2151 | 1.2978 | -0.0827 | better | 1.2207 | 1.3011 | -0.0804 | better |
| B6 | teacher | 1.2184 | — | — | — | 1.2110 | — | — | — | 1.2339 | — | — | — |
| B7 | student | 1.2617 | 1.3159 | -0.0542 | better | 1.3205 | 1.3012 | +0.0193 | flat | — | 1.3325 | — | — |
| B7 | teacher | 1.2444 | — | — | — | 1.2780 | — | — | — | — | — | — | — |
| B8 | student | 1.2857 | 1.3074 | -0.0217 | flat | 1.3157 | 1.3368 | -0.0211 | flat | — | — | — | — |
| B8 | teacher | 1.2865 | — | — | — | 1.3239 | — | — | — | — | — | — | — |
| B9 | student | 1.2791 | 1.5579 | -0.2788 | better | 1.3299 | 1.4548 | -0.1249 | better | — | 1.3308 | — | — |
| B9 | teacher | 1.2728 | — | — | — | 1.3094 | — | — | — | — | — | — | — |
| B10 | student | 1.2669 | 1.3791 | -0.1122 | better | 1.2403 | 1.3914 | -0.1511 | better | 1.2624 | — | — | — |
| B10 | teacher | 1.2730 | — | — | — | 1.2499 | — | — | — | 1.2440 | — | — | — |


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


### Runs completed

```
backbones  9 legs this round: B8 0 -> 100k, eight cells 100k -> 200k
heads      30,000 steps, seed 20260722, --grad-clip 1.0, batch 256, lr 1e-3
evals      97 GIFT-Eval configs, strategy B4, horizon 16
failed     0
```

### Spend

Credit **$11.45**, floor $5.50. Box `47557391` (cf373-dual) was destroyed at
2026-08-13 11:36Z by `scripts/r3_reap.sh`, once it had verified all 730
checkpoint files on elisa. vast.ai now reports: `No running instances found.`
Every head and every eval after that ran on elisa and cost nothing.
