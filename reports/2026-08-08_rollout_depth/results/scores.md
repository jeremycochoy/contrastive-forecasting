## Did the card's criteria pass?

| cell | med+long, 42 configs | short, 55 configs | PRIMARY | full-97 Δ | SECONDARY |
|---|---|---|---|---|---|
| A1 | -5.9% | -1.7% | **PASS** | -0.0426 | **PASS** |
| A2 | -17.6% | -4.3% | **PASS** | -0.1434 | **PASS** |
| A3 | -3.2% | +19.6% | fail | +0.1089 | fail |
| A4 | -13.5% | -6.5% | **PASS** | -0.1144 | **PASS** |
| B1 | -9.4% | -3.9% | **PASS** | -0.0735 | **PASS** |
| B2 | +1.0% | +12.6% | fail | +0.0929 | fail |
| B3 | -7.9% | -5.0% | **PASS** | -0.0780 | **PASS** |
| B4 | -19.5% | +5.0% | fail | -0.0874 | **PASS** |
| B5 | -3.6% | +5.1% | fail | +0.0164 | fail |
| B6 | -10.7% | -2.9% | **PASS** | -0.0827 | **PASS** |
| B7 | -2.9% | +4.9% | fail | +0.0193 | fail |
| B8 | -10.8% | +6.1% | fail | -0.0211 | fail |
| B9 | -21.4% | +2.6% | fail | -0.1249 | **PASS** |
| B10 | -15.4% | -7.2% | **PASS** | -0.1511 | **PASS** |

**7 of 14 cells meet the primary criterion at bb100k, and 9 of 14 meet the secondary one.** At bb40k it is 8 and 9 of 14. At bb200k it is 3 and 3 of 4. On the teacher head at bb100k, where only group A publishes a baseline, it is 3 and 3 of 4.

Primary: medium+long at least 5% better AND short losing less than 2%. Secondary: full-97 Δ at or below −0.0384, the head-seed band. Δ is `k = 3` minus the cell's published `k = 0`, so negative is a gain. Student head at bb100k, the stop every one of the 14 cells reached.

The count is over CELLS. A1 and B3 share one student model, so the 14 cells hold 13 student models. The secondary criterion therefore counts one model fewer than it counts cells.

**Every cell here ran once, on one backbone seed.** The spread over the rows does not rank the recipes.

## Collapse watch

The first line of a cell is the mean over the last 10% of the run. The second line is the lowest value over the run's second half.

`ff` is `cos(f_t, h_{t+1})` and `cos_err_dj` is `1 − cos(f^(j)_t, h_{t+1+j})`, so `cos_err_d0` is `1 − ff` and `cos_err_dj` is the card's per-depth `ff`. A collapsed latent points one way, so `u_batchtime` runs toward zero WHILE `ff` runs toward 1. It is that pair, not `ff` alone, that separates collapse from a good forecast.

**Not logged: `qk_logit_maxabs`.** No run in this study writes that column at any depth, so this study does not watch it.

| arm | k | `ff` | `cos_err_d0` | `cos_err_d1` | `cos_err_d2` | `cos_err_d3` | `u_batchtime` on `h_t` | `u_batchtime` on `e_t` |
|---|---|---|---|---|---|---|---|---|
| B9 | 0 | 0.3838<br>0.3594 | — | — | — | — | 0.7782<br>0.7561 | 0.3174<br>0.2039 |
| B9 | 3 | 0.4776<br>0.4250 | 0.5224<br>0.5000 | 0.6384<br>0.6042 | 0.7043<br>0.6706 | 0.7451<br>0.7132 | 0.3892<br>0.2808 | 0.1184<br>0.0950 |
| B1 | 0 | 0.5226<br>0.4204 | — | — | — | — | 0.3904<br>0.2118 | 0.3696<br>0.1992 |
| B1 | 3 | 0.6347<br>0.4832 | 0.3653<br>0.2469 | 0.4471<br>0.3029 | 0.4734<br>0.3109 | 0.4857<br>0.3166 | 0.1526<br>0.1231 | 0.1125<br>0.0968 |
| B5·s1 ✗ | 0 | 0.2946<br>0.2679 | — | — | — | — | 0.9312<br>0.8393 | 0.0423<br>0.0301 |
| B5·s1 ✗ | 3 | 0.2824<br>0.2578 | 0.7176<br>0.6555 | 0.7354<br>0.6785 | 0.7516<br>0.6901 | 0.7611<br>0.7047 | 0.9354<br>0.8693 | 0.0624<br>0.0525 |
| B5·s2 | 0 | 0.3060<br>0.2674 | — | — | — | — | 0.9296<br>0.8405 | 0.0443<br>0.0381 |
| B5·s2 | 3 | 0.3037<br>0.2717 | 0.6963<br>0.6270 | 0.7138<br>0.6555 | 0.7289<br>0.6750 | 0.7403<br>0.6886 | 0.9250<br>0.8409 | 0.0515<br>0.0328 |
| A3 | 0 | 0.9279<br>0.6831 | — | — | — | — | 0.1445<br>0.1127 | 0.0370<br>0.0284 |
| A3 | 1 | 0.8120<br>0.6187 | 0.1880<br>0.0800 | 0.2272<br>0.1008 | — | — | 0.1791<br>0.1166 | 0.0697<br>0.0374 |
| A3 | 3 | 0.8790<br>0.8505 | 0.1210<br>0.0676 | 0.1352<br>0.0725 | 0.1508<br>0.0788 | 0.1592<br>0.0820 | 0.1730<br>0.0800 | 0.0561<br>0.0372 |

The lowest `u_batchtime` any arm reaches over its second half is 0.0284, on `u_batchtime_e`, A3 at k = 0. One direction would give `1/H` = 0.0156 at `d_model = 64`, so that arm sits 1.8× above it. No arm reaches zero at any depth.

On `h_t`, 1 of the 5 arms that trained both depths ends the deeper run below half its own `k = 0` usage: B1 0.3904 → 0.1526. That is a reading and not a verdict. No arm reaches zero, and this study runs no control that separates a lower usage from a worse score.

## What this study cannot support

| the claim | what stops it |
|---|---|
| Any group-A delta against a published `k = 0` | The card's baseline validity gate fails on group A: A3 misses its published number by 0.0294 against a gate of 0.0002. The card then asks for the `k = 0` side of every group-A cell to be retrained, and this study reads those baselines from the parent report. |
| That `k = 3` helps, or that it hurts | This study trained both depths on 4 arms, and they do not point one way: B9 -0.2791, B1 -0.1175, B5·s2 +0.0575, A3 +0.1429 (`depth_response.png`). Each is one draw in the backbone seed, so this study reads a direction and not a per-recipe ranking. |
| That the gain is the depth alone | B1 carries the `L_align` ×4 re-weighting control, and the re-weighting moves the score on its own. The annex's B1 table and its figure print the share of the move, per head. |
| That one of the two pays more than the other | The re-weighting's move and the depth's move sit inside each other's 95% intervals, in the same B1 table in the annex. That cell measures both and ranks neither. |
| Any per-cell verdict | Every cell is n = 1 in the backbone seed. The ±0.0384 band bounds the HEAD seed alone, and backbone-seed variance is unmeasured. |
| That depth 3 is the right depth | Only `k = 3` ran on the 14 cells. One ladder holds a second depth, on A3, and its `k = 1` delta covers zero: -0.0195 [-0.0537, +0.0148] on the student. |
| The per-horizon criterion of the card, the issue this study answers, at scale | This study trained the `k = 0` side on 4 arms, and only at bb40k. Every other pair reads its baseline from a parent report, so it is a screen and not a test. |
| That `k = 3` leads at 200k | 4 cells hold a published `k = 0` at 200k. A2 by -0.1079, B6 by -0.0804, B1 by -0.0643 lead it. B2 by +0.1054 loses it, against a largest gain of -0.1079, so the 4 cells do not point one way. |
| The cost of the depth | Two solo probes agree. The annex step-time tables carry them. A3's reading covers 127 of its 273 timing windows, so it is not comparable to them. |
| That the 200k reading is unconditional | The extend rule reads the bb40k-to-bb100k contrast, which the Protocol calls not head-matched. It fired inside its own ±0.0384 band on 4 stopped cells, and both manual overrides extended. |

## Tables

### Coverage

The card names 14 cells. This study scored **14 of them**: A1, A2, A3, A4, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10. Every cell carries a number.

| configuration | cell | loss terms that use `f` | depths trained | stops scored |
|---|---|---|---|---|
| arm5 (L_rep, tau_rep 1 + L_align on the student, no CPC, EMA 0.9 to 1.0) | A1 | L_align only | k = 3 | bb40k, bb100k |
| arm6_v2 (L_rep MoCo keys, tau_rep 0.1 + L_align on the teacher, CPC, no SIGReg on e, EMA 0.9 to 1.0) | A2 | L_align + CPC auxiliary | k = 3 | bb40k, bb100k, bb200k |
| arm6_v2 (L_rep MoCo keys, tau_rep 1 + L_align on the teacher, no CPC, EMA 0.9 to 1.0) | A3 | L_align only | k = 0, k = 1, k = 3 | bb40k, bb100k, bb200k |
| arm6_v2 (L_rep MoCo keys, tau_rep 1 + L_align on the student, no CPC, EMA 0.9 to 1.0) | A4 | L_align only | k = 3 | bb40k, bb100k, bb200k |
| arm6_v2 (L_rep MoCo keys, tau_rep 1 + L_align on the student, no CPC, EMA 0.9) | B1 | L_align only | k = 0, k = 3 | bb40k, bb100k, bb200k |
| arm6_v2 (L_rep MoCo keys, tau_rep 1 + L_align on the teacher, no CPC, EMA 0.9) | B2 | L_align only | k = 3 | bb40k, bb100k, bb200k |
| arm5 (L_rep, tau_rep 1 + L_align on the student, no CPC, EMA 0.9) | B3 | L_align only | k = 3 | bb40k, bb100k |
| arm5 (L_rep, tau_rep 1 + L_align on the teacher, no CPC, EMA 0.9) | B4 | L_align only | k = 3 | bb40k, bb100k, bb200k |
| arm4 (pooled contrastive over batch and channels, MoCo negatives, floor subtracted, tau 0.1, no CPC, no SIGReg on e, EMA 0.9) | B5 | pooled xshh_allt, floor subtracted | k = 0, k = 3 | bb40k, bb100k |
| arm6_v2 (L_rep MoCo keys, tau_rep 0.1 + L_align on the student, no CPC, EMA 0.9) | B6 | L_align only | k = 3 | bb40k, bb100k, bb200k |
| arm6_v2 (L_rep MoCo keys, tau_rep 0.1 + L_align on the teacher, no CPC, EMA 0.9) | B7 | L_align only | k = 3 | bb40k, bb100k |
| arm6_v2 (L_rep MoCo keys, tau_rep 0.1 + L_align on the teacher, CPC, no SIGReg on e, EMA 0.9) | B8 | L_align + CPC auxiliary | k = 3 | bb40k, bb100k |
| arm1 (split L_pred + L_rep, tau 0.1, CPC, no SIGReg on e, EMA 0.9) | B9 | split L_pred + CPC auxiliary | k = 0, k = 3 | bb40k, bb100k |
| arm6_v2 (L_rep MoCo keys, tau_rep 0.1 + L_align on the student, CPC, no SIGReg on e, EMA 0.9) | B10 | L_align + CPC auxiliary | k = 3 | bb40k, bb100k, bb200k |

Stops scored: bb40k, bb100k, bb200k. The card's extend rule reads a cell's bb40k number against its bb100k number, so it fires only where this study has both.

### This study's k = 3 against the published k = 0

GM-Relative MASE over the same 97 GIFT-Eval configs, strategy B4, horizon 16. Δ is this study minus the published number, so negative is a gain. A verdict reads Δ against the ±0.0384 head-seed band: closer than that is `flat`. A dash is a number no parent published. ‡ marks the two cells that share one student model. The second line of a verdict cell is its 95% paired dataset-cluster interval.

At bb100k, the stop every one of the 14 cells reached, counted over distinct models. Student head: 13 distinct models, **8 better, 3 flat, 2 worse**. Teacher head, group A only: 4 distinct models, **3 better, 0 flat, 1 worse**.

| cell | head | 40k k=3 | 40k pub | Δ | | 100k k=3 | 100k pub | Δ | | 200k k=3 | 200k pub | Δ | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | student ‡ | 1.1305 | 1.2596 | -0.1291 | better<br>[-0.1966, -0.0758] | 1.1676 | 1.2102 | -0.0426 | better<br>[-0.0835, -0.0069] | — | 1.1910 | — | — |
| A1 | teacher | 1.1318 | 1.2347 | -0.1029 | better<br>[-0.1590, -0.0560] | 1.1565 | 1.2407 | -0.0842 | better<br>[-0.1396, -0.0314] | — | — | — | — |
| A2 | student | 1.2735 | 1.4238 | -0.1503 | better<br>[-0.2357, -0.0762] | 1.2479 | 1.3913 | -0.1434 | better<br>[-0.2112, -0.0820] | 1.2507 | 1.3586 | -0.1079 | better<br>[-0.1653, -0.0546] |
| A2 | teacher | 1.2753 | 1.4177 | -0.1424 | better<br>[-0.2301, -0.0659] | 1.2514 | 1.3746 | -0.1232 | better<br>[-0.1841, -0.0660] | 1.2500 | 1.3459 | -0.0959 | better<br>[-0.1472, -0.0462] |
| A3 | student | 1.3618 | 1.1895 | +0.1723 | worse<br>[+0.1159, +0.2454] | 1.3010 | 1.1921 | +0.1089 | worse<br>[+0.0627, +0.1672] | 1.3998 | — | — | — |
| A3 | teacher | 1.3521 | 1.1793 | +0.1728 | worse<br>[+0.1161, +0.2480] | 1.3151 | 1.1963 | +0.1188 | worse<br>[+0.0672, +0.1857] | 1.2913 | — | — | — |
| A4 | student | 1.0862 | 1.1603 | -0.0741 | better<br>[-0.1305, -0.0268] | 1.0801 | 1.1945 | -0.1144 | better<br>[-0.1763, -0.0648] | 1.0660 | — | — | — |
| A4 | teacher | 1.0855 | 1.1544 | -0.0689 | better<br>[-0.1223, -0.0249] | 1.0874 | 1.1837 | -0.0963 | better<br>[-0.1505, -0.0506] | 1.0828 | — | — | — |
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

### Stop reasons: what the extend rule read at each cell

The rule reads one cell's bb40k number against its bb100k number, per head. A head that moved down earns the second 100,000 steps. A head that moved up stops. Both columns are bb100k minus bb40k, so negative is an improvement. It held 6 cells at 100k. `last stop` and `ended by` are the parent report's two columns: where each cell finished, and what finished it.

| cell | 40k→100k student | 40k→100k teacher | decision | last stop | ended by | why |
|---|---|---|---|---|---|---|
| A1 | +0.0371 | +0.0248 | **stop at 100k** | bb100k | extend rule | both heads moved up |
| A2 | -0.0256 | -0.0239 | **extend both heads** | bb200k | ladder ceiling | both heads moved down |
| A3 | -0.0608 | -0.0370 | **extend both heads** | bb200k | ladder ceiling | both heads moved down |
| A4 | -0.0061 | +0.0019 | **extend both heads** | bb200k | ladder ceiling | the student head moved down. The teacher head moved +0.0019, 5% of the ±0.0384 head-seed band, so the rule decides nothing there. Extended by hand, on free hardware |
| B1 | +0.0030 | -0.0051 | **extend both heads** | bb200k | ladder ceiling | the card's call: both moves sit inside the ±0.0384 head-seed band, so the rule decides nothing |
| B2 | -0.0533 | -0.0924 | **extend both heads** | bb200k | ladder ceiling | both heads moved down |
| B3 | +0.0371 | +0.0276 | **stop at 100k** | bb100k | extend rule | both heads moved up |
| B4 | -0.0530 | -0.0591 | **extend both heads** | bb200k | ladder ceiling | both heads moved down |
| B5 | +0.0179 | +0.0212 | **stop at 100k** | bb100k | extend rule | both heads moved up |
| B6 | -0.0146 | -0.0074 | **extend both heads** | bb200k | ladder ceiling | both heads moved down |
| B7 | +0.0587 | +0.0336 | **stop at 100k** | bb100k | extend rule | both heads moved up |
| B8 | +0.0300 | +0.0374 | **stop at 100k** | bb100k | extend rule | both heads moved up |
| B9 | +0.0508 | +0.0365 | **stop at 100k** | bb100k | extend rule | both heads moved up |
| B10 | -0.0266 | -0.0231 | **extend both heads** | bb200k | ladder ceiling | both heads moved down |

**The rule chooses which cells reach bb200k.** It sent 8 cells there. On 4 of the 6 cells it stopped (A1, B3, B5, B8) the move it read was smaller than the ±0.0384 band. Both hand overrides extended a cell.

### Glossary

| term | what it means here |
|---|---|
| the card | the issue this study answers, and the 14 cells, stops and criteria it names |
| cell | the card's short id for one of those 14 configurations, `A1`..`A4` and `B1`..`B10`. A figure or a table uses it after the configuration appears in its legend or header |
| arm | a (cell, backbone seed, machine) triple. B5 trained three, so the cell is not the unit a delta lives in |
| `k`, rollout depth | the value of `--train-rollout-depth`. It copies every loss term the forecast operator `f` enters at depths 1..`k` and sums the copies. `k = 0` is today's training |
| the fixed-point approximation | how training rolls the forecast out: the depth-`j` input is the model's own depth-`j-1` predictions, not the true prefix. It buys one parallel pass over every `t`, and it is the card's alternative suspect to the objective |
| bb40k, bb100k, bb200k | backbone step 40,000 / 100,000 / 200,000. bb40k is the one stop every run here reached |
| GM-Relative MASE | geometric mean over the 97 GIFT-Eval configs of each config's MASE divided by the seasonal-naive MASE. Lower is better. 1.0 is seasonal-naive parity |
| B4 eval strategy | GIFT-Eval's official evaluation strategy, the one the parent reports use |
| rollout steps at eval | how many times the eval calls `rollout_latent` on one config: `ceil(prediction_length / 16)`, since B4 asks for one token per patch of the horizon and the function takes one autoregressive step per token. It is a property of the config, not of the run |
| student / teacher head | the quantile head is trained twice per backbone, once on the student encoder and once on its EMA copy, the teacher. The two are separate measurements of one backbone |
| f-bearing term | the loss term that the forecast operator `f` enters. `--train-rollout-depth K` duplicates it at depth 1..K |
| `rep_only` | the representation loss with no forecast term |
| `L_align` | the term that aligns `f`'s output with the future latent |
| `L_pred` | the predictive contrastive term, split from the representation term |
| `xshh_allt` | negatives pooled across the batch and across channels, taken over every time index |
| `u_batchtime` | dimension usage of a latent over the pooled (batch × time) sample axis: `1 / (H · mean off-diagonal squared cosine)`, capped at 1. 1.0 is all `H` dimensions in use and a value near `1/H` is one direction. `h_t` is the encoder latent, `e_t` the embedding it reads |
| collapse | the latent falling onto few directions, so `u_batchtime` runs toward zero. The card watches for it because a model can win the deeper f-bearing terms by flattening `f` |
| `arm1 nse` | split L_pred + L_rep, tau 0.1, CPC, no SIGReg on e. Cell B9 |
| `arm4 combab` | pooled contrastive over batch and channels, MoCo negatives, floor subtracted, tau 0.1, no CPC, no SIGReg on e. Cell B5. Note: its launcher's own label says tau 1.0, and its `--tau 1.0` sits before the shared `--tau 0.10`, so argparse kept 0.10 |
| `arm5 combab` | L_rep, tau_rep 1 + L_align, no CPC. Cells A1, B3, B4 |
| `arm6_v2 combab` | L_rep MoCo keys, tau_rep 1 + L_align, no CPC. Cells A3, A4, B1, B2 |
| `arm6_v2 ncpc` | L_rep MoCo keys, tau_rep 0.1 + L_align, no CPC. Cells B6, B7 |
| `arm6_v2 nse` | L_rep MoCo keys, tau_rep 0.1 + L_align, CPC, no SIGReg on e. Cells A2, B8, B10 |
| the align target | `L_align` compares `f`'s output against the student encoder's future latent or against the EMA teacher's. Two cells that share an arm and differ only here are two configurations |
| head-seed band ±0.0384 | how far the head seed alone moved a score in `ema_sched_ladder.md`, pooled. It bounds the head seed and nothing else |
| dataset-cluster | the resampling unit of every interval here. `<ds>/short`, `/medium` and `/long` are three configs of one series, so the bootstrap resamples the dataset, not the config |
| `mixup` | the count of examples the batch mixer touched in a 200-step window. Two runs on one data order print one count |
| ✗ | a retracted arm: its `k = 0` baseline is a rented-box artifact, so its depth delta is withdrawn |


## Annex tables

### The stop ladder, cell by cell

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
| A4 | teacher | 1.0855 | 1.0874 | 1.0828 | -0.0046 | [-0.0199, +0.0123] | -0.4% | extended by hand. The rule's move is inside the band |
| B1 | student | 1.0850 | 1.0881 | 1.1009 | +0.0128 | [+0.0001, +0.0284] | +1.2% |  |
| B1 | teacher | 1.0948 | 1.0897 | 1.1001 | +0.0104 | [-0.0037, +0.0280] | +1.0% |  |
| B2 | student | 1.3976 | 1.3443 | 1.2904 | -0.0539 | [-0.0935, -0.0197] | -4.0% |  |
| B2 | teacher | 1.4041 | 1.3117 | 1.2825 | -0.0292 | [-0.0604, -0.0016] | -2.2% |  |
| B3 | student | 1.1305 | 1.1676 | — | — | — | — | the extend rule held this cell at 100k |
| B3 | teacher | 1.1343 | 1.1618 | — | — | — | — | the extend rule held this cell at 100k |
| B4 | student | 1.3334 | 1.2804 | 1.3182 | +0.0379 | [+0.0089, +0.0742] | +3.0% |  |
| B4 | teacher | 1.3339 | 1.2748 | 1.3202 | +0.0454 | [+0.0181, +0.0807] | +3.6% |  |
| B5 | student | 1.3204 | 1.3383 | — | — | — | — | the extend rule held this cell at 100k |
| B5 | teacher | 1.3216 | 1.3428 | — | — | — | — | the extend rule held this cell at 100k |
| B6 | student | 1.2297 | 1.2151 | 1.2207 | +0.0056 | [-0.0101, +0.0212] | +0.5% |  |
| B6 | teacher | 1.2184 | 1.2110 | 1.2339 | +0.0230 | [+0.0032, +0.0440] | +1.9% |  |
| B7 | student | 1.2617 | 1.3205 | — | — | — | — | the extend rule held this cell at 100k |
| B7 | teacher | 1.2444 | 1.2780 | — | — | — | — | the extend rule held this cell at 100k |
| B8 | student | 1.2857 | 1.3157 | — | — | — | — | trained from step 0, scored at bb100k only |
| B8 | teacher | 1.2865 | 1.3239 | — | — | — | — | trained from step 0, scored at bb100k only |
| B9 | student | 1.2791 | 1.3299 | — | — | — | — | the extend rule held this cell at 100k |
| B9 | teacher | 1.2728 | 1.3094 | — | — | — | — | the extend rule held this cell at 100k |
| B10 | student | 1.2669 | 1.2403 | 1.2624 | +0.0221 | [+0.0032, +0.0481] | +1.8% |  |
| B10 | teacher | 1.2730 | 1.2499 | 1.2440 | -0.0059 | [-0.0220, +0.0105] | -0.5% |  |

### A3's two draws, the numbers

A3 at bb200k reads 1.3998 on the student and 1.2913 on the teacher, off one backbone file. That 0.1084 gap is the largest in the grid. It is 6.5x the next-largest in group A (0.0168), and 2.6x the largest of the other 35 gaps (0.0425). Every gap in the grid is in [`results/head_gap.tsv`](results/head_gap.tsv).

The second draw changes two things: the head seed, and the computer that trained the head. Draw 1 trained on the rented computer, draw 2 on elisa. Both read the same 200,000-step backbone checkpoint, the rented computer's original and elisa's synced copy of it. Held across the two draws: 30,000 head steps, the recipe, and the 97-config eval, which ran on elisa's cores for both. Only elisa's copy carries a recorded md5 (`9f0e8da71ff595523d2bf0dabdf80445`, [`results/eval/A3_k3_bb200k_student_s20260723/backbone_md5.txt`](results/eval/A3_k3_bb200k_student_s20260723/backbone_md5.txt)). The rented computer was released before anyone could checksum its original.

| draw | head seed | GM-Relative MASE | against draw 1 |
|---|---|---|---|
| 1, student | 20260722 | 1.3998 | — |
| 2, student | 20260723 | 1.4098 | +0.0100 |
| teacher | 20260722 | 1.2913 | -0.1084 |

**The two draws agree.** They sit 0.0100 apart [-0.0163, +0.0378], so 1.3998 is not a bad draw. The student/teacher gap survives the redraw at -0.1185 [-0.1819, -0.0718], teacher minus student. The two draws used different computers, so this agreement bounds the head seed and the computer together, not the seed alone.

A3's is the ladder's largest reversal. It is not the only one. 5 of the 8 three-stop student trajectories reverse at bb200k, in the stop-ladder table above.

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

**A1/B3 hold one student, not two.** arm5 combab (L_rep, tau_rep 1 + L_align, no CPC) aligns to the student and carries no MoCo keys, so no loss term reads the EMA encoder and the regime sends no gradient into the student. One student number for both cells is the right answer. It is ONE measurement: one cell's student row does not replicate the other's. The teacher side differs at every stop, and the teacher numbers do too.

**A2/B8, A3/B2, A4/B1 hold two students.** Their arms carry `--moco-rep-keys`, whose keys come from the EMA encoder, or align to the teacher. Either path reaches the student's gradient, so the regime moves it.

### The A1/B3 duplicate, re-run end to end

Each row trains a fresh student head from the checkpoint its own cell names, at seed 20260722. It then runs the 97 configs into `results/eval/<cell>rep_…`, a directory no other cell writes. A path that ignored the cell would land the re-run on the other cell's number.

| cell | stop | backbone md5 | first pass | re-run | Δ |
|---|---|---|---|---|---|
| A1 | bb40k | `f99fa42c` | 1.1305 | 1.1447 | +0.0141 |
| A1 | bb100k | `dbd23cbe` | 1.1676 | 1.1610 | -0.0066 |
| B3 | bb40k | `b3a51f06` | 1.1305 | 1.1447 | +0.0141 |
| B3 | bb100k | `0efbb813` | 1.1676 | 1.1610 | -0.0066 |

The largest re-run move is 0.0141. The two cells carry different backbone md5s and reproduce their own first-pass numbers, so the head and the eval read the file each cell names. The duplicate is the student weights, not the path.

### Reproduction of the published k = 0

Same cell, same recipe, same head seed 20260722, same 97-config B4 eval, student head. Rows are grouped by computer.

A row at the parents' own backbone seed 20260520 must meet the card's 0.0002. A row at any other seed must meet the seed band.

| backbone | seed | computer | published k = 0 | retrained k = 0 | \|Δ\| | gate | verdict |
|---|---|---|---|---|---|---|---|
| B1 | 20260520 | elisa | 1.2025 | 1.2025 | 0.0000 | 0.0002, the card | PASS |
| B5·s3 | 20260520 | elisa | 1.2748 | 1.2751 | 0.0003 | 0.0002, the card | at the re-run floor |
| B9 | 20260520 | elisa | 1.5579 | 1.5583 | 0.0004 | 0.0002, the card | at the re-run floor |
| B5·s2 | 20260521 | elisa | 1.2748 | 1.2716 | 0.0032 | 0.0230, the seed band | inside the seed band |
| A3 | 20260520 | vast box d | 1.1895 | 1.2189 | 0.0294 | 0.0002, the card | FAIL |
| B5·s1 ✗ | 20260520 | vast box d | 1.2748 | 1.3917 | 0.1169 | 0.0002, the card | FAIL |
| B5·pub | 20260520 | the parent report's box | 1.2748 | 1.2751 | 0.0003 | 0.0002, the card | at the re-run floor |

This comparison cannot resolve two things. The head and the eval move the score by 0.0003, which is what `B5·pub` moves it while training nothing. The parents' four printed decimals add 0.0001. Together they give 0.0004: a |Δ| at or below that is a run this pipeline cannot separate from the published one. The card's gate of 0.0002 is stricter than that.

The seed band is 0.0230. This study measured a seed change once: `B5·s2` against `B5·s3`, one computer, one recipe, +0.0035 [-0.0183, +0.0230]. The band is the far end of that interval. The interval covers the pair's eval sample and not the seeds. So the band is a floor on what a seed can move, not a bound on it. B5·s2 is the only row it gates, because every other row carries the parents' own seed.

`B5·pub` is not a training. It puts this study's head and eval on the parent report's own published B5 checkpoint. Its row therefore bounds the head and the eval, not the trainer. `B5·s3` is a training, at the protocol seed, on elisa. Its 97-config eval output is byte-identical to `B5·pub`'s (`results/eval/G7_B5_k0_e_bb40k_student/all_results.csv` against `results/eval/G1_B5pub_bb40k_student/all_results.csv`). So the elisa retrain reproduced the parent's backbone exactly, and the 0.0003 both rows carry is the head and the eval.

**The card's baseline validity gate, group by group.** It retrains one cell of the group at `k = 0` on this study's code and asks for the published number to within 0.0002. Group A: A3 at `k = 0`, on vast box d, misses its published number by 0.0294. **FAIL**. Group B: B1 at `k = 0`, on elisa, misses its published number by 0.0000. **PASS**.

On a failure the card asks for a retrain of the `k = 0` side of every cell of that group. It must not come from the parent report. This study did not do that for group A. So every group-A delta against a published `k = 0` is a screen and not a test.

### Depth response, against each arm's own k = 0

| arm | seed | same computer? | head | k | k = 0 | this k | Δ | all | short | med+long | criterion |
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

**This table is the only place the card's criterion runs as a test.** Every row here trains its own `k = 0`, and every row is at one stop, bb40k. The card also asks about bb100k and bb200k. This study trained no `k = 0` at either stop, so there the report has the screen and nothing else. The same criterion runs over every pair of the published-baseline table as well, where it is a screen because the `k = 0` side comes from a parent report: 25 of 41 pairs meet it, and 10 of 18 at bb100k ([`results/criterion_screen.csv`](results/criterion_screen.csv)).

`same computer?` records where the two runs trained. The B5 table below measures that change alone, at one seed, at 0.1166, and the backbone seed at 0.0035. Both are nuisance draws.

✗ marks a retracted row: B5·s1's `k = 0` trained on a rented box and misses its published value by 0.1169. `B5·s3` retrains it at the same seed on elisa and lands 0.0003 away. The baseline the -5.1% rests on is therefore a rented-box artifact, and the delta is retracted.

Head-seed band ±0.0384 (`ema_sched_ladder.md`, pooled). It bounds the head seed alone. It does not bound the computer. It does not bound the BACKBONE seed either: this study holds one backbone seed in 14 cells and one replicate of it (B5·s2 against B5·s3, at k = 0, at bb40k). Backbone-seed variance is therefore unmeasured. Every better / flat / worse verdict in this report rests on a band that bounds one of the two seeds in play.

The depths trained are k = 1, k = 3, and only k = 3 ran on the 14 cells. One ladder holds more than a single depth: A3's, the cell where k = 3 does the most damage. Its k = 1 interval covers zero. So this study supports **depth 3 moves the score**. It does NOT support *depth 3 is the right depth*: one cell measures a second depth, and no cell measures a third.

### Paired dataset-cluster bootstrap, per horizon subset

The resampling unit is the dataset. `<ds>/short`, `/medium` and `/long` are three configs of one series, so they are not independent draws. 95% percentile interval over 10,000 resamples. Each interval is over one run pair's 97 configs, so it bounds the eval sample and not run-to-run variance.

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

arm4 (pooled contrastive over batch and channels, MoCo negatives, floor subtracted, tau 0.1, no CPC, no SIGReg on e, EMA 0.9) [B5] trained three times on one recipe, one code snapshot, one head seed and one eval. They differ by backbone seed and by machine, and each contrast below names which of the two it changes. The machine contrast is the larger of the two, and each contrast is one run pair.

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

### B1: is the win the depth, or the weight?

B1 carries `L_align` as its only f-bearing term. Its `k = 3` run therefore multiplies that term's weight against the f-free terms by 4, as well as adding depth. The `L_align x4` row applies the re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 3 |
|---|---|---|---|
| student | 1.2025 | 1.1513<br>-0.0512 [-0.1001, -0.0023] | 1.0850<br>-0.1175 [-0.1801, -0.0615] |
| teacher | 1.2001 | 1.1482<br>-0.0519 [-0.0987, -0.0066] | 1.0948<br>-0.1053 [-0.1661, -0.0515] |

Second line of each cell: the difference against `k = 0` and its 95% paired dataset-cluster interval.

Every column trained on elisa at backbone seed 20260520, on the same head budget. This is the study's one such table, so it may divide one column by another.

| head | the re-weighting<br>k = 0 → x4 | the depth<br>x4 → k = 3 | total<br>k = 0 → k = 3 | the re-weighting's share |
|---|---|---|---|---|
| student | -0.0512 | -0.0663 | -0.1175 | 44% |
| teacher | -0.0519 | -0.0534 | -0.1053 | 49% |

### A3: is the damage the depth, or the weight?

Summing the depths multiplies `L_align`'s weight against the f-free terms by k + 1. The `L_align x4` row applies that re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 1 | k = 3 |
|---|---|---|---|---|
| student | 1.2189 | 1.2590<br>+0.0401 [+0.0116, +0.0767] | 1.1995<br>-0.0195 [-0.0537, +0.0148] | 1.3618<br>+0.1429 [+0.0893, +0.2122] |
| teacher | 1.2184 | 1.2558<br>+0.0374 [+0.0058, +0.0756] | 1.2063<br>-0.0121 [-0.0479, +0.0261] | 1.3521<br>+0.1337 [+0.0839, +0.2004] |

Second line of each cell: the difference against `k = 0` and its 95% paired dataset-cluster interval.

Where each column trained: A3_k0: vast box d · G3_A3_k0_aw4: elisa · G3_A3_k1: elisa · A3_k3: vast box b. The columns are separate draws, so read the two controls as direction and not as magnitude. This table does not divide one column by another.

### What the depth costs

Median `fwd + bwd` per step, from each run's own trainer log. A median is a cost of the depth only where the run had the card to itself, so the table says which did. `run_provenance.py` reads that off the driver logs and [`results/steptime_solo.csv`](results/steptime_solo.csv) carries it per run. A3's `k = 3` shared vast box b with a clone of itself up to step 14,800. Its 131.5 ms is the median over the 127 windows after that.

| arm | f-bearing term | k | machine | card | fwd+bwd | alone? |
|---|---|---|---|---|---|---|
| B9 | split L_pred + L_rep, tau 0.1, CPC, no SIGReg on e | 0 | elisa | RTX 4090 | 212.6 ms, shared | no, another backbone for 96% of the run and head training for 4% of it |
| B9 | split L_pred + L_rep, tau 0.1, CPC, no SIGReg on e | 3 | vast box c | RTX 4090 | 425.2 ms | yes |
| B1 | L_rep MoCo keys, tau_rep 1 + L_align, no CPC | 0 | elisa | RTX 4090 | 178.6 ms, shared | no, another backbone for 100% of the run and head training for 100% of it |
| B1 | L_rep MoCo keys, tau_rep 1 + L_align, no CPC | 3 | elisa | RTX 4090 | 235.1 ms, shared | no, another backbone for 68% of the run and head training for 100% of it |
| B5·s1 | pooled contrastive, MoCo negatives, floor subtracted, tau 0.1, no CPC, no SIGReg on e | 0 | vast box d | RTX 5090 | 117.6 ms | yes |
| B5·s1 | pooled contrastive, MoCo negatives, floor subtracted, tau 0.1, no CPC, no SIGReg on e | 3 | vast box a | RTX 5090 | 301.9 ms | yes |
| B5·s2 | pooled contrastive, MoCo negatives, floor subtracted, tau 0.1, no CPC, no SIGReg on e | 0 | elisa | RTX 4090 | 201.1 ms, shared | no, another backbone for 100% of the run and head training for 98% of it |
| B5·s2 | pooled contrastive, MoCo negatives, floor subtracted, tau 0.1, no CPC, no SIGReg on e | 3 | elisa | RTX 4090 | 500.9 ms, shared | no, another backbone for 43% of the run and head training for 100% of it |
| A3 | L_rep MoCo keys, tau_rep 1 + L_align, no CPC | 0 | vast box d | RTX 5090 | 115.9 ms | yes |
| A3 | L_rep MoCo keys, tau_rep 1 + L_align, no CPC | 1 | elisa | RTX 4090 | 214.7 ms, shared | no, another backbone for 72% of the run and head training for 100% of it |
| A3 | L_rep MoCo keys, tau_rep 1 + L_align, no CPC | 3 | vast box b | RTX 5090 | 131.5 ms | yes |

The two probes that agree:

| probe | k = 0 | k = 3 | change | what the two sides hold | source |
|---|---|---|---|---|---|
| B5·s1, over its own run | 117.6 ms | 301.9 ms | +157% | each side solo on its own box, vast box d → vast box a | [`results/steptime_solo.csv`](results/steptime_solo.csv) |
| B5, alternating on one elisa card | 190.2 ms | 509.9 ms | +168% | one card, 3 reps of 600 steps | [`results/steptime_B5_solo.log`](results/steptime_B5_solo.log) |

A3 reads +13% (115.9 ms against 131.5 ms) and is not comparable to those two: its `k = 3` median covers 127 of its 273 windows. **Carry +157% to +168% and do not carry the low row.** No cell of the 14 has a same-card `k = 0` / `k = 3` pair. Such a pair is what would settle it.

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

