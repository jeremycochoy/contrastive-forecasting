«Agent ExperimentRunner claude-opus-5 writing»

## Item 3's answer: neither the depth nor the re-weight alone. Item 6's gap now has its interval.

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(`results/`, `plots/`, `scripts/`, `sync/`)
**Commit:** `94db5e1a`

### The brief's premise is out of date

The brief reported `G_B1_k0_aw4` with no eval, no score file and nothing
running since 20:32. That was true at 20:40 on 13 August. Both evals ran the
same evening. The teacher finished at 22:07 and the student at 22:40. Item 6's
redraw finished earlier still, at 18:20. All three are committed.

```
results/score_G_B1_k0_aw4_bb40k_student.txt        1.1513
results/score_G_B1_k0_aw4_bb40k_teacher.txt        1.1482
results/score_A3_k3_bb200k_student_s20260723.txt   1.4098
```

Each of the three eval directories holds 97 config rows, strategy B4, horizon
16. `verify_denominator` reads one seasonal-naive denominator over all 99
evals, md5 `a86ef40144eee950866b027d876ce75e`. The three new cells divide by
the same column as every other cell.

### One label correction, because it changes the reading

The brief quoted B1 k=0 as 1.0850 student / 1.0948 teacher. Those are B1
**k = 3** at bb40k. The k=0 baseline is **1.2025 / 1.2001**
(`G6_B1_k0_bb40k`). The -0.1175 runs 1.2025 to 1.0850. The control sits
between the two.

### Runs completed

```
this session   0 training runs, 0 evals, $0.00, nothing rented, elisa only
study total   99 evals, 97 GIFT-Eval configs each, strategy B4, horizon 16
              72 of 72 deliverables over 14 cells x 3 stops x 2 heads
failed         0
```

`vastrun-balance` reads **$11.45**, unchanged. No cell with a score was
retrained.

### Headline numbers

```
item 3  B1 k = 0                    1.2025 student   1.2001 teacher
item 3  B1 k = 0, L_align x4        1.1513 student   1.1482 teacher   <- the control
item 3  B1 k = 3                    1.0850 student   1.0948 teacher
item 6  A3 bb200k student, draw 1   1.3998   seed 20260722, on the box
item 6  A3 bb200k student, draw 2   1.4098   seed 20260723, on elisa   <- the redraw
item 6  A3 bb200k teacher           1.2913   seed 20260722, on the box
```

## Item 3 — the control's verdict

B1 carries `L_align` as its only f-bearing term. Its `k = 3` run therefore
multiplies that term's weight by 4 as well as adding horizons. The control
applies the same 4x weight at `k = 0`, with no depth at all. All six points
trained on elisa at backbone seed 20260520, so the machine is held.

| head | the re-weight<br>k=0 to x4 | the depth<br>x4 to k=3 | total<br>k=0 to k=3 | the re-weight's share |
|---|---|---|---|---|
| student | **-0.0512** [-0.1001, -0.0023] | **-0.0663** [-0.1070, -0.0331] | -0.1175 [-0.1801, -0.0615] | **44%** |
| teacher | **-0.0519** [-0.0987, -0.0066] | **-0.0534** [-0.0874, -0.0237] | -0.1053 [-0.1661, -0.0515] | **49%** |

95% paired dataset-cluster bootstrap, 10,000 resamples, seed 20260809, over
the pair's 97 configs. The resampling unit is the dataset, not the config.

**Plainly: the win is not the re-weight.** The 4x `L_align` re-weight alone
reproduces 44% of B1's -0.1175 on the student and 49% on the teacher. That is
not most of it, so by the brief's own rule the answer is the depth. But the
split is near even, and both halves exclude zero on both heads. The honest
statement is that **neither carries the win alone, and the extra horizons are
the larger half, narrowly.**

One caveat on what "depth" means here. `k = 3` puts its four copies of
`L_align` on t+1..t+4. The control puts all four on t+1. So the depth segment
is the extra HORIZONS, not depth net of everything else.

The re-weight's own effect is entirely at the longer horizons: -0.1400
[-0.2267, -0.0662] on medium and long, and -0.0009 [-0.0483, +0.0460] on
short.

The same control on A3 moves the score the wrong way, +0.0401 [+0.0116,
+0.0767] student. A3's four points each trained on a different box from at
least one other, so A3 gives a direction and not a magnitude. B1 gives sizes.

## Item 6 — the redraw, and both draws against the teacher

The reseed completed. `results/gap6_a3_reseed.out` logs the whole run: head
training started 16:02:54 on elisa GPU 1, seed 20260723, 30,000 steps,
returned rc=0 at 17:08:52. The 97-config eval returned rc=0 at 18:20:31 and
printed 1.4098.

Both draws read the same 200,000-step backbone. Only elisa's copy carries a
recorded md5 (`9f0e8da71ff595523d2bf0dabdf80445`). The box was released before
its original could be checksummed.

| contrast | delta | 95% CI | better in |
|---|---|---|---|
| draw 2 against draw 1 | **+0.0100** | [-0.0163, +0.0378] | 22.7% |
| teacher against draw 1 | **-0.1084** | [-0.1648, -0.0671] | 100% |
| teacher against draw 2 | **-0.1185** | [-0.1819, -0.0718] | 100% |

**The two draws agree.** They sit 0.0100 apart, 26% of the ±0.0384 head-seed
band, and the interval covers zero. So 1.3998 is not a bad draw. The two draws
also sit on two machines, so this bounds the head seed and the machine
together.

**The student/teacher gap survives.** Both draws lose to the teacher by about
0.11, and both intervals exclude zero on every resample. Two head seeds put
A3's bb200k student above its teacher. The gap is a property of that student
encoder, not of the draw. Draw 1 and the teacher trained on the same box, so
their 0.1084 holds the machine.

The two teacher contrasts are new this session. The study stated both gaps as
point values and never put an interval on either. They are in
[`results/final_check.csv`](https://github.com/jeremycochoy/contrastive-forecasting/blob/feature/contrastive-forecasting-373/reports/2026-08-08_rollout_depth/results/final_check.csv),
and the report now carries them.

## Cross-check

`scripts/final_check.py` is a fresh implementation against the same
definitions. It recomputes all nine GM-Relative MASE scores that items 3 and 6
rest on, from the per-config CSVs and the shared seasonal-naive reference, and
re-derives the six committed intervals. **All nine scores and all six
intervals match to four decimals.** Log:
[`results/final_check.log`](https://github.com/jeremycochoy/contrastive-forecasting/blob/feature/contrastive-forecasting-373/reports/2026-08-08_rollout_depth/results/final_check.log).

`scripts/verify_close.sh` runs five standing checks. All pass.

## The review's ten items, all closed

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | recomputed from one list in `tables.py`: 7 of 16 improved, mean +0.0079, median +0.0042, band covers 13 of 16. Opener, section and ladder table read that one list |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | the depth-response table states it answers the criterion for 2 machine-held arms at one stop. The screen runs over all 41 published pairs, 25 of 41 meet it. All 41 deltas now carry a 95% interval |
| 3 | in 12 of 14 cells `k = 3` is a depth change AND a 4x re-weight of `L_align` | **the B1 control above.** 44/56 student, 49/51 teacher, both halves exclude zero |
| 4 | the extend rule reads a confounded contrast, fires inside its own band, its overrides go one way | the stop-reason table states all three by name and number. The 200k verdict reads conditional on a panel selected for having improved |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied. Backbone-seed variance is unmeasured |
| 6 | A3's bb200k student is an outlier and no one re-measured it | **the redraw above.** 1.4098 against 1.3998, and both draws lose to the teacher with intervals excluding zero |
| 7 | the headline counts A1 and B3 as two cells; their student column is one measurement | the count is over distinct MODELS: 13 student models, 8 better, 3 flat, 2 worse. The shared row is marked `‡` and counted once |
| 8 | only `k = 3` ran on the 14 cells | the report supports *depth 3 moves the score*, not *depth 3 is the right depth* |
| 9 | the cost of the depth is unknown within an order of magnitude | A3's +13% row is marked not comparable and dropped. The report carries +157% to +168%, the two probes that agree |
| 10 | the ladder caption forbids the comparison the headline makes | the caption names the one table that reads the dashes and labels it a screen |

Minors, all closed: the fidelity batch's not-held-out caveat sits next to the
claim; the mechanism sections name their four-cell diagnostic set; the
`--grad-clip 1.0` exemption names the project rule it departs from and why;
the "k = 3 still leads at 200k" line is gone. B5·s3's missing teacher head
stays disclosed in the annex.

## Final coverage

GM-Relative MASE, 97 GIFT-Eval configs, strategy B4, horizon 16. S = student
head, T = teacher head.

| cell | 40k S | 40k T | 100k S | 100k T | 200k S | 200k T |
|---|---|---|---|---|---|---|
| A1 | 1.1305‡ | 1.1318 | 1.1676‡ | 1.1565 | stop | stop |
| A2 | 1.2735 | 1.2753 | 1.2479 | 1.2514 | 1.2507 | 1.2500 |
| A3 | 1.3618 | 1.3521 | 1.3010 | 1.3151 | 1.3998 | 1.2913 |
| A4 | 1.0862 | 1.0855 | 1.0801 | 1.0874 | 1.0660 | 1.0828 |
| B1 | 1.0850 | 1.0948 | 1.0881 | 1.0897 | 1.1009 | 1.1001 |
| B2 | 1.3976 | 1.4041 | 1.3443 | 1.3117 | 1.2904 | 1.2825 |
| B3 | 1.1305‡ | 1.1343 | 1.1676‡ | 1.1618 | stop | stop |
| B4 | 1.3334 | 1.3339 | 1.2804 | 1.2748 | 1.3182 | 1.3202 |
| B5 | 1.3204 | 1.3216 | 1.3383 | 1.3428 | stop | stop |
| B6 | 1.2297 | 1.2184 | 1.2151 | 1.2110 | 1.2207 | 1.2339 |
| B7 | 1.2617 | 1.2444 | 1.3205 | 1.2780 | stop | stop |
| B8 | 1.2857 | 1.2865 | 1.3157 | 1.3239 | stop | stop |
| B9 | 1.2791 | 1.2728 | 1.3299 | 1.3094 | stop | stop |
| B10 | 1.2669 | 1.2730 | 1.2403 | 1.2499 | 1.2624 | 1.2440 |

```
deliverables 72   done 72   running 0   queued 0   NOT STARTED 0
(+12 stops, not deliverables)
```

‡ A1 and B3 hold one student model, printed twice per stop. The 72
deliverables therefore hold 70 distinct measurements. Their teacher columns
are two models and are counted twice.

Controls and replicates outside the grid, all at bb40k:

| cell | student | teacher |
|---|---|---|
| B1 k=0 | 1.2025 | 1.2001 |
| B1 k=0, L_align x4 | 1.1513 | 1.1482 |
| A3 k=0 | 1.2189 | 1.2184 |
| A3 k=0, L_align x4 | 1.2590 | 1.2558 |
| B5 k=0 | 1.3917 | 1.3719 |
| B5·s2 k=0 | 1.2716 | 1.2661 |
| B5 k=0, encoder-only | 1.2751 | not trained |
| B9 k=0 | 1.5583 | 1.5599 |
| A3 bb200k student, draw 2 | 1.4098 | — |

Nothing is running. Nothing is rented. The GPUs on elisa carry no work from
this study.
