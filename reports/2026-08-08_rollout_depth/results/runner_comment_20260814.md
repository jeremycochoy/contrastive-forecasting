«Agent ExperimentRunner claude-opus-5 writing»

## Item 3: the extra horizons are the larger half, narrowly. Item 6: the two draws agree.

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(`results/`, `plots/`, `scripts/`, `sync/`)
**Commit:** `4f4c14f9`

### The brief's premise

The brief states that `G_B1_k0_aw4` has no eval, that no `score_G_B1_k0_aw4_*`
file exists, and that item 6 has no reseed log. All three were true at 20:40 on
13 August. All three were false by 22:40 the same evening. Both jobs ran on
elisa and both are committed. **Nothing is left to run.**

The brief also labels B1 k = 0 as 1.0850 / 1.0948. Those numbers are B1
**k = 3** at bb40k. The k = 0 baseline is **1.2025 / 1.2001**. The -0.1175
runs 1.2025 down to 1.0850.

## Item 3 — the decisive control

B1 carries `L_align` as its only f-bearing term. A `k = 3` run adds one whole
copy of that term per depth, so it multiplies the term's total weight by 4 as
well as adding horizons. The control applies the same 4x weight at `k = 0`,
with no depth at all. The launcher record reads `arg=arm6_v2_combab k=0
seed=20260520 suffix='_aw4' gap_args='--align-loss-weight 4.0'`.

| head | k = 0 | k = 0, `L_align` x4 | k = 3 |
|---|---|---|---|
| student | 1.2025 | **1.1513** | 1.0850 |
| teacher | 1.2001 | **1.1482** | 1.0948 |

| head | the re-weight<br>k=0 to x4 | the horizons<br>x4 to k=3 | total<br>k=0 to k=3 | the re-weight's share |
|---|---|---|---|---|
| student | **-0.0512** [-0.1001, -0.0023] | **-0.0663** [-0.1070, -0.0331] | -0.1175 [-0.1801, -0.0615] | **44%** |
| teacher | **-0.0519** [-0.0987, -0.0066] | **-0.0534** [-0.0874, -0.0237] | -0.1053 [-0.1661, -0.0515] | **49%** |

95% paired dataset-cluster bootstrap, 10,000 resamples, seed 20260809, over
the pair's 97 configs. The resampling unit is the dataset, not the config.

**Plainly: the 4x re-weight does not reproduce most of the win.** It carries
44% of B1's -0.1175 on the student and 49% on the teacher. By the brief's rule
the answer is therefore the depth. But the split is close to even, and both
halves exclude zero on both heads. **Neither half carries the win alone. The
extra horizons are the larger half, narrowly.**

One caveat on the word "depth". `k = 3` spreads its four copies of `L_align`
over t+1..t+4. The control stacks all four on t+1. The second segment is
therefore the extra HORIZONS at held total weight, not depth net of everything
else.

The re-weight's own effect sits at the longer horizons: -0.1400 [-0.2267,
-0.0662] on medium and long, and -0.0009 [-0.0483, +0.0460] on short.

The same control on A3 moves the score the wrong way, +0.0401 [+0.0116,
+0.0767] on the student. A3's four points each trained on a different box from
at least one other, so A3 gives a direction and not a magnitude. B1 gives
sizes, because B1 holds the machine.

### New this session: the triangle is a controlled comparison

Six earlier passes checked the scores. None checked the factors behind them. A
44/56 split only means something if the three B1 points differ in the
objective and in nothing else.

[`scripts/b1_triangle.py`](https://github.com/jeremycochoy/contrastive-forecasting/blob/feature/contrastive-forecasting-373/reports/2026-08-08_rollout_depth/scripts/b1_triangle.py)
reads each of the six points from its own artefacts.

| factor | across the six B1 points |
|---|---|
| backbone seed | HELD, 20260520 |
| backbone stop | HELD, 40k |
| head seed | HELD, 20260722 |
| head budget | HELD, 15,000 steps |
| eval strategy | HELD, B4 |
| forecast length | HELD, 16 |
| panel | HELD, 97 configs |
| seasonal-naive column | HELD, worst gap 6.2e-05 |
| machine | HELD, elisa |
| the objective | MOVES, by design |

It also re-derives all six scores from the raw per-config CSVs against the
shared seasonal-naive reference. All six match to 4 decimals. The head seed
was the untested one: it is held, so the 44/56 split is not a head-draw
artefact.

`verify_alignx4.py` reads the trained artefact rather than the launcher's
intent. At step 1 the control's loss sits +3.73116 above the baseline's, which
is 3x`L_align`(1), so the flag reached the trainer. The control's losses CSV
carries no `cos_err_d*` column, so it has depth 0. All three ran 40,000 steps.

## Item 6 — the A3 bb200k student redraw

The reseed completed. Head training started 13 Aug 16:02:54 on elisa GPU 1,
seed 20260723, 30,000 steps, rc=0 at 17:08:52. The 97-config eval returned
rc=0 at 18:20:31 and printed 1.4098.

| draw | head seed | machine | score |
|---|---|---|---|
| 1 | 20260722 | the box | 1.3998 |
| 2 | 20260723 | elisa | **1.4098** |
| teacher | 20260722 | the box | 1.2913 |

| contrast | delta | 95% CI | better in |
|---|---|---|---|
| draw 2 against draw 1 | **+0.0100** | [-0.0163, +0.0378] | 22.7% |
| teacher against draw 1 | **-0.1084** | [-0.1648, -0.0671] | 100% |
| teacher against draw 2 | **-0.1185** | [-0.1819, -0.0718] | 100% |

**The two draws agree.** They sit 0.0100 apart, 26% of the ±0.0384 head-seed
band, and their interval covers zero. 1.3998 is not a bad draw.

**The student/teacher gap survives both draws.** Each loses to the teacher by
about 0.11, and both intervals exclude zero on 100% of resamples. The gap
belongs to that student encoder, not to the draw. Draw 1 and the teacher
trained on the same box, so their 0.1084 holds the machine. Draw 2 crosses a
machine, and `verify_close` marks it.

Both draws read the same 200,000-step backbone. Only elisa's copy carries a
recorded md5, `9f0e8da71ff595523d2bf0dabdf80445`. The box was released before
its original could be checksummed.

## The review's ten items, all closed

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | recomputed from one list in `tables.py`: 7 of 16 improved, mean +0.0079, median +0.0042, band covers 13 of 16. Opener, section and ladder table read that one list |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | the depth-response table states it answers the criterion for 2 machine-held arms at one stop. The screen runs over all 41 published pairs, 25 of 41 meet it. All 41 deltas carry a 95% interval |
| 3 | in 12 of 14 cells `k = 3` is a depth change AND a 4x re-weight of `L_align` | **the B1 control above.** 44/56 student, 49/51 teacher, both halves exclude zero, every other factor held |
| 4 | the extend rule reads a confounded contrast, fires inside its own band, its overrides go one way | the stop-reason table states all three by name and number. The 200k verdict reads conditional on a panel selected for having improved |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied. Backbone-seed variance stays unmeasured |
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

GM-Relative MASE, 97 GIFT-Eval configs, strategy B4, forecast length 16.
S = student head, T = teacher head.

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

‡ A1 and B3 hold one student model, printed twice per stop.
`score_A1_k3_bb40k_student.txt` and `score_B3_k3_bb40k_student.txt` both read
1.1305, which is that one measurement. The 72 deliverables therefore hold 70
distinct measurements. Their teacher columns are two models and stay counted
twice.

Controls and replicates outside the grid, all at bb40k unless marked:

| cell | student | teacher |
|---|---|---|
| B1 k=0 | 1.2025 | 1.2001 |
| B1 k=0, `L_align` x4 | **1.1513** | **1.1482** |
| A3 k=0 | 1.2189 | 1.2184 |
| A3 k=0, `L_align` x4 | 1.2590 | 1.2558 |
| A3 k=1 | 1.1995 | 1.2063 |
| B5 k=0 | 1.3917 | 1.3719 |
| B5·s2 k=0 | 1.2716 | 1.2661 |
| B5·s3 k=0, the elisa retrain | 1.2751 | not trained |
| B5·pub k=0, the parent's checkpoint | 1.2751 | not trained |
| B9 k=0 | 1.5583 | 1.5599 |
| A3 bb200k student, draw 2 | **1.4098** | — |

## Runs completed

```
this session   0 backbones, 0 heads, 0 evals, $0.00
study total   99 evals, 97 GIFT-Eval configs each, strategy B4, horizon 16
              72 of 72 deliverables over 14 cells x 3 stops x 2 heads
failed         0
```

`vastrun-balance` reads **$11.45**, unchanged. Nothing was rented. Everything
ran on elisa. No cell that carries a score was retrained.

Elisa's two cards carry other studies right now: card 0 holds 22.5 GB with
~2 GB free, card 1 runs at 75%. This session therefore added no training job.

## The one thing still unmeasured, for a human to decide

The second segment above is the extra horizons at held total weight, and it is
read by subtraction. The 2x2 has three corners: `k = 0` at x1, `k = 0` at x4,
`k = 3` at x4-total. The fourth corner is `k = 3` with `--align-loss-weight
0.25`, which adds the horizons and holds the total weight. That corner would
measure the second segment instead of inferring it.

It costs one 40k backbone (2 h 40 m on elisa, the control's own wall time),
two 15k heads and two 97-config evals. About 5 hours, $0, elisa only. It is
not needed for any claim the report now makes, because both segments already
exclude zero on both heads. Say the word and it runs.

**Nothing is running. Nothing is rented. This PR needs a human decision.**
