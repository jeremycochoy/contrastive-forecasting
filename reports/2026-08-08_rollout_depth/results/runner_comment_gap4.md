«Agent ExperimentRunner claude-opus-5 writing»

## Item 3 answers: the re-weight is not most of the win. B1 cannot promote the depth over it either.

**Experiment directory:** `reports/2026-08-08_rollout_depth/` (`results/`, `plots/`, `scripts/`, `sync/`, `curves/`)
**Commit:** `b9e3f376`

### Runs completed

```
this session   0 training runs, 0 evals, $0.00, nothing rented
study total    99 evals, 97 GIFT-Eval configs each, strategy B4, horizon 16
               99 score files, one per eval
               14 of 14 cells scored, 72 of 72 deliverables
               (36 cell-stops x 2 heads; the extend rule holds 6 cells at bb100k)
in flight      1 backbone, the fourth 2x2 corner, on elisa card 1
failed         0
```

`vastrun-balance` stays **$11.45**. No cell that carries a score was retrained.

### The two flagged jobs were already on disk

Both had completed before this session opened. The review's snapshot was
taken before they landed.

| item | artefact | written | evidence |
|---|---|---|---|
| 3 | `score_G_B1_k0_aw4_bb40k_teacher.txt` | 08-13 22:07 | `results/eval/G_B1_k0_aw4_bb40k_teacher/eval_local.log` |
| 3 | `score_G_B1_k0_aw4_bb40k_student.txt` | 08-13 22:40 | `results/eval/G_B1_k0_aw4_bb40k_student/eval_local.log` |
| 6 | `score_A3_k3_bb200k_student_s20260723.txt` | 08-13 18:20 | `results/gap6_a3_reseed.out` |

The reseed log the review could not find is `results/gap6_a3_reseed.out`. It
reads: head-train start 16:02, `rc=0` at 17:08, eval start 17:08, `rc=0` at
18:20, `DONE — GM-Relative MASE 1.4098`.

Both aw4 evals ran on the aw4 backbone's own 40k checkpoint, strategy B4,
`forecast_len=16`, 97 configs, on the one shared seasonal-naive denominator.
`scripts/runner_verify_20260814.py` re-derived all nine cells from the raw
per-config CSVs this session. The worst per-cell gap against the shared
denominator is 6.2e-05. Every score, every delta and every interval matched.

### One label correction, because it changes the arithmetic

The review quoted B1 k=0 as 1.0850 student / 1.0948 teacher. Those are B1
**k = 3** at bb40k. The k=0 baseline is **1.2025 / 1.2001**
(`G6_B1_k0_bb40k`). The -0.1175 the review quotes runs 1.2025 -> 1.0850,
which is consistent with the k=3 reading and not with the k=0 one.

### Item 3 — the verdict

The review's test: if the ×4 re-weight alone reproduces **most** of B1's
-0.1175, the win is the weight; if not, the win is the depth.

**It reproduces 44%. The win is not the weight.** It does not follow that
the win is the depth. Here is the number that decides it, which the report
did not carry until now:

| head | depth segment | weight segment | depth minus weight | ranks them? |
|---|---|---|---|---|
| student | -0.0663 | -0.0512 | **-0.0150 [-0.0844, +0.0414]** | no |
| teacher | -0.0534 | -0.0519 | **-0.0015 [-0.0609, +0.0494]** | no |

Both intervals cover zero. The depth segment is larger in 67.8% of
resamples on the student and 51.6% on the teacher, which is not a ranking.
So: the re-weight is not most of the win, and B1 cannot say the depth beats
it. Both segments are real, and this cell sizes neither above the other.
(`results/gap4_which_bigger.csv`)

The four corners and the two segments:

| head | k = 0 | k = 0, `L_align` ×4 | k = 3 | the re-weight | the depth | weight's share |
|---|---|---|---|---|---|---|
| student | 1.2025 | 1.1513 | 1.0850 | -0.0512 | -0.0663 | 44% |
| teacher | 1.2001 | 1.1482 | 1.0948 | -0.0519 | -0.0534 | 49% |

95% paired dataset-cluster intervals over the 97 configs, re-derived this
session from `all_results.csv`:

- student: re-weight **-0.0512 [-0.1001, -0.0023]** (97.8%), depth **-0.0663 [-0.1070, -0.0331]** (100%), total **-0.1175 [-0.1801, -0.0615]** (100%)
- teacher: re-weight **-0.0519 [-0.0987, -0.0066]** (98.6%), depth **-0.0534 [-0.0883, -0.0240]** (100%), total **-0.1053 [-0.1661, -0.0515]** (100%)

Both segments exclude zero on both heads. The re-weight's own effect is
entirely at the longer horizons: **-0.1400 [-0.2267, -0.0662]** on medium
and long, **-0.0009 [-0.0483, +0.0460]** on short.

**One caveat on the depth segment.** It is `k3 - w`, a subtraction, and that
is the horizon effect only if the two changes add. Nothing measured so far
tests that. The fourth corner is training now to test it (below).

### Item 6 — both draws

| draw | head seed | machine | GM-Relative MASE | against draw 1 |
|---|---|---|---|---|
| 1, student | 20260722 | rented box | 1.3998 | — |
| 2, student | 20260723 | elisa | 1.4098 | +0.0100 [-0.0163, +0.0378] |
| teacher | 20260722 | rented box | 1.2913 | -0.1084 [-0.1648, -0.0671] |

**The two draws agree.** They sit 0.0100 apart, 26% of the ±0.0384
head-seed band, and the interval covers zero. 1.3998 is not a bad draw.

**The student/teacher gap survives the redraw.** Teacher beats draw 1 by
-0.1084 [-0.1648, -0.0671] and draw 2 by -0.1185 [-0.1819, -0.0718], each
on 100% of resamples. Two head seeds put A3's bb200k student above its
teacher, so the gap belongs to that student encoder and not to the draw.

Held across the two draws: the same 200,000-step backbone file (elisa's copy
carries md5 `9f0e8da7…`; the box was released before its original could be
checksummed), 30,000 head steps, the recipe, and the 97-config eval, which
ran on elisa's cores for both. The two draws also sit on two machines, so
this agreement bounds the head seed and the machine together, not the seed
alone. The report states that where the numbers are.

### The ten review items

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | one list in `tables.py` feeds opener, section and ladder: **7 of 16 improved**, mean +0.0079, median +0.0042, band covers 13 of 16 |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | the depth-response table now states it answers the criterion for **2 machine-held arms at one stop**; the criterion runs over all 41 pairs as a screen (25 of 41 meet it). **All 41 published-baseline deltas carry a 95% paired interval**, each parent CSV admitted only after it reproduces its parent's printed number to four decimals |
| 3 | in 12 of 14 cells `k = 3` is a depth change AND a 4× re-weight of `L_align` | the `L_align` ×4 control at `k = 0` on B1, scored **1.1513 / 1.1482**, plus the segment-difference interval above. The fourth corner is training to replace the subtraction with a measurement |
| 4 | the extend rule reads a confounded contrast, fires inside its own band, its overrides go one way | the stop-reason table states all three by name and number; the 200k verdict reads **conditional on a panel selected for having improved** |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied: backbone-seed variance is unmeasured |
| 6 | A3's bb200k student is an outlier and no one re-measured it | drawn a second time at seed 20260723: **1.4098**, +0.0100 [-0.0163, +0.0378] |
| 7 | the headline counts A1 and B3 as two cells; their student column is one measurement | the count is over distinct MODELS: **13 student models, 8 better, 3 flat, 2 worse**. The shared row is marked `‡` and counted once |
| 8 | only `k = 3` ran on the 14 cells | the report supports *depth 3 moves the score*, not *depth 3 is the right depth*, in the opener and at the table |
| 9 | the cost of the depth is unknown within an order of magnitude | A3's +13% row is marked **not comparable** and dropped; the report carries **+157% to +168%**, the two probes that agree |
| 10 | the ladder caption forbids the comparison the headline makes | the caption names the one table that reads the dashes and labels it a screen |

Minors, all closed: the fidelity batch's not-held-out caveat sits next to the
claim; the mechanism sections name their four-cell diagnostic set; the
`--grad-clip 1.0` exemption names the project rule it departs from and why;
the "k = 3 still leads at 200k" line is gone. B5·s3's missing teacher head
stays disclosed in the annex, no action.

### Final coverage

The card names 14 cells. This study scored **14 of 14**. Every cell carries a
number.

| cell | f-bearing term | EMA α | depths trained | stops scored |
|---|---|---|---|---|
| A1 | L_align only | scheduled | k = 3 | bb40k, bb100k |
| A2 | L_align + CPC auxiliary | scheduled | k = 3 | bb40k, bb100k, bb200k |
| A3 | L_align only | scheduled | k = 0, k = 1, k = 3 | bb40k, bb100k, bb200k |
| A4 | L_align only | scheduled | k = 3 | bb40k, bb100k, bb200k |
| B1 | L_align only | fixed 0.9 | k = 0, k = 3 | bb40k, bb100k, bb200k |
| B2 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |
| B3 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k |
| B4 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |
| B5 | pooled xshh_allt, floor subtracted | fixed 0.9 | k = 0, k = 3 | bb40k, bb100k |
| B6 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |
| B7 | L_align only | fixed 0.9 | k = 3 | bb40k, bb100k |
| B8 | L_align + CPC auxiliary | fixed 0.9 | k = 3 | bb40k, bb100k |
| B9 | split L_pred + CPC auxiliary | fixed 0.9 | k = 0, k = 3 | bb40k, bb100k |
| B10 | L_align + CPC auxiliary | fixed 0.9 | k = 3 | bb40k, bb100k, bb200k |

Plus the review's controls, outside the 14: B1's `L_align` ×4 at k = 0, A3's
same control, A3's k = 1 rung, A3's bb200k student redraw, B5's three
backbones, and the A1/B3 duplicate re-run end to end.

```
99 evals, every one over exactly 97 configs      99 / 99
score files, one per eval                        99
cells with a number                              14 / 14
deliverables, 36 cell-stops x 2 heads            72 / 72
evals missing a per-config CSV                   0
```

One eval directory holds no CSV: `G7_B5_k0_e_bb40k_teacher`. That is B5·s3's
missing teacher head, disclosed in the annex, no action.

### Still running: the fourth 2×2 corner

The depth segment above is `k3 - w`, a subtraction that assumes the two
changes add. `G_B1_k3_aw025` measures the same segment directly.

| total `L_align` weight | horizons | cell | student |
|---|---|---|---|
| 1× | t+1 | B1 k = 0 | 1.2025 |
| 4× | t+1 | B1 k = 0, aw4 | 1.1513 |
| 4× | t+1..t+4 | B1 k = 3 | 1.0850 |
| **1×** | **t+1..t+4** | **G_B1_k3_aw025** | **training** |

`k = 3` sums four copies of `L_align`, one per depth, each at
`--align-loss-weight`. 0.25 × 4 returns the total to 1.0, which is what the
k = 0 baseline carries, so the horizons are the only thing that moved.
`scripts/gap4_2x2.py` reads the four corners and reports the interaction
`k3 - w - h + k0`. An interval covering zero means the two changes add and
the published split stands.

Backbone 40,000 steps at seed 20260520 on elisa card 1, 3.7 steps/s, then two
15,000-step heads at seed 20260722 and two 97-config evals. Nothing rented.
I will post the measured segment and its interval when both evals land.
