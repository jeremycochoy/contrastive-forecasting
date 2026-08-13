«Agent ExperimentRunner claude-opus-5 writing»

## The full-study review closes: ten items, two of them measured

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
- results `results/`, plots `plots/`, scripts `scripts/`
- 99 score files on the branch, each with its own 97-config eval
- run tree `/home/jupyter/cf373_r3`, controls under `checkpoints_backup/cf-373/`

### Runs completed this session

```
backbones  1  G_B1_k0_aw4, 40,000 steps, elisa, seed 20260520, L_align x4 at k = 0
heads      3  2 on that backbone (15,000 steps, seed 20260722)
              1 A3 bb200k student redraw (30,000 steps, seed 20260723)
evals      3  97 GIFT-Eval configs each, strategy B4, horizon 16
failed     0
rented     nothing
```

### Headline numbers

```
item 3  B1 k = 0            1.2025 student   1.2001 teacher
item 3  B1 k = 0, L_align x4  1.1513 student   1.1482 teacher
item 3  B1 k = 3            1.0850 student   1.0948 teacher
item 6  A3 bb200k student, draw 1 seed 20260722   1.3998
item 6  A3 bb200k student, draw 2 seed 20260723   1.4098
item 6  A3 bb200k teacher,        seed 20260722   1.2913
```

### The review's ten items, one line each

`report` = `reports/2026-08-08_rollout_depth/rollout_depth.md`.

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | recomputed from one list in `tables.py`: **7 of 16 improved**, mean +0.0079, median +0.0042, band covers 13 of 16. The opener, the section and the ladder table now read that one list, so the count cannot drift from the table again. The A4 pair is in it |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | the depth-response table now states it answers the card's criterion for **2 machine-held arms at one stop**. The same criterion runs over all 41 published pairs as a screen (`results/criterion_screen.csv`, 25 of 41 meet it, 10 of 18 at bb100k). **All 41 published-baseline deltas now carry a 95% paired dataset-cluster interval** (`published_bootstrap.py`), each parent CSV admitted only after it reproduces its parent's printed number to four decimals |
| 3 | in 12 of 14 cells `k = 3` is a depth change AND a 4x re-weight of `L_align` | **new run below** — the `L_align x4` control at `k = 0` on B1 |
| 4 | the extend rule reads a confounded contrast, fires inside its own band, and its overrides go one way | the stop-reason table states all three, by name and by number, and the 200k verdict now reads **conditional on a panel selected for having improved** |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied: backbone-seed variance is unmeasured and every verdict rests on a band that bounds one of the two seeds in play |
| 6 | A3's bb200k student is an outlier and no one re-measured it | **new run below** — the same head drawn a second time |
| 7 | the headline counts A1 and B3 as two cells; their student column is one measurement | the count is now over distinct MODELS: **13 student models, 8 better, 3 flat, 2 worse**. The shared row is marked `‡` and counted once |
| 8 | only `k = 3` ran on the 14 cells | the report states it supports *depth 3 moves the score* and NOT *depth 3 is the right depth*, in the opener and at the table |
| 9 | the cost of the depth is unknown within an order of magnitude | A3's +13% row is marked **not comparable** and dropped from the carried range. The report carries **+157% to +168%**, the two probes that agree |
| 10 | the ladder caption forbids the comparison the headline makes | the caption now names the one table that reads the dashes and labels it a screen; the verdict table carries the same warning where it is stated |

Minors, all closed: the fidelity batch's not-held-out caveat now sits next to
the claim; the mechanism sections name their four-cell diagnostic set; the
`--grad-clip 1.0` exemption names the project rule it departs from and why;
the "k = 3 still leads at 200k" line is gone, replaced by 3 of 4 lead and B2
loses by more than any of the three gains. B5·s3's missing teacher head stays
disclosed in the annex, no action.

## Item 3 — the decisive control

### B1: is the win the depth, or the weight?

B1 carries `L_align` as its only f-bearing term, so its `k = 3` run multiplies that term's weight against the f-free terms by 4 as well as adding depth. The `L_align x4` row applies the re-weighting at k = 0, with no depth at all.

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

## Item 6 — the redrawn head

### A3's bb200k student, drawn twice

A3 at bb200k reads 1.3998 on the student and 1.2913 on the teacher, off one backbone file. That 0.1084 gap is 6.5x the next-largest in group A (0.0168) and 2.6x the largest anywhere (0.0425). Every gap in the grid is in [`results/head_gap.tsv`](results/head_gap.tsv).

The second draw changes the head seed and nothing else: same backbone file, same 30,000 steps, same recipe, same 97-config eval.

| draw | head seed | GM-Relative MASE | against draw 1 |
|---|---|---|---|
| 1, student | 20260722 | 1.3998 | — |
| 2, student | 20260723 | 1.4098 | +0.0100 |
| teacher | 20260722 | 1.2913 | -0.1084 |

**The two draws agree.** They sit 0.0100 apart [-0.0163, +0.0378], 26% of the ±0.0384 head-seed band, and the second draw is the higher of the two. So 1.3998 is not a bad draw. The interval covers zero, and its far end lands on the imported band, so this head behaves like the heads that band was measured on.

The student/teacher gap survives the redraw at 0.1185, 3.1x the band. Two head seeds put A3's bb200k student above its teacher, so the gap is a property of that student encoder and not of the draw.

The ladder's largest move reads +0.1088 [+0.0656, +0.1667] off the second draw, against +0.0988 off the first. Both exclude zero.

A3's is also the ladder's largest reversal, but it is not the only one: 5 of the 8 three-stop student trajectories turn round at bb200k.

| cell | bb40k | bb100k | bb200k | bb200k − bb100k | shape |
|---|---|---|---|---|---|
| A2 | 1.2735 | 1.2479 | 1.2507 | +0.0028 | turns round |
| A3 | 1.3618 | 1.3010 | 1.3998 | +0.0988 | turns round |
| A4 | 1.0862 | 1.0801 | 1.0660 | -0.0141 | monotone |
| B1 | 1.0850 | 1.0881 | 1.1009 | +0.0128 | monotone |
| B2 | 1.3976 | 1.3443 | 1.2904 | -0.0539 | monotone |
| B4 | 1.3334 | 1.2804 | 1.3182 | +0.0378 | turns round |
| B6 | 1.2297 | 1.2151 | 1.2207 | +0.0056 | turns round |
| B10 | 1.2669 | 1.2403 | 1.2624 | +0.0221 | turns round |


The redraw does not land near the teacher's 1.2913 and it does not move
toward it. So the two lines the review put at risk both stand:

- **A3's student degrades at bb200k.** 1.3618 → 1.3010 → 1.3998, and →
  1.4098 on the second draw. The ladder's +0.0988 is +0.1088 read off draw 2.
- **A3's student/teacher gap is real.** 0.1084 on draw 1, 0.1185 on draw 2,
  against a next-largest of 0.0168 in group A and 0.0425 anywhere. Two head
  seeds put that student above its teacher by ~3x the band, so the gap is a
  property of that student encoder rather than of one head draw.

What this closes, and what it does not. It removes "one bad draw" as the
explanation. It does not explain the gap: A3 is the cell where `k = 3` does
the most damage, and this study has no second backbone seed on it.

## Coverage — 14 cells x 3 stops x 2 heads

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

deliverables 72   done 72   running 0   queued 0   NOT STARTED 0   (+12 stops, not deliverables)
done=number in hand  run=own head/eval running  bb-run=backbone training now  plan=queued, not started  MISS-e=eval not run  MISS-h=head not trained  MISS-t=backbone not trained  stop=not a deliverable this round
‡ A1 and B3 hold one student model, printed twice per stop. The 72 deliverables therefore hold 70 distinct measurements. Their teacher columns are two models and are counted twice.

## What the study can and cannot support

The review's own list, re-read against the closed items and the two runs.

**Can support.**

- Training the forecaster on its own output at depth 3 moves GM-Relative MASE
  by more than the head seed does, in most cells, in both directions.
- One machine-held, seed-held, head-budget-matched pair exists in the grid:
  B1 at bb40k, -0.1175, CI [-0.1801, -0.0615].
- On B1, the one cell where the depth wins, the re-weighting that comes with it carries 44% of the student's -0.1175. Holding `L_align`'s total weight at 4 and dropping the depth to 0 reads 1.1513 against 1.2025.
- The composed operator's rollout fidelity rises with depth on the four arms
  measured, including two whose score falls. Depth changes the operator and
  the score does not follow it.
- Coverage: all 14 recipes train and score at k = 3, at every stop they were
  meant to reach, on both heads. 72 of 72 deliverables, no cell failed.
- Every delta against a published k = 0 now carries a 95% paired
  dataset-cluster interval. All 41 of them, each parent CSV admitted only
  after it reproduced its parent's printed number to four decimals.
- A3's bb200k student is not one bad head draw. Two seeds, 1.3998 and 1.4098.

**Cannot support.**

- That either the depth or the re-weighting alone wins on B1. The control splits the move between them and one cell cannot say which generalises.
- Any per-cell verdict. Every cell is n = 1 in the backbone seed, and the
  ±0.0384 band used to judge it bounds the HEAD seed. Backbone-seed variance
  is unmeasured everywhere in this study.
- "9 of 14 better" as a rate. It is 8 of 13 distinct student models, judged
  against baselines this study did not retrain on its own machine, at a
  threshold whose band bounds a different seed. The report labels it a
  screen.
- Whether depth 3 helps at bb100k or bb200k. No cell holds a same-machine,
  same-seed `k = 0` at either stop. The one clean pair is at bb40k, and that
  cell then gets worse with more backbone steps: B1 student 1.0850 → 1.0881
  → 1.1009.
- "The second 100,000 steps buy nothing" as a general claim. The panel is
  selected on an improving first leg and the two hand overrides went the same
  way. Read it as conditional on that panel. Within it: 7 of 16 improved,
  mean +0.0079, median +0.0042, band covers 13 of 16.
- Any ranking of the 14 recipes. The better/worse split tracks each cell's
  published baseline as much as its own `k = 3` number.
- That depth 3 is the right depth. Only `k = 3` ran on the 14 cells. `k = 1`
  ran on A3 alone.
- What the depth costs, to better than +157% to +168%. Two probes agree
  there; A3's +13% row crosses a box change and is dropped.

### Spend

Credit **$11.45**, floor $5.50. Nothing was rented this session: the backbone, the three heads and the three evals all ran on elisa.
vast.ai reports: `No running instances found.`
