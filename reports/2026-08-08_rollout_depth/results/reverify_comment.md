«Agent ExperimentRunner claude-opus-5 writing»

## Both flagged items had already landed. This session re-derived them from the raw artefacts.

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(`results/`, `plots/`, `scripts/`, `sync/`)

The review reported item 3 stalled one step from its answer, with no eval run
and no score file. That state was true when it was written and is no longer
true. The two `G_B1_k0_aw4` evals ran on elisa between 20:32 and 22:40 on
Aug 13, and item 6's redraw ran earlier the same day. Both are on the branch.

This session trained nothing, evaluated nothing and rented nothing. It
re-derived every number the two items rest on, from the per-config CSVs, and
re-ran all five checks.

### Runs completed

```
this session   0 training runs, 0 evals, $0.00, nothing rented
study total   99 evals, 97 GIFT-Eval configs each, strategy B4, horizon 16
              99 score files, one per eval
              72 of 72 deliverables over 14 cells x 3 stops x 2 heads
failed         0
```

`vastrun-status` returns "No running instances found." `vastrun-balance`
reads **$11.45**, unchanged, floor $5.50. Nothing on the GPUs belongs to
this study.

### Headline numbers

```
item 3  B1 k = 0                     1.2025 student   1.2001 teacher
item 3  B1 k = 0, L_align x4         1.1513 student   1.1482 teacher   <- the control
item 3  B1 k = 3                     1.0850 student   1.0948 teacher
item 6  A3 bb200k student, draw 1    1.3998   seed 20260722, on the box
item 6  A3 bb200k student, draw 2    1.4098   seed 20260723, on elisa   <- the redraw
item 6  A3 bb200k teacher            1.2913   seed 20260722, on the box
```

## Item 3 — the decisive control, and its verdict

The review set the test: if the ×4 re-weight alone reproduces most of B1's
-0.1175, the win is the weight; if it does not, the win is the depth.

**It reproduces 44%. So by that test the win is not the weight.** The honest
reading is narrower than the test allows: depth carries 56% and the weight
carries 44%, both intervals exclude zero, and the two shares sit inside each
other's intervals. Neither segment alone reproduces the win, and this cell
cannot rank them.

| head | k = 0 | k = 0, `L_align` ×4 | k = 3 | the re-weighting<br>k = 0 → ×4 | the depth<br>×4 → k = 3 | the weight's share |
|---|---|---|---|---|---|---|
| student | 1.2025 | 1.1513 | 1.0850 | -0.0512 | -0.0663 | 44% |
| teacher | 1.2001 | 1.1482 | 1.0948 | -0.0519 | -0.0534 | 49% |

95% paired dataset-cluster intervals over the 97 eval configs, re-derived this
session from the per-config CSVs:

- student: re-weighting **-0.0512 [-0.1001, -0.0023]**, depth **-0.0663 [-0.1070, -0.0331]**, total **-0.1175 [-0.1801, -0.0615]**
- teacher: re-weighting **-0.0519 [-0.0987, -0.0066]**, depth **-0.0534 [-0.0874, -0.0237]**, total **-0.1053 [-0.1661, -0.0515]**

**The two segments do not buy the same horizons.** Split by horizon, the
re-weighting's whole gain sits at medium and long, and depth pays at both.

| segment, student | short (n=55) | medium+long (n=42) |
|---|---|---|
| the re-weighting | -0.0009 [-0.0483, +0.0460] | -0.1400 [-0.2267, -0.0662] |
| the depth | -0.0547 [-0.0840, -0.0310] | -0.0845 [-0.1783, -0.0136] |

The control is clean. `verify_alignx4` reads the flag off the loss curve
rather than the launcher: the ×4 arm sits +3.73116 above `k = 0` at step 1,
which is 3 × `L_align(1)`, it carries no `cos_err_d*` column, so no depth, and
all three arms logged 40,000 steps. Every column trained on elisa at backbone
seed 20260520, then 15,000 head steps at seed 20260722, then the same 97
configs. `verify_provenance` marks all six item-3 pairs machine-held.

What the control cannot separate: `k = 3` puts its four copies of `L_align` on
t+1..t+4 and the control puts all four on t+1. The depth segment is therefore
the extra HORIZONS at a held total weight, not depth net of everything else.

## Item 6 — the A3 redraw: it completed

It never started under the queue, which is why the review found no log there.
`gap_worker.sh` ran it outside the queue. It has since completed, and the
timings are in its own artefacts: head training started 16:02:54 on elisa GPU
1 at seed 20260723, the eval merged 97 configs at 18:20:29, and the score
committed at 18:24. Both draws trained 30,000 head steps, the study's budget
for a 200k stop.

| | draw 1 | draw 2 | teacher |
|---|---|---|---|
| head seed | 20260722 | 20260723 | 20260722 |
| head trained on | the rented box | elisa | the rented box |
| score | 1.3998 | 1.4098 | 1.2913 |

Both draws, against the original and against the teacher:

| contrast | Δ | 95% paired interval |
|---|---|---|
| draw 2 against the original **1.3998** | **+0.0100** | [-0.0163, +0.0378] |
| draw 1 against the teacher **1.2913** | **+0.1084** | [+0.0671, +0.1648] |
| draw 2 against the teacher **1.2913** | **+0.1185** | [+0.0718, +0.1819] |

**1.3998 is not a bad draw.** The redraw sits 0.0100 above it, 26% of the
±0.0384 head-seed band, and it is the higher of the two. The interval covers
zero and its far end lands on the imported band.

**The student/teacher gap survives both draws.** Draw 1 sits 0.1084 above the
teacher and draw 2 sits 0.1185 above it. Both intervals exclude zero. So the
gap belongs to the student encoder, not to the draw.

The redraw changes the machine as well as the seed, so its 0.0100 bounds the
two together. Draw 1 and the teacher both trained on the box, so their 0.1084
is the machine-held one. `plots/a3_reseed.png` names each point's machine.

## The five checks, all re-run this session

| check | what it reads | result |
|---|---|---|
| `verify_scores` | every score against its own 97-config eval, two ways | **99 of 99 reproduce**, worst deviation 5.13e-05 against a 1.07e-04 derived allowance |
| `verify_coverage` | the 14 × 3 × 2 grid, rebuilt from the score files alone | **72 of 72**, none missing, none extra |
| `verify_alignx4` | item 3's ×4 weight, off the loss curve rather than the launcher | **the flag reached the objective**, +3.73116 at step 1, no `cos_err_d*` column |
| `verify_provenance` | each head's training machine, from the backbone path in its own log | **item 3 is machine-held, item 6's two pairs cross** |
| `verify_denominator` | the `SN_MASE` column, across evals rather than within one | **one denominator over 99 evals**, md5 `a86ef40144eee950866b027d876ce75e` |

`scripts/verify_close.sh` runs all five. All five pass. Every artefact it
rewrites came back byte-identical, so nothing moved since the close.

All nine intervals quoted above were recomputed this session with
`paired_bootstrap.py` straight from `all_results.csv`. Every one matches the
report to the printed decimal.

## The review's ten items, all closed

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | recomputed from one list in `tables.py`: **7 of 16 improved**, mean +0.0079, median +0.0042, band covers 13 of 16 |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | the criterion runs over all 41 published pairs as a screen (25 of 41 meet it); **all 41 deltas carry a 95% paired interval** |
| 3 | in 12 of 14 cells `k = 3` is a depth change AND a 4× re-weight of `L_align` | **measured** — the control above. Both segments pay, 44% / 56% |
| 4 | the extend rule reads a confounded contrast and fires inside its own band | all three faults named by number; the 200k verdict reads conditional on a panel selected for having improved |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied |
| 6 | A3's bb200k student is an outlier and no one re-measured it | **measured** — 1.4098 on a second seed. Its provenance is corrected |
| 7 | the headline counts A1 and B3 as two cells; their student column is one measurement | the count is over distinct MODELS: **13 student models, 8 better, 3 flat, 2 worse**. The shared row is marked `‡` and counted once |
| 8 | only `k = 3` ran on the 14 cells | the report supports *depth 3 moves the score*, not *depth 3 is the right depth* |
| 9 | the cost of the depth is unknown within an order of magnitude | A3's +13% row is dropped as not comparable; the report carries **+157% to +168%** |
| 10 | the ladder caption forbids the comparison the headline makes | the caption names the one table that reads the dashes and labels it a screen |

Minors, all closed: the fidelity batch's not-held-out caveat sits next to the
claim; the mechanism sections name their four-cell diagnostic set; the
`--grad-clip 1.0` exemption names the project rule it departs from and why;
the "k = 3 still leads at 200k" line is gone. B5·s3's missing teacher head
stays disclosed in the annex.

## Coverage — 14 cells × 3 stops × 2 heads

Rebuilt from the score files alone by `verify_coverage`.

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
(+12 stops, not deliverables this round)
```

‡ A1 and B3 hold one student model, printed twice per stop, and counted once.
The 72 deliverables therefore hold 70 distinct measurements. Their teacher
columns are two models and stay counted twice.

Plus the three controls outside the grid: `G_B1_k0_aw4` student and teacher,
and `A3_k3_bb200k_student_s20260723`. 99 score files in all.
