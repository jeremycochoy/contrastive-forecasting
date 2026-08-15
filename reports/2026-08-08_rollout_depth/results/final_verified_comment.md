«Agent ExperimentRunner claude-opus-5 writing»

## The study closes: every number re-derived from its own artefacts, one sentence corrected

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(`results/`, `plots/`, `scripts/`, `sync/`)

The two gap runs landed in the sessions above. This session trained nothing,
evaluated nothing and rented nothing. It re-derived the study's numbers from
the raw artefacts, found one wrong sentence, and fixed it at the generator.

### Runs completed, across the whole study

```
score files  99   each re-derived this session from its own eval
evals        99   97 GIFT-Eval configs each, strategy B4, horizon 16
heads        99   one per score, student or teacher, on its own checkpoint
deliverables 72   of 72, over the 14 cells x 3 stops x 2 heads grid
failed        0
this session  0 runs, $0.00, nothing rented
```

### Headline numbers

```
item 3  B1 k = 0                       1.2025 student   1.2001 teacher
item 3  B1 k = 0, L_align x4           1.1513 student   1.1482 teacher
item 3  B1 k = 3                       1.0850 student   1.0948 teacher
item 6  A3 bb200k student, draw 1      1.3998   seed 20260722, on the box
item 6  A3 bb200k student, draw 2      1.4098   seed 20260723, on elisa
item 6  A3 bb200k teacher              1.2913   seed 20260722, on the box
```

## Verification: the numbers re-derive from the artefacts

`scripts/verify_close.sh` runs four checks. Each writes its own log. All pass.

| check | what it reads | result |
|---|---|---|
| `verify_scores` | every score file against its own 97-config eval, two ways: the geometric mean of the per-config `Relative` column in `summary.txt`, and that file's `MASE` column against `eval_metrics/MASE[0.5]` in the harness CSV | **99 of 99 reproduce.** Worst deviation 5.13e-05 against a 1.07e-04 allowance |
| `verify_coverage` | the 14 x 3 x 2 grid rebuilt from the score files alone, so it does not read the queue state that wrote `coverage.md` | **72 of 72.** None missing, none scored that this round did not owe |
| `verify_alignx4` | item 3's x4 weight read off the loss curve, not the launcher | **the flag reached the objective.** +3.73116 over its own `k = 0` at step 1, and no `cos_err_d*` column |
| `verify_provenance` | the training machine of every head, from the backbone path in its own log | **item 3 is machine-held, item 6 is not.** See below |

**The tolerance is derived, not chosen.** `summary.txt` prints to 4 decimals,
so recomputing a geometric mean from that column carries rounding. A first
pass with a flat 5e-5 failed two files at 5.1e-05, which is the print and not
a drift. The check now derives its own allowance,
`bound = GM * mean(5e-5 / r_i) + 5e-5`, and all 99 pass inside it.

**The preflight proved intent; the loss curve proves effect.**
`results/gap3_preflight.txt` records the flags the launcher meant to pass, and
the backbone log does not echo them. So the control was checked against its
trained artefact instead. It shares seed 20260520 and batch order with its own
`k = 0` baseline, and at step 1 it sits +3.73116 above it, so
`L_align(1) ~= 1.244` and the 4x is real. It writes no `cos_err_d*` column
where `k = 3` writes four. The weight moved and the depth did not. All three
columns logged 40,000 steps.

## The correction

The report said A3's second draw "changes the head seed and nothing else".
**It does not.**

| | draw 1 | draw 2 | teacher |
|---|---|---|---|
| head seed | 20260722 | 20260723 | 20260722 |
| head trained on | the rented box | elisa | the rented box |
| backbone read | `/root/cf373_runs/…` | `/home/jupyter/cf373_r3/sync/…` | `/root/cf373_runs/…` |
| score | 1.3998 | 1.4098 | 1.2913 |

A3's 200k leg trained on the box (`queue/bb_A3_200k.machine` = `rem:1`), so
the box held the original backbone and elisa holds the synced copy. Only
elisa's copy carries a recorded md5,
`9f0e8da71ff595523d2bf0dabdf80445`; the box was released before its original
could be checksummed. Both evals ran on elisa's cores over the same 97 configs.

**The verdict holds, and it gains.** Two head seeds on two machines land
0.0100 apart, 26% of the ±0.0384 band. So 1.3998 is not a bad draw, and that
agreement now bounds the head seed and the machine together rather than the
seed alone. The student/teacher gap keeps its machine-held evidence: draw 1
and the teacher both trained on the box, so their 0.1084 holds the machine.
The redraw's 0.1185 crosses machines.

**Item 3 is unaffected.** The sweep checked all 100 eval directories for other
divided pairs that cross machines. Item 3's six columns are all on elisa, and
so are both A3 depth-0 controls. Only item 6's two pairs cross.

```
| pair                                 | left  | right | machine |
| item 3, re-weighting, student        | elisa | elisa | HELD    |
| item 3, depth, student               | elisa | elisa | HELD    |
| item 3, total, student               | elisa | elisa | HELD    |
| item 3, re-weighting, teacher        | elisa | elisa | HELD    |
| item 3, depth, teacher               | elisa | elisa | HELD    |
| item 3, total, teacher               | elisa | elisa | HELD    |
| item 6, head seed, student           | box   | elisa | CROSSES |
| item 6, student vs teacher, draw 1   | box   | box   | HELD    |
| item 6, student vs teacher, draw 2   | elisa | box   | CROSSES |
| A3 depth control, student            | elisa | elisa | HELD    |
| A3 re-weighting control, student     | elisa | elisa | HELD    |
```

49 of the 100 eval directories carry no head log, because rounds 1 and 2 wrote
the machine to their launch logs instead. The check reports that rather than
guessing.

The fix lands in `scripts/tables.py`, the generator, so the report and the
close comment cannot drift apart. `plots/a3_reseed.png`'s legend now names
each point's machine, so the figure carries the caveat on its own.

## The review's ten items

`report` = `reports/2026-08-08_rollout_depth/rollout_depth.md`.

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | recomputed from one list in `tables.py`: **7 of 16 improved**, mean +0.0079, median +0.0042, band covers 13 of 16. Opener, section and ladder table read that one list |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | the depth-response table states it answers the card's criterion for **2 machine-held arms at one stop**. The criterion runs over all 41 published pairs as a screen (`results/criterion_screen.csv`, 25 of 41 meet it). **All 41 deltas carry a 95% paired dataset-cluster interval** |
| 3 | in 12 of 14 cells `k = 3` is a depth change AND a 4x re-weight of `L_align` | **measured** — the control below. Both segments pay |
| 4 | the extend rule reads a confounded contrast and fires inside its own band | the stop-reason table states all three faults by name and number; the 200k verdict reads **conditional on a panel selected for having improved** |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied |
| 6 | A3's bb200k student is an outlier and no one re-measured it | **measured** — the same head drawn again, 1.4098. Its provenance is corrected above |
| 7 | the headline counts A1 and B3 as two cells; their student column is one measurement | the count is over distinct MODELS: **13 student models, 8 better, 3 flat, 2 worse**. The shared row is marked `‡` and counted once |
| 8 | only `k = 3` ran on the 14 cells | the report supports *depth 3 moves the score*, not *depth 3 is the right depth* |
| 9 | the cost of the depth is unknown within an order of magnitude | A3's +13% row is marked not comparable and dropped. The report carries **+157% to +168%** |
| 10 | the ladder caption forbids the comparison the headline makes | the caption names the one table that reads the dashes and labels it a screen |

Minors, all closed: the fidelity batch's not-held-out caveat sits next to the
claim; the mechanism sections name their four-cell diagnostic set; the
`--grad-clip 1.0` exemption names the project rule it departs from and why;
the "k = 3 still leads at 200k" line is gone. B5·s3's missing teacher head
stays disclosed in the annex.

## Item 3 — the decisive control

B1 carries `L_align` as its only f-bearing term, so its `k = 3` run multiplies
that term's weight against the f-free terms by 4 as well as adding depth. The
`L_align x4` row applies the re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 3 | the re-weighting<br>k = 0 → x4 | the depth<br>x4 → k = 3 | share |
|---|---|---|---|---|---|---|
| student | 1.2025 | 1.1513 | 1.0850 | -0.0512 | -0.0663 | 44% |
| teacher | 1.2001 | 1.1482 | 1.0948 | -0.0519 | -0.0534 | 49% |

Intervals, 95% paired dataset-cluster over the 97 eval configs:

- student: re-weighting [-0.1001, -0.0023], depth [-0.1070, -0.0331], total [-0.1801, -0.0615]
- teacher: re-weighting [-0.0987, -0.0066], depth [-0.0874, -0.0237], total [-0.1661, -0.0515]

**Both pay.** The re-weighting carries 44% of the student's -0.1175 and the
extra horizons carry the rest. Neither alone accounts for the win.

Every column trained on elisa at backbone seed 20260520 on the same head
budget: 15,000 head steps at seed 20260722, then 97 GIFT-Eval configs. All six
heads are machine-held, confirmed from their own logs. This is the study's one
machine-held, seed-held, head-budget-matched set, so it may divide one column
by another.

What it cannot separate: `k = 3` puts its four copies of `L_align` on four
horizons and `k = 0` x4 puts all four on t+1. So the depth column is the extra
HORIZONS at a held total weight, not depth net of everything else.

## Coverage — 14 cells x 3 stops x 2 heads

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

‡ A1 and B3 hold one student model, printed twice per stop. The 72
deliverables therefore hold 70 distinct measurements. Their teacher columns
are two models and are counted twice.

## What the study can and cannot support

**Can support.**

- Training the forecaster on its own output at depth 3 moves GM-Relative MASE
  by more than the head seed does, in most cells, in both directions.
- One machine-held, seed-held, head-budget-matched pair exists in the grid:
  B1 at bb40k, -0.1175, CI [-0.1801, -0.0615].
- On B1, the one cell where the depth wins, the re-weighting that comes with
  it carries 44% of the student's -0.1175. Holding `L_align`'s total weight at
  4 and dropping the depth to 0 reads 1.1513 against 1.2025. All six columns
  of that comparison trained on one machine, confirmed from their own logs.
- The composed operator's rollout fidelity rises with depth on the four arms
  measured, including two whose score falls. Depth changes the operator and
  the score does not follow it.
- Coverage: all 14 recipes train and score at k = 3, at every stop they were
  meant to reach, on both heads. 72 of 72 deliverables, no cell failed.
- Every delta against a published k = 0 carries a 95% paired dataset-cluster
  interval. All 41, each parent CSV admitted only after it reproduced its
  parent's printed number to four decimals.
- Every one of the 99 score files re-derives from its own 97-config eval.
- A3's bb200k student is not one bad head draw. Two seeds, on two machines,
  1.3998 and 1.4098.

**Cannot support.**

- That either the depth or the re-weighting alone wins on B1. The control
  splits the move between them and one cell cannot say which generalises.
- That A3's redraw isolates the head seed. It changes the machine too, so its
  0.0100 bounds the two together.
- Any per-cell verdict. Every cell is n = 1 in the backbone seed, and the
  ±0.0384 band used to judge it bounds the HEAD seed. Backbone-seed variance
  is unmeasured everywhere in this study.
- "9 of 14 better" as a rate. It is 8 of 13 distinct student models, judged
  against baselines this study did not retrain on its own machine, at a
  threshold whose band bounds a different seed. The report labels it a screen.
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
- What the depth costs, to better than +157% to +168%. Two probes agree there;
  A3's +13% row crosses a box change and is dropped.

### Spend

Credit **$11.45**, floor $5.50. This session spent $0.00: it trained nothing
and rented nothing. `vastrun-status` returns "No running instances found." The
queue is empty, so no card sits idle on unfinished work.
