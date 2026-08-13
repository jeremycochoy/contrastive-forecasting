«Agent ExperimentRunner claude-opus-5 writing»

## The close holds, and a fifth check now covers the denominator

**Experiment directory:** `reports/2026-08-08_rollout_depth/`
(`results/`, `plots/`, `scripts/`, `sync/`)

Both gap runs had already landed when this session opened. It trained nothing,
evaluated nothing and rented nothing. It re-ran every check, found one thing
the study asserted but never verified, and added the check for it.

### Runs completed

```
this session   0 training runs, 0 evals, $0.00, nothing rented
study total   99 evals, 97 GIFT-Eval configs each, strategy B4, horizon 16
              99 score files, one per eval
              72 of 72 deliverables over 14 cells x 3 stops x 2 heads
failed         0
```

`vastrun-status` returns "No running instances found." Credit **$11.45**,
unchanged, floor $5.50.

### Headline numbers

```
item 3  B1 k = 0                     1.2025 student   1.2001 teacher
item 3  B1 k = 0, L_align x4         1.1513 student   1.1482 teacher   <- the control
item 3  B1 k = 3                     1.0850 student   1.0948 teacher
item 6  A3 bb200k student, draw 1    1.3998   seed 20260722, on the box
item 6  A3 bb200k student, draw 2    1.4098   seed 20260723, on elisa   <- the redraw
item 6  A3 bb200k teacher            1.2913   seed 20260722, on the box
```

## Item 3 — the decisive control, and what it says

B1 carries `L_align` as its only f-bearing term, so its `k = 3` run multiplies
that term's weight against the f-free terms by 4 as well as adding depth. The
`L_align x4` row applies the re-weighting at k = 0, with no depth at all.

| head | k = 0 | k = 0, `L_align` x4 | k = 3 | the re-weighting<br>k = 0 → x4 | the depth<br>x4 → k = 3 | share |
|---|---|---|---|---|---|---|
| student | 1.2025 | 1.1513 | 1.0850 | -0.0512 | -0.0663 | 44% |
| teacher | 1.2001 | 1.1482 | 1.0948 | -0.0519 | -0.0534 | 49% |

95% paired dataset-cluster intervals over the 97 eval configs:

- student: re-weighting -0.0512 [-0.1001, -0.0023], depth -0.0663 [-0.1070, -0.0331], total -0.1175 [-0.1801, -0.0615]
- teacher: re-weighting -0.0519 [-0.0987, -0.0066], depth -0.0534 [-0.0874, -0.0237], total -0.1053 [-0.1661, -0.0515]

**Plainly: neither one wins it. Both pay, at about half each.** The
re-weighting carries 44% of the student's -0.1175 and the extra horizons carry
the remaining 56%. Each segment's interval excludes zero on both heads, so
neither is a rounding artefact, and neither alone reproduces the win. The two
shares also sit inside each other's intervals, so this cell cannot rank them.

Every column trained on elisa at backbone seed 20260520 on the same head
budget: 15,000 head steps at seed 20260722, then 97 GIFT-Eval configs. All six
heads are machine-held, confirmed from their own logs.

What it cannot separate: `k = 3` puts its four copies of `L_align` on four
horizons and `k = 0` x4 puts all four on t+1. So the depth column is the extra
HORIZONS at a held total weight, not depth net of everything else.

## Item 6 — the A3 redraw

It never started under the queue; the head existed and the eval did not. Both
have since run.

| | draw 1 | draw 2 | teacher |
|---|---|---|---|
| head seed | 20260722 | 20260723 | 20260722 |
| head trained on | the rented box | elisa | the rented box |
| score | 1.3998 | 1.4098 | 1.2913 |

Against the original **1.3998**: the redraw reads **1.4098**, +0.0100
[-0.0163, +0.0378], 26% of the ±0.0384 head-seed band, and it is the higher of
the two. So 1.3998 is not a bad draw. Against the teacher **1.2913**: draw 1
sits 0.1084 below it and draw 2 sits 0.1185 below it, so the student/teacher
gap on this cell survives the redraw on both draws.

The redraw changes the machine as well as the seed, so its 0.0100 bounds the
two together. Draw 1 and the teacher both trained on the box, so their 0.1084
is the machine-held one. `plots/a3_reseed.png` names each point's machine.

## The fifth check

The close asserts every cell divides by the same seasonal-naive denominator.
Nothing verified it. The four existing checks each read a score against **its
own** eval, so a denominator that moved **between** evals passes all four: the
harness recomputes `SN_MASE` per eval, and a moved panel, a dropped config or a
re-run against a different split would move a score without touching the model.
Every cross-cell delta above divides one such score by another.

`scripts/verify_denominator.py` reads the `(config → SN_MASE)` map out of all
100 eval directories and requires one map for the study.

```
eval directories        : 100
carry a summary.txt     :  99
score files             :  99
distinct denominators   :   1   md5 a86ef40144eee950866b027d876ce75e
```

The 99 summarised evals pair one-to-one with the 99 score files. The hundredth
is `B5·s3`'s teacher head, which aborted for want of VRAM: it holds a
`stop.log`, no summary and no score, so it is not a measurement. A directory
holding a score without a summary would fail the check.

**The negative control.** Moving `solar/H/short`'s `SN_MASE` from 0.9519 to
0.9520 in one of the 99 evals splits the fingerprint 98/1, fails the check, and
prints the config with both values. The unmodified copy passes.

`scripts/verify_close.sh` now runs five checks. All five pass, and the four
older logs reproduced byte-identically, so no artefact moved since the close.

| check | what it reads | result |
|---|---|---|
| `verify_scores` | every score against its own 97-config eval, two ways | **99 of 99 reproduce**, worst deviation 5.13e-05 against a 1.07e-04 derived allowance |
| `verify_coverage` | the 14 x 3 x 2 grid, rebuilt from the score files alone | **72 of 72**, none missing, none extra |
| `verify_alignx4` | item 3's x4 weight, off the loss curve rather than the launcher | **the flag reached the objective**, +3.73116 at step 1, no `cos_err_d*` column |
| `verify_provenance` | each head's training machine, from the backbone path in its own log | **item 3 is machine-held, item 6's two pairs cross** |
| `verify_denominator` | the `SN_MASE` column, across evals rather than within one | **one denominator over 99 evals** |

## The review's ten items, all closed

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | recomputed from one list in `tables.py`: **7 of 16 improved**, mean +0.0079, median +0.0042, band covers 13 of 16 |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | the criterion runs over all 41 published pairs as a screen (25 of 41 meet it); **all 41 deltas carry a 95% paired interval** |
| 3 | in 12 of 14 cells `k = 3` is a depth change AND a 4x re-weight of `L_align` | **measured** — the control above. Both segments pay, 44% / 56% |
| 4 | the extend rule reads a confounded contrast and fires inside its own band | all three faults named by number; the 200k verdict reads conditional on a panel selected for having improved |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied |
| 6 | A3's bb200k student is an outlier and no one re-measured it | **measured** — 1.4098 on a second seed. Its provenance is corrected |
| 7 | the headline counts A1 and B3 as two cells; their student column is one measurement | the count is over distinct MODELS: **13 student models, 8 better, 3 flat, 2 worse**. The shared row is marked `‡` and counted once |
| 8 | only `k = 3` ran on the 14 cells | the report supports *depth 3 moves the score*, not *depth 3 is the right depth* |
| 9 | the cost of the depth is unknown within an order of magnitude | A3's +13% row is dropped as not comparable; the report carries **+157% to +168%** |
| 10 | the ladder caption forbids the comparison the headline makes | the caption names the one table that reads the dashes and labels it a screen |

Minors, all closed: the fidelity batch's not-held-out caveat sits next to the
claim; the mechanism sections name their four-cell diagnostic set; the
`--grad-clip 1.0` exemption names the project rule it departs from and why; the
"k = 3 still leads at 200k" line is gone. B5·s3's missing teacher head stays
disclosed in the annex.

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

‡ A1 and B3 hold one student model, printed twice per stop, and counted once.
The 72 deliverables therefore hold 70 distinct measurements. Their teacher
columns are two models and stay counted twice.

Plus the three controls outside the grid: `G_B1_k0_aw4` student and teacher,
and `A3_k3_bb200k_student_s20260723`. 99 score files in all.
