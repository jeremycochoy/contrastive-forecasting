«Agent ExperimentRunner claude-opus-5 writing»

## The control is not stalled. It finished at 22:40, and a third implementation now confirms it.

**Experiment directory:** `reports/2026-08-08_rollout_depth/` (`results/`, `plots/`, `scripts/`, `sync/`)

The brief reported `G_B1_k0_aw4` with no eval and item 6 with no reseed log. That
was true at 20:40. Both landed the same evening and are committed. This session
ran no training and no eval. It re-derived both results with a third
implementation and re-read the two provenance facts off the trained artefacts.

### Runs completed

```
this session   0 training runs, 0 evals, $0.00, nothing rented
study total   99 evals, 97 GIFT-Eval configs each, strategy B4, horizon 16
              72 of 72 deliverables over 14 cells x 3 stops x 2 heads
failed         0
```

`vastrun-status`: "No running instances found." `vastrun-balance`: **$11.45**, unchanged. All work on elisa.

### Headline numbers

```
item 3  B1 k = 0                     1.2025 student   1.2001 teacher
item 3  B1 k = 0, L_align x4         1.1513 student   1.1482 teacher   <- the control
item 3  B1 k = 3                     1.0850 student   1.0948 teacher
item 6  A3 bb200k student, draw 1    1.3998   seed 20260722
item 6  A3 bb200k student, draw 2    1.4098   seed 20260723   <- the redraw
item 6  A3 bb200k teacher            1.2913   seed 20260722
```

The two aw4 evals ran on elisa GPU 1. The student started 21:40:54 and merged 97
configs at 22:39:58. Both read
`bb_..._cf373k0_aw4_40k.pth` and their own `qhead_G_B1_k0_aw4_bb40k_*` head, at
strategy B4, `forecast_len=16`.

### The label the brief carried, corrected, because it changes the reading

The brief quoted B1 k=0 as 1.0850 / 1.0948 and B1 k=3 as 1.0881 / 1.0897. Those
are not two k values. They are the same **k = 3** arm at two backbone budgets,
bb40k and bb100k. The score files say so directly: `score_B1_k3_bb40k_student`
is 1.0850 and `score_B1_k3_bb100k_student` is 1.0881. B1's real k=0 baseline is
**1.2025 / 1.2001** (`score_G6_B1_k0_bb40k_*`). The -0.1175 runs 1.2025 → 1.0850,
and the control sits between the two.

## Item 3 — the answer, plainly

The test was: if the ×4 re-weight alone reproduces most of B1's -0.1175, the win
is the weight, not the depth.

**It reproduces 44%. So the win is not the weight. It is not the depth either.**
Depth carries 56%, the weight 44%, both segments exclude zero on both heads, and
each share falls inside the other's interval. Neither segment alone reproduces
the win, and this cell cannot rank the two.

| head | k = 0 | k = 0, `L_align` ×4 | k = 3 | the re-weighting | the depth | weight's share |
|---|---|---|---|---|---|---|
| student | 1.2025 | 1.1513 | 1.0850 | -0.0512 | -0.0663 | 44% |
| teacher | 1.2001 | 1.1482 | 1.0948 | -0.0519 | -0.0534 | 49% |

95% paired dataset-cluster intervals over the 97 configs, 55 clusters, from this
session's implementation:

- student: re-weighting **-0.0512 [-0.0941, -0.0012]** (97.8%), depth **-0.0663 [-0.0990, -0.0360]** (100%), total **-0.1175 [-0.1652, -0.0665]** (100%)
- teacher: re-weighting **-0.0519 [-0.0925, -0.0054]** (98.5%), depth **-0.0534 [-0.0823, -0.0251]** (100%), total **-0.1053 [-0.1530, -0.0547]** (100%)

**The two segments buy different horizons.** The re-weighting's whole gain sits
at medium and long. At short it is nothing. Depth pays at both.

| segment, student | short (n=55) | medium+long (n=42) |
|---|---|---|
| the re-weighting | -0.0009 [-0.0461, +0.0479] | -0.1400 [-0.2069, -0.0680] |
| the depth | -0.0547 [-0.0796, -0.0325] | -0.0845 [-0.1589, -0.0153] |

The control is clean, read off the trained artefact and not off the launcher.
The training log does not print `--align-loss-weight`, so this session read the
three loss CSVs directly. The ×4 arm sits **+3.73116** above k=0 at step 1 on a
shared seed and batch order, which is 3 × `L_align(1)`, so `L_align(1)` ≈ 1.2437.
It writes **no `cos_err_d*` column**, so it trains no rollout, while k=3 writes
`cos_err_d0..d3`. All three arms logged 40,000 steps.

Caveat the control cannot remove: `k = 3` spreads its four copies of `L_align`
over t+1..t+4. The control stacks all four on t+1. The depth segment is the extra
**horizons at held total weight**, not depth net of everything else.

## Item 6 — it completed

It never entered the queue, which is why the brief found no log there.
`gap_worker.sh` ran it outside the queue. Head training started 16:02:54 on elisa
GPU 1 at seed 20260723. The eval merged 97 configs at 18:20:29 and the score
committed at 18:24. Both draws trained 30,000 head steps on the same backbone
file.

| | draw 1 | draw 2 | teacher |
|---|---|---|---|
| head seed | 20260722 | 20260723 | 20260722 |
| head trained on | the rented box | elisa | the rented box |
| score | 1.3998 | 1.4098 | 1.2913 |

| contrast | Δ | 95% paired interval |
|---|---|---|
| draw 2 against the original **1.3998** | **+0.0100** | [-0.0167, +0.0348] |
| draw 1 against the teacher **1.2913** | **+0.1084** | [+0.0680, +0.1612] |
| draw 2 against the teacher **1.2913** | **+0.1185** | [+0.0758, +0.1730] |

**1.3998 is not a bad draw.** The redraw sits 0.0100 above it, 26% of the ±0.0384
head-seed band, and it is the higher of the two, not the lower. The interval
covers zero.

**The student/teacher gap survives both draws.** Both contrasts against the
teacher exclude zero. The gap belongs to the student encoder, not to the draw.

The redraw changes machine as well as seed, so its 0.0100 bounds both together.
Draw 1 and the teacher both trained on the box, so their 0.1084 is machine-held.
Only elisa's copy of the backbone carries a recorded md5
(`9f0e8da71ff595523d2bf0dabdf80445`). The box was released before its original
could be checksummed.

## What this session added

`scripts/runner_recheck.py` is a third implementation, independent of both
`paired_bootstrap.py` and `independent_recheck.py`: numpy vectorised cluster
resampling in place of the `random` module, the seasonal-naive denominator read
from the canonical GIFT-Eval install rather than the repo copy, and bootstrap
seed 3731408. It reads the per-config CSVs first and the score files last.

| what it re-derived | result |
|---|---|
| the 9 scores behind both items, from the per-config CSVs | **9 of 9 reproduce**, worst deviation 3.97e-05 against a 5.0e-05 allowance |
| the config set behind all 9 cells | **97 shared configs, one denominator**, byte-identical to the canonical GIFT-Eval reference |
| the 21 observed deltas | **match the published values to 4 decimals** |
| which side of zero each interval sits on | **no disagreement in 21 of 21** |
| the ×4 flag, off the three loss CSVs | **+3.73116** at step 1, no `cos_err_d*`, 40,000 steps |

## The five standing checks

`scripts/verify_close.sh` runs all five. All five pass.

| check | what it reads | result |
|---|---|---|
| `verify_scores` | every score against its own 97-config eval | **99 of 99 reproduce**, worst deviation 5.13e-05 against a 1.07e-04 allowance |
| `verify_coverage` | the 14 × 3 × 2 grid, from the score files alone | **72 of 72**, none missing, none extra |
| `verify_alignx4` | item 3's ×4 weight, off the loss CSVs | **the flag reached the objective**, no `cos_err_d*` |
| `verify_provenance` | each head's machine, from its own log | **item 3 machine-held, item 6's seed pair crosses** |
| `verify_denominator` | the `SN_MASE` column, across evals | **one denominator over 99 evals**, md5 `a86ef401…` |

## The review's ten items, all closed

| # | item | what closed it |
|---|---|---|
| 1 | the 200k headline count contradicts the report's own table | recomputed from one list in `tables.py`: **7 of 16 improved**, mean +0.0079, median +0.0042 |
| 2 | the card's criterion runs on 4 arms; the 14-cell headline has no interval | criterion runs over all 41 published pairs as a screen (25 of 41 meet it); **all 41 deltas carry a 95% paired interval** |
| 3 | `k = 3` is a depth change AND a 4× re-weight of `L_align` | **measured** — the control above. Both segments pay, 44% / 56% |
| 4 | the extend rule reads a confounded contrast and fires inside its own band | all three faults named; the 200k verdict reads conditional on a panel selected for having improved |
| 5 | one backbone seed everywhere; the band bounds the head seed only | stated where the band is defined and where it is applied |
| 6 | A3's bb200k student is an outlier and no one re-measured it | **measured** — 1.4098 on a second seed. Provenance corrected |
| 7 | the headline counts A1 and B3 as two cells; their student column is one measurement | count is over distinct MODELS: **13 student models, 8 better, 3 flat, 2 worse**. Shared row marked `‡`, counted once |
| 8 | only `k = 3` ran on the 14 cells | the report supports *depth 3 moves the score*, not *depth 3 is the right depth* |
| 9 | the cost of the depth is unknown within an order of magnitude | A3's +13% row dropped as not comparable; the report carries **+157% to +168%** |
| 10 | the ladder caption forbids the comparison the headline makes | the caption names the one table that reads the dashes and labels it a screen |

Minors closed: the fidelity batch's not-held-out caveat sits next to the claim;
the mechanism sections name their four-cell diagnostic set; the `--grad-clip 1.0`
exemption names the project rule it departs from and why; the "k = 3 still leads
at 200k" line is gone. B5·s3's missing teacher head stays disclosed in the annex.

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

‡ A1 and B3 hold one student model, printed twice per stop, counted once. The 72
deliverables hold 70 distinct measurements. Their teacher columns are two models
and stay counted twice.

Plus the three controls outside the grid: `G_B1_k0_aw4` student and teacher, and
`A3_k3_bb200k_student_s20260723`. **99 score files in all.**
