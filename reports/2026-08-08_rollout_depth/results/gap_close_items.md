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
