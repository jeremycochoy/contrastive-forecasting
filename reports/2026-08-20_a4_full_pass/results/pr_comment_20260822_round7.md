«Agent ExperimentRunner claude-opus-5 writing»

Experiment directory: `reports/2026-08-20_a4_full_pass/`

The card is complete. Both heads carry a score at all four stops, and the
eleven review items are closed.

## Runs completed

| run | state |
|---|---|
| driver leg to 300,000 steps | done in 8.5 h |
| driver leg to 450,000 steps | done in 11.7 h |
| driver leg to 665,000 steps | done in 14.1 h |
| 200k, 300k and 450k bands, three head seeds, both heads | 18 draws, scored |
| 665k stop, student head, seed 20260722 | scored 18:33 UTC |
| 665k stop, teacher head, seed 20260722 | scored 20:55 UTC |
| 665k head-seed band, student and teacher | skipped by rule, about 16 GPU-hours kept |
| paired bootstrap, 6 comparisons | run on CPU |
| goal metrics, 20 (stop, head, seed) rows | read back on CPU |

## Headline numbers

GM-Relative MASE, per-stop mean over the head-seed draws:

| stop | student | teacher | draws each |
|---|---:|---:|---:|
| 200,000 | 1.0651 | 1.0800 | 3 |
| 300,000 | 1.0864 | 1.1009 | 3 |
| 450,000 | 1.0743 | 1.0952 | 3 |
| 665,000 | 1.0783 | 1.1038 | 1 |

**The answer to the card is no.** This run did not improve past 200,000
steps. Every stop after 200,000 steps is worse than 200,000 steps, on both
heads. The student rises 0.0132 and the teacher rises 0.0237 from their
200,000-step band means. The pooled head-seed standard deviation is 0.0029,
so those are 4.6 and 8.2 of it.

The paired bootstrap over the 97 configs agrees. 200k to 665k, student:
+0.0123, 95% CI [+0.0009, +0.0240], improved in 1.7% of resamples. Teacher:
+0.0210, 95% CI [+0.0037, +0.0385], improved in 0.8%.

![full pass](../plots/full_pass.png)

## What closed

1. **The 665,000-step stop is complete on disk.** Both scores, both 97-row
   eval CSVs, both summaries, the eval logs and the leg losses CSV are
   committed. The figure is redrawn.
2. **The figure plots the band mean.** The two lines join the per-stop
   means. At 450,000 steps the old line ran through 1.0691 while the tables
   carried 1.0743.
3. **The pooled ribbon is gone.** Every head-seed draw is a small dot in its
   head colour, so the spread is measured and not summarised. The pooled
   number stays in `results/head_band.txt`. The 665,000-step points carry
   one draw each, so they carry no spread.
4. **The band rule, its window and its date are on disk.** The rule went in
   at 12:52 UTC on 2026-08-22 with the leg at step 631,200 and no score
   written. `band_decision.py --offsets` bounds what the skip did not
   measure: the student band mean lands between 1.0774 and 1.0835, which is
   4.2 to 6.3 pooled standard deviations above the 200,000-step mean.
   Artefacts: `results/band_665k_decision.txt`,
   `results/band_665k_offsets.txt`.
5. **The teacher case is decided.** `band_decision.py --head teacher` reads
   its own center off `head_band.csv`, 1.0800, so its window is
   [1.0700, 1.0900]. The teacher scored 1.1038, which is 0.0238 outside.
   Verdict SKIP, so one draw reads on its own and the teacher comparison is
   not undecided. `watchdog.sh` runs that call every tick, so the verdict
   never depended on an agent. Artefacts:
   `results/band_665k_teacher_decision.txt`,
   `results/band_665k_teacher_offsets.txt`.
6. **Both promised analyses ran and are committed.**
   `results/stop_bootstrap.csv` holds six paired comparisons.
   `results/metrics_table.csv` holds 20 rows and
   `results/metrics_table_means.csv` holds the per-stop means. Both cost
   CPU seconds. The recomputed GM-Relative MASE agrees with every published
   score, so one seasonal-naive denominator is in play.
7. **No trend between the last three points.** The figure draws no trend
   line. The student moves -0.0121 from 300k to 450k, which is 4.2 pooled
   standard deviations, then +0.0040 from 450k to 665k, which is 1.4 of them
   and inside the widest measured range.
8. **One run, one backbone seed.** The learning rate stayed at 0.001 across
   every leg, with no schedule flag, and `train.py` prints `lr=0.001` at the
   start of all four legs. The EMA momentum ramp is anchored to a fixed
   100,000 steps and holds past it, so no schedule change sits inside the
   card's span. The execution log records what the report may and may not
   claim.
9. **1.0660 is a selected minimum.** Rank 1 of 99 published scores,
   runner-up 1.0801. The like-for-like comparison is band mean against band
   mean: 1.0651 at 200,000 steps against 1.0783 at 665,000 steps.
10. **No card numbers on the axes.** The rule label reads "prior best,
    1.0660" and the legend reads "rollout-depth study point".
11. **The smaller items.** 665,000 steps is 99.97% of the manifest row count
    and 99.58% of the shard arithmetic, and both counts are in the log.
    `plot_full_pass.py` writes `results/figure_caption.txt` on every run, so
    it cannot go stale again. The retracted "free null" stays out of the
    report. The launcher writes into the parent study's directory, and
    `read_back.sh` now runs `collect.sh` as well as
    `collect_replicates.sh`, so every driver pair crosses into this study
    without an agent.

## Two notes for the reviewer

- Item 9 asked for "selected minimum" in the caption. The orchestrator
  fixed the caption word for word, so that fact now lives in the execution
  log and in `results/selection_context.json` for the report to carry.
- Item 10 asked to keep the provenance line on the figure. The orchestrator
  ruled out caveat text under the axes, so it moved to
  `results/figure_provenance.txt`.

## The three goal metrics

Each move is the 200,000-step band mean against the one draw at 665,000
steps. The scale beside it is the largest head-seed range this card
measured for that metric.

| metric | largest range | student move | teacher move | reads |
|---|---:|---:|---:|---|
| GM-Relative MASE | 0.0087 | +0.0132 | +0.0237 | both outside |
| GM-MASE | 0.0122 | +0.0184 | +0.0332 | both outside |
| GM-MAPE_SN | 0.0419 | +0.0365 | -0.0303 | both inside |
| GM-CRPS_SN | 0.0083 | -0.0012 | +0.0089 | student inside |

The deliverable and GM-MASE agree that both heads are worse. GM-MAPE_SN
moves less than its own head-seed spread, and the two heads move in
opposite directions on it, so it resolves nothing at this scale.

## Tests

`tests/test_407_full_pass.py`, 250 tests, all pass.

The report comes next, from the ReportWriter agent.
