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
