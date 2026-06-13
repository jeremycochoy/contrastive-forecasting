# Stop-gradient shrinks the encoder-depth penalty toward zero but never flips it

**Question.** A SimSiam/BYOL-style stop-gradient on the encoder side of the InfoNCE
positive reliably *improved* downstream transfer for one recipe (#339). Separately,
growing the encoder from 3 to 6 layers reliably *hurt* transfer for the no-bottleneck
recipe *without* the stop-grad (#336). If the stop-grad changes what the encoder learns,
the extra depth that hurt without it might start to pay. Does the stop-grad **flip the
sign of the encoder-depth knob** — and how does it interact with the forecaster bottleneck?

Four arms share the GRU patch-embedding, d_model 384 / 6 heads, the crossfade-triplet
data mix, 12,500 steps at batch 1024, and seed 20260520. They differ only in three knobs:

| # | encoder | forecaster | stop-grad | role |
|---|---|---|:--:|---|
| 1 | 6-layer | 128-wide bottleneck | no | base+triplet (#336) |
| 2 | 3-layer | 6-layer full-width | yes | the #339 winner |
| 3 | 6-layer | 6-layer full-width | yes | **new** — depth, full-width |
| 4 | 6-layer | 128-wide bottleneck | yes | **new** — base + stop-grad |

**Result.** Two findings, both in the figure below.

![Left: GM-Relative MASE per arm × head × checkpoint — every arm sits near 1.16–1.21
except arm 4 (stop-grad + bottleneck), which collapses above 2.2 at the last checkpoint
while remaining normal at best-loss. Right: the encoder-depth step (enc3→enc6) as a
paired-bootstrap change with its 90% interval — without stop-grad (red) the step is a
penalty on three of four cells (up to +0.13; the 6-layer best-loss cell is a smaller,
non-significant +0.015); with stop-grad (green) it shrinks toward zero (≤+0.035) and
never reverses.](plots/gm_summary.png)

1. **Stop-grad nearly cancels the depth penalty but does not reverse it.** With the
   full-width forecaster, growing the encoder 3→6 layers under stop-grad is tied at the
   best-loss checkpoint (both heads, intervals straddle zero) and only mildly worse at
   the last checkpoint — a fraction of the reliable penalty the same step incurs *without*
   the stop-grad. So the stop-grad neutralises most of the damage from extra depth, yet
   the deeper encoder still never *beats* the shallower one: capacity does not start to pay.
2. **The bottleneck forecaster under stop-grad collapses at full training.** Arm 4 is
   indistinguishable from the no-stop-grad base at best-loss, but by the last checkpoint
   its forecasts are worse than the seasonal-naive baseline (GM-Relative MASE > 2 on both
   heads). The full-width arm shows no such collapse — the failure is specific to the
   bottleneck-plus-stop-grad combination, and only at full training.

*Forecast error is **GM-Relative MASE**: the geometric mean, over the GIFT-Eval
benchmark's 97 forecasting tasks, of a model's error divided by the seasonal-naive
forecast's error. Lower is better; 1.0 is seasonal-naive.*

| arm | 2-layer head, best / last | 6-layer head, best / last |
|---|--:|--:|
| 1 — bn, no stop-grad | 1.186 / 1.187 | 1.185 / 1.190 |
| 2 — enc3, full, sg (#339) | 1.177 / 1.180 | **1.159** / 1.163 |
| 3 — enc6, full, sg | 1.180 / 1.213 | 1.161 / 1.193 |
| 4 — enc6, bn, sg | 1.200 / **2.265** | 1.183 / **2.201** |

Each pairwise change carries a **paired-bootstrap** 90% interval (resample the 97-task
list with repeats, score both arms on each resample so per-task difficulty cancels). The
depth step under stop-grad (arm 2→3) is ns at best-loss on both heads and reliably worse
by ≈+0.03 at last; the same step without stop-grad (#336's no-bottleneck arms) is reliably
worse on three of four cells, by up to +0.13. Adding stop-grad to the base recipe (arm 1→4) is ns at
best-loss and reliably worse by ≈+1.0 at last.

## Training dynamics: transfer peaks early; only the bottleneck collapses after

![Training metrics, log-log, the two new arms (solid) against the #339 stop-grad arm and
the no-stop-grad base (dashed). The bottleneck arm's floor-subtracted contrastive loss
(top-left) bottoms early and then climbs; both stop-grad arms hold far more batch-wise
embedding dimensions in use (bottom-centre, U_batch) than the no-stop-grad
base.](plots/training_metrics.png)

For arm 4 the downstream score tracks the pretext loss across the two measured checkpoints. Its floor-subtracted
contrastive loss reaches its minimum at step ~1000 and then *rises* to a plateau; the
best-loss checkpoint (step ~1000) transfers normally (~1.18–1.20) while the fully-trained
checkpoint transfers far worse (~2.2). The full-width arm's loss rises after its own early
minimum too, yet its last checkpoint stays close to its best (~1.16→~1.19) — so the shared
late-training loss rise coincides with a transfer collapse only in the bottleneck arm.
Both stop-grad arms sit in the
high-rank regime #339 identified — the encoder keeps many more batch-wise dimensions in
use than the no-stop-grad base — so the collapse is not a rank collapse; it is specific to
how the bottleneck forecaster and the late-training encoder interact.

## Protocol

One backbone per arm, single seed, 12,500 steps at batch 1024 on one RTX 4090. Arms 1 and
2 are reused unchanged from #336 and #339; arms 3 and 4 add only the listed capacity change
plus the `--stopgrad-positive-h` flag. Each finished backbone is frozen and scored by
training a fresh quantile forecasting head on top — once with two transformer layers, once
with six — and evaluating on GIFT-Eval's 97 tasks at two backbone checkpoints: **best-loss**
(lowest smoothed contrastive loss; step ~1000 for arm 4, ~1300 for arm 3) and **last** (12,500).
The eval pipeline was cross-checked against #336's published numbers (it reproduces arm 1's
per-task MASE to four decimals), so the arm-4 collapse is a real measurement, not a pipeline
artifact.

The last-checkpoint head was trained two independent ways — a 10k re-adapt from the
best-loss head, and a fresh 30k head trained directly on the last backbone — because the
best-loss checkpoint lands so early (step ~1000) that a short re-adapt could in principle
underfit. The two protocols agree within ±0.01 GM on every cell, so each last-checkpoint
score reflects the backbone, not the head-adaptation choice; the table reports the re-adapt
head for consistency with arms 1–2.

## What we learned

- **Stop-grad does not make extra encoder depth pay** — it shrinks the depth penalty from
  reliable on three of four cells (up to +0.13 without it) to near-zero (≤+0.035), but the
  deeper encoder never beats the shallower one. The #339 gain is about the stop-grad itself, not a capacity
  that the stop-grad unlocks.
- **A forecaster bottleneck and the encoder-side stop-grad do not mix at full training.**
  Together they are fine early but collapse below seasonal-naive by the last checkpoint —
  a failure neither the full-width stop-grad arm nor the no-stop-grad bottleneck base
  shows. The forecaster width is the only knob that differs between the collapsing arm (4)
  and the stable one (3).
- **Use the best-loss checkpoint for these stop-grad recipes.** Transfer peaks early (step
  ~1000–1300) and degrades with further training, sharply so for the bottleneck arm; the
  smoothed-loss minimum is a good selector.

## Follow-up

The two new arms reach their best transfer at step ~1000–1300, an order of magnitude before the
12,500-step budget — a **stop-grad + early-stop** card would test whether the full recipe
line's downstream numbers can be matched at a fraction of the compute. Separately, the
bottleneck collapse motivates a **forecaster-width sweep under stop-grad** to locate where
stability breaks.
