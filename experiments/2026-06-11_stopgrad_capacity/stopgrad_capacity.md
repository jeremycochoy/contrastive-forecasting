# Stop-gradient shrinks the encoder-depth penalty toward zero but never flips it

**Question.** A SimSiam/BYOL-style stop-gradient on the encoder side of the InfoNCE positive
improved transfer in a prior run, and separately, with no stop-grad, growing the encoder from
3 to 6 layers *hurt* transfer. Does adding the stop-grad flip the sign of that depth knob, so
the extra capacity starts to pay, and how does it interact with the forecaster bottleneck?

## The design

![The four arms as a comparison chain. From arm 2 (enc3·full·sg, the prior stop-grad winner),
a depth step gives the new arm 3 (enc6·full·sg) and a further width step gives the new arm 4
(enc6·bottleneck·sg). Arm 1 (enc6·bottleneck, the no-stop-grad reference) is arm 4 with the
stop-grad removed — the control that isolates stop-grad's role in the
collapse.](plots/design_grid.png)

All four share the GRU patch-embedding, d_model 384 / 6 heads, the crossfade-triplet mix,
12,500 steps at batch 1024, and seed 20260520.

## Result

![GM-Relative MASE grouped by architecture, four bars (head × checkpoint) per arm, collapsed
bars clipped at 1.4. Only arm 4 splits within-arm — best-loss with the pack, last-checkpoint
collapsed; the other three arms are flat across all four cells.](plots/gm_by_arch.png)

Forecaster width is the only knob differing between the collapsed arm 4 and the stable arm 3.

![Left: GM-Relative MASE per arm × head × checkpoint (collapsed bars clipped at 1.4, true
value labelled). Every arm sits near 1.16–1.21 except arm 4, whose last-checkpoint bars jump
above 2.2 while its best-loss bars stay with the pack. Right: the depth step enc3→enc6 as a
paired-bootstrap Δ with 90% interval — without stop-grad (red) it is a penalty on three of
four cells; with stop-grad (green) it shrinks toward zero and never crosses
below.](plots/gm_summary.png)

Stop-grad neutralises most of the depth penalty, but the deeper encoder never *beats* the
shallower one: the extra capacity does not start to pay.

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97 forecasting
tasks, of a model's error divided by the seasonal-naive forecast's error. Lower is better;
1.0 is seasonal-naive. Each pairwise Δ carries a **paired-bootstrap** 90% interval — resample
the 97-task list with repeats and score both arms on each resample so per-task difficulty
cancels.*

| arm | 2-layer head, best / last | 6-layer head, best / last |
|---|--:|--:|
| 1 — bn, no stop-grad | 1.186 / 1.187 | 1.185 / 1.190 |
| 2 — enc3, full, sg (prior winner) | 1.177 / 1.180 | **1.159** / 1.163 |
| 3 — enc6, full, sg | 1.180 / 1.213 | 1.161 / 1.193 |
| 4 — enc6, bn, sg | 1.200 / **2.265** | 1.183 / **2.201** |

## Training dynamics: transfer peaks early; only the bottleneck collapses after

![Training metrics, log-log: the two new arms (solid) against the prior stop-grad arm and the
no-stop-grad base (dashed). The bottleneck arm's floor-subtracted contrastive loss (top-left)
bottoms early then climbs; both stop-grad arms hold far more batch-wise embedding dimensions
in use (bottom-centre, U_batch) than the no-stop-grad base.](plots/training_metrics.png)

Both stop-grad arms' losses rise after an early minimum, yet only the bottleneck arm's
transfer collapses, so the late-training loss rise coincides with the collapse but does not by
itself explain it. Both stop-grad arms keep many batch-wise dimensions in use, so the collapse
is not a rank collapse.

## Protocol

We train one backbone per arm, single seed, on one RTX 4090. Arms 1 and 2 are reused
unchanged from the two prior runs; arms 3 and 4 are the new ones, each adding its capacity
change plus the encoder-side stop-grad (`--stopgrad-positive-h`).

To score a backbone, we freeze it and train a fresh quantile forecasting head on top, once
with two transformer layers and once with six. We then evaluate on GIFT-Eval's 97 tasks at
two checkpoints: the **best-loss** one (the lowest smoothed contrastive loss, around step
1,000 for arm 4 and 1,300 for arm 3) and the **last** one (step 12,500).

Two checks make the collapse trustworthy. First, the pipeline reproduces arm 1's per-task
MASE to four decimals against its previously published numbers. Second, we train the
last-checkpoint head two ways, a 10k re-adapt from the best-loss head and a fresh 30k head;
they agree to within 0.011 GM on every cell, so the collapse is not an artifact of how the
head was trained. The table reports the re-adapt head.
