# Stop-gradient shrinks the encoder-depth penalty toward zero but never flips it

**Question.** #339 found a SimSiam/BYOL-style stop-gradient on the encoder side of the
InfoNCE positive improved transfer; #336 (no stop-grad) found growing the encoder 3→6 layers
*hurt* transfer. Does adding the stop-grad flip the sign of that depth knob — make the extra
capacity start to pay — and how does it interact with the forecaster bottleneck?

## The design

![The four arms as a comparison chain. From arm 2 (enc3·full·sg, the reused #339 stop-grad
winner), a depth step gives the new arm 3 (enc6·full·sg) and a further width step gives the
new arm 4 (enc6·bottleneck·sg). Arm 1 (enc6·bottleneck, reused #336) is arm 4 with the
stop-grad removed — the control that isolates stop-grad's role in the
collapse.](plots/design_grid.png)

All four share the GRU patch-embedding, d_model 384 / 6 heads, the crossfade-triplet mix,
12,500 steps at batch 1024, and seed 20260520.

## Result

![GM-Relative MASE grouped by architecture, four bars (head × checkpoint) per arm, collapsed
bars clipped at 1.4. Only arm 4 splits within-arm — best-loss with the pack, last-checkpoint
collapsed; the other three arms are flat across all four cells.](plots/gm_by_arch.png)

The collapse is specific to the bottleneck-plus-stop-grad combination and only at full
training: forecaster width is the only knob differing between arm 4 and the stable arm 3.

![Left: GM-Relative MASE per arm × head × checkpoint (collapsed bars clipped at 1.4, true
value labelled). Every arm sits at 1.16–1.21 except arm 4, whose last-checkpoint bars jump
above 2.2 while its best-loss bars stay with the pack. Right: the depth step enc3→enc6 as a
paired-bootstrap Δ with 90% interval — without stop-grad (red) it is a penalty on three of
four cells; with stop-grad (green) it shrinks toward zero and never crosses
below.](plots/gm_summary.png)

Stop-grad neutralises most of the depth penalty (green intervals straddle or barely clear
zero) but the deeper encoder never *beats* the shallower one — capacity does not start to
pay.

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97 forecasting
tasks, of a model's error divided by the seasonal-naive forecast's error. Lower is better;
1.0 is seasonal-naive. Each pairwise Δ carries a **paired-bootstrap** 90% interval — resample
the 97-task list with repeats and score both arms on each resample so per-task difficulty
cancels.*

| arm | 2-layer head, best / last | 6-layer head, best / last |
|---|--:|--:|
| 1 — bn, no stop-grad | 1.186 / 1.187 | 1.185 / 1.190 |
| 2 — enc3, full, sg (#339) | 1.177 / 1.180 | **1.159** / 1.163 |
| 3 — enc6, full, sg | 1.180 / 1.213 | 1.161 / 1.193 |
| 4 — enc6, bn, sg | 1.200 / **2.265** | 1.183 / **2.201** |

## Training dynamics: transfer peaks early; only the bottleneck collapses after

![Training metrics, log-log: the two new arms (solid) against the #339 stop-grad arm and the
no-stop-grad base (dashed). The bottleneck arm's floor-subtracted contrastive loss (top-left)
bottoms early then climbs; both stop-grad arms hold far more batch-wise embedding dimensions
in use (bottom-centre, U_batch) than the no-stop-grad base.](plots/training_metrics.png)

Both stop-grad arms' losses rise after an early minimum, yet only the bottleneck arm's
transfer collapses — so the late-training loss rise coincides with the collapse but does not
by itself explain it. Both stop-grad arms keep many batch-wise dimensions in use (high
U_batch), so the collapse is not a rank collapse; it is specific to how the bottleneck
forecaster and the late-training encoder interact.

## Protocol

One backbone per arm, single seed, 12,500 steps at batch 1024 on one RTX 4090. Arms 1–2 are
reused unchanged from #336/#339; arms 3–4 add the listed capacity change plus
`--stopgrad-positive-h`. Each frozen backbone is scored by training a fresh quantile
forecasting head on top — once with two transformer layers, once with six — and evaluating on
GIFT-Eval's 97 tasks at two checkpoints: **best-loss** (lowest smoothed contrastive loss;
step ~1000 for arm 4, ~1300 for arm 3) and **last** (12,500). The eval pipeline reproduces
arm 1's per-task MASE to four decimals against #336's published numbers, so the arm-4
collapse is a real measurement. The last-checkpoint head was trained two ways — a 10k
re-adapt from the best-loss head and a fresh 30k head — which agree within 0.011 GM on every
cell; the table reports the re-adapt head.

## What we learned

- **Stop-grad does not make extra encoder depth pay.** It shrinks the depth penalty from
  reliable on three of four cells (up to +0.13 without it) to near-zero (≤+0.035 with it), but
  the deeper encoder never beats the shallower one. The #339 gain is the stop-grad itself, not
  unlocked capacity.
- **Forecaster bottleneck and encoder-side stop-grad do not mix at full training** — fine
  early, collapse below seasonal-naive by the last checkpoint, a failure neither the
  full-width stop-grad arm nor the no-stop-grad bottleneck base shows.
- **Use the best-loss checkpoint for these stop-grad recipes.** Transfer peaks at step
  ~1000–1300 and degrades after; the smoothed-loss minimum is a good selector.
