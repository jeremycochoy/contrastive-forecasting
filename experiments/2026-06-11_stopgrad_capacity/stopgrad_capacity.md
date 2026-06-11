<!-- DRAFT SKELETON (#341) — Question/Arms/Protocol are final (design facts);
Result, Training dynamics, and What we learned are PLACEHOLDERS filled after
training + GIFT-Eval complete. Title finalized from the verdict. Do not open the
PR until the placeholders are replaced and the report checklist passes. -->

# [TITLE — finalize from results, single declarative sentence stating the verdict]

**Question.** Adding a stop-gradient on the encoder side of the InfoNCE positive
(detaching the future encoding h_{t+1} in the positive term) reliably improves
downstream transfer for one backbone recipe (#339, PR #340). A separate result
(#336) found that **growing the encoder from 3 to 6 layers makes transfer
reliably worse** for the no-bottleneck recipe *without* the stop-grad. Does the
stop-grad change what the encoder learns enough to **flip the sign of that
capacity knob** — i.e. does extra encoder depth (and the bottleneck choice) help
under the stop-grad where it hurt without it?

## Arms

All four arms share the data, optimizer, and contrastive loss; they differ only
in encoder depth, forecaster width, and whether the encoder-side stop-grad is on.
Every arm uses a GRU patch-embedding, an EWMA input norm, d_model 384 with 6
attention heads, the crossfade-triplet synthetic mix, and 12,500 steps at batch
1024 (seed 20260520).

| # | encoder | forecaster | stop-grad | role | source |
|---|---|---|---|:--:|---|
| 1 | 6-layer | 128-wide bottleneck (4 heads) | no | base+triplet | #336 |
| 2 | 3-layer | 6-layer, full width (no bottleneck) | yes | the #339 winner | #339 |
| 3 | 6-layer | 6-layer, full width (no bottleneck) | yes | **new** | this |
| 4 | 6-layer | 128-wide bottleneck (4 heads) | yes | **new** (= arm 1 + stop-grad) | this |

The three contrasts that test the hypothesis:

- **Arm 3 vs arm 2** — encoder 3→6 layers, holding no-bottleneck + stop-grad.
  Without the stop-grad this step was reliably *worse* (#336); does it flip?
- **Arm 4 vs arm 1** — the stop-grad switched on for the base recipe (6-layer
  encoder + bottleneck forecaster), everything else equal.
- **Arm 4 vs arm 2** — bottleneck + 6-layer encoder vs full-width + 3-layer
  encoder, both with the stop-grad.

<!-- RESULT — placeholder. Fill after evals: gm_summary.png + the GM table +
verdict on each contrast above. State the interpretation; let the figure/table
carry the numbers. -->

## Result

*[pending training + GIFT-Eval]*

<!-- TRAINING DYNAMICS — placeholder. training_metrics.png (log-log, the two new
arms vs the #339 stop-grad arm and base+triplet) + at most a few sentences of
interpretation. -->

## Training dynamics

*[pending training]*

## Protocol

One backbone per arm, single seed (20260520), 12,500 steps at batch 1024 on one
RTX 4090. Arms 1 and 2 are reused unchanged from #336 and #339; arms 3 and 4 add
only the listed capacity change (and, for both, the `--stopgrad-positive-h`
flag). Each finished backbone is frozen and scored by training a fresh quantile
forecasting head on top — once with two transformer layers, once with six;
identical head hyperparameters and eval data across arms — on GIFT-Eval's 97
tasks, at two backbone checkpoints: **best-loss** (the step with the lowest
smoothed contrastive loss) and **last** (12,500). Forecast error is
**GM-Relative MASE**: the geometric mean, over the 97 tasks, of a model's error
divided by the seasonal-naive forecast's error (lower is better; 1.0 is
seasonal-naive). Each pairwise change carries a **paired-bootstrap** 90% interval
(resample the 97-task list with repeats, score both arms on each resample so
per-task difficulty cancels). One backbone per arm, so the intervals quantify
task-set noise, not seed noise.

<!-- WHAT WE LEARNED — placeholder. Bullets tied to the verdict; flag any
hypothesis explicitly. -->

## What we learned

*[pending]*
