# A regime crossfade as a contrastive hard negative

**Verdict.** The best backbone recipe so far trains by a contrastive rule — a sample's
forecast should resemble its own future and not its present — and blocks the lazy way to
satisfy that rule (a position code that tells time steps apart without encoding anything
forecastable) by mixing in synthetic series that copy a real past then veer into a different
future. This experiment adds a second, complementary hard example: a **regime crossfade** —
take two real windows from the same batch, keep one's past and the other's future, and blend
smoothly between them. It changes training as intended — the crossfade is a genuinely harder
negative, and the backbone responds by separating future from present more strongly — **but it
does not reliably improve forecasting.** Both forecasting heads drop by about one part in a
hundred, yet the uncertainty on each drop comfortably spans zero. The recipe's existing
negatives already capture the signal; this extra hard example buys nothing we can measure.

*Forecast error is **GM-Relative MASE**: the geometric mean, over the benchmark's 97 tasks, of
a model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is the
seasonal-naive baseline.*

![Forecast error on the benchmark for each forecasting head, best recipe vs the same recipe
with the crossfade added. The bars are within a whisker of each other; the dashed line is the
strongest prior backbone, the dotted line seasonal-naive.](plots/gm_summary.png)

## What we asked

A contrastive backbone can cheat its training loss with an **indexing shortcut**: a per-step
code, shared across every series, that makes time steps distinguishable without learning
anything you could forecast from. The current-best recipe denies that shortcut by pairing each
real sample with a **synthetic series that shares its past then diverges into a different
future** — so position alone can no longer predict the future — and scores its negatives with a
loss that additionally pushes apart every pair of *different* series at *every* time step.

The crossfade attacks the same shortcut from a different angle, with **real** futures instead of
synthetic ones. For one new window we keep the past of real window A and the future of real
window B, ramping from one to the other across a random transition; both A and B stay in the
batch as their own samples. So the new window shares its past with one neighbour and its future
with another — to tell the three apart the backbone must read content, not position.

![The primitive on one example pair: the crossfade (black) copies window A's past, ramps across
a transition band, and copies window B's future; below, the blend weight rises 0 → 1. Each
window is z-normalised first; one transition is drawn per sample (a random midpoint and a width
spanning sharp to gradual) and shared across channels.](plots/crossfade_schematic.png)

## What happened

The crossfade rows are a tenth of each batch (the batch becomes 80 % real, 10 % forked
synthetic, 10 % crossfade); nothing else changes. On the benchmark the point estimates move the
right way — the 2-layer head from 1.222 to 1.208, the 6-layer head from 1.191 to 1.178, the
latter nominally the best single score we have recorded — but neither move is reliable.

![Change in forecast error from adding the crossfade, per head. Both bars are negative
(crossfade a touch better), but both 90 % intervals cross zero, so neither is distinguishable
from no change.](plots/delta.png)

To separate a real improvement from luck we use a **paired bootstrap** over the 97 tasks: draw
the 97 tasks with replacement 2000 times; for each draw, score *both* models on the *same* drawn
tasks and take the difference of their geometric-mean errors; report the range covering the
middle 90 % of those 2000 differences. Scoring both models on the same resampled tasks cancels
per-task difficulty, so the interval reflects the model change alone. Here both intervals
straddle zero — the ~0.013 gains are within task-set noise.

| forecasting head | best recipe | + crossfade | change | 90 % interval |
|---|--:|--:|--:|:--:|
| 2-layer | 1.222 | **1.208** | −0.014 | (−0.040, +0.012) |
| 6-layer | 1.191 | **1.178** | −0.013 | (−0.039, +0.012) |

## A harder negative that the forecaster can't use

The crossfade is not inert — it measurably reshapes training. The clearest read-out is the
**gap**: how much more a forecast resembles the true future than the present (higher is a
sharper representation). With the crossfade the gap climbs higher than the best recipe alone
(~1.18 vs ~1.03) and the contrastive loss settles higher — sharing a real past *and* a real
future with batch-mates is a harder separation than the synthetic fork alone, and the backbone
is pushed to encode the future more strongly. The representation stays healthy throughout
(different series keep repelling — the cross-series similarity stays flat near zero, no
collapse).

![Training curves, with-crossfade (solid) vs best-recipe (dashed), from step 100. Left: the
contrastive loss falls cleanly on both. Middle: the gap climbs higher with the crossfade.
Right: different series stay near-orthogonal throughout — no collapse.](plots/training_curves_loglog.png)

That a larger gap does not bring a lower forecast error is the experiment's one durable lesson:
the gap is a *training* signal, not a *forecasting* one. The crossfade makes the backbone work
harder at the contrastive task without giving the downstream forecaster anything new.

## Protocol

One change from the best recipe: a tenth of every batch is regime-crossfade windows, blended
from the real windows already in that batch (so they cost no extra data). Everything else is
held identical — the same backbone, the same large contrastive batch with all samples pooled as
one negative set, the same two attention normalisations that batch needs to train, the same
synthetic-fork tenth, optimiser, learning rate, temperature, and seed. Each frozen backbone is
scored by training a fresh small forecasting head — once 2-layer, once 6-layer — on it and
evaluating on the GIFT-Eval benchmark (the 97-task full set and an 11-task fast subset). A
single backbone seed was trained, so the interval above is over tasks, not seeds.

## What we learned & follow-up

A real-window regime crossfade is a sound idea for a hard negative and demonstrably bends the
contrastive geometry, but on top of an already-strong negative set it does not convert into a
reliable forecasting gain. The consistent (if tiny) positive lean on both heads is the only
hint of an effect; the cheapest way to confirm or dismiss it would be to repeat the run with
two or three backbone seeds, which a paired test across seeds could resolve where a single seed
cannot. Absent that, the result is a clean neutral: not worth adding to the recipe.
