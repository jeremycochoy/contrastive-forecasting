# A regime crossfade as a contrastive hard negative

We took our best forecasting backbone and gave it a new kind of hard training example, built by
splicing the recent past of one real series onto the future of another. It did what we intended
to the training and made the learning task harder, yet it left the forecasts no better than they
already were. The picture below is the whole result: the recipe we started from, and the same
recipe with the new example mixed in, land on top of each other.

*Forecast error is **GM-Relative MASE**: the geometric mean, over the benchmark's 97 tasks, of a
model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is the
seasonal-naive baseline.*

![Forecast error for each forecasting head, best recipe versus the same recipe with the crossfade
added. The pairs sit within a whisker of each other. The dashed line is the strongest backbone
from before this line of work; the dotted line is seasonal-naive.](plots/gm_summary.png)

The rest of this note is what that new example actually is, why we thought it would help, and how
we know the sliver of difference left over is noise.

## What we tried

A contrastive backbone learns from a single promise: a window's forecast should look like its own
future and not like its present. There is a lazy way to keep that promise. Instead of reading the
data, the model can just track where it sits in time, giving every series the same internal
clock, and the clock by itself is enough to tell future from present. The recipe we began with
takes that shortcut away by mixing in synthetic series that copy a real past and then drift into a
different future, so that two windows can share a past and still owe you different answers.

The crossfade goes after the same shortcut with real data instead of synthetic. Take two real
windows from the same batch, keep the past of one and the future of the other, and fade gently
from the first into the second; both windows also remain in the batch as themselves. The spliced
window therefore shares its past with one neighbour and its future with another, and the only way
to keep the three apart is to read the content rather than the position.

![The primitive on one example pair: the crossfade (black) copies window A's past, ramps across a
transition band, and copies window B's future; below, the blend weight rises from 0 to 1. Each
window is normalised first, and the midpoint and width of the transition are drawn fresh for every
sample and shared across channels.](plots/crossfade_schematic.png)

## What we got

The crossfade windows are a tenth of every batch, and nothing else about the recipe changes. The
point estimates do move the right way: the 2-layer forecasting head improves from 1.222 to 1.208,
the 6-layer from 1.191 to 1.178, the latter nominally the best single score we have on record. But
neither move is large enough to trust.

![Change in forecast error from adding the crossfade. Both bars are negative, a touch better, but
both 90% intervals cross zero, so neither is distinguishable from no change at all.](plots/delta.png)

To tell a real improvement from luck we use a **paired bootstrap** over the 97 tasks. Draw the 97
tasks with replacement two thousand times; for each draw, score both models on the same drawn
tasks and take the difference of their geometric-mean errors; then report the range that holds the
middle 90% of those two thousand differences. Because both models are always scored on the very
same resampled tasks, the difficulty of any one task cancels out, and what is left reflects only
the difference between the models. Here both ranges sit comfortably across zero, so the gains of
about a hundredth are inside the noise.

| forecasting head | best recipe | + crossfade | change | 90% interval |
|---|--:|--:|--:|:--:|
| 2-layer | 1.222 | **1.208** | −0.014 | (−0.040, +0.012) |
| 6-layer | 1.191 | **1.178** | −0.013 | (−0.039, +0.012) |

Splitting the same scores by data domain tells the story with a little more texture. The
crossfade pulls a few small domains in and pushes one large domain out, and the two cancel.

![Forecast error by data domain, both heads. The two profiles nearly coincide. The crossfade
helps a little on Healthcare and Econ/Fin and costs a little on Web/CloudOps, leaving the
task-weighted total flat.](plots/perdomain.png)

The places it helps are the smallest and noisiest domains, only a handful of tasks each; the one
domain where it clearly costs something is several times larger. No domain shows the kind of
broad, steady gain that would survive a careful look.

## A harder negative the forecaster cannot use

The crossfade is not inert; it measurably reshapes the training. The clearest read-out is the
**gap**, how much more a forecast resembles the true future than the present, where a larger gap
means a sharper representation. With the crossfade the gap climbs higher than the recipe reaches
on its own, from about 1.03 to about 1.18, and the contrastive loss settles higher to match.
Sharing a real past and a real future with batch-mates is a harder separation than the synthetic
look-alike alone, and the backbone answers by encoding the future more strongly. The
representation stays healthy the whole way: different series keep pushing apart, with none of the
collapse this larger training batch can otherwise fall into.

![Training curves, with the crossfade (solid) against the recipe alone (dashed), from step 100.
Left, the contrastive loss falls cleanly on both. Middle, the gap climbs higher with the
crossfade. Right, different series stay near-orthogonal throughout, so nothing has
collapsed.](plots/training_curves_loglog.png)

That a larger gap buys no lower forecast error is the lasting lesson here. The gap measures how
well the backbone solves its own training puzzle, not how well it forecasts; the crossfade makes
the puzzle harder without handing the forecaster anything it can use.

## Protocol

One change from the best recipe: a tenth of every batch is crossfade windows, built from the real
windows already in that batch, so they cost no extra data. Everything else is held fixed: the
same backbone, the same large contrastive batch with all of its samples pooled into a single set
of negatives, the same two attention normalisations that batch needs in order to train without
diverging, the same tenth of synthetic look-alikes, and the same optimiser, learning rate,
temperature, and random seed. Each finished backbone is frozen and scored by training a fresh
forecasting head on top of it, once with two layers and once with six, and evaluating on the GIFT-Eval
benchmark of 97 tasks (with an 11-task subset for quick checks). Only one backbone was trained, so
the intervals above are taken over tasks, not over training seeds.

## What we learned

A real-window crossfade is a sound idea for a hard negative, and it does bend the contrastive
geometry the way we hoped. On top of an already-strong set of negatives, though, that extra effort
never reaches the forecaster. The small, consistent lean toward the crossfade on both heads is the
only hint of an effect; the cheapest way to confirm it or rule it out would be to train the recipe
again under two or three different random seeds, which a test paired across seeds could settle
where a single seed cannot. Until then the honest reading is a clean tie, and a tie is not worth
the extra machinery.
