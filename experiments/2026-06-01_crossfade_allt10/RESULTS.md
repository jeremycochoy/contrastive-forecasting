# A regime crossfade as a contrastive hard negative

We took our best forecasting backbone and added a new kind of training example, built by splicing
the recent past of one real series onto the future of another. Averaged across the benchmark it
changed nothing we can measure: the model with the new example and the model without it score the
same, well inside the uncertainty of the comparison. Domain by domain, though, the crossfade
reliably improves some kinds of series and reliably worsens another, and across the full benchmark
those gains and losses cancel. On this single training seed the clearest of those moves is what
looks like a real improvement on healthcare series, the most promising lead to follow up with more
seeds.

*Forecast error is **GM-Relative MASE**: the geometric mean, over the benchmark's 97 tasks, of a
model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is the
seasonal-naive baseline.*

![Forecast error for each forecasting head, best recipe versus the same recipe with the crossfade
added. The two bars in each pair are nearly identical. The dashed line is the strongest backbone
from before this line of work; the dotted line is seasonal-naive.](plots/gm_summary.png)

## What we got, overall

The crossfade windows are 10% of every batch. Both point estimates come out slightly lower with the
crossfade, the 2-layer head from 1.222 to 1.208 and the 6-layer from 1.191 to 1.178, but a
difference that small needs a check. A **paired bootstrap** is that check: it asks whether the
difference would survive if the benchmark had drawn a different set of tasks. We rebuild the task
list many times by sampling the 97 tasks at random, with repeats, recompute the gap between the two
models' average errors on each rebuilt list, and read off how far that gap moves. *Paired* means
both models are scored on the same rebuilt list, so per-task difficulty cancels and only the gap
between the two models is left. Both spreads cover zero, so neither overall difference is larger
than the luck of which tasks the benchmark happens to contain.

| forecasting head | best recipe | + crossfade | change | 90% interval |
|---|--:|--:|--:|:--:|
| 2-layer | 1.222 | 1.208 | −0.014 | (−0.040, +0.012) |
| 6-layer | 1.191 | 1.178 | −0.013 | (−0.039, +0.012) |

The 6-layer 1.178 is the lowest single figure we have on record, below the recipe's own 1.191, but
the interval says not to lean on it.

## What we got, by domain

That benchmark-wide flatness hides a real redistribution. Running the same paired bootstrap inside
each data domain, the crossfade reliably improves Healthcare and Transport on both heads and
reliably worsens Web/CloudOps on both; the rest stays within noise.

![Per-domain change in error from the crossfade, both heads, with the 90% paired-bootstrap interval
for each domain (the domain's task count in brackets). Green is a reliable improvement, the whole
interval below zero; red a reliable worsening; grey within noise.](plots/perdomain_delta.png)

The reliable wins sit on the smallest domains, where even a one-sided interval is wide (Healthcare
is five tasks), while the reliable loss sits on a domain four times that size. Combined by task
count across the whole benchmark, the two sides cancel, which is the flat overall number above. So
the crossfade is not adding accuracy; it is moving it from one kind of series to another.

In absolute terms all three profiles sit close together, with the parent experiment's strongest
per-domain arm carried along for reference: it improves the same domains, but the crossfade is the
worst of the three on Web/CloudOps.

![Forecast error by domain, both heads, best recipe versus the crossfade versus the parent's
0.8%-fork arm. Closer to the centre is better; the dotted ring is seasonal-naive.](plots/perdomain.png)

## How the crossfade works

A contrastive backbone learns from a single promise: a window's forecast should look like its own
future and not like its present. There is, in principle, a lazy way to keep that promise. Rather
than read the data, the model can track where it sits in time, giving every series the same
internal clock, and the clock by itself separates present from future. The recipe we began with is
built to deny that shortcut: it mixes in synthetic series that copy a real past and then drift into
a different future, so two windows can share a past and still owe you different answers.

The crossfade goes after the same shortcut with real data. Take two real windows from the same
batch, keep the past of one and the future of the other, and fade gently from the first into the
second; both windows also remain in the batch as themselves. The spliced window then shares its
past with one neighbour and its future with another, so position no longer separates the three.

![The primitive on one example pair: the crossfade (black) copies window A's past, ramps across a
transition band, and copies window B's future; below, the blend weight rises from 0 to 1. Each
window is normalised first, and the midpoint and width of the transition are drawn fresh for every
sample and shared across channels.](plots/crossfade_schematic.png)

## What we observed in the training dynamics

The crossfade does change the training. The contrastive loss settles a little higher, and so does
the **gap** between two cosine similarities: how much more a forecast resembles the true future than
the present. The gap climbs from about 1.03 without the crossfade to about 1.18 with it.

![Training curves, crossfade (solid) versus the recipe alone (dashed), from step 100. Left, the
contrastive loss falls cleanly on both and settles a little higher with the crossfade. Right, the
gap climbs higher with the crossfade.](plots/training_curves_loglog.png)

But the gap is a difference of two parts, and splitting it shows the change is one-sided. The
forecast-to-future similarity, the part that would actually help forecasting, ends slightly *lower*
with the crossfade, 0.98 against the recipe's 0.99; it is the forecast-to-present similarity that
drops much further below zero. So the wider gap is the model pushing its forecast away from the
present, not pulling it closer to the future. Different series stay near-orthogonal throughout, so
nothing has collapsed.

![The four contrastive cosines through training (log step), best recipe (dashed) vs crossfade
(solid). Top-left is the forecast-to-future match, nearly identical on both. Top-right and
bottom-left, the forecast-to-present and future-to-present similarities both drop further with the
crossfade. Bottom-right, different series stay near zero, so nothing collapsed.](plots/cosines.png)

The training signal moved and the benchmark-wide forecast did not, and the split says why: the
crossfade sharpened the cheap half of the gap, the part that does not need a better forecast.

## Protocol

One change from the best recipe: 10% of every batch is crossfade windows, built from the real
windows already in that batch, so they cost no extra data. Everything else is held fixed: the same
backbone, the same large contrastive batch with all of its samples pooled into a single set of
negatives, the same two attention normalisations the recipe uses at this batch size, the same 10%
of synthetic series, and the same optimiser, learning rate, temperature, and random seed. Each
finished backbone is frozen and scored by training a fresh forecasting head on top of it, once with
two layers and once with six, and evaluating on the GIFT-Eval benchmark of 97 tasks (with an
11-task subset for quick checks). Only one backbone was trained, so every interval here is taken
over tasks, not over training seeds.

## What we learned

The crossfade is neither inert nor a free win. It reliably shifts where the backbone is accurate,
helping a couple of domains and hurting another, and across the benchmark those moves cancel. If the
goal is a lower benchmark-wide number, this is a tie and not worth the added complexity. If the goal
were the domains it helps, the trade might be worth making, though on a single training seed the
honest reading of even the per-domain effects is "promising, not settled." The cheapest way to firm
any of it up is to repeat the run under two or three seeds, which a test paired across seeds could
settle where a single seed cannot.
