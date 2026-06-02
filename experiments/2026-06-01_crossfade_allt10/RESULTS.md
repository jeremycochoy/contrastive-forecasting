# A regime crossfade as a contrastive hard negative

We took our best forecasting backbone and added a new kind of training example, built by splicing
the recent past of one real series onto the future of another. Averaged across the benchmark it
changed nothing we can measure: the model with the new example and the model without it score the
same, well inside the uncertainty of the comparison. That flat average is not the whole story,
though. Domain by domain the crossfade reliably improves some kinds of series and reliably worsens
another, and across the full benchmark those gains and losses come out even. On this single
training seed the clearest of those moves is what looks like a genuine improvement on healthcare
series, and it is the most promising thread to pull if a run with more seeds confirms it.

*Forecast error is **GM-Relative MASE**: the geometric mean, over the benchmark's 97 tasks, of a
model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is the
seasonal-naive baseline.*

![Forecast error for each forecasting head, best recipe versus the same recipe with the crossfade
added. The pairs sit within a whisker of each other. The dashed line is the strongest backbone
from before this line of work; the dotted line is seasonal-naive.](plots/gm_summary.png)

The rest of this note is what the new example is, what the benchmark-wide numbers are, and where,
domain by domain, the crossfade actually moves the needle.

## What we tried

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

## What we got, overall

The crossfade windows are a tenth of every batch, and nothing else about the recipe changes. The
point estimates land a touch lower, the 2-layer head from 1.222 to 1.208 and the 6-layer from 1.191
to 1.178, but to weigh a move that small against task-set luck we use a **paired bootstrap**. Draw
the 97 tasks with replacement two thousand times; for each draw, score both models on the same
drawn tasks and take the difference of their geometric-mean errors; then report the range that
holds the middle 90% of those differences. Because both models always see the same resampled tasks,
per-task difficulty cancels, and the range reflects only the difference between the models. Both
ranges sit squarely across zero, so neither overall move is distinguishable from no change.

![Change in forecast error from adding the crossfade. Both bars are negative, a touch better, but
both 90% intervals cross zero, so neither is distinguishable from no change.](plots/delta.png)

| forecasting head | best recipe | + crossfade | change | 90% interval |
|---|--:|--:|--:|:--:|
| 2-layer | 1.222 | 1.208 | −0.014 | (−0.040, +0.012) |
| 6-layer | 1.191 | 1.178 | −0.013 | (−0.039, +0.012) |

The 6-layer 1.178 is nominally the lowest single figure we have on record, below the recipe's own
1.191, but the interval says not to lean on it.

## What we got, by domain

That benchmark-wide flatness hides a real redistribution. Running the same paired bootstrap inside
each data domain, three effects clear zero on both heads, and they do not all point the same way.

![Forecast error by domain, both heads, with the parent experiment's strongest per-domain arm (its
0.8%-fork variant) carried along for reference. Closer to the centre is better. The crossfade (solid
blue) sits inside the best recipe on Healthcare, Transport and Econ/Fin and outside it on
Web/CloudOps.](plots/perdomain.png)

| domain | tasks | change, 2-layer | change, 6-layer |
|---|--:|--:|--:|
| Healthcare | 5 | **−0.20** | **−0.27** |
| Econ/Fin | 6 | −0.08 | −0.07 |
| Transport | 15 | **−0.04** | **−0.03** |
| Energy | 32 | 0.00 | −0.02 |
| Sales | 4 | +0.02 | +0.01 |
| Nature | 15 | +0.01 | **+0.02** |
| Web/CloudOps | 20 | **+0.03** | **+0.07** |

*Negative means the crossfade is better. **Bold** marks a change whose whole 90% paired-bootstrap
interval lies on one side of zero; the rest are within noise.*

Healthcare and Transport improve reliably on both heads, Web/CloudOps worsens reliably on both, and
everything else stays within noise. The reliable wins sit on the smallest domains, where even a
one-sided interval is wide (Healthcare is five tasks), while the reliable loss sits on a domain four
times that size. Combined by task count over the whole benchmark, the two sides land even, which is
the flat overall number above. So the crossfade is not adding accuracy; it is moving it from one
kind of series to another. The parent experiment's strongest per-domain arm bends the same domains
inward, but the crossfade gives up the most of the three on Web/CloudOps.

## What changed in training

The crossfade does change the contrastive training, visibly. The **gap**, how much more a forecast
resembles the true future than the present, settles higher with the crossfade than the recipe
reaches on its own, from about 1.03 to about 1.18, and the contrastive loss settles higher
alongside it. Throughout, different series stay near-orthogonal, so nothing has collapsed.

![Training curves, with the crossfade (solid) against the recipe alone (dashed), from step 100.
Left, the contrastive loss falls cleanly on both, settling a little higher with the crossfade.
Middle, the gap climbs higher with the crossfade. Right, different series stay near-orthogonal
throughout, so nothing has collapsed.](plots/training_curves_loglog.png)

The training signal moved and the benchmark-wide forecast did not. From one run we cannot say which
way the causation runs, only that a higher gap did not, by itself, buy a lower overall error.

## Protocol

One change from the best recipe: a tenth of every batch is crossfade windows, built from the real
windows already in that batch, so they cost no extra data. Everything else is held fixed: the same
backbone, the same large contrastive batch with all of its samples pooled into a single set of
negatives, the same two attention normalisations the recipe uses at this batch size, the same tenth
of synthetic look-alikes, and the same optimiser, learning rate, temperature, and random seed. Each
finished backbone is frozen and scored by training a fresh forecasting head on top of it, once with
two layers and once with six, and evaluating on the GIFT-Eval benchmark of 97 tasks (with an
11-task subset for quick checks). Only one backbone was trained, so every interval here is taken
over tasks, not over training seeds.

## What we learned

The crossfade is neither inert nor a free win. It reliably shifts where the backbone is accurate,
helping a couple of domains and hurting another, and across the benchmark those moves cancel. If the
goal is a lower benchmark-wide number, this is a tie and not worth the extra machinery. If the goal
were the domains it helps, the trade might be worth making, though on a single training seed the
honest reading of even the per-domain effects is "promising, not settled." The cheapest way to firm
any of it up is to repeat the run under two or three seeds, which a test paired across seeds could
settle where a single seed cannot.
