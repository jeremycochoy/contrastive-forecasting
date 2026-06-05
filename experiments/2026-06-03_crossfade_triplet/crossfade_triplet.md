# An explicit crossfade triplet on a wider, shallower backbone

We took our strongest per-domain backbone recipe (the base) and changed three things at once, each
expected to help the downstream forecast: we replaced the earlier regime-crossfade slice with an
**explicit triplet** (both parent windows, z-normalised, plus their blend, all added to the batch);
we **removed the forecaster bottleneck** so the forecaster runs at the encoder's full width; and we
**halved the pre-forecaster encoder** from six layers to three. On a single training seed the
combined arm does not improve the benchmark forecast: both forecasting heads come out slightly worse
than the base, by less than the task-set noise, and the arm also trails the previous best recipe.

*Forecast error is **GM-Relative MASE**: the geometric mean, over the benchmark's 97 tasks, of a
model's error divided by the seasonal-naive forecast's error. Lower is better; 1.0 is
seasonal-naive.*

![Forecast error for each head, the base versus the combined arm. In each pair the arm (dark) is a
touch higher than the base (light). The dotted line is seasonal-naive.](plots/gm_summary.png)

## What we got, overall

Both point estimates move the wrong way — the combined arm is slightly worse on both heads. A
**paired bootstrap** over the 97 tasks (resample the task list with repeats many times, scoring both
models on each resampled list so per-task difficulty cancels) puts a 90% interval on each change.
Both intervals cover zero, so neither aggregate difference is larger than the luck of which tasks the
benchmark happens to contain.

| forecasting head | base | combined arm | change | 90% interval |
|---|--:|--:|--:|:--:|
| 2-layer | 1.213 | 1.220 | +0.007 | (−0.014, +0.028) |
| 6-layer | 1.198 | 1.211 | +0.013 | (−0.012, +0.036) |

The arm is also worse than the previous best recipe (a 10% crossfade slice on a richer-synthetic
base: 1.208 / 1.178). And it carries **more** parameters than the base — 16.7M against 12.7M, because
widening the forecaster outweighs dropping three encoder layers — so the larger model is no more
accurate.

## What we got, by domain

Splitting the aggregate by domain shows a redistribution.

![Per-domain change in error from the combined arm, both heads, with the 90% paired-bootstrap
interval for each domain (task count in brackets). Green is a reliable improvement, the whole
interval below zero; red a reliable worsening; grey within noise.](plots/perdomain_delta.png)

The only effect reliable on **both** heads is a worsening on Web/CloudOps (+0.064 and +0.079 on a
20-task domain), the largest single move. On one head each, the arm reliably worsens Transport
(2-layer) and Sales (6-layer) and reliably improves Healthcare (2-layer). Econ/Fin shows the largest
nominal improvement on both heads, but on six tasks its interval is far too wide to call. By task
count the reliable losses are larger than the single reliable win, which is consistent with the
slightly-worse — though noisy — aggregate.

## How the arm works

A contrastive backbone is promised that a window's forecast should look like its own future, not its
present; a crossfade denies the lazy shortcut of tracking position in time by splicing one real
window's past onto another's future, so position no longer separates the spliced window from its
neighbours. The earlier recipe added only the blend C and left its parents as their raw selves in the
batch. Here we add the **triplet** — both parents z-normalised (A_norm, B_norm) alongside the blend C
— so C sits in the batch with its two exact, same-scale parents: A_norm shares C's past, B_norm
shares C's future. Only this one triplet (three windows) is added per step, on top of the base
recipe's natural batch.

![The crossfade triplet on one example pair: both parents are z-normalised and added to the batch
(A_norm, B_norm), together with the blend C, which copies A's past, ramps across a transition band,
and copies B's future. Below, the blend weight rises 0→1 across the band.](plots/triplet_schematic.png)

The other two changes are architectural: the forecaster drops its 128-wide bottleneck and runs at the
encoder width (384), and the pre-forecaster encoder goes from six causal layers to three.

## Training dynamics

The arm trains cleanly and does not collapse.

![Training metrics, log-log, combined arm (solid) vs the base (dashed), from step 100. Top, lower is
better: contrastive loss, the ratio gap (1−ff)/(1−fp) where ff is the forecast-to-future cosine and
fp the forecast-to-present cosine, and 1−R²_naive, then 1−R²_random — all decay toward zero on both.
Bottom, higher is better: U_batch and U_temporal, the fraction of embedding dimensions actually
used.](plots/training_metrics.png)

The contrastive loss falls, the ratio gap and the skill metrics (R² against naive and random
baselines) converge to the same neighbourhood as the base, and the used-dimension counts stay well
above collapse. The forecast-to-future cosine reaches ~0.99 on both, and different series stay
near-orthogonal throughout (cross-series cosine ≲0.003), so nothing collapsed. The combined arm's
contrastive loss settles slightly higher late in training. Each backbone is evaluated at its
**best-loss checkpoint** (the step with the lowest contrastive loss).

## Protocol

Three changes from the base recipe (explicit crossfade triplet; full-width forecaster; three encoder
layers), everything else held fixed. The base trains a contrastive backbone at batch 1024 with all
samples pooled into one set of negatives, on a mix that is 99.2% real series and 0.8% synthetic
**forked-ARMA** (pairs of ARMA series that share a prefix then diverge — a hard negative for
position), under an **all-time** contrastive loss (it separates different series at every time lag,
not just aligned positions). Held fixed with it: the two stabilisers the base needs at this batch
size — **QK-norm** and an **attention-output norm** — and **floor-subtraction** (re-basing the loss
by its uniformity floor), plus the same optimiser, learning rate, temperature, and random seed. The
finished backbone is frozen and scored by training a fresh forecasting head on top — once with two
layers, once with six — and evaluating on GIFT-Eval's 97 tasks (with an 11-task subset for quick
checks). One backbone was trained, so every interval here is taken over tasks, not over training
seeds. Because three things changed together, the result is their **joint** effect and cannot be
attributed to any single change.

## What we learned

The three changes, applied together, do not lift the downstream forecast on this seed: both heads are
slightly worse than the base and the whole arm trails the previous best recipe, though the aggregate
move is within task-set noise. The only effect reliable on both heads is a worsening on Web/CloudOps.
The cheapest way to firm up the small aggregate move — or to learn which of the three changes is
responsible — is to run the changes one at a time, across two or three seeds.
