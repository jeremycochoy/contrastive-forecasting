# Crossfade triplet on the 0.8%-fork base — disentangling triplet, bottleneck, and encoder depth

## Question

Issue #328 changed three things at once on top of the strongest per-domain backbone (the
**base**: 6-layer encoder, a 128-wide bottleneck forecaster, no crossfade): it added an explicit
crossfade **triplet** to the batch, **dropped the forecaster bottleneck** so the forecaster runs at
the encoder width (384), and **shrank the encoder from six layers to three**. Does this combined
arm — **L3+nobn+triplet** — improve the downstream forecast, and which of the three changes is
responsible?

*Forecast error is **GM-Relative MASE**: the geometric mean, over GIFT-Eval's 97 tasks, of a model's
error divided by the seasonal-naive forecast's error. Lower is better; 1.0 = seasonal-naive. Each
frozen backbone is scored by training a quantile forecasting head on top — once with two layers
(**2L**), once with six (**6L**) — and evaluating on all 97 tasks. A **paired bootstrap** over the
97 tasks (resample the task list with repeats, score both models on each resample so per-task
difficulty cancels) gives a 90% interval on each arm's difference from the base. One backbone seed
per arm, so every interval is over tasks, not over training seeds.*

## Result — checkpoint selection decides the verdict

The combined arm **L3+nobn+triplet** is neutral at its best-loss checkpoint but reliably better than
the base after full training (the last checkpoint).

| arm vs base | head | best-loss GM | Δ (90% CI) | last GM | Δ (90% CI) |
|---|---|--:|:--:|--:|:--:|
| L3+nobn+triplet | 2L | 1.220 | +0.007 (−0.014, +0.028) | **1.181** | **−0.032 (−0.050, −0.014)** |
| L3+nobn+triplet | 6L | 1.211 | +0.013 (−0.012, +0.036) | **1.170** | **−0.029 (−0.049, −0.011)** |

Base GM is 1.213 (2L) and 1.198 (6L). At the best-loss checkpoint both intervals cover zero, so the
arm is within task-set noise. At the last checkpoint both intervals sit fully below zero on both
heads, so the arm reliably beats the base. Selecting the backbone by lowest contrastive loss hid
this gain; evaluating at full training revealed it. (The two checkpoint columns also differ in head
protocol — fresh 30k-step heads at best-loss, the same heads re-adapted a further 10k steps at
last, identically for every arm; see Protocol.)

## Disentanglement — the gain tracks the triplet, not bottleneck removal

Running the three changes one at a time (same base, same seed, same budget) separates their effects.
The figure below reads every arm trained to the standard 12.5k budget against the base at both
checkpoints and both heads; this section covers the component arms, the next section the two arms
that add the triplet to other hosts, and the 25k extension is read separately below.

![Change in GM-Relative MASE vs the base for every 12.5k-budget arm at the best-loss (solid) and
last (hatched) checkpoints, 2L and 6L heads. Bars below zero are improvements; green and red mark a
90% interval clear of zero (reliably better / reliably worse), grey is within noise.](plots/gm_summary.png)

| arm | head | best-loss Δ (90% CI) | last Δ (90% CI) |
|---|---|:--:|:--:|
| L3 (3L encoder only) | 2L | +0.027 (+0.006, +0.049) worse | −0.002 (−0.025, +0.020) ns |
| L3 | 6L | +0.017 (−0.003, +0.037) ns | +0.064 (+0.040, +0.088) worse |
| L6+nobn (full-width forecaster) | 2L | +0.041 (+0.015, +0.067) worse | +0.062 (+0.046, +0.078) worse |
| L6+nobn | 6L | +0.077 (+0.042, +0.109) worse | +0.021 (+0.001, +0.039) worse |
| L3+nobn | 2L | −0.001 (−0.020, +0.019) ns | +0.003 (−0.015, +0.020) ns |
| L3+nobn | 6L | +0.019 (−0.012, +0.049) ns | −0.015 (−0.029, −0.002) better |
| **L3+nobn+triplet** | 2L | +0.007 (−0.014, +0.028) ns | **−0.032 (−0.050, −0.014) better** |
| **L3+nobn+triplet** | 6L | +0.013 (−0.012, +0.036) ns | **−0.029 (−0.049, −0.011) better** |

Reading down the table:

- **Dropping the bottleneck alone (L6+nobn)** is reliably worse than the base on both heads at both
  checkpoints — the only arm reliably worse in all four head × checkpoint cells.
- **Shrinking the encoder alone (L3)** is inconsistent: reliably worse at 2L-best and at 6L-last,
  neutral elsewhere.
- **L3+nobn** (both architecture changes, no triplet) is neutral except 6L-last, where it is
  marginally better (−0.015, upper bound −0.002).
- **L3+nobn+triplet** is the only arm in this table reliably better on both heads, and only at full
  training.

Adding the triplet is the only change that turns the arm into a reliable both-head improvement, so
the full-training gain tracks the triplet rather than the architecture changes alone. The direct
paired contrast of L3+nobn+triplet against L3+nobn at the last checkpoint puts the triplet's own
increment at −0.035 (−0.050, −0.020) at 2L (reliable) and −0.014 (−0.030, +0.002) at 6L — same
direction, within noise, on top of L3+nobn's own marginal 6L-last gain (−0.015).

## Triplet isolation — the triplet alone is sufficient on the base

The headline arm bundles the triplet with two architecture changes. Two further arms isolate the
triplet's effect: each row below differs from its own reference by the triplet alone.

| arm | difference from its reference | result |
|---|---|---|
| **base+triplet** | adds only the triplet to the unmodified base | **Reliably better at both heads at best-loss, and at 2L robust to checkpoint choice**: 2L best 1.186 (Δ −0.027 [−0.042, −0.011]), 2L last 1.187 (Δ −0.025 [−0.042, −0.008]), 6L best 1.185 (Δ −0.013 [−0.025, −0.002]), 6L last 1.190 (Δ −0.008 [−0.020, +0.003], same direction, ns). |
| L6+nobn+triplet | adds the triplet to L6+nobn | Ends reliably worse than the base on both heads and degrades with training: 2L 1.274 → 1.313 (last Δ +0.100), 6L 1.225 → 1.260 (last Δ +0.062). |

Two readings follow.

**The triplet alone is sufficient.** On the unmodified base it is reliably better at both heads at
the best-loss checkpoint, without any architecture change — and unlike the headline arm, whose
verdict flips between checkpoints, its 2L gain does not depend on checkpoint selection (best −0.027,
last −0.025, both reliable; at 6L the last checkpoint keeps the direction at −0.008 but is within
noise). Its 2L gain is comparable to the headline arm's (−0.032 at last); the architecture changes
are not needed for the effect. The predecessor #326 (a C-only crossfade slice on the 10%-fork base)
gave ≈−0.013 over its base — the same direction, smaller.

**The failure case is the no-bottleneck 6-layer configuration, not 6-layer encoders.** The triplet
helps on the base (6-layer encoder, with bottleneck) and, at full training, on L3+nobn — but at full
training it reliably hurts L6+nobn (direct paired contrast against L6+nobn: 2L +0.039
[+0.017, +0.062], 6L +0.042 [+0.017, +0.067]), the one configuration that is already reliably worse
than the base on its own.

## Train-longer hurts

Extending the combined backbone from 12,500 to 25,000 steps (fresh data for the extra steps,
equivalent to a single 25k-step run) does not help. This eval uses a fresh 30k head — the same
protocol as the best-loss evals — so the controlled comparisons are against the base and against the
fresh-head best-loss checkpoint (step 6,400: 1.220 / 1.211): the step-25k backbone is reliably worse
than the base on both heads, and worse than the best-loss point reliably at 2L (same direction,
within noise, at 6L).

| arm (step 25k, fresh head) | head | GM | Δ vs base (90% CI) | Δ vs step-6.4k (90% CI) |
|---|---|--:|:--:|:--:|
| L3+nobn+triplet @25k | 2L | 1.249 | +0.037 (+0.018, +0.055) worse | +0.030 (+0.002, +0.059) worse |
| L3+nobn+triplet @25k | 6L | 1.222 | +0.024 (+0.013, +0.035) worse | +0.011 (−0.014, +0.037) ns |

25k steps is ≈0.6 of one epoch (the pretrain set `small_v1` holds 42,571,692 windows; at 1,016 real
rows/step one pass ≈ 41,900 steps), so the degradation arrives within the first epoch — data
starvation is not the cause. *Protocol note:* the 12.5k headline result (1.181 / 1.170) used a
re-adapted head, not a fresh one, so it is not on the same head-protocol curve as the fresh-head
points here and is not used for the train-longer contrast.

## Training dynamics — pretext loss does not predict transfer

The combined arm trains cleanly and does not collapse. The panels below overlay all arms on log-log
axes.

![Training dynamics, log-log, all arms vs the base (dashed grey), from step 100 with a 120-step
moving average. Lower is better for: contrastive loss above its InfoNCE floor; the ratio gap
(1−ff)/(1−fp), where ff is the forecast-to-future cosine and fp the forecast-to-present cosine; and
1−R²_naive and 1−R²_random, the share of the next latent the forecaster leaves unexplained relative
to a latent-persistence / random-pair baseline. Higher is better for: U_batch and U_temporal, the
fraction of embedding dimensions in use.](plots/disentangle_metrics.png)

At step 12,500 the extremes of the pretext task sit in the opposite order downstream:

- The **triplet raises the contrastive loss** (1.065 for L3+nobn+triplet vs 0.895 for the base) —
  the hardest pretext among all arms — yet it is the arm that improves downstream.
- **Dropping the bottleneck lowers the contrastive loss and raises used dimensions** (L6+nobn
  reaches loss 0.662 and U_batch 0.197, the easiest pretext and the most-used embedding among the
  no-triplet arms) — yet it is the only arm reliably worse than the base in all four cells.

On this benchmark a lower contrastive loss is not a sign of a better forecasting backbone. A harder
pretext is not required for a gain either: base+triplet leaves the pretext loss essentially at the
base value (0.881 vs 0.895) and still transfers better. No arm collapsed: every used-dimension count
stays well above zero and forecast-to-future cosine reaches ~0.99.

## Protocol

Each arm starts from the base recipe and changes only the stated components, holding everything else
fixed: the base trains a contrastive backbone at global batch 1024 with all samples pooled into one
set of negatives, on a mix that is 99.2% real series and 0.8% synthetic **forked-ARMA** (pairs of
ARMA series that share a prefix then diverge — a hard negative for position), under an **all-time**
contrastive loss (it separates different series at every time lag, not just aligned positions). Held
fixed: the two batch-1024 stabilisers (**QK-norm** and an **attention-output norm**),
**floor-subtraction** (re-basing the loss by its InfoNCE floor), the optimiser, learning rate,
temperature, and the random seed. The **triplet** adds, per step, one z-normalised window pair
(A_norm, B_norm) and their blend C on top of the natural batch — C copies A's past, ramps across a
transition band, and copies B's future, so position no longer separates the spliced window from its
neighbours; the annex shows one example.

Each backbone is scored frozen, under two head protocols. At the **best-loss checkpoint** (lowest
contrastive loss) a fresh 2L and a fresh 6L quantile head are trained on top (30k steps, batch 256)
— the same fresh-head protocol as the base's own score, which uses the base's fully-trained backbone
(12,500 steps). At the **last checkpoint** (step 12,500) the arm's best-loss head is resumed and
re-adapted for 10k steps on the final backbone — the same re-adapt protocol for every arm. The 25k
extension is scored with a fresh 30k head. Heads are evaluated on GIFT-Eval (forecast strategy
**B4**: roll the forecast out in latent space, decode each step with the head) over the full 97
tasks, and every delta is paired against the same base score. Because the headline arm changes three
things together, its delta against the base measures their joint effect; the disentanglement arms
isolate the components.

## What we learned

1. **Checkpoint selection flips the verdict.** L3+nobn+triplet is neutral at best-loss (2L +0.007,
   6L +0.013, both CIs straddle zero) but reliably beats the base at full training (2L −0.032
   [−0.050, −0.014]; 6L −0.029 [−0.049, −0.011]).
2. **The gain tracks the triplet.** Dropping the bottleneck alone is reliably worse on both heads;
   shrinking the encoder alone is inconsistent; of the disentanglement arms, only the full
   combination with the triplet is reliably better on both heads, and only at full training.
3. **The triplet alone is sufficient on the base.** base+triplet is reliably better at both heads at
   best-loss and keeps its 2L gain at the last checkpoint (−0.027 / −0.025); no architecture change
   is needed.
4. **Train-longer hurts.** Extending to 25k reliably worsens both heads against the base (2L +0.037,
   6L +0.024) and sits above the fresh-head best-loss point (reliably at 2L) — all within the first
   epoch.
5. **Pretext loss does not predict transfer in either direction.** Dropping the bottleneck gives the
   easiest pretext among the no-triplet arms yet is reliably worse in all four cells; the
   headline arm raises the loss and transfers better; base+triplet leaves the loss essentially
   unchanged (0.881 vs 0.895) and still transfers reliably better. Checkpoint and arm selection
   should not be driven by the contrastive loss.

## Follow-up — two candidate recipes and a stop-gradient test

Two configurations improve on the base and carry forward: **base+triplet** (128-wide bottleneck
kept) — the smaller gain but robust to checkpoint choice at 2L, and the simplest change — and
**L3+nobn+triplet** (full-width forecaster) — the best single number (6L last 1.170) but only at
full training, and it needed the encoder shrink to work.

*Hypothesis — beyond what the data shows directly.* The L6+nobn results are consistent with the
bottleneck protecting the objective from being exploited: with a full-width forecaster and no other
change, the contrastive loss falls to 0.662 (base: 0.895) while the arm is reliably worse than the
base in all four cells, i.e. the extra width appears to serve the pretext rather than the forecast.
If the objective were fixed so that a full-width forecaster cannot exploit it, the no-bottleneck line
would be the preferred direction — the forecaster would learn at the encoder's width instead of
through a 128-wide pinch. The planned stop-gradient run (stop-grad on the encoder side of the
positive term, numerator and denominator) is exactly such an attempt.

---

## Annex

### Per-domain breakdown (combined arm vs base)

Both splits below break the combined arm's aggregate delta out by GIFT-Eval domain (task counts in
brackets): the benchmark-wide paired bootstrap restricted to each domain's tasks.

![Per-domain change in error for the combined arm at its last checkpoint, both heads, with the 90%
paired-bootstrap interval per domain. Green = reliable improvement, red = reliable worsening, grey =
within noise.](plots/perdomain_delta_last.png)

At the last checkpoint — where the aggregate gain is — the only both-head reliable effect is an
improvement on Energy, the largest domain (−0.066 at 2L, −0.089 at 6L, 32 tasks). Nature and
Healthcare improve reliably at 2L, Sales at 6L; Transport worsens reliably at 2L (+0.036).

![Per-domain change in error for the combined arm at its best-loss checkpoint, both heads, with the
90% paired-bootstrap interval per domain. Green = reliable improvement, red = reliable worsening,
grey = within noise.](plots/perdomain_delta.png)

At the best-loss checkpoint — where the aggregate is neutral — the only both-head reliable effect is
a worsening on Web/CloudOps (+0.064 at 2L, +0.079 at 6L); that worsening is gone at the last
checkpoint (2L −0.000, 6L +0.011, both within noise). On one head each, the arm reliably worsens
Transport (2L) and Sales (6L) and reliably improves Healthcare (2L). Econ/Fin shows the largest
nominal improvement on both heads but, on six tasks, its interval is far too wide to call.

### Crossfade triplet schematic

![The crossfade triplet on one example pair: both parents are z-normalised and added to the batch
(A_norm, B_norm), together with the blend C, which copies A's past, ramps across a transition band,
and copies B's future. Below, the blend weight rises 0→1 across the band.](plots/triplet_schematic.png)

### Arm definitions and parameter counts

| arm | encoder layers | forecaster width | triplet | backbone params |
|---|--:|--:|:--:|--:|
| base | 6 | 128 (bottleneck) | no | 12.7M |
| L3 | 3 | 128 (bottleneck) | no | 7.4M |
| L6+nobn | 6 | 384 (full width) | no | 22.1M |
| L3+nobn | 3 | 384 (full width) | no | 16.7M |
| L3+nobn+triplet (#328) | 3 | 384 (full width) | yes | 16.7M |
| L6+nobn+triplet | 6 | 384 (full width) | yes | 22.1M |
| base+triplet | 6 | 128 (bottleneck) | yes | 12.7M |

The triplet changes only the training batch, so each triplet arm matches its host's parameter count.
The combined #328 arm carries 16.7M parameters against the base's 12.7M — widening the forecaster
outweighs dropping three encoder layers — so its last-checkpoint gain comes with more parameters
than the base. base+triplet improves on the base at the base's own 12.7M.
