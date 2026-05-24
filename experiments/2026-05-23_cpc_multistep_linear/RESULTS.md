# #316 — Does multi-step forecasting (k=12) improve β?

## Question

The backbone is trained by **contrastive forecasting**: a causal encoder turns
each patch of a series into a latent *h_t*; a small **forecaster** predicts the
next latent; an InfoNCE loss pulls the prediction toward the true future latent
*h_{t+1}* and away from negatives (other times, channels, series). The backbone
is then frozen and a quantile head is trained on its latents for **GIFT-Eval**,
scored as **GM-Relative MASE** — the geometric mean over 97 configs of
(model MASE ÷ seasonal-naive MASE); **lower is better**, 1.0 = seasonal naive.

The strongest recipe on this line, **β** (a 1-layer causal-transformer
forecaster with a d=128 bottleneck, AdamW β2=0.98), reaches **1.3272**, still
+2.7 % over the champion **v11c (1.292)**. β predicts only the **next** latent
(*k = 1*). Contrastive Predictive Coding (van den Oord et al. 2018) predicts
**several** steps ahead, on the theory that it forces the latent to hold more
forecastable structure. So:

> Holding β's architecture and negatives fixed, does predicting **k = 12** steps
> ahead instead of one improve transfer — or not change it, or hurt?

## Result (and an important caveat)

**Single-seed, k=12 transfers worse — but a two-seed check shows that margin is
inside the metric's seed noise, so the transfer verdict is _suggestive, not
established_.** Replacing β's single forecaster with **K = 12 transformer-1L
heads** (each *architecturally identical* to β's forecaster; head *k* predicts
*h_{t+k}*), keeping β's negatives unchanged and changing **only** the number of
forecast steps (at k=1 the loss is byte-identical to β), gives full-97 GM-MASE
**1.4781** (small head) vs β's 1.3272 — apparently +11.4 % worse, and the same
single-seed ordering repeats in all three forecaster families (below).

**The catch:** when we trained a *second seed* of one arm and evaluated it, the
two seeds came out **0.49 GM-MASE apart** (§"How reliable is this") — larger
than every k-effect in the study. So the single-seed k-effects are within
run-to-run noise; the downstream numbers can only say the trend *looks*
negative, not that k=12 reliably hurts. **What is robust** (a large, within-run,
multi-seed-consistent effect) is the *mechanism*: k=12 drives the encoder latent
to a markedly higher-dimensional, diffuse representation than k=1.

![gm summary](plots/gm_summary.png)

### The single-seed trend points one way (in every variant)

To rule out the forecaster *type* and the *negative set* as explanations, the
same k=1→k=12 change was run in three families. In all three **single-seed**
runs, k=12 is nominally worse than k=1 (small-head full-97 GM-MASE; lower =
better):

| forecaster family | k = 1 | k = 12 | k effect |
|---|---:|---:|---:|
| **transformer-1L heads, β negatives**  (k=1 ≡ β) | 1.3272 | 1.4781 | +0.151 |
| linear heads, β negatives                        | 1.4248 | 1.6635 | +0.239 |
| linear heads, CPC negatives                       | 1.4313 | 1.5240 · **2.0137**¹ | +0.09 … +0.58¹ |

_References: v11c = 1.292, β = 1.3272, (B) = 1.3572 (all small-head full-97)._
_¹ The CPC-neg k=12 cell has **two seeds** (1.5240 and 2.0137) — see the next
section. That 0.49 spread is the reason these k-effects are flagged as
suggestive, not significant._

The single-seed ordering is consistent, and a secondary, orthogonal effect is
visible: the **transformer forecaster scores better than the linear one even at
k=1** (β = 1.3272 vs linear k=1 ≈ 1.43). But before reading anything into the
+0.09…+0.24 k-effects, look at how far apart two seeds of the *same* arm land.

![k trend](plots/k_trend.png)

### How reliable is this? A two-seed check

Both seeds of the linear+CPC-neg k=12 arm were trained the full 50 k steps and
converge to **near-identical pretraining metrics** — contrastive loss ≈ 0.06,
forecast gap ≈ 0.77, AUC ≈ 1.0 for both. Yet their **downstream GM-MASE differs
by 0.49**:

| linear+CPC-neg k=12 | pretrain loss | forecast gap | full-97 GM-MASE |
|---|---:|---:|---:|
| seed 20260520 | ≈0.058 | ≈0.77 | **1.5240** |
| seed 20260523 | ≈0.060 | ≈0.77 | **2.0137** |

The second seed's q-head was trained from the same fixed seed on the same data,
so the gap is driven by the *backbone*, not the head; and its resume (after a
mid-run interruption) was clean — no loss spike, no step discontinuity, AUC
continuous — so it is a legitimate, fully-converged seed, not a damaged run.

**Implication.** The downstream metric carries **~0.5 GM-MASE of seed variance**
for this recipe, *decoupled from how well the contrastive objective converged*.
That dwarfs the +0.09…+0.24 k-effects above and is ~25× the ±0.02 between-run
figure assumed from #307 (which clearly does not transfer to this less-stable
recipe class). So the single-seed k-effects are **not statistically
established**: the trend is suggestive, and a multi-seed replication — most
valuably of the headline transformer arm and of β — is needed before claiming
k=12 reliably hurts transfer.

### Per domain (full GIFT-Eval), v11c reference

Across the 7 GIFT-Eval domains, the k=12 transformer-head ring (single seed)
sits **outside** (worse than) both β and v11c on 6 of 7 — tying only on
Web/CloudOps. The single-seed deficit is broad, not a domain-specific trade-off
— though, per the seed-variance caveat above, the *size* of that deficit is not
firmly pinned down.

![per-domain radar](plots/perdomain_radar.png)

### Why: multi-step keeps the latent diffuse; β concentrates it

A natural hypothesis (raised in review) is that the multi-step objective forces
the latent to *collapse* onto a low-dimensional, linearly-extrapolable subspace.
The data **refutes it — the opposite happens.** Measuring the **participation
ratio** of the encoder latent (effective dimensionality, PR = (Σλ)²/Σλ² of the
latent covariance; 1 = one direction, H = 384 = full) on a fixed real-data batch
across training:

| latent dim-usage @ 50k | k = 1 | k = 12 |
|---|---:|---:|
| transformer head (k=1 ≡ β) | **3.2** | **55.3** |
| linear head, β negatives | **4.9** | **51.4** |

The **k=1** recipes *concentrate* the latent over training, collapsing to **~3–5
effective dimensions**; the **k=12** recipes hold it **~10–17× wider (~50)**.
Predicting 12 steps ahead requires the latent to stay linearly extrapolable far
out, so it remains diffuse; predicting only the next step lets the encoder pack
variance into a few sharp, highly-predictive directions. The best-transferring
recipe is the most concentrated one — and the trend is identical in both
forecaster families, so it is the *horizon*, not the head type.

![dim usage](plots/dim_usage.png)

This agrees with the **forecast gap** — cos(f_t, h_{t+1}) − cos(f_t, h_t), how
much more the forecast resembles the next latent than the current one — which
settles at **~1.09 for k=1** but only **~0.65 for k=12**: the k=12 latent is both
higher-dimensional *and* less sharply one-step-predictable. β's tight,
low-dimensional latent is the best transferrer on this line; the diffuse k=12
latent the worst single-seed — a plausible dim-usage→transfer link, though (per
the seed-variance caveat) the *size* of that transfer gap is not nailed down.

![training curves](plots/training_curves.png)

## Protocol

**One axis changes from β: the number of forecast steps.** β's forecaster is a
single 1-layer causal transformer (d=128 bottleneck, 4 heads) predicting
*h_{t+1}*. The k=12 arm replaces it with **K = 12 such heads** in parallel, head
*k* mapping the encoder output *h_t* to a prediction of *h_{t+k}*. The loss is
β's exact `cosine_similarity_batch_full_hh_negs` negative pool (encoder
all-time + cross-channel + cross-batch, batch-pooled) with the positive
**averaged over the K horizons**; at K=1 it is byte-for-byte β (verified
numerically). Everything else = β: 6-layer causal encoder, DropKey 0.70, τ=0.10,
β2=0.98, lr 1e-3, 50 k steps, global batch 256, single channel, fp32.

**Control families** (to isolate the k effect from the head/negatives): the
same k=1→k=12 change with **linear** heads (W_k: H→H) under **β negatives** and
under **CPC-canonical negatives**. β itself is the k=1 transformer/β-negs cell.

**Downstream (unchanged from β / #315).** Each frozen backbone → a quantile
forecasting head (forecast length 16, reconstruction = forecaster, 30 k steps),
**full-97 + triage-11 GM-MASE**. The headline arm (#1) is evaluated with both a
small 2-layer and a 6-layer head; the control families use the small head for
the trend comparison.

**Precision = fp32** (the v11c champion's precision; the multi-step objective is
unstable under β's fp16 body at lr 1e-3 — detail in EXECUTION_LOG.md).

## What we learned

1. **Robust — multi-step makes the latent diffuse, not collapsed.** k=12 trains
   as a healthy contrastive model, and (contra a natural low-rank-collapse
   hypothesis) its encoder latent is *higher*-dimensional than β's — effective
   dim **~50 vs ~3**, a 10–17× within-run effect that holds across forecaster
   families. The multi-step constraint keeps the latent spread out and
   longer-horizon-linear, away from the sharp, low-dimensional 1-step structure
   (forecast gap ~0.65 vs ~1.09) that β concentrates into. This is the solid
   result, and it argues against the CPC "predict-further → richer latent"
   intuition: the extra dimensions are not useful for transfer.
2. **Suggestive — k=12 *appears* to transfer worse, but it is not established.**
   Every single-seed k=12 arm scores nominally worse than its k=1 counterpart
   (+0.09…+0.24 GM-MASE). But those effect sizes sit *inside* the measured seed
   variance, so this is a consistent trend, not a confirmed verdict.
3. **Methodological — single-seed GM-MASE is unreliable for this recipe class.**
   Two equally-converged seeds of one arm differ by **0.49 GM-MASE** — far more
   than any k-effect, and decoupled from pretraining loss/gap/AUC. Comparisons
   at this scale need ≥2 seeds; the ±0.02 figure from #307 does not generalize
   to these higher-MASE, less-stable recipes.
4. **Single-seed aside:** the transformer forecaster scores better than the
   linear one even at k=1 (β 1.3272 vs linear ≈1.43) — but with ~0.5 seed noise,
   treat this as indicative only.

## Limitations

- **Seed variance is large and only partly measured — the main limitation.**
  The one arm with two seeds spans **0.49 GM-MASE** (1.5240 / 2.0137); every
  other cell is single-seed. So the +0.09…+0.24 k-effects are not statistically
  established. A clean ≥2-seed replication of the headline transformer arm (and
  of β) is the natural — and necessary — next step to turn the suggestive trend
  into a verdict.
- The k=12 arm bundles K=12 heads (more parameters) with the multi-step
  objective; the control families share the K-head structure, so the consistent
  trend isolates the objective, but a same-parameter single-head multi-target
  variant is untested.
- fp16 (β's body precision) is unavailable for this objective (diverges at
  lr 1e-3); the comparison is fp32-matched to v11c, not to β's fp16.
- The dim-usage probe is one participation-ratio measurement on a single fixed
  batch; absolute PR is batch-dependent, but the **relative** k=1≪k=12 gap is
  large, monotonic over training, and identical in both families.

## Annex

### Metric & references
full-97 / triage-11 **GM-Relative MASE**, geometric mean of (model ÷
seasonal-naive) MASE; lower better. v11c = 1.292, β = 1.3272, (B) = 1.3572.
#1 also: 6L-head full-97 = 1.4212.

### Code
Branch `experiment/2026-05-23-cpc-multistep-linear`. `forecaster_kind` ∈
{transformer (β), cpc (#1, K transformer-1L heads), linear_cpc (#2/#3, K linear
heads)}; losses `cpc_multistep` (β negatives) and `cpc_multistep_cpcnegs`
(CPC-canonical), both = β's loss at K=1 for their family. CPC heads auto-detected
downstream. Launchers `scripts/elisa_run.sh` (#1) and `scripts/elisa_run_linear.sh`
(#2/#3); `scripts/downstream.sh`; figures `scripts/plot_results.py` and
`scripts/dim_usage.py` (participation ratio vs step); CPU tests
`scripts/test_cpc.py` (incl. k=1 ≡ β `full_hh_negs`, 7.468332).
