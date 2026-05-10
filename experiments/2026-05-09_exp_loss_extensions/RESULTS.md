# Experiment: loss_extensions — square cross-batch negatives

## Goal

Compare AUC and Top-1 of the new `cosine_similarity_batch_square` loss
against the baseline `cosine_similarity_batch` loss, at τ=0.10 and τ=0.20.

## What we changed

`cosine_similarity_batch_square` adds two cross-batch negative edges on
top of `cosine_similarity_batch`: **neg_cross_batch_forecast** (forecast
embedding of element b vs forecast of b′≠b at the same t) and
**neg_cross_batch_embedding** (context h_{b,t+1} vs h_{b′,t+1}).
Together they tile the diagonal that the base loss leaves untouched,
forming a 2×2 square of negatives instead of a 1×2 rectangle.

## Protocol

| Setting | Value |
|---|---|
| Arms | 4: {baseline, square} × {τ=0.10, τ=0.20} |
| Steps | baselines 50 000, square arms 100 000, batch 256 |
| Dataset | GIFT-pretrain-full-4096 |
| Encoder | GRU, d=384, 6 heads, 6 layers |
| RevNorm | EWMA span=128, mixup p=0.3 |

Baselines = τ=0.10/τ=0.20 arms from prior tau-sweep (identical hyperparams
except loss). Square arms are now extended to 100 k so we can see whether
they reach the same plateau as the baselines and how long it takes.

## What the AUC / Top-1 curves show

![AUC and Top-1](plots/4arm_auc_top1.png)

Inside the 0–15 k overlap window all four arms cluster tightly within
run-to-run noise. Past 15 k all curves continue ramping; the baselines
hit a flat plateau by ~40 k (AUC ≈ 0.903, Top-1 ≈ 0.758) and stay
there through 50 k. The squares ramp more slowly — at 50 k they are
still very slightly below the baselines (Δ AUC ≈ −0.001 at τ=0.10,
≈ −0.003 at τ=0.20) — but by their own late plateau (90–100 k) both
square arms have closed the gap to within noise of the baseline
plateau. Headline: **square is not worse at convergence, just slower
to get there.**

## The same data on log-log

![Log-log convergence](plots/4arm_logscale.png)

The log-x view exposes the early-training ramp and confirms what the
linear-scale plot suggests: all 4 arms ride essentially the same scaling
curve. The squares' "delay" relative to the baselines is small in
log-step terms — they need maybe 1.5–2× as many steps to reach the same
AUC, not 10×.

## Where the loss change is most visible: dimension usage

![U_batch and U_temporal](plots/4arm_dim_usage.png)

`U_batch` and `U_temporal` measure how many embedding dimensions are
actively used along the batch and temporal axes respectively (higher =
more dimensions in use, lower = more collapsed). Through 50 k the square
loss systematically lowers both at both τ — the gap is very visible
between baseline τ=0.10 (U_batch ≈ 0.114) and square τ=0.10 (U_batch
≈ 0.069 at 15 k, ≈ 0.081 at 50 k). With the square arms extended to
100 k that gap continues to close: square τ=0.10 reaches U_batch ≈ 0.087
by 100 k (vs baseline 0.114 at 50 k — still lower) and square τ=0.20
reaches U_batch ≈ 0.093 (vs baseline 0.089 at 50 k — now matched / very
slightly higher). So the "square reduces batch-axis dim usage" claim
that looked clean at 15 k is partly a "square hasn't converged yet"
artifact. By the squares' own late plateau the τ=0.20 dim-usage gap
has disappeared and the τ=0.10 gap has shrunk noticeably. Still a side
metric, not the objective we score on.

## Late-window means

Per-step values are noisy by ±0.02 AUC, so the headline numbers below
are means over 10 k-step windows. Two windows reported for the squares:
the **common** window (40 001–50 000, where all 4 arms have data) is
the apples-to-apples comparison; the **own** window (90 001–100 000)
is each square arm's own late plateau, after the additional 50 k of
training the baselines didn't get.

| Arm | Window | AUC | Top-1 | U_batch | U_temporal |
|---|---|---:|---:|---:|---:|
| baseline τ=0.10 | 40 001–50 000  | 0.9034 | 0.7573 | 0.1136 | 0.0571 |
| baseline τ=0.20 | 40 001–50 000  | 0.9039 | 0.7588 | 0.0890 | 0.0423 |
| square   τ=0.10 | 40 001–50 000  | 0.9024 | 0.7571 | 0.0814 | 0.0408 |
| square   τ=0.20 | 40 001–50 000  | 0.9014 | 0.7555 | 0.0891 | 0.0416 |
| square   τ=0.10 | 90 001–100 000 | 0.9040 | 0.7607 | 0.0868 | 0.0430 |
| square   τ=0.20 | 90 001–100 000 | 0.9031 | 0.7590 | 0.0930 | 0.0429 |

At 50 k the squares are still ~0.001–0.003 AUC below baseline. By 100 k
square τ=0.10 has slightly **overtaken** baseline τ=0.10 (+0.0006 AUC,
+0.0034 Top-1) and square τ=0.20 has closed to within noise of baseline
τ=0.20 (−0.0008 AUC, +0.0002 Top-1). All differences are well inside
the ±0.02 per-step noise band, but the systematic late-window means
are stable enough to read direction.

## Statistical tests on AUC / Top-1 (Welch t, overlap window 5 001–50 000, n=45 000 each)

| Comparison | Δ AUC | p_AUC | Δ Top-1 | p_Top-1 |
|---|---:|---:|---:|---:|
| baseline τ=0.10 vs square τ=0.10 | +0.0005 | 3.8e-05 | −0.0002 | 5.0e-01 |
| baseline τ=0.20 vs square τ=0.20 | +0.0031 | 7.9e-173 | +0.0046 | 2.0e-130 |
| baseline τ=0.10 vs baseline τ=0.20 (sanity) | −0.0010 | 1.3e-20 | −0.0024 | 1.4e-33 |
| square τ=0.10 vs square τ=0.20 | +0.0016 | 5.5e-47 | +0.0024 | 1.9e-36 |

**Caveat:** samples are consecutive training steps from a single run,
not i.i.d.; effective sample size is much smaller than n, so Welch
p-values are anti-conservative. The huge p-values at τ=0.20 are still
real — the effect is consistent across the whole 5 k–50 k window — but
treat absolute Δ values as the load-bearing thing, not the p numbers.
Window includes the entire region where the squares are slow-to-converge,
so the Δ overstates the late-plateau gap; see the 90 001–100 000 row of
the late-window table for the actual converged comparison.

## Bottom line

- **τ=0.10:** square reaches and slightly overtakes baseline at its own
  late plateau (AUC 0.9040 vs 0.9034; Top-1 0.7607 vs 0.7573 at
  90–100 k vs baseline 40–50 k). At equal compute (50 k step count)
  square is marginally behind (Δ AUC −0.0010), but the difference is
  inside the per-step noise.
- **τ=0.20:** square also catches up (AUC 0.9031 vs baseline 0.9039;
  Top-1 0.7590 vs 0.7588 at 90–100 k). At equal 50 k compute square
  is clearly behind (Δ AUC −0.0025). The "square is significantly
  worse" verdict from the 15 k report was a convergence-speed artifact,
  not a final-plateau difference.
- **Convergence cost:** squares need roughly 1.5–2× the steps to hit
  the baseline plateau. So the extra edges add training-time cost
  without an end-state benefit on AUC / Top-1.
- **Side effect on dim usage:** the early-training "square reduces
  U_batch" claim still holds qualitatively but shrinks substantially
  with more training. By 100 k the τ=0.20 U_batch gap has disappeared;
  the τ=0.10 gap has roughly halved (0.087 vs 0.114). Lower U_batch
  is not necessarily desirable — baselines with higher U_batch reach
  the same AUC.
- **Practical take:** at this model scale and dataset the extra
  cross-batch negatives are a wash on the metrics we score on, and
  cost compute. Not a recommended default, but also not harmful at
  convergence.
