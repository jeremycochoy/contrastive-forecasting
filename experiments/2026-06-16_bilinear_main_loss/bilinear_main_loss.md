# Learnable bilinear W vs temperature in the main contrastive loss

The CPC auxiliary term scores with a learnable log-bilinear `exp(eᵀ W₁ h)`,
no temperature. Inspired by that, we replace the main contrastive loss's
temperature-scaled dot product `exp(uᵀv / τ)` (τ = 0.10) with a learnable
log-bilinear `exp(uᵀ W v)`. The training-time W-free reference loss
`loss_tau_ref` (a τ=0.07 normalised-InfoNCE computed identically across arms)
diverged above the τ baseline by step 3,500 (17.3 baseline vs 25.8 / 29.2
bilinear); training was stopped at 3,500 / 3,900 steps and downstream-evaluated
with a 2-layer quantile head. Both bilinear placements (W applied to the
encoder target `h`, run-1; W applied to the autoregressive forecast `f`,
run-2 — equivalent up to a transpose of W) perform worse than the τ baseline
+ CPC.

## Result

![GIFT-Eval full-97 GM-Relative MASE (geometric mean over the 97 tasks of a
model's error divided by seasonal-naive; lower is better, 1.0 = seasonal-naive).
2L quantile head, best-loss checkpoint. Left: per-arm GM with the 90% bootstrap
CI (2000 resamples, seed 0). Right: paired-bootstrap Δ = GM(arm) − GM(τ
baseline) with 90% CI. Both bilinear arms sit above the baseline; both CIs
exclude zero.](plots/gm_summary.png)

**Verdict: both bilinear placements are worse than the τ baseline.** Both
paired-bootstrap CIs are strictly above zero.

## Training curves

![Six log-log panels. **Total training loss** (raw value; for the bilinear
arms the InfoNCE floor of 9.412 is added back since `W` can amplify scores
past it). **loss_tau_ref**: a τ=0.07 normalised-InfoNCE reference computed
identically across arms — comparable regardless of training objective.
**CPC InfoNCE term**: the auxiliary CPC loss value. **ratio gap**:
(1 − `ff`) / (1 − `fp`), the f↔f vs f↔positive log-energy ratio (↓→0 better).
**U_batch**: a batch-wise used-dim proxy (higher is better). **1 − R²_naive**:
residual proportion of a naive-forecast R² (lower is better). Blue: bilinear
W on `h`. Green: bilinear W on `f`. Red dashed: τ baseline +
CPC.](plots/training_dynamics.png)

`loss_tau_ref` is the load-bearing comparison. At step 3,500 (the last step
both bilinear arms reached) it is **17.3** for the τ baseline, **25.8** for
run-1, **29.2** for run-2. The ratio_gap, U_batch, and 1−R²_naive panels
also place both bilinear arms above the baseline across the overlapping
range.

## Result table

GM-Relative MASE (GIFT-Eval full-97, 2L quantile head, best-loss checkpoint).
Paired bootstrap over the 97 tasks, 2000 resamples, seed 0.

| arm | GM-Relative MASE | Δ vs τ-baseline + CPC | 90% CI |
|---|--:|--:|--|
| τ-baseline + CPC (reference) | **1.168** | — | — |
| bilinear W on `h` (run-1) | **2.270** | +1.102 | [+0.772, +1.490] |
| bilinear W on `f` (run-2) | **2.395** | +1.228 | [+0.886, +1.624] |

Scope: the original issue scope was 2L and 6L heads at best-loss and last;
this report covers 2L best-loss only. The bilinear-arm training-time signals
(both `loss_tau_ref` and the W-free cosine gap) diverged above the τ baseline
within the first 3,500 steps, and downstream eval at that point gave Δ
> +1.1 with CIs well above zero — large enough that the 6L head and the
later "last" checkpoint were not run.

## Protocol

A backbone here is patch-embedding (a GRU) → forecaster (a 6-layer causal
transformer; no encoder stack, `--num-encoder-layers 0`), trained by a
contrastive objective on the `xshh_allt` loss shape — the project's main
contrastive loss form, with cross-channel, cross-batch, within-series
all-time and cross-series all-time negatives, positive-in-denominator, and
floor subtraction. To score it we freeze it, train a fresh 2-layer
transformer quantile forecasting head (the prior head recipe, 30,000 steps),
and evaluate on GIFT-Eval's 97 tasks.

Single seed (20260520), one RTX 4090, the prior no-encoder + CPC recipe
(same code, machine, and seed; only the main-loss similarity differs):
d_model 384 / 6 heads, the crossfade-triplet allt·0.8% data mix, qk-norm,
attention-output norm, the encoder-side positive stop-gradient, the CPC
InfoNCE auxiliary at weight 1.0, batch 1024. Two arms differ in which side
of the bilinear `W` the positive applies:

- **run-1: W on `h`** — positive `s(f_t, h_{t+1}) = (W h_{t+1}) · f_t`. The
  cross-batch f↔h negative scores `(W f_t) · h'_{t+1} = f_tᵀ Wᵀ h'_{t+1}`.
- **run-2: W on `f`** — positive `s(f_t, h_{t+1}) = (W f_t) · h_{t+1}`. The
  cross-batch f↔h negative is `f_tᵀ Wᵀ h'_{t+1}` again.

`W` is H×H (H=384), one per run, initialised to (1/τ₀)·I with τ₀ = 0.10 — so
step 0 reproduces the τ baseline — and excluded from weight decay, matching
the fixed scalar temperature it replaces. Training horizons: 3,500 steps
(run-1), 3,900 steps (run-2). For run-1 the predicted next encoder latent
is `Wᵀ f_t` (the maximum-correlation direction of `h` under `(W h) · f`);
for run-2 it is `W f_t`. The downstream loader applies `Wᵀ`; for run-2 the
saved `main_w.weight` is transposed before loading so the same code computes
`W f_t`. Per-task results, W matrices, and training logs are reproducible
from this experiment's `results/` and `runs/` directories.

Single-seed caveat: the verdict rests on one backbone seed per arm. The
matched-step `loss_tau_ref` and `gap` differences are large in absolute
terms (+8.4 and +11.9 respectively, on a scale where the τ baseline sits at
17.3), and the downstream Δ is +1.1 with a CI lower bound of +0.77 — large
enough that seed noise alone is unlikely to flip the verdict.

## Annex: W at the saved checkpoints

`τ_eff = 1/mean(diag W)`; `‖W_off‖_F = ‖W − diag(W)‖_F`.

| arm | ckpt (step) | ‖W‖_F | × init | τ_eff | ‖W_off‖_F / ‖W‖_F | ‖W − Wᵀ‖_F / ‖W‖_F | cond(W) |
|---|---|--:|--:|--:|--:|--:|--:|
| init (W = (1/τ₀)·I) | — | 195.96 | 1.00 | 0.100 | 0.000 | 0.000 | 1.0 |
| run-1 (W on h) | best_gap (step 400) | 201.79 | 1.03 | 0.101 | 0.274 | 0.379 | 35.3 |
| run-1 (W on h) | 2,000-step periodic | 251.64 | 1.28 | 0.108 | 0.693 | 1.052 | 874.8 |
| run-1 (W on h) | best_loss (step 3,500) | 262.40 | 1.34 | 0.110 | 0.733 | 1.112 | 3,553 |
| run-2 (W on f) | best_gap (step 1,700) | 202.52 | 1.03 | 0.104 | 0.372 | 0.325 | 112.3 |
| run-2 (W on f) | 2,000-step periodic | 205.45 | 1.05 | 0.105 | 0.418 | 0.388 | 724.0 |
| run-2 (W on f) | best_loss (step 3,700) | 211.94 | 1.08 | 0.106 | 0.487 | 0.488 | 67.9 |
