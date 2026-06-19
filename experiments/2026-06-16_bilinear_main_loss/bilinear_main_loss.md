# Learnable bilinear W vs temperature in the main contrastive loss

The main contrastive loss scores a forecast against a candidate embedding with
a temperature-scaled dot product, `exp(uᵀv / τ)`, τ = 0.10 fixed. The CPC
InfoNCE auxiliary term used as a regulariser in the prior no-encoder + CPC
backbone instead scores with a *learnable log-bilinear*, `exp(eᵀ W₁ h)`, no
temperature — the matrix `W₁` carries the scale. This experiment asks whether
giving the **main** term the same treatment helps: replace `exp(uᵀv / τ)`
with `exp(uᵀ W v)`, drop τ, make `W` learnable.

*A backbone here is patch-embedding (a GRU) → forecaster (a 6-layer causal
transformer; no encoder stack, `--num-encoder-layers 0`), trained by the
contrastive objective to predict the next token's embedding. To score it we
freeze it, train a fresh quantile forecasting head, and evaluate on GIFT-Eval.
**GM-Relative MASE** is the geometric mean, over GIFT-Eval's 97 tasks, of a
model's error divided by the seasonal-naive forecast's error; lower is better,
1.0 = seasonal-naive. `W` is an H×H matrix (H = 384) initialised to (1/τ₀)·I
so training starts exactly at the τ = 0.10 baseline.*

For a free learnable `W`, the bilinear can be written either as
`(W h_{t+1}) · f_t` (W on the encoder-side target — *run-1*) or as
`(W f_t) · h_{t+1}` (W on the autoregressive forecast — *run-2*). The two
forms are equivalent up to a transpose of `W`; both are evaluated here.

## Result

![GIFT-Eval full-97, 2L quantile head, best-loss checkpoint. Left: per-arm
GM-Relative MASE with the 90% bootstrap CI (2000 resamples). Right:
paired-bootstrap Δ = GM(arm) − GM(τ baseline) point estimate with 90% CI.
Both bilinear arms sit above the baseline; both CIs exclude zero.](plots/gm_summary.png)

| arm | GM-Relative MASE | Δ vs τ-baseline + CPC (point) | 90% CI (paired bootstrap, 2000 resamples, seed 0) |
|---|--:|--:|--|
| τ-baseline + CPC (reference) | **1.168** | — | — |
| bilinear W on `h` (run-1) | **2.270** | +1.102 | [+0.772, +1.490] |
| bilinear W on `f` (run-2) | **2.395** | +1.228 | [+0.886, +1.624] |

**Verdict: worse than the τ baseline, in both bilinear placements.** Each CI is
strictly above zero; the GM-Relative MASE roughly doubles (1.168 → 2.27 / 2.40).

## What the learned W became

Three checkpoints exist per arm (the 2,000-step periodic save, the best
gap-ratio, and the best loss). All values are computed from the saved
`main_w.weight`.

| arm | ckpt (step) | ‖W‖_F | × init | τ_eff = 1/mean(diag) | ‖W_off‖_F / ‖W‖_F | ‖W − Wᵀ‖_F / ‖W‖_F | cond(W) |
|---|---|--:|--:|--:|--:|--:|--:|
| init reference (W = (1/τ₀)·I) | — | 195.96 | 1.00 | 0.100 | 0.000 | 0.000 | 1.0 |
| run-1 (W on h) | best_gap (step 400) | 201.79 | 1.03 | 0.101 | 0.274 | 0.379 | 35.3 |
| run-1 (W on h) | 2,000-step periodic | 251.64 | 1.28 | 0.108 | 0.693 | 1.052 | 874.8 |
| run-1 (W on h) | best_loss (step 3,500) | 262.40 | 1.34 | 0.110 | 0.733 | 1.112 | 3,553 |
| run-2 (W on f) | best_gap (step 1,700) | 202.52 | 1.03 | 0.104 | 0.372 | 0.325 | 112.3 |
| run-2 (W on f) | 2,000-step periodic | 205.45 | 1.05 | 0.105 | 0.418 | 0.388 | 724.0 |
| run-2 (W on f) | best_loss (step 3,700) | 211.94 | 1.08 | 0.106 | 0.487 | 0.488 | 67.9 |

In both arms `τ_eff` stays within ±10% of the init τ₀ = 0.10 across all saved
checkpoints; the Frobenius norm grows at most 1.34× over the init. The
off-diagonal norm ratio rises from 0 (init) to 0.733 in run-1 and 0.487 in
run-2 at best_loss; the antisymmetric norm ratio rises from 0 to 1.112 in
run-1 and 0.488 in run-2. The condition number rises to 3,553 (run-1) and to
112.3 then 67.9 (run-2).

## Training curves

![Six log-log panels. **Total training loss**: the floor-subtracted training
objective; the InfoNCE floor (the lower-bound value at uniform scores) is
9.412 for the configured τ = 0.10, B = 1024, T = 256, our loss_shape; it is
added back to the bilinear arms because their loss can dip below it (W can
amplify scores past the floor), so what is plotted is the raw ≥-0 InfoNCE
value. **loss_tau_ref**: the W-free, τ=0.07 normalised-InfoNCE reference,
computed identically across arms; comparable regardless of which loss the
arm was trained with. **CPC InfoNCE term**: the auxiliary CPC loss value
(same construction in all three arms). **ratio gap**: (1 − `ff`) / (1 − `fp`),
the discriminative ratio between the forecast↔forecast and the forecast↔positive
log-energies; lower is better. **U_batch**: the batch-wise used-dim
proxy (higher is better). **1 − R²_naive**: the residual proportion of a
naive-forecast R² (lower is better). Blue: bilinear W on `h` (run-1).
Green: bilinear W on `f` (run-2). Red dashed: τ baseline +
CPC.](plots/training_dynamics.png)

The W-free reference loss is the load-bearing comparison since it is computed
identically across arms. At step 3,500 (the last step of run-1; run-2 also
covers this step), `loss_tau_ref` is **17.3** for the τ baseline, **25.8** for
run-1, and **29.2** for run-2 — both bilinear arms are higher than the
baseline at the matched step. The ratio gap, U_batch, and 1 − R²_naive panels
all show the bilinear arms above (worse than) the baseline at the matched-step
range where all three runs overlap.

## Protocol

Single seed (20260520), one RTX 4090, the prior no-encoder + CPC recipe (same
code, machine, and seed; only the main-loss similarity differs): GRU
patch-embedding, d_model 384 / 6 heads, a 6-layer full-width forecaster, no
encoder stack, the crossfade-triplet allt·0.8% data mix, qk-norm,
attention-output norm, the `xshh_allt` contrastive loss with
positive-in-denominator and floor subtraction, the encoder-side positive
stop-gradient, the CPC InfoNCE auxiliary at weight 1.0, batch 1024. The only
change is the main loss's similarity: `exp(uᵀv / τ)` → `exp(uᵀ W v)`. Two
arms differ in which side of the bilinear `W` is applied to the positive:

- **run-1: W on `h`** — positive `s(f_t, h_{t+1}) = (W h_{t+1}) · f_t`. The
  cross-batch f↔h negative scores `(W f_t) · h'_{t+1} = f_tᵀ Wᵀ h'_{t+1}`.
- **run-2: W on `f`** — positive `s(f_t, h_{t+1}) = (W f_t) · h_{t+1}`. The
  cross-batch f↔h negative is `f_tᵀ Wᵀ h'_{t+1}` again.

Training horizon: 3,500 steps for run-1 and 3,900 steps for run-2 (≈ 1/3 of
the 12,500-step schedule used by the τ baseline). `W` is H×H (H=384), one
per run, initialised to (1/τ₀)·I with τ₀ = 0.10 — so step 0 reproduces the τ
baseline exactly — and excluded from weight decay, matching the fixed scalar
temperature it replaces.

To score a backbone we freeze it, train a fresh 2-layer transformer quantile
forecasting head (30,000 steps, the prior head recipe), and evaluate on
GIFT-Eval's 97 tasks at the best-loss checkpoint. Each arm gets its own head,
trained on the backbone's actual representation. For run-1 the predicted
next encoder latent is `Wᵀ f_t` (the maximum-correlation direction of `h`
under `(W h) · f`); for run-2 it is `W f_t`. At downstream time the predicted
latent is computed by applying `Wᵀ` (run-1) or `W` (run-2) to the forecaster
output, both inside `extract_forecaster_latents` and per step inside
`rollout_latent`. To keep one inference code path, the downstream loader
applies `Wᵀ`; for run-2 the saved `main_w.weight` is transposed before
loading so this code applies the trained `W` to `f_t` as required.

The τ baseline numbers come from the prior + CPC arm: same code, machine,
seed, and recipe, scored by the same eval, so the comparison changes the
main-loss similarity and the training horizon. Per-task results and W stats
are reproducible from the artifacts in this experiment's `results/` and
`runs/` directories.

## The change

For two L2-normalised vectors `u` (a forecast) and `v` (a candidate
embedding), the main loss previously scored every pair with a
temperature-scaled dot product and assembled a normalised InfoNCE over the
positive and the cross-channel, cross-batch, within-series-all-time and
cross-series-all-time negatives:

```
score(u, v) = uᵀv / τ           (τ = 0.10, fixed)
```

This experiment replaces it with a learnable log-bilinear with no temperature:

```
score(u, v) = uᵀ W v            (W ∈ ℝ^{H×H}, learnable, no τ)
```

`W` is registered as an `nn.Linear(H, H, bias=False)` in the backbone, init
`(1/τ₀)·I`, and excluded from AdamW weight decay (the scalar τ it replaces
is fixed, not decayed). The cross-batch f↔h negative and the
latent-uniformity (h↔h) terms use the same `W` as the positive in each arm.
The `xs_allt` cross-series Gram pre-projects its anchor by `W` and runs the
inner kernel at τ=1, leaving the existing chunked / fused autograd backward
untouched. The CPC auxiliary term keeps its own separate `W₁`, unchanged.
At `W = (1/τ)·I` every term equals the τ baseline byte-for-byte.
