# Learnable bilinear W vs temperature in the main contrastive loss

The CPC auxiliary term scores with a learnable log-bilinear, `exp(eᵀ W₁ h)`,
no temperature. Inspired by that, this experiment replaces the main
contrastive loss's temperature-scaled dot product `exp(uᵀv / τ)` (τ = 0.10)
with a learnable log-bilinear `exp(uᵀ W v)`, τ dropped. Both placements (W
applied to the encoder target `h`, run-1; W applied to the autoregressive
forecast `f`, run-2 — equivalent up to a transpose of W) were trained on the
same recipe as the prior no-encoder + CPC baseline. The training-dynamics
curves degraded below the baseline early; training was stopped at step ~3,500
(run-1) and ~3,900 (run-2) and the backbones were evaluated. Both bilinear
arms perform worse than the τ baseline + CPC.

## Result

![GIFT-Eval full-97 GM-Relative MASE for the τ-baseline + CPC and the two
bilinear-W arms (2L quantile head, best-loss checkpoint). Left: per-arm GM
with the 90% bootstrap CI (2000 resamples, seed 0). Right: paired-bootstrap
Δ = GM(arm) − GM(τ baseline) with 90% CI. Both bilinear arms sit above the
baseline; both CIs exclude zero.](plots/gm_summary.png)

**Verdict: both bilinear placements are worse than the τ baseline.** Both
paired-bootstrap CIs are strictly above zero.

## Training curves

![Six log-log panels of training-time signals — total training loss (raw,
floor added back for the bilinear arms whose loss can dip below the InfoNCE
floor), W-free reference loss `loss_tau_ref` (τ=0.07, computed identically
across arms), CPC InfoNCE term, ratio gap (1−ff)/(1−fp), batch-wise used-dim
proxy U_batch, 1 − R²_naive. Blue: bilinear W on `h`. Green: bilinear W on
`f`. Red dashed: τ baseline + CPC.](plots/training_dynamics.png)

The W-free reference loss `loss_tau_ref` is the load-bearing comparison
since it is computed identically across arms. At step 3,500 (the last step
both bilinear arms reached) it is **17.3** for the τ baseline, **25.8** for
run-1, and **29.2** for run-2.

## Result table

GM-Relative MASE (GIFT-Eval full-97, 2L head, best-loss checkpoint). Paired
bootstrap over the 97-task list, 2000 resamples, seed 0.

| arm | GM-Relative MASE | Δ vs τ-baseline + CPC | 90% CI |
|---|--:|--:|--|
| τ-baseline + CPC (reference) | **1.168** | — | — |
| bilinear W on `h` (run-1) | **2.270** | +1.102 | [+0.772, +1.490] |
| bilinear W on `f` (run-2) | **2.395** | +1.228 | [+0.886, +1.624] |

## Protocol

A backbone here is patch-embedding (a GRU) → forecaster (a 6-layer causal
transformer; no encoder stack, `--num-encoder-layers 0`), trained by the
contrastive objective to predict the next token's embedding. To score it we
freeze it, train a fresh 2-layer transformer quantile forecasting head (the
prior head recipe, 30,000 steps), and evaluate on GIFT-Eval's 97 tasks at the
best-loss checkpoint. **GM-Relative MASE** is the geometric mean, over those
97 tasks, of a model's error divided by the seasonal-naive forecast's error;
lower is better, 1.0 = seasonal-naive.

Single seed (20260520), one RTX 4090, the prior no-encoder + CPC recipe (same
code, machine, and seed; only the main-loss similarity differs): d_model 384
/ 6 heads, the crossfade-triplet allt·0.8% data mix, qk-norm,
attention-output norm, the `xshh_allt` contrastive loss with
positive-in-denominator and floor subtraction, the encoder-side positive
stop-gradient, the CPC InfoNCE auxiliary at weight 1.0, batch 1024. The only
change is the main loss's similarity. Two arms differ in which side of the
bilinear `W` is applied to the positive:

- **run-1: W on `h`** — positive `s(f_t, h_{t+1}) = (W h_{t+1}) · f_t`. The
  cross-batch f↔h negative scores `(W f_t) · h'_{t+1} = f_tᵀ Wᵀ h'_{t+1}`.
- **run-2: W on `f`** — positive `s(f_t, h_{t+1}) = (W f_t) · h_{t+1}`. The
  cross-batch f↔h negative is `f_tᵀ Wᵀ h'_{t+1}` again.

`W` is H×H (H=384), one per run, initialised to (1/τ₀)·I with τ₀ = 0.10 — so
step 0 reproduces the τ baseline — and excluded from weight decay, matching
the fixed scalar temperature it replaces. Training horizon: 3,500 steps for
run-1, 3,900 steps for run-2 (the τ-baseline schedule is 12,500 steps).

For run-1 the predicted next encoder latent is `Wᵀ f_t` (the
maximum-correlation direction of `h` under `(W h) · f`); for run-2 it is
`W f_t`. The downstream loader applies `Wᵀ`; for run-2 the saved
`main_w.weight` is transposed before loading so the same code computes
`W f_t`. Per-task results, W matrices, and training logs are reproducible
from this experiment's `results/` and `runs/` directories.

## The change

The main loss previously scored every pair with `score(u, v) = uᵀv / τ`
(τ = 0.10, fixed) and assembled a normalised InfoNCE over the positive and
the cross-channel, cross-batch, within-series-all-time and
cross-series-all-time negatives. This experiment replaces it with the
learnable log-bilinear `score(u, v) = uᵀ W v`, with W ∈ ℝ^{H×H} learnable
and no temperature. `W` is registered as an `nn.Linear(H, H, bias=False)` in
the backbone, init `(1/τ₀)·I`, excluded from AdamW weight decay. The
cross-batch f↔h negative and the latent-uniformity (h↔h) terms use the same
`W` as the positive in each arm; the `xs_allt` cross-series Gram
pre-projects its anchor by `W` and runs the inner kernel at τ=1, leaving
the existing chunked / fused autograd backward untouched. The CPC auxiliary
term keeps its own separate `W₁`, unchanged. At `W = (1/τ)·I` every term
equals the τ baseline exactly.

## Annex: W at the saved checkpoints

`τ_eff = 1/mean(diag W)`; `‖W_off‖_F` is `‖W − diag(W)‖_F`.

| arm | ckpt (step) | ‖W‖_F | × init | τ_eff | ‖W_off‖_F / ‖W‖_F | ‖W − Wᵀ‖_F / ‖W‖_F | cond(W) |
|---|---|--:|--:|--:|--:|--:|--:|
| init (W = (1/τ₀)·I) | — | 195.96 | 1.00 | 0.100 | 0.000 | 0.000 | 1.0 |
| run-1 (W on h) | best_gap (step 400) | 201.79 | 1.03 | 0.101 | 0.274 | 0.379 | 35.3 |
| run-1 (W on h) | 2,000-step periodic | 251.64 | 1.28 | 0.108 | 0.693 | 1.052 | 874.8 |
| run-1 (W on h) | best_loss (step 3,500) | 262.40 | 1.34 | 0.110 | 0.733 | 1.112 | 3,553 |
| run-2 (W on f) | best_gap (step 1,700) | 202.52 | 1.03 | 0.104 | 0.372 | 0.325 | 112.3 |
| run-2 (W on f) | 2,000-step periodic | 205.45 | 1.05 | 0.105 | 0.418 | 0.388 | 724.0 |
| run-2 (W on f) | best_loss (step 3,700) | 211.94 | 1.08 | 0.106 | 0.487 | 0.488 | 67.9 |
