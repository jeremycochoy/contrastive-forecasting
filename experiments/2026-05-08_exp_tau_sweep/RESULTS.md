# τ-sweep — RESULTS

**Headline:** 6-arm τ-sweep over τ ∈ {0.03, 0.05, 0.07, 0.10, 0.20, learnable_init=0.10}.
**τ=0.20 wins held-out eval on 4/6 metrics** (R²_random, R²_naive, AUC,
Top-1). The learnable-τ arm slid downward from init=0.10 to τ≈0.069 by
step 15k and lands near τ=0.07's metric values — i.e. it does **not**
discover the τ=0.20 optimum within a single 15k run.

## Setup

Identical architecture/HP except `--tau`. Architecture matches
backbone-beta: T_RAW=4096, C=1, d_model=384, num_layers=6, n_heads=6,
freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind=ewma span=128,
loss=cosine_similarity_batch, batch_size=256, mixup_p=0.3, AdamW
lr=1e-3 wd=0.1 β1=0.9 β2=0.98. **15,000 steps per arm** (cut from 50k
after arm 1 — τ=0.03 — plateaued by step ~5k with no further upward
trend through step 22.4k).

Arms 1–4 (τ ∈ {0.03, 0.05, 0.07, 0.10}) and the learnable-τ arm trained
on elisa GPU 1 (RTX 4090). Arm 5 (τ=0.20) original trained on a
vast.ai 5090 spot. Arm 5 v2 (τ=0.20 redo) trained on a separate
vast.ai 5090; the python process died at step 7800 (process gone from
`ps`, no traceback in log) — `_FINAL.pth` was copied from the step-7700
`_best_loss.pth`.

All 6 backbones evaluated by
[`scripts/eval_tau_sweep_metrics_v2.py`](scripts/eval_tau_sweep_metrics_v2.py)
on a fixed held-out HF batch (B=256, eval seed=0, skip=50M wrapping
to row 7,260,000 because 50M > 42.7M total).

## Final eval-batch values (held-out)

| backbone                   | τ       | R²_random | R²_naive | U_t    | U_b        | AUC        | Top-1      |
|----------------------------|---------|-----------|----------|--------|------------|------------|------------|
| tau_sweep_0_03             | 0.03    | 0.7633    | 0.6914   | 0.0078 | 0.0099     | 0.8890     | 0.7369     |
| tau_sweep_0_05             | 0.05    | 0.7254    | 0.6479   | 0.0183 | 0.0319     | 0.8859     | 0.7319     |
| tau_sweep_0_07             | 0.07    | 0.6938    | 0.6208   | 0.0327 | 0.0639     | 0.8908     | 0.7407     |
| tau_sweep_0_10             | 0.10    | 0.6671    | 0.6075   | 0.0506 | **0.1028** | 0.8915     | 0.7449     |
| tau_sweep_0_20 (orig 15k)  | 0.20    | **0.7731**| **0.7256**| 0.0386| 0.0850     | **0.8938** | **0.7470** |
| tau_sweep_learnable_0_10   | 0.10→0.069 | 0.6919  | 0.6185   | 0.0320 | 0.0638     | 0.8908     | 0.7396     |

Source: [`results/tau_sweep_metrics_v2.csv`](results/tau_sweep_metrics_v2.csv).

Reference (backbone-beta_167k on the same held-out batch):
R²_random 0.6839, R²_naive 0.6080, U_t 0.0375, U_b 0.0762,
AUC 0.8966, Top-1 0.7531.

## Per-metric winner

| metric     | winning τ | value  |
|------------|-----------|--------|
| R²_random  | 0.20      | 0.7731 |
| R²_naive   | 0.20      | 0.7256 |
| U_temporal | 0.10      | 0.0506 |
| U_batch    | 0.10      | 0.1028 |
| AUC        | 0.20      | 0.8938 |
| Top-1      | 0.20      | 0.7470 |

## Range across τ

| metric     | min                  | max                  | range  |
|------------|----------------------|----------------------|--------|
| R²_random  | 0.6671 (τ=0.10)      | 0.7731 (τ=0.20)      | 0.106  |
| R²_naive   | 0.6075 (τ=0.10)      | 0.7256 (τ=0.20)      | 0.118  |
| U_temporal | 0.0078 (τ=0.03)      | 0.0506 (τ=0.10)      | 0.043  |
| U_batch    | 0.0099 (τ=0.03)      | 0.1028 (τ=0.10)      | 0.093  |
| AUC        | 0.8859 (τ=0.05)      | 0.8938 (τ=0.20)      | 0.0079 |
| Top-1      | 0.7319 (τ=0.05)      | 0.7470 (τ=0.20)      | 0.0151 |

R² and U-metric ranges are large; AUC and Top-1 ranges are small (sub-1%
and ~1.5%).

## Trajectories

![trajectories](plots/tau_sweep_v2_trajectories.png)

6-panel training trajectories (1000-step MA), 5 arms with per-step CSVs:
{0.03, 0.05, 0.07, 0.10, learnable_init0.10}, plus τ=0.20 v2 (partial,
ends at the last available CSV step). AUC y-zoom (0.89, 0.91) and
Top-1 y-zoom (0.72, 0.76) make arm-to-arm separation legible. AUC
trajectories overlap closely; Top-1 separates the τ=0.10 / learnable arms
above the sharper-τ pack.

## Comparison

![comparison](plots/tau_sweep_v2_comparison.png)

Bar chart of the held-out eval values per backbone, with backbone-beta_167k
reference as a dashed line. The eval CSV also contains an
`exp2_residual_silu_tau_0_10` row from a separate Exp-2 encoder probe
which is not part of this τ-sweep and is not plotted here.

## In-training vs held-out gap (5 arms with full trajectory CSVs)

In-training values are computed under `model.train()` (dropout active)
on the just-trained batch. Held-out values are computed under
`model.eval()` (dropout off) on a different batch. The two differ by
metric and arm; mechanism not investigated.

| backbone                 | last step | in-training loss | in-training AUC | in-training Top-1 | in-training U_b |
|--------------------------|-----------|------------------|-----------------|-------------------|------------------|
| tau_sweep_0_03           | 23000     | 6.7313           | 0.9133          | 0.7669            | 0.0102           |
| tau_sweep_0_05           | 15000     | 6.9970           | 0.9177          | 0.7660            | 0.0288           |
| tau_sweep_0_07           | 15000     | 7.0201           | 0.9171          | 0.7694            | 0.0598           |
| tau_sweep_0_10           | 15000     | 7.0414           | 0.9199          | 0.7765            | 0.0939           |
| tau_sweep_learnable_0_10 | 15000     | 6.9749           | 0.9206          | 0.7745            | 0.0591           |

(τ=0.20 original 15k arm has no per-step trajectory CSV — lost to the
spot-stop event documented under Caveats.)

## Picked winner: τ=0.20

The original τ=0.20 arm wins 4 of 6 held-out metrics (R²_rand, R²_naive,
AUC, Top-1). U_t and U_b favour τ=0.10 — those measure
representation-spread, not predictive quality, so they are weighted
secondary for downstream MASE.

The learnable-τ arm slid from init=0.10 to τ=0.0692 (log_inv_tau=2.671)
over 15k steps. Its held-out values land near τ=0.07 on every metric
(R²_random 0.6919 vs τ=0.07's 0.6938; AUC 0.8908 = τ=0.07's 0.8908;
U_b 0.0638 vs τ=0.07's 0.0639) — coherent with the converged τ value
sitting just below 0.07. Within a single 15k run, gradient pressure
pulls τ down, not up to the 0.20 optimum.

**Recommendation: keep using fixed τ for the next backbone, with
τ=0.20 as the picked value.** Re-evaluate learnable-τ if/when a wider
init sweep (e.g. starting from τ=0.30 or τ=0.50) gets run.

## Caveats

- **τ=0.20 original trajectory CSV lost.** The original vast spot
  auto-stopped on completion before any sync_loop pulled the per-step
  losses CSV (a one-shot DONE-marker scp-back was wired up instead).
  `_FINAL.pth` was preserved (it had been pulled at DONE) so the
  held-out eval row is authoritative; only the per-step trajectory is
  missing. Postmortem and the resulting addition to
  [`REMOTE_LAUNCH_CHECKLIST`](../REMOTE_LAUNCH_CHECKLIST.md) document
  the fix.
- **τ=0.20 v2 partial.** The v2 redo's python process died at step
  7800 (process absent from `ps`, no traceback in log); `_FINAL.pth`
  was copied from `_best_loss.pth` (step 7700). The v2 row in the
  eval table is from this 7800-step partial — directly comparable in
  scoring protocol but only ~52% trained vs the other arms. v2 trails
  the original τ=0.20 on every metric, consistent with under-training
  rather than rejection of τ=0.20 itself.
- **Vast spot teardown costs:** $1.51 (Exp 2 residual_silu, separate)
  and $0.97 (τ=0.20 v2). Both instances destroyed after final pulls.

## Open

- The τ=0.20 v2 retraining (in flight as of this consolidation) will
  replace the partial 7800-step trajectory once the full 15k run
  completes; the v2 plots above will be regenerated then.
- **Proxy MASE per arm.** The
  [`scripts/run_tau_sweep_proxy.sh`](scripts/run_tau_sweep_proxy.sh)
  recipe trains an R3_E4 head on each backbone for downstream
  GIFT-Eval; it has not been run yet.
- Whether AUC/Top-1/U-metric ranks predict downstream MASE rank for
  this set of 6 arms specifically. The proxy correlation analysis at
  `experiments/2026-05-05_exp_qhead_improvements/results/backbone_proxy_correlation.csv`
  is over a *different* set of 5 backbones (n=5, directional ρ);
  applying its conclusions here would be extrapolation.

## Outcome

τ=0.20 picked as the fixed temperature for the next backbone. Learnable
τ deferred to a future wider-init sweep. Proxy MASE evaluation on the
6 arms remains an open follow-up.
