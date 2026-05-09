# Exp 2 — encoder swap (gru vs residual_silu) @ τ=0.10

## Setup

Single-arm comparison: residual_silu encoder vs the gru encoder used in
the τ-sweep (Exp 1). All other recipe ingredients identical to Exp 1
arm `tau_sweep_0_10`:

- `T_RAW=4096`, `C=1`, `d_model=384`, `num_layers=6`, `n_heads=6`
- `freq_emb_dim=3`, `seasonality_emb_dim=3`
- `RevEWMNorm span=128`
- `loss=cosine_similarity_batch`, fixed τ=0.10
- `mixup_p=0.3`, `batch_size=256`
- AdamW lr 1e-3 wd 0.1, β=(0.9, 0.98)
- 15,000 steps, no grad clip

The only thing that changed is `--encoder-type residual_silu` (vs
`gru` in Exp 1).

- gru baseline: `tau_sweep_0_10` (15k, elisa, from Exp 1).
- residual_silu: `exp2_residual_silu_tau_0_10` (15k, vast 36368320 RTX
  5090). Vast destroyed after final sync. Total spend on the residual_silu
  run: **$1.51** (uptime 2h 12m).

## Held-out eval (FINAL.pth)

Same held-out HF batch as Exp 1 (skip=50M wrap to row 7,260,000;
B=256; eval seed=0). Eval script:
[`experiments/2026-05-08_exp_tau_sweep/scripts/eval_tau_sweep_metrics_v2.py`](../2026-05-08_exp_tau_sweep/scripts/eval_tau_sweep_metrics_v2.py).
Results CSV:
[`experiments/2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_v2.csv`](../2026-05-08_exp_tau_sweep/results/tau_sweep_metrics_v2.csv).

| backbone                       | encoder        | R²_rand | R²_naive | U_t    | U_b    | AUC    | Top1   |
|--------------------------------|----------------|---------|----------|--------|--------|--------|--------|
| tau_sweep_0_10                 | gru            | **0.6671** | **0.6075** | **0.0506** | **0.1028** | **0.8915** | **0.7449** |
| exp2_residual_silu_tau_0_10    | residual_silu  | 0.6737  | 0.5997   | 0.0336 | 0.0574 | 0.8864 | 0.7318 |

Bold = winner per metric.

Reference (backbone-beta_167k on the same held-out batch):
R²_rand 0.6839, R²_naive 0.6080, U_t 0.0375, U_b 0.0762,
AUC 0.8966, Top1 0.7531.

## Last in-training step values (from per-step CSVs)

| backbone                       | last step | loss   | AUC_train | Top1_train | U_b_train |
|--------------------------------|-----------|--------|-----------|------------|-----------|
| tau_sweep_0_10                 | 15000     | 7.0414 | 0.9199    | 0.7765     | 0.0939    |
| exp2_residual_silu_tau_0_10    | 15000     | 7.4514 | 0.8895    | 0.7183     | 0.0597    |

The residual_silu run lands at a higher contrastive loss (7.45 vs 7.04)
and lower in-training AUC/Top1 across the entire trajectory; the
held-out gap is consistent with the in-training gap.

## Plots

- [`plots/encoder_comparison.png`](plots/encoder_comparison.png) — 6-panel
  trajectory comparison (1000-step MA), gru vs residual_silu over the
  full 15k steps. AUC zoom 0.88–0.93; Top-1 zoom 0.70–0.78. Star
  markers at step 15000 show the held-out FINAL.pth eval values.

## Picked winner: gru

The gru encoder beats residual_silu on **5 of 6 held-out metrics**:
- AUC (0.8915 vs 0.8864), Top1 (0.7449 vs 0.7318) — gru wins by
  ~0.005 AUC and ~0.013 Top1.
- U_temporal (0.0506 vs 0.0336) and U_batch (0.1028 vs 0.0574) — gru
  uses substantially more representation dimensions on this held-out
  batch.
- R²_naive (0.6075 vs 0.5997) — narrow gru win.

residual_silu wins only R²_random (0.6737 vs 0.6671), which measures
forecast usefulness vs a random latent baseline; combined with the
worse R²_naive, this suggests residual_silu's forecasts are slightly
more "absolutely" predictive but less informative beyond the persistence
baseline.

The training trajectories (15k steps each) confirm the held-out finding.
On 1000-step MA, residual_silu starts faster (AUC ~0.67 by step 1000 vs
gru ~0.54), but gru catches up by step ~2.1k (AUC) and ~2.5k (Top1) and
stays ahead from step ~5k onward (mean gru − residual_silu = +0.0070
AUC, +0.0150 Top1). The gap does not appear to be closing at step 15k.
A longer run is unlikely to flip the order.

**Recommendation: keep gru as the default encoder for the next
backbone.** No further encoder ablations planned this round.

## What this report does not cover

- Other encoder types (mlp, mlp_wide, conv) — only one alternative was
  tested in this experiment.
- Encoder × τ interactions — residual_silu was tested only at τ=0.10.
  If the picked τ from Exp 1 RESULTS_v2 (τ=0.20) interacts unfavourably
  with gru, residual_silu @ τ=0.20 might still be relevant; not
  measured.
- Proxy MASE — same blocker as Exp 1.
