# τ-sweep — RESULTS

## Setup

5 fixed-τ arms in {0.03, 0.05, 0.07, 0.10, 0.20}, identical
architecture/HP except `--tau`. Architecture matches backbone-beta:
T_RAW=4096, C=1, d_model=384, num_layers=6, n_heads=6,
freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind=ewma span=128,
loss=cosine_similarity_batch, batch_size=256, mixup_p=0.3, AdamW
lr=1e-3 wd=0.1 β1=0.9 β2=0.98. **15,000 steps per arm** (cut from
50,000 after arm 1 metrics oscillated around stable means by step ~5,000
with no further upward trend through step 22,400 when arm 1 was
killed early to switch all arms to a 15k budget).

Arms 1–4 trained on elisa GPU 1 (RTX 4090). Arm 5 (τ=0.20) trained
on a vast.ai 5090 spot in parallel.

`tau_sweep_<safe>_FINAL.pth` for all 5 arms checked into
`sync_tau_sweep/checkpoints/`. Eval performed by
[`scripts/eval_tau_sweep_metrics.py`](scripts/eval_tau_sweep_metrics.py)
on a fixed held-out HF batch (B=256, seed=0, skip=50M wrapping to row
7,260,000 because 50M > 42.7M total).

## Final eval-batch values

| τ | R²_random | R²_naive | U_t | U_b | AUC | Top1 |
|---|---|---|---|---|---|---|
| 0.03 | 0.7633 | 0.6914 | 0.0078 | 0.0099 | 0.8890 | 0.7369 |
| 0.05 | 0.7254 | 0.6479 | 0.0183 | 0.0319 | 0.8859 | 0.7319 |
| 0.07 | 0.6938 | 0.6208 | 0.0327 | 0.0639 | 0.8908 | 0.7407 |
| 0.10 | 0.6671 | 0.6075 | 0.0506 | **0.1028** | 0.8915 | 0.7449 |
| 0.20 | **0.7731** | **0.7256** | 0.0386 | 0.0850 | **0.8938** | **0.7470** |

Source: `results/tau_sweep_metrics.csv`. Reference (backbone-beta_167k
on the same held-out batch): R²_random 0.6839, R²_naive 0.6080,
U_t 0.0375, U_b 0.0762, AUC 0.8966, Top1 0.7531.

## Plots

- [`plots/tau_sweep_comparison.png`](plots/tau_sweep_comparison.png) —
  6-panel final-eval-value vs τ.
- [`plots/tau_sweep_trajectories.png`](plots/tau_sweep_trajectories.png) —
  6-panel training trajectories (4 arms; τ=0.20 trajectory CSV lost
  to vast spot preemption, marked ★ at step 15k).

## Per-metric winner (held-out eval-batch)

| metric | winning τ | value |
|---|---|---|
| R²_random | 0.20 | 0.7731 |
| R²_naive | 0.20 | 0.7256 |
| U_temporal | 0.10 | 0.0506 |
| U_batch | 0.10 | 0.1028 |
| AUC | 0.20 | 0.8938 |
| Top1 | 0.20 | 0.7470 |

## Range per metric

| metric | min | max | range |
|---|---|---|---|
| R²_random | 0.6671 (τ=0.10) | 0.7731 (τ=0.20) | 0.106 |
| R²_naive | 0.6075 (τ=0.10) | 0.7256 (τ=0.20) | 0.118 |
| U_temporal | 0.0078 (τ=0.03) | 0.0506 (τ=0.10) | 0.043 |
| U_batch | 0.0099 (τ=0.03) | 0.1028 (τ=0.10) | 0.093 |
| AUC | 0.8859 (τ=0.05) | 0.8938 (τ=0.20) | 0.0079 |
| Top1 | 0.7319 (τ=0.05) | 0.7470 (τ=0.20) | 0.0151 |

## In-training vs held-out gap (4 arms with trajectories)

The held-out eval values can be compared to each arm's last-step
in-training values from the trajectory CSVs (4 of 5 arms have
trajectory CSVs locally). Examples:

| τ | metric | held-out | last in-training step value |
|---|---|---|---|
| 0.10 | AUC | 0.8915 | 0.9199 (step 15000) |
| 0.10 | U_b | 0.1028 | 0.0939 (step 15000) |
| 0.07 | AUC | 0.8908 | 0.9171 (step 15000) |
| 0.07 | U_b | 0.0639 | 0.0598 (step 15000) |

The in-training values are computed under `model.train()` (dropout
active) on the just-trained batch. The held-out values are computed
under `model.eval()` (dropout off) on a different batch. The two are
different by metric and arm; we have not measured the cause.

## Open data point

τ=0.20 trajectory CSV was lost when the original vast spot
auto-stopped on completion before any sync_loop pulled the file (one-shot
DONE-marker scp-back was wired up instead). The FINAL.pth was
preserved (it had been pulled at DONE), so the eval-batch row is
authoritative; only the per-step training trajectory is missing.
Postmortem and the resulting [`REMOTE_LAUNCH_CHECKLIST`](../REMOTE_LAUNCH_CHECKLIST.md)
addition to CLAUDE.md document the fix.

## What this report does not cover

- Proxy MASE per arm. The
  [`scripts/run_tau_sweep_proxy.sh`](scripts/run_tau_sweep_proxy.sh)
  recipe trains an R3_E4 head on each backbone for downstream
  GIFT-Eval; it has not been run yet.
- Whether AUC/Top1/U-metric ranks predict downstream MASE rank for
  this set of 5 arms specifically. The proxy correlation analysis at
  `experiments/2026-05-05_exp_qhead_improvements/results/backbone_proxy_correlation.csv`
  is over a *different* set of 5 backbones (n=5, directional ρ);
  applying its conclusions to this τ sweep would be extrapolation.
