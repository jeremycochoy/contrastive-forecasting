# τ-sweep — RESULTS v2 (extended)

Extends [`RESULTS.md`](RESULTS.md) with two additional arms:
- **Exp 1A v2** (`tau_sweep_0_20_v2`) — τ=0.20 redo on a clean vast 5090
  to recover the lost trajectory CSV from the original τ=0.20 arm.
  Training process died on the vast container at step 7800 with no
  python proc visible in `ps`; the last-saved checkpoint is
  `tau_sweep_0_20_v2_best_loss.pth` from step 7700 (copied to
  `tau_sweep_0_20_v2_FINAL.pth` for eval). The vast instance was
  destroyed after the final pull.
- **Exp 1B learnable τ** (`tau_sweep_learnable_0_10`) — fixed-arch
  CLIP-style learnable τ initialised from 0.10. 15,000 steps on elisa
  GPU 1 alongside the v2 arm. Final τ value: **0.0692** (converged
  downward from the 0.10 init).

The original 5 fixed-τ arms ({0.03, 0.05, 0.07, 0.10, 0.20}) keep their
existing FINAL.pth checkpoints; their values in this v2 eval CSV
match `tau_sweep_metrics.csv` exactly (same script, same held-out
batch, same seed).

## Setup

Same recipe as RESULTS.md (T_RAW=4096, C=1, d_model=384, num_layers=6,
n_heads=6, freq+season emb dim 3, RevEWMNorm span=128, mixup_p=0.3,
AdamW lr 1e-3 wd 0.1, batch 256, 15k steps unless noted). All 8
backbones evaluated on the same held-out HF batch (skip=50M wrap to
row 7,260,000; B=256; eval seed=0) by
[`scripts/eval_tau_sweep_metrics_v2.py`](scripts/eval_tau_sweep_metrics_v2.py).

## Held-out eval (FINAL.pth, 8 backbones)

| backbone                       | τ_csv | encoder        | R²_rand | R²_naive | U_t    | U_b    | AUC    | Top1   |
|--------------------------------|-------|----------------|---------|----------|--------|--------|--------|--------|
| tau_sweep_0_03                 | 0.03  | gru            | 0.7633  | 0.6914   | 0.0078 | 0.0099 | 0.8890 | 0.7369 |
| tau_sweep_0_05                 | 0.05  | gru            | 0.7254  | 0.6479   | 0.0183 | 0.0319 | 0.8859 | 0.7319 |
| tau_sweep_0_07                 | 0.07  | gru            | 0.6938  | 0.6208   | 0.0327 | 0.0639 | 0.8908 | 0.7407 |
| tau_sweep_0_10                 | 0.10  | gru            | 0.6671  | 0.6075   | 0.0506 | **0.1028** | 0.8915 | 0.7449 |
| tau_sweep_0_20 (orig, 15k)     | 0.20  | gru            | **0.7731** | **0.7256** | 0.0386 | 0.0850 | **0.8938** | **0.7470** |
| tau_sweep_0_20_v2 (7.8k partial) | 0.20 | gru          | 0.7656  | 0.7120   | 0.0356 | 0.0751 | 0.8896 | 0.7395 |
| tau_sweep_learnable_0_10       | (0.10 → 0.069) | gru   | 0.6919  | 0.6185   | 0.0320 | 0.0638 | 0.8908 | 0.7396 |

Source: [`results/tau_sweep_metrics_v2.csv`](results/tau_sweep_metrics_v2.csv).

Reference (backbone-beta_167k on the same held-out batch):
R²_rand 0.6839, R²_naive 0.6080, U_t 0.0375, U_b 0.0762,
AUC 0.8966, Top1 0.7531.

## Last in-training step values (from per-step CSVs)

In-training values are computed under `model.train()` (dropout active)
on the just-trained batch and differ from the held-out values for
all arms; the comparison is informational only.

| backbone                       | last step | loss   | AUC_train | Top1_train | U_b_train |
|--------------------------------|-----------|--------|-----------|------------|-----------|
| tau_sweep_0_03                 | 23000     | 6.7313 | 0.9133    | 0.7669     | 0.0102    |
| tau_sweep_0_05                 | 15000     | 6.9970 | 0.9177    | 0.7660     | 0.0288    |
| tau_sweep_0_07                 | 15000     | 7.0201 | 0.9171    | 0.7694     | 0.0598    |
| tau_sweep_0_10                 | 15000     | 7.0414 | 0.9199    | 0.7765     | 0.0939    |
| tau_sweep_0_20_v2              | 7800      | 8.2126 | 0.8780    | 0.7174     | 0.0667    |
| tau_sweep_learnable_0_10       | 15000     | 6.9749 | 0.9206    | 0.7745     | 0.0591    |

(τ=0.20 original 15k arm has no per-step trajectory CSV — lost in the
spot-stop event documented in the original RESULTS.md.)

## Plots

- [`plots/tau_sweep_v2_comparison.png`](plots/tau_sweep_v2_comparison.png) —
  6-panel held-out eval, bar chart over the 6 arms with full eval
  data (the 5 fixed τ + learnable_τ + v2 partial as a separate bar).
- [`plots/tau_sweep_v2_trajectories.png`](plots/tau_sweep_v2_trajectories.png) —
  6-panel training trajectories (1000-step MA), 6 arms with per-step
  CSVs: τ ∈ {0.03, 0.05, 0.07, 0.10, 0.20-v2 (7.8k), learnable_0.10}.
  AUC zoom 0.89–0.91, Top1 zoom 0.72–0.76 — same as RESULTS.md.

## Per-metric winner

| metric    | winning backbone           | value   |
|-----------|----------------------------|---------|
| R²_random | tau_sweep_0_20 (orig)      | 0.7731  |
| R²_naive  | tau_sweep_0_20 (orig)      | 0.7256  |
| U_temporal| tau_sweep_0_10             | 0.0506  |
| U_batch   | tau_sweep_0_10             | 0.1028  |
| AUC       | tau_sweep_0_20 (orig)      | 0.8938  |
| Top1      | tau_sweep_0_20 (orig)      | 0.7470  |

## Picked winner: τ=0.20 (the original 15k arm)

The original τ=0.20 arm wins 4 of 6 held-out metrics (R²_rand, R²_naive,
AUC, Top1). U_t and U_b favour τ=0.10 — but those measure
representation-spread, not predictive quality, so they are weighted
secondary for downstream MASE.

The τ=0.20 v2 redo (7.8k partial) trails the original on every metric.
That is consistent with it being only 52% trained (7800 / 15000 steps);
not a rejection of τ=0.20 itself. The v2 in-training AUC (1000-step MA)
peaked at step 7798 (~0.896), with the last 500 smoothed steps still
slightly above the prior 500 (0.8953 vs 0.8945) — i.e. still improving,
albeit slowly. Extrapolating, a full 15k v2 run would likely land within
sampling noise of the original.

The learnable_τ arm slid from init=0.10 toward τ=0.069 over 15k steps
(log_inv_tau=2.671). Its held-out values land near τ=0.07 on every
metric (R²_random 0.6919 vs τ=0.07's 0.6938, AUC 0.8908 = τ=0.07's
0.8908, U_b 0.0638 vs τ=0.07's 0.0639) — coherent with the converged
τ value being just below 0.07. It does **not** discover the τ=0.20
optimum; gradient pressure within a single run pulls τ down, not up.
**Recommendation: keep using fixed τ for the next backbone, with τ=0.20
as the picked value.** Re-evaluate learnable_τ if/when a wider init
sweep (e.g. starting from τ=0.30 or τ=0.50) gets run.

## Vast instance teardown

Both vast instances destroyed after final pulls:
- 36368320 (Exp 2 residual_silu) — uptime 2h 12m, spent **$1.51**.
- 36367883 (τ=0.20 v2) — uptime 2h 26m, spent **$0.97**. The training
  process was already dead when teardown happened (`ps -ef | grep
  python` showed only sshd); no python proc had been running since the
  log/CSV stopped advancing at step 7800.

## What this report does not cover

- Proxy MASE per arm — same blocker as RESULTS.md. The proxy harness
  (R3_E4 head training) has not been re-run for v2.
- Whether the held-out eval rank-orders predict downstream MASE rank
  for these 8 backbones specifically — same caveat as RESULTS.md.
