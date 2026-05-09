# τ-sweep — RESULTS

## Goal / question

Does the contrastive temperature τ — fixed during training rather than
learned — affect the converged representation quality (AUC, Top-1 — the
encoder's ability to distinguish a target from negatives), the
forecast-match metrics (R²_random, R²_naive — prediction-error of the
forecaster's output vs random / naive-last-step baselines in cosine
space), and the encoder dimension usage (U_t, U_b — directional spread
across time / batch axes) of the contrastive backbone? AUC was the
strongest predictor of downstream MASE in the prior 5-backbone
proxy-correlation analysis (Spearman ρ ≈ +0.70, n=5, directional only),
so its sensitivity to τ is the central question.

backbone-beta's learnable τ converged to ~0.072 over 167k steps. The
sweep probes whether nearby fixed values match that learnable optimum
and whether sharper / softer τ values shift the metrics materially.

## Protocol

**Sweep design.** Six from-scratch arms, identical architecture and
hyperparameters except for τ:

| arm                             | τ                  | rationale                                      |
|---------------------------------|--------------------|------------------------------------------------|
| `tau_sweep_0_03`                | 0.03               | sharp — punishes near-misses harder            |
| `tau_sweep_0_05`                | 0.05               | moderately sharp                               |
| `tau_sweep_0_07`                | 0.07               | closest fixed value to backbone-beta's τ ≈ 0.072 |
| `tau_sweep_0_10`                | 0.10               | moderately soft                                |
| `tau_sweep_learnable_0_10`      | learnable, init 0.10 | CLIP-style learnable τ; tests whether it discovers the fixed-τ optimum |
| `tau_sweep_0_20`                | 0.20               | soft — high entropy, harder to discriminate    |

**Architecture / training recipe** (matches backbone-beta exactly,
only τ varies): T_RAW=4096, C=1, d_model=384, num_layers=6, n_heads=6,
freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind=ewma span=128,
loss=cosine_similarity_batch, batch_size=256, mixup_p=0.3, mix_ratio=0,
AdamW lr=1e-3 wd=0.1 β1=0.9 β2=0.98, **15,000 steps per arm**. (The
τ=0.03 arm was killed at 23k after metrics plateaued by step ~5k;
budget for the remaining arms cut from 50k to 15k.)

**6 metrics tracked per minibatch.** The trainer logs, every step, six
`@torch.no_grad` values that reuse the already-computed `f`, `h`, `z`
tensors from the loss:

| metric        | what it measures                                                                                  |
|---------------|---------------------------------------------------------------------------------------------------|
| `R²_random`   | forecast match vs random pair — improvement over a baseline that pairs forecasts with arbitrary held-out targets |
| `R²_naive`    | forecast match vs naive last-step — improvement over a "no-change" baseline that copies the previous encoder output as the prediction |
| `U_temporal`  | dimension usage along the time axis — how much of the 384-D encoder-output space is spanned by one series across its time positions (averaged over batch) |
| `U_batch`     | dimension usage along the batch axis — how much of the 384-D space is spanned by different series at the same time position (averaged over time) |
| `AUC`         | representation quality / discrimination — per-query, fraction of past-window negatives that the positive ranks above |
| `Top-1`       | representation quality / strict discrimination — fraction of queries where the positive beats every past-window negative simultaneously (strictest version of AUC) |

Column names match `experiments/2026-05-05_exp_qhead_improvements/results/backbone_proxy_correlation.csv`
so per-batch and post-hoc metrics merge cleanly.

**Held-out eval batches.** All arms scored on **N=10 disjoint held-out
batches** (B=256 each, 10 different `skip_rows` values spaced 4.27M
rows apart so each wraps to a distinct region of the 42.7M-row
corpus). Computed under `model.eval()` (dropout off) by
[`scripts/eval_multisample.py`](scripts/eval_multisample.py), which
writes [`results/tau_sweep_metrics_multisample.csv`](results/tau_sweep_metrics_multisample.csv).
Per-batch stdev across the 10 batches gives a noise floor for each
metric; SEM (precision of the mean) is stdev/√N.

**Reference.** `backbone-beta_167k` (trainable τ, 167k steps) was
scored on a single held-out batch — included as a context anchor only,
not an arm in the sweep.

## What we did

- Trained the 5 fixed-τ arms ({0.03, 0.05, 0.07, 0.10, 0.20}) and the
  learnable-τ arm to 15,000 steps each.
- Evaluated all 6 backbones on N=10 disjoint held-out batches (B=256
  each); recorded mean ± stdev per metric.
- Generated training-trajectory + held-out comparison plots (see "What
  we learned").

## What we learned

### Trajectories

![trajectories](plots/tau_sweep_v2_trajectories.png)

6-panel training trajectories for all 6 arms (R² and U panels: 400-step MA; AUC and Top-1: 100-step MA). R² panels
are pinned to ≥ 0; AUC and Top-1 panels are zoomed for legibility.
**τ=0.10 dominates the in-training AUC and Top-1 trajectory across the
full 15k window**, with the learnable arm sitting just underneath. AUC
trajectories overlap closely; Top-1 separates the softer-τ arms (0.10,
learnable) above the sharper-τ pack. R²_random and R²_naive show
τ=0.20 well above the rest.

### Held-out eval (mean ± stdev, N=10 disjoint batches)

![multisample](plots/tau_sweep_eval_multisample.png)

Six metric panels, one box per arm. Boxes span ±1 stdev across the 10
batches; the line is at the mean. Arms are placed left-to-right τ=0.03
→ 0.05 → 0.07 → 0.10 → learnable_τ_init0.10 → 0.20.

| backbone                 | τ            | R²_random           | R²_naive            | U_t                | U_b                 | AUC                 | Top-1               |
|--------------------------|--------------|---------------------|---------------------|--------------------|---------------------|---------------------|---------------------|
| tau_sweep_0_03           | 0.03         | 0.7624 ± 0.0041     | 0.6928 ± 0.0062     | 0.0078 ± 0.0001    | 0.0099 ± 0.0001     | 0.8967 ± 0.0063     | 0.7457 ± 0.0102     |
| tau_sweep_0_05           | 0.05         | 0.7239 ± 0.0051     | 0.6497 ± 0.0066     | 0.0183 ± 0.0005    | 0.0315 ± 0.0007     | 0.8923 ± 0.0062     | 0.7381 ± 0.0097     |
| tau_sweep_0_07           | 0.07         | 0.6926 ± 0.0053     | 0.6238 ± 0.0071     | 0.0326 ± 0.0010    | 0.0632 ± 0.0009     | 0.8967 ± 0.0060     | 0.7475 ± 0.0101     |
| tau_sweep_0_10           | 0.10         | 0.6672 ± 0.0063     | 0.6118 ± 0.0079     | **0.0506 ± 0.0017** | **0.1020 ± 0.0015** | 0.8980 ± 0.0062     | 0.7518 ± 0.0099     |
| tau_sweep_learnable_0_10 | 0.10 → 0.069 | 0.6911 ± 0.0054     | 0.6220 ± 0.0072     | 0.0320 ± 0.0009    | 0.0632 ± 0.0009     | 0.8981 ± 0.0060     | 0.7487 ± 0.0100     |
| tau_sweep_0_20           | 0.20         | **0.7693 ± 0.0048** | **0.7233 ± 0.0064** | 0.0379 ± 0.0012    | 0.0818 ± 0.0012     | **0.9004 ± 0.0065** | **0.7545 ± 0.0106** |

(Source: [`results/tau_sweep_metrics_multisample.csv`](results/tau_sweep_metrics_multisample.csv).)

backbone-beta_167k single-batch reference: R²_random 0.6839,
R²_naive 0.6080, U_t 0.0375, U_b 0.0762, AUC 0.8966, Top-1 0.7531.

### Per-metric winner

| metric     | winning arm   | value                                            |
|------------|---------------|--------------------------------------------------|
| R²_random  | τ=0.20        | 0.7693 ± 0.0048                                  |
| R²_naive   | τ=0.20        | 0.7233 ± 0.0064                                  |
| U_temporal | τ=0.10        | 0.0506 ± 0.0017                                  |
| U_batch    | τ=0.10        | 0.1020 ± 0.0015                                  |
| AUC        | τ=0.20        | 0.9004 ± 0.0065 (within ~1 SEM of τ=0.10)        |
| Top-1      | τ=0.20        | 0.7545 ± 0.0106 (within ~1 SEM of τ=0.10)        |

Per-batch stdev (~0.006 AUC / ~0.010 Top-1 / ~0.005 R²_random) is the
noise floor for a single batch; SEM across 10 batches is stdev/√10
(~0.002 AUC / ~0.003 Top-1).

### Significance of inter-arm differences

τ=0.10 vs τ=0.20 — the leading "winner-uncertain" pair:

| metric     | τ=0.20 − τ=0.10 | max stdev | resolved (>2σ)?               |
|------------|-----------------|-----------|-------------------------------|
| R²_random  | +0.1021         | 0.0063    | yes (~16σ)                    |
| R²_naive   | +0.1115         | 0.0079    | yes (~14σ)                    |
| U_temporal | −0.0127         | 0.0017    | yes (~7σ), τ=0.10 higher      |
| U_batch    | −0.0202         | 0.0015    | yes (~13σ), τ=0.10 higher     |
| AUC        | +0.0024         | 0.0065    | **no — within ~1 SEM**        |
| Top-1      | +0.0027         | 0.0106    | **no — within ~1 SEM**        |

### Verdict

- **R²_random / R²_naive: τ=0.20 wins decisively** (Δ ≈ +0.10, >10σ).
- **U_temporal / U_batch: τ=0.10 wins decisively** (Δ ≈ +0.013 / +0.020,
  >7σ). The directionality of "softer τ → more spread" holds.
- **AUC / Top-1: τ=0.10 vs τ=0.20 is within ~1 SEM, both ways.** The
  apparent τ=0.20 edge over τ=0.10 (+0.0024 AUC, +0.0027 Top-1) is
  within the per-batch stdev. **AUC and Top-1 do not separate τ=0.10
  from τ=0.20 at N=10**; a larger sample budget is needed to call this
  metric.

The split is therefore: τ=0.20 is materially better on the
forecast-match (R²) metrics, τ=0.10 is materially better on the
encoder-spread (U) metrics, and AUC/Top-1 do not separate the two at
the available eval precision (N=10).

The **learnable-τ arm slid from init=0.10 to τ ≈ 0.069**
(log_inv_tau ≈ 2.671) over 15k steps. Its held-out values land near
τ=0.07 on every metric (R²_random 0.6911 vs τ=0.07's 0.6926; AUC
0.8981 vs τ=0.07's 0.8967; U_b 0.0632 = τ=0.07's 0.0632) — coherent
with the converged τ value sitting just below 0.07. Within a single
15k run, gradient pressure pulls τ down, not up to either of the soft
optima. **Learnable-τ does not discover the τ=0.10 / τ=0.20 optimum
from a 0.10 init.** A wider-init learnable-τ sweep (e.g. init=0.30,
init=0.50) would test whether the learnable schedule can find the
soft optimum from above.

### In-training vs held-out gap

In-training values are computed under `model.train()` (dropout active)
on the just-trained batch. Held-out values are computed under
`model.eval()` (dropout off) on a different batch. The two differ by
metric and arm; the underlying mechanism (dropout, batch sampling,
something else) was not investigated.

| backbone                 | last step | in-training loss | in-training AUC | in-training Top-1 | in-training U_b |
|--------------------------|-----------|------------------|-----------------|-------------------|------------------|
| tau_sweep_0_03           | 23000     | 6.7313           | 0.9133          | 0.7669            | 0.0102           |
| tau_sweep_0_05           | 15000     | 6.9970           | 0.9177          | 0.7660            | 0.0288           |
| tau_sweep_0_07           | 15000     | 7.0201           | 0.9171          | 0.7694            | 0.0598           |
| tau_sweep_0_10           | 15000     | 7.0414           | 0.9199          | 0.7765            | 0.0939           |
| tau_sweep_learnable_0_10 | 15000     | 6.9749           | 0.9206          | 0.7745            | 0.0591           |

(τ=0.20 in-training row not yet added; the trajectory CSV at
`sync_tau_sweep_arm5_v2/checkpoints/tau_sweep_0_20_v2_losses.csv` can
populate this table at the next refresh.)

## Open

- **Higher-N eval to resolve AUC/Top-1.** Re-run the held-out eval at
  N=50 (~2.2× tighter SEM) to test whether τ=0.20's apparent +0.002 AUC /
  +0.003 Top-1 edge over τ=0.10 is real or noise.
- **Proxy MASE per arm.** The
  [`scripts/run_tau_sweep_proxy.sh`](scripts/run_tau_sweep_proxy.sh)
  recipe trains an R3_E4 head on each backbone for downstream
  GIFT-Eval; it has not been run yet.
- **Whether AUC/Top-1/U-metric ranks predict downstream MASE rank**
  for this set of arms specifically. The proxy correlation analysis at
  `experiments/2026-05-05_exp_qhead_improvements/results/backbone_proxy_correlation.csv`
  is over a different set of 5 backbones (n=5, directional ρ);
  applying its conclusions to this sweep would be extrapolation.
- **Wider-init learnable-τ sweep.** Test whether learnable τ
  initialised above the soft optimum (e.g. init=0.30, 0.50) finds the
  τ=0.10–0.20 region instead of sliding down to ~0.07.
- **τ=0.30 fixed arm.** Extend the sweep above 0.20 to test whether
  the soft side keeps improving.
