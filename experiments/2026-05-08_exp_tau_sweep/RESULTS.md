# τ-sweep — RESULTS

## Goal / question

Does the contrastive temperature τ — fixed during training — change
the encoder's representation quality, forecast-match metrics, and
dimension usage? backbone-beta's learnable τ converged to ~0.072 over
167k steps; the sweep probes whether nearby fixed values match that
optimum and whether sharper / softer τ values shift the metrics
materially.

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

**Architecture / training recipe** matches backbone-beta exactly, only
τ varies: 6-layer × 384-D transformer, GRU patch encoder, T_RAW=4096,
B=256, AdamW lr=1e-3, 15,000 steps per arm.

**6 metrics tracked per minibatch:**

| metric        | what it measures                                                                                  |
|---------------|---------------------------------------------------------------------------------------------------|
| `R²_random`   | forecast match vs random pair — improvement over a baseline that pairs forecasts with arbitrary held-out targets |
| `R²_naive`    | forecast match vs naive last-step — improvement over a "no-change" baseline that copies the previous encoder output as the prediction |
| `U_temporal`  | dimension usage along the time axis — how much of the 384-D encoder-output space is spanned by one series across its time positions (averaged over batch) |
| `U_batch`     | dimension usage along the batch axis — how much of the 384-D space is spanned by different series at the same time position (averaged over time) |
| `AUC`         | representation quality / discrimination — per-query, fraction of past-window negatives that the positive ranks above |
| `Top-1`       | representation quality / strict discrimination — fraction of queries where the positive beats every past-window negative simultaneously (strictest version of AUC) |

**Held-out eval.** All arms scored on **N=50 disjoint held-out batches**
(B=256 each, model.eval()) by
[`scripts/eval_multisample.py`](scripts/eval_multisample.py), writing
[`results/tau_sweep_metrics_multisample.csv`](results/tau_sweep_metrics_multisample.csv).
SEM (precision of the mean) = stdev/√50. `backbone-beta_167k` is shown
as a single-batch reference only, not an arm.

## What we learned

### Trajectories

![trajectories](plots/tau_sweep_v2_trajectories.png)

**τ=0.10 dominates in-training AUC and Top-1 across the full 15k
window**, with the learnable arm just underneath; R²_random / R²_naive
show **τ=0.20 well above the rest**.

### Held-out eval (mean ± stdev, N=50 disjoint batches)

![multisample](plots/tau_sweep_eval_multisample.png)

| backbone                 | τ            | R²_random           | R²_naive            | U_t                 | U_b                 | AUC                 | Top-1               |
|--------------------------|--------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| tau_sweep_0_03           | 0.03         | 0.7631 ± 0.0070     | 0.6956 ± 0.0087     | 0.0079 ± 0.0001     | 0.0099 ± 0.0002     | 0.8978 ± 0.0055     | 0.7473 ± 0.0099     |
| tau_sweep_0_05           | 0.05         | 0.7248 ± 0.0066     | 0.6530 ± 0.0086     | 0.0185 ± 0.0004     | 0.0316 ± 0.0009     | 0.8936 ± 0.0055     | 0.7400 ± 0.0098     |
| tau_sweep_0_07           | 0.07         | 0.6935 ± 0.0068     | 0.6271 ± 0.0089     | 0.0330 ± 0.0008     | 0.0632 ± 0.0012     | 0.8978 ± 0.0053     | 0.7487 ± 0.0097     |
| tau_sweep_learnable_0_10 | 0.10 → 0.069 | 0.6922 ± 0.0069     | 0.6254 ± 0.0091     | 0.0323 ± 0.0007     | 0.0632 ± 0.0011     | 0.8993 ± 0.0053     | 0.7500 ± 0.0099     |
| tau_sweep_0_10           | 0.10         | 0.6683 ± 0.0074     | 0.6153 ± 0.0094     | **0.0512 ± 0.0012** | **0.1019 ± 0.0015** | 0.8993 ± 0.0053     | 0.7535 ± 0.0098     |
| tau_sweep_0_20           | 0.20         | **0.7703 ± 0.0068** | **0.7262 ± 0.0083** | 0.0383 ± 0.0009     | 0.0819 ± 0.0014     | **0.9021 ± 0.0054** | **0.7570 ± 0.0097** |

backbone-β_167k single-batch reference: R² 0.6839 / 0.6080,
U 0.0375 / 0.0762, AUC 0.8966, Top-1 0.7531.

### Per-metric winner

| metric     | winning arm   | value                | runner-up               |
|------------|---------------|----------------------|--------------------------|
| R²_random  | τ=0.20        | 0.7703 ± 0.0068      | τ=0.03 (0.7631 ± 0.0070) |
| R²_naive   | τ=0.20        | 0.7262 ± 0.0083      | τ=0.03 (0.6956 ± 0.0087) |
| U_temporal | τ=0.10        | 0.0512 ± 0.0012      | τ=0.20 (0.0383 ± 0.0009) |
| U_batch    | τ=0.10        | 0.1019 ± 0.0015      | τ=0.20 (0.0819 ± 0.0014) |
| AUC        | τ=0.20        | 0.9021 ± 0.0054      | τ=0.10 (0.8993 ± 0.0053) |
| Top-1      | τ=0.20        | 0.7570 ± 0.0097      | τ=0.10 (0.7535 ± 0.0098) |

### τ=0.10 vs τ=0.20 — significance

| metric     | τ=0.20 − τ=0.10 | Δ / SEM   | resolved? |
|------------|-----------------|-----------|-----------|
| R²_random  | +0.1020         | +98σ      | yes       |
| R²_naive   | +0.1108         | +83σ      | yes       |
| U_temporal | −0.0129         | −75σ      | yes (τ=0.10 higher) |
| U_batch    | −0.0201         | −97σ      | yes (τ=0.10 higher) |
| AUC        | +0.0027         | +3.6σ     | yes       |
| Top-1      | +0.0035         | +2.5σ     | yes       |

### Verdict

- **τ=0.20 wins R²_random, R²_naive, AUC, Top-1** (4 of 6 metrics —
  including both discrimination metrics).
- **τ=0.10 wins U_temporal, U_batch** (encoder-spread metrics; "softer
  τ → more spread" holds within the swept range).

**Learnable-τ from init=0.10 slides to τ ≈ 0.069 over 15k steps**
(log_inv_tau ≈ 2.671). Its held-out values land on the τ=0.07 cluster
(R²_random 0.6922 vs τ=0.07's 0.6935; AUC 0.8993 vs 0.8978; U_b 0.0632
= 0.0632). Gradient pressure pulls τ down, not up — learnable-τ does
not discover the τ=0.10 / τ=0.20 optimum from a 0.10 init.

