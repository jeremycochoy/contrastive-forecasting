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

**Held-out eval batches.** All arms scored on **N=50 disjoint held-out
batches** (B=256 each, 10 different `skip_rows` values spaced 4.27M
rows apart so each wraps to a distinct region of the 42.7M-row
corpus). Computed under `model.eval()` (dropout off) by
[`scripts/eval_multisample.py`](scripts/eval_multisample.py), which
writes [`results/tau_sweep_metrics_multisample.csv`](results/tau_sweep_metrics_multisample.csv).
Per-batch stdev across the 50 batches gives a noise floor for each
metric; SEM (precision of the mean) is stdev/√N.

**Reference.** `backbone-beta_167k` (trainable τ, 167k steps) was
scored on a single held-out batch — included as a context anchor only,
not an arm in the sweep.

## What we did

- Trained the 5 fixed-τ arms ({0.03, 0.05, 0.07, 0.10, 0.20}) and the
  learnable-τ arm to 15,000 steps each.
- Evaluated all 6 backbones on N=50 disjoint held-out batches (B=256
  each); recorded mean ± stdev per metric.
- Generated training-trajectory + held-out comparison plots (see "What
  we learned").

## What we learned

### Trajectories

![trajectories](plots/tau_sweep_v2_trajectories.png)

6-panel training trajectories for all 6 arms (R² and U panels: 400-step MA; AUC and Top-1: 125-step MA). R² panels
are zoomed (R²_random 0.60–0.85, R²_naive 0.45–0.80); AUC and Top-1
panels are zoomed for legibility.
**τ=0.10 dominates the in-training AUC and Top-1 trajectory across the
full 15k window**, with the learnable arm sitting just underneath. AUC
trajectories overlap closely; Top-1 separates the softer-τ arms (0.10,
learnable) above the sharper-τ pack. R²_random and R²_naive show
τ=0.20 well above the rest.

### Held-out eval (mean ± stdev, N=50 disjoint batches)

![multisample](plots/tau_sweep_eval_multisample.png)

Six metric panels, one box per arm. Boxes span ±1 stdev across the 50
batches; the line is at the mean. Arms are placed left-to-right by τ
value, with the learnable-τ arm next to its converged value (≈ 0.07,
not its 0.10 init): τ=0.03 → 0.05 → 0.07 → learnable → 0.10 → 0.20.

| backbone                 | τ            | R²_random           | R²_naive            | U_t                 | U_b                 | AUC                 | Top-1               |
|--------------------------|--------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| tau_sweep_0_03           | 0.03         | 0.7631 ± 0.0070     | 0.6956 ± 0.0087     | 0.0079 ± 0.0001     | 0.0099 ± 0.0002     | 0.8978 ± 0.0055     | 0.7473 ± 0.0099     |
| tau_sweep_0_05           | 0.05         | 0.7248 ± 0.0066     | 0.6530 ± 0.0086     | 0.0185 ± 0.0004     | 0.0316 ± 0.0009     | 0.8936 ± 0.0055     | 0.7400 ± 0.0098     |
| tau_sweep_0_07           | 0.07         | 0.6935 ± 0.0068     | 0.6271 ± 0.0089     | 0.0330 ± 0.0008     | 0.0632 ± 0.0012     | 0.8978 ± 0.0053     | 0.7487 ± 0.0097     |
| tau_sweep_learnable_0_10 | 0.10 → 0.069 | 0.6922 ± 0.0069     | 0.6254 ± 0.0091     | 0.0323 ± 0.0007     | 0.0632 ± 0.0011     | 0.8993 ± 0.0053     | 0.7500 ± 0.0099     |
| tau_sweep_0_10           | 0.10         | 0.6683 ± 0.0074     | 0.6153 ± 0.0094     | **0.0512 ± 0.0012** | **0.1019 ± 0.0015** | 0.8993 ± 0.0053     | 0.7535 ± 0.0098     |
| tau_sweep_0_20           | 0.20         | **0.7703 ± 0.0068** | **0.7262 ± 0.0083** | 0.0383 ± 0.0009     | 0.0819 ± 0.0014     | **0.9021 ± 0.0054** | **0.7570 ± 0.0097** |

(Source: [`results/tau_sweep_metrics_multisample.csv`](results/tau_sweep_metrics_multisample.csv).)

backbone-beta_167k single-batch reference: R²_random 0.6839,
R²_naive 0.6080, U_t 0.0375, U_b 0.0762, AUC 0.8966, Top-1 0.7531.

### Per-metric winner

| metric     | winning arm   | value                | runner-up               |
|------------|---------------|----------------------|--------------------------|
| R²_random  | τ=0.20        | 0.7703 ± 0.0068      | τ=0.03 (0.7631 ± 0.0070) |
| R²_naive   | τ=0.20        | 0.7262 ± 0.0083      | τ=0.03 (0.6956 ± 0.0087) |
| U_temporal | τ=0.10        | 0.0512 ± 0.0012      | τ=0.20 (0.0383 ± 0.0009) |
| U_batch    | τ=0.10        | 0.1019 ± 0.0015      | τ=0.20 (0.0819 ± 0.0014) |
| AUC        | τ=0.20        | 0.9021 ± 0.0054      | τ=0.10 (0.8993 ± 0.0053) |
| Top-1      | τ=0.20        | 0.7570 ± 0.0097      | τ=0.10 (0.7535 ± 0.0098) |

### Significance of inter-arm differences

τ=0.10 vs τ=0.20 — the leading pair. With N=50 batches, the per-batch
stdev (σ) gives the per-sample noise floor; SEM = σ/√50 is the
precision of the mean. A difference is "resolved" if Δ exceeds ~2 SEM.

| metric     | τ=0.20 − τ=0.10 | per-batch σ | SEM (σ/√50) | Δ / SEM   | resolved? |
|------------|-----------------|-------------|-------------|-----------|-----------|
| R²_random  | +0.1020         | 0.0074      | 0.0010      | +98σ      | yes       |
| R²_naive   | +0.1108         | 0.0094      | 0.0013      | +83σ      | yes       |
| U_temporal | −0.0129         | 0.0012      | 0.0002      | −75σ      | yes (τ=0.10 higher) |
| U_batch    | −0.0201         | 0.0015      | 0.0002      | −97σ      | yes (τ=0.10 higher) |
| AUC        | +0.0027         | 0.0054      | 0.0008      | +3.6σ     | **yes (new)** |
| Top-1      | +0.0035         | 0.0098      | 0.0014      | +2.5σ     | **yes (new)** |

The N=10 verdict had AUC and Top-1 within ~1 SEM. With N=50, the
SEM tightens by √5 ≈ 2.2× and AUC / Top-1 separate by 3.6 / 2.5 SEM
respectively — τ=0.20 wins both.

### Verdict

- **R²_random / R²_naive: τ=0.20 wins decisively** (Δ ≈ +0.10, >80 SEM).
- **U_temporal / U_batch: τ=0.10 wins decisively** (Δ ≈ +0.013 /
  +0.020, >70 SEM each). The directionality of "softer τ → more
  spread" holds.
- **AUC / Top-1: τ=0.20 wins** (Δ = +0.0027 AUC at +3.6 SEM,
  +0.0035 Top-1 at +2.5 SEM). N=50 was needed to resolve this; at N=10
  the difference was within ~1 SEM.

The split is therefore: τ=0.20 is the better recipe on 4 of 6 metrics
(R²_random, R²_naive, AUC, Top-1) — including the discrimination
metrics that the prior proxy-correlation analysis found most
predictive of downstream MASE — while τ=0.10 wins on the encoder-
spread (U) metrics.

The **learnable-τ arm slid from init=0.10 to τ ≈ 0.069**
(log_inv_tau ≈ 2.671) over 15k steps. Its held-out values land near
τ=0.07 on every metric (R²_random 0.6922 vs τ=0.07's 0.6935; AUC
0.8993 vs τ=0.07's 0.8978; U_b 0.0632 = τ=0.07's 0.0632) — coherent
with the converged τ value sitting just below 0.07. Within a single
15k run, gradient pressure pulls τ down, not up to either of the soft
optima. **Learnable-τ does not discover the τ=0.10 / τ=0.20 optimum
from a 0.10 init.**

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
