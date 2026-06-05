# Side-investigation: which cheap backbone metric predicts downstream MASE?

This is a **different question** from the head-improvement thread in
[`../exp_qhead_improvements.md`](../exp_qhead_improvements.md) and is
parked here so it does not break that thread. Every head experiment in
the main report shares **one** frozen backbone (backbone-beta, step
167k), so nothing below changes any head result. The question here is:
*if we ever want to compare candidate backbones without running the
slow downstream GIFT-Eval, which cheap self-supervised metric on a
held-out batch tracks downstream MASE best?*

All metrics are computed on the same fixed held-out HF batch
(skip=50M, B=256, seed=0) by `scripts/eval_backbone_metrics.py` /
`scripts/eval_backbones_cross.py`. Metric glossary:

- **R²_random / R²_naive** = `1 − Q`, where `Q = mean_b e(forecast, target) / mean_b e(reference)` and the reference is a random latent (R²_random) or the naive last-value latent (R²_naive). R²=0 → no better than the baseline; R²=1 → exact. Higher is better. Stored as `Q` in `../results/backbone_metrics_trajectory.csv`.
- **u_temporal / u_batch** = latent dimension-usage (fraction of latent dims actively used) along the temporal / batch axis. Reported as-is from the eval script.
- **auc / top1** = forecast→future retrieval AUC and top-1 accuracy on the held-out batch (does a window's forecast retrieve its own future against in-batch negatives). Higher is better.

## Backbone-beta metric trajectory across its own training

How the single backbone evolved across its 50k→167k training (one
held-out batch per checkpoint; this is **not** a per-head comparison).

![Six self-supervised backbone metrics vs training step, 50k→167k, on the fixed held-out batch. All six oscillate within a narrow band with no monotone trend.](../plots/backbone_metrics_curve.png)

The trajectory is essentially flat across the whole window — between
50k and 167k every metric's drift is within run-to-run batch noise:
r2_random Δ=+0.0015, r2_naive Δ=−0.0017, u_temporal Δ=+0.0005,
u_batch Δ=+0.0010, auc Δ=+0.0054, top1 Δ=+0.0092. Per-step band widths
(max−min over 50k–167k): r2_random 0.0345, r2_naive 0.0406,
u_temporal 0.0053, u_batch 0.0167, auc 0.0044, top1 0.0088.

| step | r2_random | r2_naive | u_temporal | u_batch | auc | top1 |
|---|---|---|---|---|---|---|
| 50000 | 0.6823 | 0.6097 | 0.0371 | 0.0752 | 0.8911 | 0.7440 |
| 60000 | 0.7149 | 0.6421 | 0.0327 | 0.0653 | 0.8955 | 0.7509 |
| 70000 | 0.6867 | 0.6145 | 0.0364 | 0.0733 | 0.8949 | 0.7496 |
| 80000 | 0.6996 | 0.6266 | 0.0349 | 0.0692 | 0.8967 | 0.7521 |
| 90000 | 0.6979 | 0.6255 | 0.0347 | 0.0680 | 0.8923 | 0.7447 |
| 100000 | 0.6970 | 0.6209 | 0.0322 | 0.0595 | 0.8937 | 0.7471 |
| 110000 | 0.6814 | 0.6052 | 0.0362 | 0.0707 | 0.8946 | 0.7486 |
| 120000 | 0.7085 | 0.6309 | 0.0323 | 0.0627 | 0.8954 | 0.7518 |
| 130000 | 0.7020 | 0.6272 | 0.0341 | 0.0661 | 0.8953 | 0.7508 |
| 140000 | 0.6972 | 0.6239 | 0.0346 | 0.0674 | 0.8957 | 0.7536 |
| 150000 | 0.6803 | 0.6015 | 0.0368 | 0.0723 | 0.8923 | 0.7454 |
| 160000 | 0.7002 | 0.6257 | 0.0351 | 0.0693 | 0.8952 | 0.7510 |
| 167000 | 0.6839 | 0.6080 | 0.0375 | 0.0762 | 0.8966 | 0.7531 |

R² columns are recomputed as `1 − Q` from the committed `Q` values in
`../results/backbone_metrics_trajectory.csv` (verified:
`1 − q_random(167k) = 0.6839`, `1 − q_naive(167k) = 0.6080`, matching
the cross-run table below).

## Cross-backbone comparison (best checkpoint per run)

The `best_loss` checkpoint (or the highest periodic save where no
`best_loss` was emitted) of each completed backbone run, on the same
fixed held-out batch. Architecture is held constant across runs
(C=1, H=384, 6L, nhead=6); runs differ in HP (β2, weight_decay,
learnable τ on/off), schedule, and total length. Source:
`../results/backbone_metrics_cross.csv`.

| name | r2_random | r2_naive | u_temporal | u_batch | auc | top1 |
|---|---|---|---|---|---|---|
| moirai_hp_FINAL_run1 | 0.6759 | 0.6091 | 0.0403 | 0.0754 | 0.8902 | 0.7402 |
| backbone_beta_167k | 0.6839 | 0.6080 | 0.0375 | 0.0762 | 0.8966 | 0.7531 |
| FRESH_50k | 0.6951 | 0.6244 | 0.0341 | 0.0670 | 0.8922 | 0.7468 |
| moirai_hp_early | 0.6983 | 0.6319 | 0.0338 | 0.0659 | 0.8929 | 0.7432 |
| learnable_tau | 0.7634 | 0.6952 | 0.0134 | 0.0205 | 0.8888 | 0.7365 |

`learnable_tau` has by far the highest R² values (0.7634 / 0.6952) and
the lowest dimension-usage — note this for the proxy test below, where
high R² does **not** translate into the best downstream MASE.

## Proxy test: train an identical head on each backbone, then GIFT-Eval

To anchor the cheap metrics to the real objective, an R3_E4-recipe head
(6L causal transformer + Moirai HP + cosine + 30k steps, **no**
`e_then_f`) was trained on each of the five backbones above, and each
head was triage-evaluated on the same 11-config subset
(`scripts/run_eval_proxy.sh`). Source:
`../results/backbone_proxy_correlation.csv`.

| name | proxy_mase | r2_random | r2_naive | u_temporal | u_batch | auc | top1 |
|---|---|---|---|---|---|---|---|
| backbone_beta_167k | 1.0166 | 0.6839 | 0.6080 | 0.0375 | 0.0762 | 0.8966 | 0.7531 |
| moirai_hp_early | 1.0259 | 0.6983 | 0.6319 | 0.0338 | 0.0659 | 0.8929 | 0.7432 |
| learnable_tau | 1.0278 | 0.7634 | 0.6952 | 0.0134 | 0.0205 | 0.8888 | 0.7365 |
| FRESH_50k | 1.0285 | 0.6951 | 0.6244 | 0.0341 | 0.0670 | 0.8922 | 0.7468 |
| moirai_hp_FINAL_run1 | 1.0940 | 0.6759 | 0.6091 | 0.0403 | 0.0754 | 0.8902 | 0.7402 |

**Spearman ρ, n=5, recomputed from the CSV** with `scipy.stats.spearmanr`.
Convention: `+ρ` means *better metric quality predicts lower proxy
MASE* — i.e. ρ is taken between each metric's value and `proxy_mase`,
then the sign is flipped for the higher-is-better metrics (auc, top1,
R²_random, R²_naive) so the sign is comparable across all metrics.

| metric | Spearman ρ (quality vs proxy_mase) |
|---|---|
| auc | +0.70 |
| top1 | +0.50 |
| r2_random | +0.30 |
| u_temporal | +0.30 |
| u_batch | −0.10 |
| r2_naive | −0.10 |

**AUC is the best predictor (ρ = +0.70).** In this set AUC ranks the
five backbones in the same order as proxy_mase except for one adjacent
swap (FRESH_50k vs learnable_tau, whose MASE differ by only 0.0007).
The R² metrics do **not** track downstream MASE: `learnable_tau` has
the highest R² of the five (0.7634 / 0.6952) yet is only third on
proxy_mase, and the proxy_mase winner (`backbone_beta_167k`) has the
second-lowest R²_random and the lowest R²_naive. n=5 is tiny — this is
directional only.

> Correction vs an earlier draft of this table: three of the six ρ
> values in the old draft (u_batch +0.40, r2_naive +0.30,
> u_temporal −0.10) could not be reproduced from the committed CSV
> under any single consistent ranking convention; the values above are
> the `scipy.stats.spearmanr` output under the stated convention. The
> headline (auc best at +0.70, top1 next at +0.50, R² metrics weak) is
> unchanged.

Recompute with:

```python
import pandas as pd
from scipy.stats import spearmanr
df = pd.read_csv("../results/backbone_proxy_correlation.csv")
higher_better = {"auc", "top1", "r2_random", "r2_naive"}
for m in ["auc", "top1", "r2_random", "u_temporal", "u_batch", "r2_naive"]:
    rho = spearmanr(df[m], df["proxy_mase"]).statistic
    print(m, round(-rho if m in higher_better else rho, 2))
```
