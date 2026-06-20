# EMA target on the contrastive positive: helps the early 2L checkpoint, hurts both late checkpoints

**Question.** The enc3+CPC recipe trains a 3-layer encoder + 6-layer
forecaster with a contrastive loss `sim(h_{t+1}, f_{t+1})` whose encoder
side `h_{t+1}` is *detached* (a hard stop-grad on the target). Replace that
hard stop-grad with a slowly-moving teacher: the GRU patch-embedding and
the 3-layer encoder each get a non-trained EMA copy (τ = 0.99, constant);
the teacher's `h^T_{t+1}` becomes the contrastive positive; everything
else — the forecaster, the negatives, and the CPC InfoNCE auxiliary term
(`exp(e_{t+1}ᵀ W₁ h_t)`, van den Oord et al. 2018) — stays on the student.
Does that improve downstream forecasting on GIFT-Eval's 97 tasks?

**Answer.** Mixed. At the early ("best-loss") checkpoint, the EMA target
reliably helps the 2L head; the 6L head is unchanged. At the last
checkpoint, both heads are reliably worse. Single seed; uncertainty is
task-spread via paired bootstrap over the 97 tasks, not seed-spread.

## Result

![Left: GM-Relative MASE per head × checkpoint. Right: paired-bootstrap Δ
with 90% CI (negative ⇒ EMA-target better; green = reliably better, red =
reliably worse, grey = ns).](plots/gm_summary.png)

| recipe | 2L best | 2L last | 6L best | 6L last |
|---|--:|--:|--:|--:|
| stop-grad on positive | 1.185 | 1.153 | 1.158 | 1.144 |
| EMA-target on positive | **1.161** | **1.182** | 1.158 | **1.160** |

*GM-Relative MASE: geometric mean over the 97 tasks of (model error) /
(seasonal-naive error); lower is better, 1.0 is seasonal-naive.*

![Per-domain GM relative MASE, two q-head depths. Grey = stop-grad
baseline, blue = EMA-target; solid = best-loss, dashed = last; ring at
1.0 = seasonal-naive.](plots/perdomain_radar.png)

## Training dynamics

![Training metrics, log-log (blue = stop-grad baseline, green =
EMA-target). The total and "contrastive only" panels are not directly
comparable across arms — under EMA the positive cosine is measured against
a moving teacher, so the value reflects a different objective. `loss_tau_ref`
(a fixed-τ=0.07 normalized-InfoNCE diagnostic computed identically
student-side in both runs) and the CPC term are the apples-to-apples
curves.](plots/training_dynamics.png)

The EMA arm's `loss_tau_ref` sits above the baseline throughout training,
and its CPC term's per-step median (steps ≥ 1k) is 0.0056 vs 0.0004 for
the baseline — about 14× higher. The "best-loss" criterion catches the
EMA arm at step 1,100 vs the baseline's step 3,800.

## Method and recipe

The recipe is exactly the enc3+CPC stop-grad-positive arm (reused as the
baseline column above): GRU patch-embedding → 3-layer transformer encoder →
6-layer forecaster, d_model 384, the same data mix and contrastive loss
shape, the CPC auxiliary at λ=1 with its learnable bilinear `W₁`,
τ = 0.10, batch 1024, 12,500 steps, seed 20260520. Two flag swaps make
this arm: drop `--stopgrad-positive-h`, add `--ema-embedding --ema-encoder
--ema-tau 0.99`.

The EMA path is a deep copy of the student's patch-embedding and encoder
at step 0, marked `requires_grad=False`, held in `eval()` so dropkey and
dropout do not affect the target. Each step the teacher pulls toward the
student: `θ_T ← τ·θ_T + (1−τ)·θ_S` (half-life `ln(0.5)/ln(0.99)` ≈ 69
steps). At the loss, the main-contrastive positive's encoder side comes
from the teacher; negatives keep the student. The forecaster is fully
online — it is the predictor — and the CPC term is unchanged from the
baseline (all-student, no stop-grad). Teacher parameters live in the
checkpoint state_dict; head-training and eval strip `teacher_*.*` before
strict-loading the backbone.

Each scored cell freezes the backbone, then trains a fresh quantile
forecasting head — once with two transformer layers, once with six — and
evaluates on GIFT-Eval's 97 tasks at the **best-loss** checkpoint and the
**last** one (step 12,500). The baseline cells are reproduced from
`experiments/2026-06-13_cpc_infonce_aux/` (the enc3+CPC arm of
`cpc_infonce_aux.md`); analysis reuses that report's GM and
paired-bootstrap code unchanged.

**Next:** the issue's arm 2 — `--moco-negatives` — sends the
main-contrastive negatives through the teacher path too, closing the
positive/negative-source mismatch left open here.

## Annex — paired-bootstrap Δ (90% interval)

Δ = GM(EMA-target) − GM(stop-grad baseline); negative ⇒ EMA-target better.
Paired bootstrap over the 97 tasks at a single training seed; spread is
across tasks, not seeds.

| cell | Δ (90% CI) | verdict |
|---|--:|---|
| 2L best | **−0.023 (−0.040, −0.007)** | reliably better |
| 2L last | **+0.029 (+0.011, +0.045)** | reliably worse |
| 6L best | −0.001 (−0.016, +0.016) | ns |
| 6L last | **+0.016 (+0.004, +0.029)** | reliably worse |
