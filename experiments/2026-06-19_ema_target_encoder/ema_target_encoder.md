# An EMA target keeps more rank alive but only the early 2L cell transfers better

**Question.** Replacing the contrastive positive's hard stop-grad
(`sim(stopgrad(h_{t+1}), f_{t+1})`) with a *slowly-moving* teacher should,
in principle, hold onto a richer representation of the input — the prior
hard stop-grad already widened batch-wise dimension usage 4× ([report](
../2026-06-10_stopgrad_positive/stopgrad_positive.md)); a teacher should
preserve more. Does it, and does that show up downstream?

We swap the stop-grad for an EMA teacher: the GRU patch-embedding and the
3-layer encoder each get a non-trained EMA copy (τ = 0.99, constant); the
teacher's `h^T_{t+1}` becomes the positive. Forecaster, negatives, and the
CPC InfoNCE auxiliary (`exp(e_{t+1}ᵀ W₁ h_t)`, van den Oord et al. 2018)
stay on the student.

**Result.** Yes on rank, mixed on transfer. The teacher ends training
holding noticeably more of the embedding rank alive — on the temporal
axis especially. Downstream, though, only the early 2L checkpoint
improves reliably; the 6L best is unchanged, and both *late* checkpoints
are reliably worse.

![Left: GM-Relative MASE per head × checkpoint, hard stop-grad (grey) vs
EMA target (blue). Right: paired-bootstrap Δ with 90% interval — green
below zero is reliably better, red above is reliably worse, grey
straddles.](plots/gm_summary.png)

*GM-Relative MASE: geometric mean over GIFT-Eval's 97 tasks of model error
divided by seasonal-naive error. Lower is better; 1.0 is seasonal-naive.*

| backbone | checkpoint | 2-layer head | Δ (90% interval) | 6-layer head | Δ (90% interval) |
|---|---|--:|:--:|--:|:--:|
| hard stop-grad | best-loss | 1.185 | — | 1.158 | — |
| hard stop-grad | last | 1.153 | — | 1.144 | — |
| EMA target | best-loss | **1.161** | −0.023 (−0.040, −0.007) | 1.158 | −0.001 (−0.016, +0.016) |
| EMA target | last | 1.182 | +0.029 (+0.011, +0.045) | 1.160 | +0.016 (+0.004, +0.029) |

![Per-domain GM-Relative MASE, two q-head depths. Grey = hard stop-grad,
blue = EMA target; solid = best-loss, dashed = last. Ring at 1.0 =
seasonal-naive.](plots/perdomain_radar.png)


## Training dynamics

![Log-log training metrics, blue = hard stop-grad, green = EMA target. The
total/contrastive panels are not apples-to-apples — the EMA arm's positive
is measured against a moving target. `loss_tau_ref` (a fixed-τ=0.07
normalized-InfoNCE diagnostic, student-side, identical in both runs) and
the CPC term are the comparable curves.](plots/training_dynamics.png)

- **`loss_tau_ref` stays above the baseline throughout, and the CPC term
  is about an order of magnitude higher.** Both diagnostics are computed
  identically student-side in both runs, so they measure the same student
  output under the same scoring rule.
- **`U_batch` and `U_temporal` end higher than the baseline.** The
  temporal axis gains more, and the baseline's `U_temporal` was the lower
  of the two to begin with.
- **`ff` is lower and `fp` is higher than the baseline** — the forecast
  is less aligned with the future and more aligned with the present.

## Protocol

One backbone per arm, single seed (20260520), 12,500 steps at batch 1024;
first 10k single-GPU, then resumed on two RTX 4090s (per-rank 512, global
1024 via gathered-loss — same objective as single-GPU). Baseline =
`experiments/2026-06-13_cpc_infonce_aux/` (enc3+CPC, stop-grad on the
positive). This arm drops `--stopgrad-positive-h` and adds `--ema-embedding
--ema-encoder --ema-tau 0.99`; nothing else changes.

The teacher is a deep copy of the student's patch-embedding and 3-layer
encoder at step 0, `requires_grad=False`, held in `eval()`. Each step:
`θ_T ← τ·θ_T + (1−τ)·θ_S` (half-life ≈ 69 steps). Teacher parameters ride
in the checkpoint state_dict; head-training and eval strip `teacher_*.*`
before strict-loading.

Each scored cell freezes the backbone, trains a fresh quantile head (2L
and 6L), and evaluates on GIFT-Eval's 97 tasks at **best-loss** and at
**last** (step 12,500). The bootstrap intervals quantify task-set noise,
not seed noise.

## Follow-up

Issue **arm 2** (`--moco-negatives`) sends the main-contrastive negatives
through the teacher too, so the positive and the negatives share one
slowly-moving space.
