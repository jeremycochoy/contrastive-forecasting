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

The split is broad. At best-loss the EMA arm sits inside the baseline on
most domains; by the last checkpoint it has moved outside on most.

## Training dynamics

![Log-log training metrics, blue = hard stop-grad, green = EMA target. The
total/contrastive panels are not apples-to-apples — the EMA arm's positive
is measured against a moving target. `loss_tau_ref` (a fixed-τ=0.07
normalized-InfoNCE diagnostic, student-side, identical in both runs) and
the CPC term are the comparable curves.](plots/training_dynamics.png)

- **The teacher makes the contrastive task harder for the student.**
  `loss_tau_ref` stays above the baseline throughout, and the CPC term is
  about an order of magnitude higher — the next-step shot is harder when
  the target keeps drifting.
- **More dimensions stay in use**, and the gain is bigger on the temporal
  axis the hard stop-grad had left room on.
- **The positive alignment loosens.** The forecast tracks present and
  future about equally rather than only future.

*Hypothesis (consistent with the curves, untested causally):* the
late-checkpoint regression is the cost of the looser positive — rank gain
is the dominant factor early on, but by 12,500 steps the student's
next-step direction is no longer crisp enough for the 6-layer head to
read off a clean signal.

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

If the hypothesis is right, the issue's **arm 2** (`--moco-negatives`,
sending the main-contrastive *negatives* through the teacher too) sharpens
the late-checkpoint positive without giving up the rank gain.
