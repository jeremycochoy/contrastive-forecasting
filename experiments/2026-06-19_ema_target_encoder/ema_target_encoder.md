# An EMA target on the positive keeps more dimensions alive, but transfer only improves on the early 2L cell

**Question.** A SimSiam/BYOL-style hard stop-grad on the encoder side of
the contrastive positive (`sim(stopgrad(h_{t+1}), f_{t+1})`) is already in
the strongest recipe in this line: enc3+CPC with `--stopgrad-positive-h`.
The hard stop-grad keeps the encoder from being pulled toward its own
forecast, and previous work in this line showed that change widens
batch-wise dimension usage (`U_batch ~4×`, [stop-grad report](
../2026-06-10_stopgrad_positive/stopgrad_positive.md)).
A *slowly-moving* teacher should preserve that property even more, and in
principle hold onto a richer, less-collapsed representation of the input.
We swap the hard stop-grad for a teacher: the GRU patch-embedding and the
3-layer encoder each get a non-trained EMA copy (τ = 0.99, constant); the
teacher's `h^T_{t+1}` becomes the positive; everything else stays on the
student (forecaster, negatives, CPC InfoNCE auxiliary `exp(e_{t+1}ᵀ W₁
h_t)` from van den Oord et al. 2018). Does the teacher preserve more
features of the input, and does that show up downstream?

**Result.** Yes on the first half, no on the second. The teacher keeps a
materially higher fraction of the embedding rank alive at end of training
(`U_batch` 0.82 vs 0.74, `U_temporal` 0.58 vs 0.40 — the temporal axis
gains 43%), and the positive alignment loosens (ff 0.50 vs 0.58, fp 0.04
vs −0.05, so the forecast tracks present and future about equally rather
than only future). Downstream this only helps one cell — the early 2L
checkpoint (GM 1.185 → 1.161, paired-bootstrap 90% interval below zero).
At the last checkpoint both heads are *reliably worse* (2L 1.153 → 1.182,
6L 1.144 → 1.160). Single seed; the bootstrap quantifies task-set noise,
not seed noise.

![Left: GM-Relative MASE per head × checkpoint, hard stop-grad (grey)
vs EMA target (blue). Right: the paired-bootstrap Δ with 90% interval —
green sits fully below zero (reliably better), red fully above (reliably
worse), grey straddles.](plots/gm_summary.png)

*GM-Relative MASE: the geometric mean over GIFT-Eval's 97 forecasting
tasks of a model's error divided by the seasonal-naive forecast's. Lower
is better; 1.0 is seasonal-naive.*

| backbone | checkpoint | 2-layer head | Δ (90% interval) | 6-layer head | Δ (90% interval) |
|---|---|--:|:--:|--:|:--:|
| hard stop-grad | best-loss | 1.185 | — | 1.158 | — |
| hard stop-grad | last | 1.153 | — | 1.144 | — |
| EMA target | best-loss | **1.161** | −0.023 (−0.040, −0.007) | 1.158 | −0.001 (−0.016, +0.016) |
| EMA target | last | 1.182 | +0.029 (+0.011, +0.045) | 1.160 | +0.016 (+0.004, +0.029) |

**Bold = reliably better than the same-cell baseline.** The early/late
split is a real reversal: the only cell the EMA target reliably *improves*
is the early one on the smaller head; both late-checkpoint cells go the
other way.

## By domain

![Per-domain GM-Relative MASE on GIFT-Eval full-97, two q-head depths
(panels). Grey = hard stop-grad, blue = EMA target; solid = best-loss,
dashed = last. Ring at 1.0 = seasonal-naive.](plots/perdomain_radar.png)

The 2L panel shows the early best-loss split: the blue solid curve sits
inside the grey solid one on most domains. By the last checkpoint (dashed)
the blue moves *outside* the grey on most domains — the same direction
flip the aggregate showed.

## Training dynamics: more rank stays in use, the contrastive task gets harder

![Training metrics, log-log (blue = hard stop-grad, green = EMA target);
loss/contrastive panels are not apples-to-apples across arms — under the
EMA target the positive cosine is measured against a moving teacher.
`loss_tau_ref` (a fixed-τ=0.07 normalized-InfoNCE diagnostic computed
identically student-side in both runs) and the CPC term *are* the
comparable curves.](plots/training_dynamics.png)

Three differences carry the story:

- **The teacher makes the contrastive task harder for the student.**
  `loss_tau_ref` sits above the baseline throughout training, and the
  CPC term's per-step median (steps ≥ 1k) is 0.0056 vs 0.0004 for the
  baseline — about 14× higher. The student's next-step prediction is a
  harder shot when the target keeps moving.
- **More dimensions stay in use.** `U_batch` ends 0.82 (vs 0.74) and
  `U_temporal` 0.58 (vs 0.40, +43%). The gain is bigger on the temporal
  axis — across-time rank is the one the hard stop-grad already left
  some room on, and the teacher fills more of it.
- **The positive alignment loosens.** ff = 0.50 vs 0.58, fp = +0.04 vs
  −0.05. Where the hard stop-grad's forecast tracks only the future, the
  teacher arm's forecast tracks present and future about equally — the
  representation is less specialised on the immediate-next-step direction.

`U_batch / U_temporal` are the fraction of embedding dimensions that vary
across batch / across time (higher = wider rank). `ff` is the
forecast-to-future cosine, `fp` the forecast-to-present.

*Hypothesis (consistent with the curves, not tested causally):* the
late-checkpoint regression is the cost of the looser positive. The teacher
keeps rank alive (good), but the student's positive direction is no
longer crisp enough by step 12,500 for the 6-layer head to read off a
clean next-step signal. The early 2L cell wins because the rank-preserving
effect is the dominant factor early on, before the alignment drift
matters.

## Protocol

One backbone per arm, single seed (20260520), 12,500 steps at batch 1024;
the first 10k steps ran single-GPU, then the run was resumed on two RTX
4090s (per-rank batch 512, global 1024 via the gathered-loss path — the
objective is identical to the single-GPU run, comm overhead aside). The
baseline arm is reused from `experiments/2026-06-13_cpc_infonce_aux/`
(enc3+CPC with `--stopgrad-positive-h`); this arm adds three flags:
`--ema-embedding --ema-encoder --ema-tau 0.99` and drops
`--stopgrad-positive-h` (the EMA teacher carries no autograd graph, so
the stop-grad on the positive is implicit). The teacher is a deep copy of
the student's patch-embedding and 3-layer encoder at step 0,
`requires_grad=False`, held in `eval()`; each step `θ_T ← τ·θ_T +
(1−τ)·θ_S` (half-life `ln(0.5)/ln(0.99)` ≈ 69 steps). Teacher parameters
ride in the checkpoint state_dict; head-training and eval strip
`teacher_*.*` before strict-loading.

Each scored cell freezes the backbone, then trains a fresh quantile
forecasting head on top — once with two transformer layers, once with
six — and evaluates on GIFT-Eval's 97 tasks at two checkpoints:
**best-loss** (the step with the lowest smoothed training loss) and
**last** (12,500). Analysis reuses the GM and paired-bootstrap code from
the stop-grad-positive and CPC reports unchanged.

## Follow-up

The hypothesis above predicts the issue's **arm 2** (`--moco-negatives`):
sending the main-contrastive negatives through the teacher too closes the
positive/negative-source mismatch and should sharpen the late-checkpoint
positive without giving up the rank gain. Worth running.
