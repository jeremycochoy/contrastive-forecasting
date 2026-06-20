# EMA-target encoder/embed on enc3+CPC: helps the early checkpoint, hurts the late one

**Question.** Our prior enc3+CPC arm trains a 3-layer encoder + 6-layer
forecaster with a contrastive loss `sim(h_{t+1}, f_{t+1})` whose
encoder-side positive `h_{t+1}` is detached (a SimSiam/BYOL-style hard
stop-grad on the target — the `--stopgrad-positive-h` flag), plus a CPC
(Contrastive Predictive Coding, van den Oord et al. 2018) auxiliary that
predicts the next encoder embedding `e_{t+1}` from the AR context `h_t`
through a learnable bilinear. We swap that hard stop-grad for a
*slowly-moving* EMA target — the GRU patch-embedding and the 3-layer
encoder each get a non-trained EMA copy (τ = 0.99 constant), and the
teacher's `h^T_{t+1}` replaces the student's as the main-contrastive
positive. The forecaster, the negatives, and the CPC term all stay on the
student. Does the EMA target transfer better than the hard stop-grad it
replaces, head-matched, on GIFT-Eval's 97 tasks?

**Answer.** Single-seed verdict (paired-bootstrap over the 97 tasks, single
backbone per arm — uncertainty is task-spread, *not* seed-spread): the
early ("best-loss") checkpoint improves with the 2L head (GM 1.185 → 1.161)
and is tied with the 6L head; the late ("last") checkpoint moves the other
way on both head sizes (2L: 1.153 → 1.182; 6L: 1.144 → 1.160). Net, the
EMA-target arm does not strictly dominate `--stopgrad-positive-h` at this
seed.

## Result

![GM-Relative MASE (geometric mean over GIFT-Eval's 97 tasks) per head ×
checkpoint, single seed, single backbone per arm. Left: stop-grad baseline
(grey) vs EMA-target (blue), 2L/6L q-head × best/last checkpoint. Right:
paired-bootstrap Δ = GM(EMA) − GM(baseline) over the 97 tasks with 90% CI
(negative ⇒ EMA better; green = reliably better, red = reliably worse, grey
= ns).](plots/gm_summary.png)

GM-Relative MASE, all cells (bold = reliable change vs the same-cell
baseline):

| arm · setup | 2L best | 2L last | 6L best | 6L last |
|---|--:|--:|--:|--:|
| enc3+CPC baseline (`--stopgrad-positive-h`) | 1.185 | 1.153 | 1.158 | 1.144 |
| enc3+CPC EMA-target (`--ema-embedding --ema-encoder`) | **1.161** | **1.182** | 1.158 | **1.160** |

*GM-Relative MASE: the geometric mean, over GIFT-Eval's 97 tasks, of a
model's error divided by the seasonal-naive forecast's — lower is better,
1.0 is seasonal-naive.*

## Training dynamics

![Training metrics, log-log (blue = baseline `--stopgrad-positive-h`,
green = EMA-target). The total and "contrastive only" panels are not
apples-to-apples across the two arms — the EMA arm's positive cosine is
measured against a moving teacher target, so the value reflects a different
objective. The `loss_tau_ref` panel (a fixed-τ=0.07 normalized-InfoNCE
diagnostic computed identically student-side in both runs) and the CPC term
panel are the comparable curves.](plots/training_dynamics.png)

`loss_tau_ref` is the same contrastive loss re-evaluated under no-grad
at a fixed canonical τ = 0.07 in normalized-InfoNCE form (positive in both
numerator and denominator) — a constant-temperature reference comparable
across runs. On that panel the EMA arm sits above the baseline throughout
training. The CPC term's per-step median (steps ≥ 1k) is 0.0056 for the EMA
arm vs 0.0004 for the baseline — about 14× higher. The EMA arm's
"best-loss" criterion lands at step 1,100, the baseline's at step 3,800.

## Protocol

One backbone, single seed, on two RTX 4090s. The EMA-target arm reuses the
prior enc3+CPC stop-grad recipe **exactly** — same backbone (GRU
patch-embedding → 3-layer encoder → 6-layer forecaster, d_model 384), same
data mix, same contrastive loss shape and CPC auxiliary, same
τ = 0.10 / batch 1024 / 12,500 steps / seed 20260520. Two flag swaps:

- **Drop** `--stopgrad-positive-h` — the teacher carries no autograd graph,
  so the stop-grad on the positive is implicit.
- **Add** `--ema-embedding --ema-encoder --ema-tau 0.99` (constant, no
  schedule).

To score the backbone we freeze it, train a fresh quantile forecasting head
(once with two transformer layers, once with six), and evaluate on
GIFT-Eval's 97 tasks at two checkpoints: the **best-loss** one (lowest
smoothed contrastive loss) and the **last** one (step 12,500). Baseline is
the previously published enc3+CPC stop-grad-positive numbers, head-trained
on the identical recipe; this report's analysis reuses that report's GM and
paired-bootstrap code unchanged.

## The change

EMA-target representation path (BYOL / JEPA pattern). Let `θ_S` denote the
student's patch-embedding + encoder parameters and `θ_T` the teacher's. At
step 0, `θ_T = θ_S`. Each step:

1. Forward the input through both the student (full path: embed → encoder →
   forecaster) and the teacher (representation only: embed → encoder).
2. The main-contrastive positive `sim(h_{t+1}, f_{t+1})` reads `h_{t+1}`
   from the teacher instead of the student. Negatives still read the
   student.
3. After `optimizer.step()`: `θ_T ← τ·θ_T + (1−τ)·θ_S` with τ = 0.99
   (half-life ≈ 69 steps).

The teacher modules are deep copies at step 0, `requires_grad=False`, kept
in `eval()` so dropkey/dropout never touch them. Teacher params are saved
in the checkpoint state_dict (clean resume); head-train and eval strip
`teacher_*.*` before strict-loading the backbone — they have no downstream
role.

**Forecaster is fully online — it is the predictor.** The CPC term is
unchanged from the baseline: `exp(e_{t+1}ᵀ W₁ h_t)` all-student, no
stop-grad. Negatives for the main-contrastive loss also stay on the
student — that mismatch is a follow-up arm (`--moco-negatives`, drawing
main-contrastive negatives from the teacher too), out of scope here.

## What we learned

The EMA-target swap does not behave like an auxiliary regulariser that
quietly improves transfer across all checkpoints, the way the CPC InfoNCE
auxiliary did. Instead, the head-matched comparison cell-by-cell points in
opposite directions:

- The EMA target *helps* downstream transfer when the backbone is scored
  early, while the teacher is still close to the student's initialisation.
- The EMA target *hurts* downstream transfer at the last checkpoint — both
  2L and 6L heads.

Across the run the CPC term is ~14× larger than under the baseline, and
the τ=0.07 student-side reference loss is also higher — both facts say
`e_{t+1}` is harder to predict from `h_t` than under the baseline.
Connecting that to the late-checkpoint drop is a *hypothesis*: the
representation that the EMA-positive optimises seems to support less
next-step structure in the encoder output, and the gap widens with
training.

**Next:** the result rules out a straight one-arm replacement of
`--stopgrad-positive-h` by the EMA target on enc3+CPC at this seed; arm 2
of the issue (`--moco-negatives`, drawing the main-contrastive negatives
from the teacher as well) is the natural follow-up.

## Annex — per-cell paired-bootstrap Δ (90% interval)

Δ = GM(EMA-target) − GM(stop-grad-positive baseline); negative ⇒ EMA beats
the baseline. The interval is a paired bootstrap over the 97 tasks
(resample with repeats, score both arms on each resample, so per-task
difficulty cancels) at a single training seed, as for the baselines — it
reflects spread across tasks, not across seeds.

| cell | EMA Δ (90% CI) | verdict |
|---|--:|---|
| 2L best | **−0.023 (−0.040, −0.007)** | EMA reliably better |
| 2L last | **+0.029 (+0.011, +0.045)** | EMA reliably worse |
| 6L best | −0.001 (−0.016, +0.016) | ns (CI straddles 0) |
| 6L last | **+0.016 (+0.004, +0.029)** | EMA reliably worse |
