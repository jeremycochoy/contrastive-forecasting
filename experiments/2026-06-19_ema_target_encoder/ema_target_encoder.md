# EMA-target encoder/embed on enc3+CPC: TBD

**Question.** The #344 enc3+CPC arm pairs the forecaster's context with the
next encoder embedding through a fixed-temperature cosine score, with the
encoder side of the positive pair stop-gradded (`--stopgrad-positive-h`).
We swap that hard stop-grad for a *slowly-moving* EMA target — the
patch-embedding and the 3-layer encoder each get a non-trained EMA copy
(τ = 0.99 constant), and the teacher's `h^T_{t+1}` replaces the student's
as the main-contrastive positive. The forecaster, the negatives, and the
CPC term all stay on the student. Does the EMA target transfer better than
the hard stop-grad it replaces, head-matched, on GIFT-Eval's 97 tasks?

**Answer.** TBD.

## Result

![Left: GM-Relative MASE, #344 enc3+CPC baseline (stop-grad positive) vs
EMA-target, per head × checkpoint. Right: EMA−baseline paired-bootstrap
Δ with 90% CI per cell (negative ⇒ EMA better).](plots/gm_summary.png)

GM-Relative MASE, all cells (bold = a reliable improvement over the same-cell
baseline):

| arm · setup | 2L best | 2L last | 6L best | 6L last |
|---|--:|--:|--:|--:|
| enc3+CPC baseline (--stopgrad-positive-h) | 1.185 | 1.153 | 1.158 | 1.144 |
| enc3+CPC EMA-target (--ema-embedding --ema-encoder) | TBD | TBD | TBD | TBD |

*GM-Relative MASE: the geometric mean, over GIFT-Eval's 97 tasks, of a model's
error divided by the seasonal-naive forecast's — lower is better, 1.0 is
seasonal-naive.*

## Training dynamics

![Training metrics, log-log (blue = #344 enc3+CPC stop-grad baseline,
green = EMA-target).](plots/training_dynamics.png)

## Protocol

One backbone, single seed, single RTX 4090. The EMA-target arm reuses the
#344 enc3+CPC recipe **exactly** — GRU patch-embedding, d_model 384 / 6
heads, 3-layer encoder, 6-layer full-width forecaster, the crossfade-triplet
allt·0.8% data mix, qk-norm, attention-output norm, the `xshh_allt`
contrastive loss (positive-in-denominator, floor subtraction), the CPC
InfoNCE auxiliary term at λ=1 with its learnable W₁, τ 0.10, batch 1024,
12,500 steps, seed 20260520 — with two flag swaps:

- **Drop** `--stopgrad-positive-h` (the teacher carries no autograd graph, so
  the stop-grad on the positive is implicit).
- **Add** `--ema-embedding --ema-encoder --ema-tau 0.99` (constant, no schedule).

The teacher path is built as deep copies of the student's patch-embedding
(GRU) and 3-layer transformer encoder at step 0, marked `requires_grad=False`,
and kept in `eval()` so dropout/dropkey never touch it. After every
`optimizer.step()`, teacher params are pulled toward the just-stepped
student: `θ_T ← τ·θ_T + (1−τ)·θ_S`. Teacher params are saved in the
checkpoint state_dict (clean resume); head-train and eval strip `teacher_*.*`
before strict-loading the backbone (no downstream role).

To score the backbone we freeze it, train a fresh quantile forecasting head
(once with two transformer layers, once with six), and evaluate on
GIFT-Eval's 97 tasks at two checkpoints: the **best-loss** one (lowest
smoothed contrastive loss, within roughly the first quarter-to-half of
training) and the **last** one (step 12,500). Baseline is the published
#344 enc3+CPC numbers, head-trained on the identical recipe.

## The change

EMA-target representation path (BYOL / JEPA pattern). Let `θ_S` denote the
student's patch-embedding + encoder parameters and `θ_T` the teacher's.
At step 0, `θ_T = θ_S`. Each step:

1. Forward the input through both the student (full path: embed → encoder
   → forecaster) and the teacher (representation only: embed → encoder).
2. The main-contrastive positive `sim(h_{t+1}, f_{t+1})` reads `h_{t+1}` from
   the teacher instead of the student. Negatives still read the student.
3. After `optimizer.step()`: `θ_T ← τ·θ_T + (1−τ)·θ_S` with τ = 0.99
   (half-life ≈ 69 steps).

**Forecaster is fully online — it is the predictor.** The CPC term is
unchanged from #344: `exp(e_{t+1}ᵀ W₁ h_t)` all-student, no stop-grad.
Negatives for the main-contrastive loss also stay on the student — that
mismatch is the next-arm follow-up (`--moco-negatives` in the issue), out
of scope here.

**Next:** TBD — outcome decides whether the moco-negatives follow-up (arm 2
in #353) is worth running.

## Annex — per-cell paired-bootstrap Δ (90% interval)

Δ = GM(EMA-target) − GM(#344 enc3+CPC baseline); negative ⇒ EMA beats the
baseline. The interval is a paired bootstrap over the 97 tasks (resample
with repeats, score both arms on each resample, so per-task difficulty
cancels) at a single training seed, as for the baselines — it reflects
spread across tasks, not across seeds.

| cell | EMA Δ (90% CI) |
|---|--:|
| 2L best | TBD |
| 2L last | TBD |
| 6L best | TBD |
| 6L last | TBD |
