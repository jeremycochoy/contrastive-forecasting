# Is the forecast↔encoder crossed negative (fₜ↔hₗ) the cause of the bottleneck-fullfh gap, or do its encoder↔encoder / forecaster↔forecaster siblings behave the same?

The bottleneck-fullfh loss-of-record adds one extra contrastive negative
— each forecast fₜ pushed away from the encoder latent hₗ at *every*
time l (the **fₜ↔hₗ** term; arm **A**). This ablation swaps it for the
two structurally identical siblings — encoder↔encoder **hₜ↔hₗ** (arm
**B**), forecaster↔forecaster **fₜ↔fₗ** (arm **C**) — and the union A+B.

**Verdict.** The siblings do **not** behave like A. On the trusted
full-97 GIFT-Eval metric, swapping fₜ↔hₗ for hₜ↔hₗ (B) lowers GM-MASE
**1.4377 → 1.3572 (−5.6%)**, B beating A in **all 7 domains**; fₜ↔fₗ (C)
gives −3.9%. Carrying both (A+B) lands at 1.4517 ≈ A — re-adding fₜ↔hₗ
on top of B erases B's gain. These three single-seed, internally
consistent points **point to fₜ↔hₗ as the harmful term** (single seed;
a per-arm CI would need multiple seeds). Not the whole gap: best (B,
1.357) is still ≈5% above v11c (1.292) and far above the same backbone
the #10 line uses (1.183 with a simple 30k head) — the deficit is
largely the bottleneck-fullfh backbone, not this loss term.

![Per-domain relative MASE — A, B, C, A+B](plots/perdomain_star.png)
*Radar chart, one spoke per GIFT-Eval domain; distance from centre = GM
relative MASE, lower = better; dotted green = seasonal naive (1.0).
(B) (blue) sits inside (A) (grey) on all seven domains; the geomean is
dominated by **Econ/Fin** (A 3.40 → B 2.79), the same single-domain
driver #300 flagged; only **Sales** (B 0.855) and **Nature** (B 0.951)
beat seasonal naive. The **gold dashed** ring is the project's best-ever
GIFT-Eval (R9_E13, #127 — a heavier 12L q-head on a different recipe,
full GM 1.029): it sits well inside every #303 arm on every domain —
the achievable frontier this loss line is still far from.*

**Held-out GM-Relative MASE** (official GIFT-Eval; standard 2L causal
q-head trained 30k on each backbone; 1.0 = seasonal naive, lower better):

| loss arm | crossed negative | triage (11) | **full (97)** | vs (A) |
|---|---|---|---|---|
| **(A)** `full_fh_negs` *(of record, #296)* | fₜ↔hₗ ∀ l≠t+1 | 1.5611 | **1.4377** | — |
| **(B)** `full_hh_negs` | hₜ↔hₗ ∀ l≠t | 1.4461 | **1.3572** | **−5.6 %** |
| **(C)** `full_ff_negs` | fₜ↔fₗ ∀ l≠t | 1.5185 | **1.3822** | −3.9 % |
| **(A)+(B)** `full_fh_hh_negs` | both | 1.5426 | **1.4517** | +1.0 % |
| *ref:* v11c (bneck-fullfh prior best) / seasonal-naive | — | — | 1.292 / 1.000 | — |
| *ref:* #10 backbone (167k) + simple GRU q-head 30k | — | — | **1.183** | — |
| *ref:* #10 backbone (167k) + xfmr-q 12L 60k = R9_E13 (#127)† | — | 0.990† | **1.029** | — |

†Both refs share the **same #10 backbone** (167k MOIRAI-HP, a
*different* line from the bottleneck-fullfh arms — not same-recipe);
they differ only in the q-head (simple 30k GRU → 1.183; heavier 12-layer
transformer 60k, e_then_f → R9_E13 1.029, the project best-ever).
R9_E13's triage 0.990 (apparently < 1.0) was optimistic vs the trusted
full-97 1.029 (its own TRIAGE_NOTE); neither beats seasonal naive in
aggregate, but both are far below every bottleneck-fullfh arm.

Single seed per arm (matched to (A) for a clean one-variable contrast);
triage (11-cfg) is ≈7–10 % noisy (#296), full-97 is the trusted metric.
The four full-97 deltas exceed the ≈3 % checkpoint spread #296 treated as
within-noise, and the sign pattern repeats on triage (B < C < A,A+B) and
in the per-domain sweep (B beats A in 7/7) — but a confidence interval
per arm would need multiple seeds (not run).

![Training curves (log–log)](plots/training_curves.png)
*Contrastive loss, latent dimension usage (u_temporal / u_batch), the
fixed-τ `loss_tau_ref` diagnostic, and 1−AUC, all log–log, four arms
overlaid (thick = rolling-median smoothed, raw faint behind —
`loss_tau_ref`/1−AUC are step-to-step spiky). (A), (B), (C) are
near-identical (final loss 2.18 / 2.18 /
2.15, `loss_tau_ref` 0.20–0.21, u_temporal ≈0.20, u_batch ≈0.16–0.17,
1−AUC ≈6e-8 — positives fully separated from negatives); (A)+(B) sits
~5% higher in contrastive loss (2.29, `loss_tau_ref` 0.23, u_temporal
0.22), as expected from its strictly larger negative set. No arm
diverged (post-warmup max loss ≤3.5). Crucially the contrastive-proxy
ranking does **not** track GM-MASE — (A) and (B) reach near-identical
proxies yet differ 5.6% in transfer — so the GM-MASE separation is not
explained by training-objective fit, echoing #296 ("training the
objective harder did not transfer").*

## Question

The bottleneck-fullfh gap (full 1.438 vs v11c 1.292, seasonal-naive
1.0) is confounded across loss / bottleneck / dropkey / precision (#296);
#300 ruled out q-head undertraining. This isolates the **loss** axis:
replace the of-record fₜ↔hₗ all-time crossed negative with each
structurally identical sibling and measure held-out GIFT-Eval.
*GM-Relative MASE* = geomean over configs of model MASE ÷ seasonal-naive
MASE (1.0 = seasonal naive, lower better).

## Protocol

Each arm is `cosine_similarity_batch` with its single l=t
forecaster–encoder negative **replaced by** the all-time crossed term
(A+B carries both) — the identical transform that defines A; logsumexp,
`--pos-in-denominator`. **Only `--loss-shape` changes** from A's
backbone-of-record (#296 run `a1`: 1L forecaster, fp16 group, 2-GPU DDP
global-batch 256, 50k, seed 20260517) — `run_ddp.sh`'s literal 2L/bf16
form diverged (#296 RESULTS.md), so the controlled test starts from the
1L/fp16 run #296/#300 evaluate. Per backbone: 30k 2L-causal q-head +
official GIFT-Eval (triage 11, full 97). Commands:
[`scripts/run_all.sh`](scripts/run_all.sh). Tests: 6 closed-form/mask
pins (`TestCrossedLossSiblings`) + 113-test loss suite green; baseline A
independently re-audited positive-free
([`scripts/verify_A_positive_exclusion.py`](scripts/verify_A_positive_exclusion.py),
all PASS). Wall-clock in the back annex.

![Contrastive loss structure over time & batch](plots/loss_diagram.png)
*The contrastive structure on a (time × batch) ladder — horizontal =
time, vertical = batch (two batch lines b, b′; each carries an f and an
h node per step). **Arrows: ▶◀ inward = positive (attract), ◀▶ outward
= negative (repel).** Green = every positive f_{·,τ}→h_{·,τ+1} on both
ladders. The loss-of-record's **(A) fₜ↔hₗ** (red, l≠t+1) is fanned in
full from one anchor f_{b,t} on the top ladder; the swapped siblings
**(B) hₜ↔hₗ** (blue, l≠t) and **(C) fₜ↔fₗ** (orange, l≠t) from one
anchor each on the bottom ladder; purple = a few cross-batch negatives
(b≠b′) — the batch axis, shared by every arm. Each crossed family is
fanned from one anchor for clarity (the loss sums over all anchors).
Single channel, so same-time cross-channel negatives don't exist here.*

## What we learned

1. **Contrastive-objective fit does not predict transfer.** A/B/C reach
   near-identical contrastive proxies (loss, `loss_tau_ref`, 1−AUC,
   dim-usage) yet differ up to 5.6% in GM-MASE; A+B has the highest
   contrastive loss (~5% above, from its larger negative set) and among
   the worst transfer. No arm diverged. The separation is held-out
   transfer, not training fit (echoes #296).
2. **fₜ↔hₗ is the harmful loss term, but the backbone — not the loss —
   is the dominant gap.** B beats A in all 7 domains and A+B collapsing
   back to ≈A pins fₜ↔hₗ as the harmful term. Yet the **same #10
   backbone** (167k, MOIRAI-HP — a different line) reaches **1.183 with a
   simple 30k GRU q-head** and **1.029 with a 12L 60k head** (radar
   gold), both far below every bottleneck-fullfh arm (best B 1.357):
   even a simple short head on a strong backbone beats all four. So the
   crossed-loss choice is a small lever inside a backbone line that is
   itself well off the project frontier.

---

### Annex — loss-arm definitions

All four are `cosine_similarity_batch` (paper loss: cross-time +
cross-channel + cross-batch negatives, logsumexp, normalized InfoNCE)
with the single l=t forecaster–encoder negative `log_neg_xy_hat`
(= cos(hₜ, fₜ)) **replaced** by an all-time crossed term, same (b, c),
masking only the degenerate pair:

| arm | `loss_shape` | replacement negative term | masked l |
|---|---|---|---|
| (A) | `cosine_similarity_batch_full_fh_negs` | cos(fₜ, hₗ) ∀ l | l = t+1 (the positive) |
| (B) | `cosine_similarity_batch_full_hh_negs` | cos(hₜ, hₗ) ∀ l | l = t (self, cos≡1) |
| (C) | `cosine_similarity_batch_full_ff_negs` | cos(fₜ, fₗ) ∀ l | l = t (self, cos≡1) |
| (A)+(B) | `cosine_similarity_batch_full_fh_hh_negs` | both (A) and (B) terms | resp. t+1 / t |

xy (cos(hₜ,h_{t+1}) cross-channel), xx (cross-channel same-time), zy
(f_{t+1}↔fₜ cross-channel), and the cross-batch term are byte-for-byte
identical to `cosine_similarity_batch` across all arms. Distinct from
`cosine_similarity_batch_square`'s batch-crossed h×h / f×f (b≠b', fixed
t): these siblings are **time-crossed** (l≠t, same b, c).

### Annex — per-domain full GM relative MASE

| domain | (A) | (B) | (C) | (A)+(B) | best |
|---|---|---|---|---|---|
| Econ/Fin | 3.397 | **2.788** | 3.488 | 3.233 | B |
| Energy | 1.646 | **1.577** | 1.589 | 1.680 | B |
| Healthcare | 1.630 | 1.576 | 1.624 | **1.507** | A+B |
| Nature | 1.010 | **0.951** | 0.958 | 1.029 | B |
| Sales | 0.898 | **0.855** | 0.873 | 0.929 | B |
| Transport | 1.100 | **1.061** | 1.086 | 1.108 | B |
| Web/CloudOps | 1.518 | 1.427 | **1.392** | 1.552 | C |
| **full GM (97)** | 1.4377 | **1.3572** | 1.3822 | 1.4517 | B |

### Annex — wall-clock breakdown

Backbone = 50k DDP (2× RTX 4090). Downstream = 30k q-head + GIFT-Eval
triage (11) + full (97), single GPU. B/C/A+B: idle box, back-to-back,
2026-05-19 (controlled). (A): #296 session, GPUs shared (reference only).

| arm | backbone 50k | q-head 30k | triage (11) | full (97) | downstream Σ |
|---|---|---|---|---|---|
| (B) `full_hh_negs` | 1 h 59 m | 1 h 10 m | 3 m | 1 h 36 m | 2 h 49 m |
| (C) `full_ff_negs` | 1 h 57 m | 1 h 10 m | 3 m | 1 h 34 m | 2 h 47 m |
| (A)+(B) `full_fh_hh_negs` | 1 h 59 m | 1 h 16 m | 3 m | 1 h 32 m | 2 h 51 m |
| (A) `full_fh_negs` *(#296, shared)* | ≈3 h 29 m | — | — | — | ≈2 h 47 m |

Phase 1 (3 backbones, serial — DDP holds both GPUs): 5 h 55 m.
Phase 2 (downstream — B+C concurrent on GPU0/GPU1, then A+B alone):
5 h 39 m. **End-to-end (3 new arms, code→eval): 11 h 34 m.** Backbone
time is loss-variant-independent (B/C/A+B all ≈1 h 58 m — the extra term
costs no measurable time); (A)'s longer ≈3 h 29 m is its shared #296
session, not the loss, so it is not a controlled timing point.

### Annex — reference models (same #10 backbone, two q-heads)

Backbone *"beta"* (`tiny_full4096_moirai_hp_FRESH_RESUME50k`, the #10
RESUME50k run): **GRU patch embedding**, **6** transformer layers,
**latent dim 384**, 6 heads, C=1, T_RAW=4096, EWMA-RevIN(span 128);
**167,000** backbone steps (fresh-init to 50k, then deterministic-resume
to 167k — the `RESUME50k` prefix is the resume-checkpoint name, *not*
the step count). Both gold references use this exact backbone, differing
only in the q-head:

- **simple head — full GM 1.183** (#10): legacy GRU quantile head,
  **30k** steps, lr 3e-4, forecast-len 16.
- **R9_E13 — full GM 1.029** (#127, radar gold): **12-layer**
  causal-transformer quantile head, width **384** (= backbone) / 6
  heads / FFN ×4 (≈1536), dropout 0.1, `e_then_f` input, forecast-len
  16; **60k** steps.

(Vs the #303 arms' 6-layer-encoder / 1-layer-d128-forecaster backbone +
30k 2-layer q-head — hence "different recipe / line".)
