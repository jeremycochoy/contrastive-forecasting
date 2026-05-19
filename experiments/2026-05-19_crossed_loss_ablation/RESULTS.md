# Is (A)'s crossed fₜ↔hₗ negative the cause of the bottleneck-fullfh gap, or do its sibling crossed terms behave the same?

**Verdict.** The siblings do **not** behave like (A). On the trusted
full-97 GIFT-Eval metric, swapping (A)'s fₜ↔hₗ all-time crossed negative
for the encoder–encoder sibling (B) hₜ↔hₗ lowers GM-MASE
**1.4377 → 1.3572 (−5.6%)** — and (B) beats (A) in **every one of the 7
domains**; the forecaster–forecaster sibling (C) fₜ↔fₗ gives −3.9%.
Carrying **both** terms, (A)+(B), lands at 1.4517 ≈ (A) — adding (A)'s
fₜ↔hₗ term back on top of (B) erases essentially all of (B)'s gain.
These three single-seed, internally-consistent points **point to fₜ↔hₗ
as the harmful loss component** (a per-arm confidence interval would need
multiple seeds — single seed here; see Follow-up). It is not the whole
gap: best (B, 1.357) is still ≈5% above v11c (1.292) and beats seasonal
naive (1.0) in only 2 of 7 domains, so the remainder stays in #296's
bottleneck/dropkey/precision confound.

![Per-domain relative MASE — A, B, C, A+B](plots/perdomain_star.png)
*Per GIFT-Eval domain; distance from centre = GM relative MASE, dashed
hexagon = seasonal naive (1.0), lower = better. (B) (blue) sits inside
(A) (grey) on all seven domains; the geomean is dominated by **Econ/Fin**
(A 3.40 → B 2.79), the same single-domain driver #300 flagged. Only
**Sales** (B 0.855) and **Nature** (B 0.951) beat seasonal naive.*

**Held-out GM-Relative MASE** (official GIFT-Eval; standard 2L causal
q-head trained 30k on each backbone; 1.0 = seasonal naive, lower better):

| loss arm | crossed negative | triage (11) | **full (97)** | vs (A) |
|---|---|---|---|---|
| **(A)** `full_fh_negs` *(of record, #296)* | fₜ↔hₗ ∀ l≠t+1 | 1.5611 | **1.4377** | — |
| **(B)** `full_hh_negs` | hₜ↔hₗ ∀ l≠t | 1.4461 | **1.3572** | **−5.6 %** |
| **(C)** `full_ff_negs` | fₜ↔fₗ ∀ l≠t | 1.5185 | **1.3822** | −3.9 % |
| **(A)+(B)** `full_fh_hh_negs` | both | 1.5426 | **1.4517** | +1.0 % |
| v11c (prior best) / seasonal-naive | — | — | 1.292 / 1.000 | — |

Single seed per arm (matched to (A) for a clean one-variable contrast);
triage (11-cfg) is ≈7–10 % noisy (#296), full-97 is the trusted metric.
The four full-97 deltas exceed the ≈3 % checkpoint spread #296 treated as
within-noise, and the sign pattern repeats on triage (B < C < A,A+B) and
in the per-domain sweep (B beats A in 7/7) — but a confidence interval
per arm would need multiple seeds (not run; see Follow-up).

![Training curves (log–log)](plots/training_curves.png)
*Contrastive loss, latent dimension usage (u_temporal / u_batch), the
fixed-τ `loss_tau_ref` diagnostic, and 1−AUC, all log–log, four arms
overlaid. (A), (B), (C) are near-identical (final loss 2.18 / 2.18 /
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

#296 (`2026-05-17_bottleneck_fullfh_ddp`) and #300 left the
bottleneck-fullfh gap (full 1.438 vs v11c 1.292, seasonal-naive 1.0)
confounded across the loss / bottleneck / dropkey / precision axes; #300
ruled out q-head undertraining as the main cause. (A) — the
recipe-of-record — adds, on top of the `cosine_similarity_batch` loss,
**fₜ↔hₗ negatives**: every forecast fₜ contrasted against the encoder
latent hₗ at *every* time position l (≠ the positive target t+1), same
(batch, channel). This isolates the **loss** axis: do the structurally
identical *sibling* crossed terms behave like (A)?

- **(B)** hₜ↔hₗ, ∀ l≠t — encoder–encoder, all-time (`full_hh_negs`)
- **(C)** fₜ↔fₗ, ∀ l≠t — forecaster–forecaster, all-time (`full_ff_negs`)
- **(A)+(B)** both (A)'s and (B)'s crossed terms (`full_fh_hh_negs`)

Vocabulary: *crossed negative* = a contrastive negative pair indexed
across the time axis (l ≠ t / l ≠ t+1) within the same (batch, channel),
as opposed to the cross-channel / cross-batch negatives already in the
baseline. *GM-Relative MASE* = geometric mean over GIFT-Eval configs of
(model MASE ÷ seasonal-naive MASE); 1.0 ties seasonal naive, lower wins.
*Transfer* = held-out GIFT-Eval, as opposed to the contrastive
pre-training objective itself.

## Protocol

Each variant is `cosine_similarity_batch` with the single l=t
forecaster–encoder negative **replaced by** its all-time crossed term —
the *identical* structural transform that defines (A)
(`cosine_similarity_batch_full_fh_negs`); (A)+(B) carries both terms.
logsumexp form, `--pos-in-denominator` (normalized InfoNCE) supported.
Implementation in `src/loss.py`; 6 closed-form / mask unit tests
(`tests/test_loss.py::TestCrossedLossSiblings`, incl. orthonormal-C1
exact-value pins of the negative composition + masks) plus the three
variants added to the stability-suite parametrization (18 further
cases); combined loss suite `test_loss.py` + `test_loss_stability.py`
= 113 passed.

**Backbone recipe — only `--loss-shape` changes.** The comparison base
is (A)'s *backbone-of-record*: the #296 orchestrator `a1` run
(`enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k`) that
#296/#300 actually evaluate — 1-layer forecaster, fp16 group (residual
fp32 / attn-ffn-conv fp16), 2-GPU DDP global-batch 256, 50k, seed
20260517, `--pos-in-denominator`. `run_ddp.sh`'s *literal* 2-layer / bf16
form diverged at ~step 1.1k (2026-05-17 RESULTS.md), so the controlled
"change only the loss" test must start from the 1L/fp16 run that
produced (A). Per backbone: a 30k 2L-causal-transformer q-head, then
official GIFT-Eval (triage 11, full 97). Exact commands:
[`scripts/run_all.sh`](scripts/run_all.sh) (recipe verbatim, only
`--loss-shape` differs across arms).

## What we learned

1. **(A)'s fₜ↔hₗ crossed term is the worst single choice.** Replacing
   it with hₜ↔hₗ (B) lowers full GM-MASE 1.4377 → 1.3572 (−5.6 %), with
   (B) below (A) in **all 7 domains**; fₜ↔fₗ (C) gives 1.3822 (−3.9 %).
2. **The effect is the fₜ↔hₗ term, not "more negatives".** (A)+(B)
   (1.4517) sits ≈(A) and ≈0.095 *worse* than (B) alone — re-introducing
   fₜ↔hₗ on top of (B) removes essentially all of (B)'s gain. The three
   single-seed measurements are mutually consistent and point to fₜ↔hₗ
   as the harmful term (not yet a multi-seed interval — see Follow-up).
3. **Contrastive-objective fit does not predict transfer.** (A), (B),
   (C) reach near-identical contrastive proxies (loss / `loss_tau_ref` /
   1−AUC / dim-usage) yet differ up to 5.6% in GM-MASE; (A)+(B) has the
   *highest* contrastive loss (~5% above, expected from its larger
   negative set) and among the *worst* transfer. No arm diverged. The
   GM-MASE separation is held-out transfer, unexplained by training fit.
4. **It is an isolated contributor, not the whole gap.** Best (B, 1.357)
   remains ≈5 % above v11c (1.292) and beats seasonal naive in only
   Sales/Nature; the rest stays in #296's bottleneck/dropkey/precision
   confound. *(Hypothesis, not measured here: contrasting fₜ against
   encoder states hₗ at every non-target lag penalises the forecast for
   resembling nearby-in-time encoder states it legitimately should
   resemble in autocorrelated series, hurting transfer; the
   same-modality hₜ↔hₗ / fₜ↔fₗ spreads do not.)*

## Follow-up

- **Multi-seed confirmation** of (B) vs (A): the −5.6 % is a single-seed
  delta; 2–3 extra seeds would put an interval on it.
- **Adopt (B) `full_hh_negs`** as the bottleneck-fullfh loss default and
  re-attack the residual ≈5 % to v11c on the still-confounded
  bottleneck / dropkey / precision axes (#296).
- **Econ/Fin** (≈2.8 even under (B)) remains the dominant geomean term —
  the single largest lever (per #300), not aggregate tuning.

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
