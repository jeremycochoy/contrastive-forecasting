<!-- DRAFT — result numbers filled once GIFT-Eval completes. -->
# Is (A)'s crossed fₜ↔hₗ negative the cause of the bottleneck-fullfh gap, or do its sibling crossed terms behave the same?

**Verdict.** _{VERDICT — filled from data: do (B) hₜ↔hₗ, (C) fₜ↔fₗ, (A)+(B) move full GM-MASE off (A)'s 1.438, and toward v11c 1.292 / seasonal-naive 1.0?}_

![Per-domain relative MASE — A, B, C, A+B](plots/perdomain_star.png)
*Per GIFT-Eval domain; distance from centre = GM relative MASE, dashed
circle = seasonal naive (1.0), lower = better. {INTERPRETATION}*

**Held-out GM-Relative MASE** (official GIFT-Eval; standard 2L causal
q-head trained 30k on each backbone; 1.0 = seasonal naive, lower better):

| loss arm | crossed negative added | triage (11) | **full (97)** |
|---|---|---|---|
| **(A)** `full_fh_negs` *(of record, #296)* | fₜ↔hₗ ∀ l≠t+1 | 1.5611 | **1.4377** |
| **(B)** `full_hh_negs` | hₜ↔hₗ ∀ l≠t | {B_TR} | **{B_GM}** |
| **(C)** `full_ff_negs` | fₜ↔fₗ ∀ l≠t | {C_TR} | **{C_GM}** |
| **(A)+(B)** `full_fh_hh_negs` | both of the above | {AB_TR} | **{AB_GM}** |
| v11c (prior best) / seasonal-naive | — | — | 1.292 / 1.000 |

_{ONE-LINE READ of the table vs the single-seed training noise (#296 saw
≈3% horizon spread, triage ≈7–10% noisy; full-97 is the trusted metric).}_

![Training curves (log–log)](plots/training_curves.png)
*Contrastive loss, latent dimension usage (u_temporal / u_batch), the
fixed-τ `loss_tau_ref` diagnostic, and 1−AUC, all log–log, four arms
overlaid. {INTERPRETATION — did any arm diverge / under-use dimensions /
fail to separate positives (AUC→1)?}*

## Question

`experiments/2026-05-17_bottleneck_fullfh_ddp` (#296) and #300 left the
bottleneck-fullfh GM-MASE gap (full 1.438 vs v11c 1.292, seasonal-naive
1.0) confounded across the loss / bottleneck / dropkey / precision axes;
#300 ruled out q-head undertraining as the main cause. (A) — the
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

## Protocol

Each variant is `cosine_similarity_batch` with the single l=t
forecaster–encoder negative **replaced by** its all-time crossed term —
the *identical* structural transform that defines (A)
(`cosine_similarity_batch_full_fh_negs`); (A)+(B) carries both terms.
logsumexp form, `--pos-in-denominator` (normalized InfoNCE) supported.
Implementation + 16 closed-form/mask unit tests: `src/loss.py`,
`tests/test_loss.py::TestCrossedLossSiblings` (full suite 113 passed).

**Backbone recipe — only `--loss-shape` changes.** The comparison base
is (A)'s *backbone-of-record*: the #296 orchestrator `a1` run
(`enc_fcst_bneck128_dk07_fullfh_norminfonce_1L_fp16_ddp128_50k`) that
#296/#300 actually evaluate — **1-layer** forecaster, fp16 group
(residual fp32 / attn-ffn-conv fp16), 2-GPU DDP global-batch 256, 50k,
seed 20260517, `--pos-in-denominator`. `run_ddp.sh`'s *literal* 2-layer
/ bf16 form diverged at ~step 1.1k (2026-05-17 RESULTS.md) and is not a
usable nor (A)-comparable recipe, so the controlled "change only the
loss" test must start from the 1L/fp16 run that produced (A). Per
backbone: a 30k 2L-causal-transformer q-head, then official GIFT-Eval
(triage 11, full 97). Exact commands:
[`scripts/run_all.sh`](scripts/run_all.sh) (recipe verbatim, only
`--loss-shape` differs across arms).

Single seed per arm (matched to (A) for a clean one-variable contrast);
single-seed training noise therefore bounds which differences are
meaningful — full-97 is the trusted metric, triage ≈7–10% noisy (#296).

## What we learned

_{FILLED FROM DATA — strictly facts; any reasoning beyond the numbers
flagged as hypothesis.}_

## Follow-up / hypothesis

_{FILLED FROM DATA.}_

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
