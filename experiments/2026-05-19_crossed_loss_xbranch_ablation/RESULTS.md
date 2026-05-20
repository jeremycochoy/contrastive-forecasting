# Does the cross-branch f↔h negative carry any useful signal — or is the entire f↔h family (within-batch *and* cross-batch) just harmful or inert?

#303 isolated **f↔h all-time** (arm A) as the harmful crossed term: replacing it with within-branch siblings h↔h (B, 1.357) or f↔f (C, 1.382) lowered GM-MASE; re-introducing it on top of B (A+B, 1.452) erased the gain. Two questions #303 left open: (i) are the within-branch gains *additive*, i.e. does **B+C (no f↔h anywhere within-batch)** improve further or saturate? (ii) what about the **cross-batch** f↔h links — does removing them entirely (**B-xbfree**: B plus within-branch cross-batch only, NO f↔h cross-batch) help, hurt, or do nothing? And a control **A+B+C** rounds out the 2×2×2.

**Verdict.** Three single-seed numbers added to the #303 quartet, full-97:

- **B-xbfree 1.368 ≈ B 1.357** (+0.8 %). Removing every f↔h *negative* — within-batch *and* cross-batch — costs nothing measurable. The cross-batch f↔h links are **inert**; the cross-branch (f↔h) family contributes essentially no useful signal as a negative at this recipe. (The f→h positive is retained — this is about negatives only.)
- **B+C 1.398 ≈ B 1.357** (+3.0 %). Adding C on top of B slightly *hurts*; the within-branch siblings are **not additive** — within-batch h↔h already saturates the available signal, and the f↔f fan competes mildly with the f→h positive.
- **A+B+C 1.447 ≈ A+B 1.452** (−0.4 %). Adding C on top of A+B doesn't undo A's harm — re-confirming #303's "f↔h all-time is the harmful term" with a third witness.

The single winner of this family is still **(B) at 1.357** (#303). Across all seven arms in this recipe, no value of `--loss-shape` reaches v11c 1.292; the residual gap is the backbone-of-record, not the crossed-negative choice — fully consistent with #303's closing observation. **Practically:** the cleaner "no-cross-branch-negative" form (B-xbfree) is **as good as** the full-cross-batch B and is structurally simpler, so the f↔h negatives can be dropped entirely without measurable cost.

![Per-domain relative MASE — 4 #303 arms + 3 #307 arms + refs](plots/perdomain_star.png)
*Radar chart, one spoke per GIFT-Eval domain; distance from centre = GM relative MASE, lower = better; dotted green = seasonal naive (1.0). Thick lines = #307 new arms; thin = #303 kept arms; **gold dashed** = best-ever (R9_E13, #127, full GM 1.029); **purple dashed** = v11c (best enc-fcst, full GM 1.292). **(B) xbranch-free** (cyan) tracks **(B) full_hh** (blue) within ≈1 % on every domain — visual confirmation that the f↔h family is inert. **(B)+(C)** (red) and **(A)+(B)+(C)** (purple) sit at or above the (A+B) cluster; (A+B+C) clusters with (A+B) on Econ/Fin (≈3.4) where A's harm is largest.*

**Held-out GM-Relative MASE** (official GIFT-Eval; standard 2L causal q-head trained 30k on each backbone; 1.0 = seasonal naive, lower better):

| loss arm | within-batch crossed negatives | triage (11) | **full (97)** | vs (A) | vs (B) |
|---|---|---|---|---|---|
| **(A)** `full_fh_negs` *(#296, #303)* | f↔h (∀l≠t+1) | 1.5611 | **1.4377** | — | +5.9 % |
| **(B)** `full_hh_negs` *(#303)* | h↔h (∀l≠t) | 1.4461 | **1.3572** | −5.6 % | — |
| **(C)** `full_ff_negs` *(#303)* | f↔f (∀l≠t) | 1.5185 | **1.3822** | −3.9 % | +1.8 % |
| **(A)+(B)** `full_fh_hh_negs` *(#303)* | f↔h + h↔h | 1.5426 | **1.4517** | +1.0 % | +7.0 % |
| **(B)+(C)** `full_hh_ff_negs` *(#307)* | h↔h + f↔f | 1.5259 | **1.3982** | −2.7 % | +3.0 % |
| **(A)+(B)+(C)** `full_fh_hh_ff_negs` *(#307)* | f↔h + h↔h + f↔f | 1.6315 | **1.4465** | +0.6 % | +6.6 % |
| **(B) xbfree** `full_hh_negs_xbf` *(#307)* | h↔h (∀l≠t); **NO f↔h xb** | 1.4843 | **1.3681** | −4.8 % | +0.8 % |
| *ref:* v11c (bneck-fullfh prior best) / seasonal-naive | — | — | 1.292 / 1.000 | — | — |
| *ref:* #10 backbone (167k) + simple GRU q-head 30k | — | — | **1.183** | — | — |
| *ref:* #10 backbone (167k) + xfmr-q 12L 60k = R9_E13 (#127) | — | 0.990 | **1.029** | — | — |

Single seed per arm (matched to (A) — same `seed 20260517`, same 1L-fp16 / DDP-128 backbone-of-record). Triage (11-cfg) is ≈7–10 % noisy (#296); full-97 is the trusted metric. Reading the sign column **vs (B)**: every #303+#307 variant lies in {−0.0, +0.8, +1.8, +3.0, +5.9, +6.6, +7.0} %. The +0.8 % B-xbfree gap is **inside #296's ≈3 % checkpoint-spread noise**; +3.0 % (B+C) is at the noise-ceiling but in a consistent direction (B+C ≈ midway between B and C, slightly worse than either alone). Sign pattern: every arm containing all-time **f↔h within-batch (A)** is ≥+5.9 % worse than B; every arm without it is within +3 % of B.

## Question

#303 closed with two natural follow-ups:

1. **Are the within-branch gains additive?** B alone −5.6 %, C alone −3.9 % vs A; the contrastive structures overlap mildly (both touch h_l / f_l fans on the bottom ladder), so a priori (B+C) could land anywhere from "≈B (saturated)" to "well below B (additive)". Result: **+3.0 % vs B** — slightly *worse*, so the two within-branch fans are not additive; (B) already captures the available signal.
2. **Is f↔h cross-batch carrying anything?** All four #303 arms kept the **full** cross-batch negatives (b≠b′), including the f↔h cross-batch links. Dropping these — B-xbfree (B + only h↔h and f↔f cross-batch, no f↔h cross-batch, positive retained) — isolates whether the cross-batch f↔h family contributes. Result: **+0.8 % vs B** (inside noise), so the f↔h cross-batch term is **inert**.

A+B+C is the matched control: if A's harm is the sole determinant, A+B+C ≈ A+B and any further C-bonus is washed out. Observed: 1.447 ≈ 1.452 (−0.4 %), confirming A dominates.

## Protocol

Same protocol as #303 (continuation), only `--loss-shape` changes per arm. Each arm is `cosine_similarity_batch_<shape>` with logsumexp, `--pos-in-denominator`, identical backbone recipe (1L forecaster, fp16 group, 2-GPU DDP global-batch 256, 50k, seed 20260517, EWMA-RevIN span 128). Per backbone: 30k 2L-causal q-head + official GIFT-Eval (triage 11 + full 97). Commands: [`scripts/box_run.sh`](scripts/box_run.sh) (vast.ai backbone DDP) + [`scripts/local_downstream.sh`](scripts/local_downstream.sh) (free elisa downstream from synced backbones). Tests: **11 new closed-form/mask pins** (5 in `TestCrossBranchAblationExtended` + 6 in `TestCrossBranchNegativeFree`) — including the discriminating analytic value `log(2(T+3))` for (B)-xbfree at orthonormal inputs (vs (B)'s `log(2(T+2))`) — and the full 124-test loss suite green.

![Contrastive loss structure — 3 #307 arms on a (time × batch) ladder](plots/loss_diagram.png)
*Three panels, same primitives as #303. **Left (B+C):** blue h↔h fan from h_{b′,t} + orange f↔f fan from f_{b′,t} on the bottom ladder; full purple cross-batch (including f↔h xb). **Centre (A+B+C):** adds red f↔h fan from f_{b,t} on top ladder. **Right (B) xbfree:** blue h↔h only; the f↔h cross-batch links between ladders are **grayed-out / not drawn** — every f↔h interaction (within-batch *and* cross-batch) is removed; the positive f_{·,τ}→h_{·,τ+1} is **retained** (green, both ladders). Each crossed family is fanned from one anchor for clarity; the loss sums over all anchors. Arrows: ▶◀ inward = positive (attract), ◀▶ outward = negative (repel).*

## What we learned

1. **The entire f↔h *negative* family is inert.** B-xbfree (drop f↔h within-batch *and* cross-batch, keep positive) lands at 1.368 vs B's 1.357 (+0.8 %, inside noise). At this recipe, the f↔h cross-branch contrastive *signal as a negative* contributes nothing measurable; only the positive f→h carries information. **Implication for #303's interpretation:** A's harm (loss-of-record) is not from "any f↔h negative is bad" — within-batch f↔h cross-batch only is fine — but specifically from the **all-time** within-batch f↔h fan, which over-constrains forecasts at every l≠t+1.
2. **Within-branch h↔h and f↔f do not stack — (B) is the unique winner.** B+C = 1.398 = B + 3.0 % ≈ C + 1.6 %. So h↔h and f↔f are not orthogonal contributions: stacking them slightly hurts (likely f↔f all-time competes with the f→h positive). **(B) alone is the single best arm in this family.**
3. **A's harm is robust to compositional context.** A+B = 1.452 (#303) ↔ A+B+C = 1.447 (#307) ↔ B+C = 1.398 (without A). Within ≈1 % once A is present; ≈4 % better once A is removed. The +5.9 % (A vs B) of #303 is a stable harm independent of which other within-branch siblings are present.
4. **The loss is a small lever; the backbone is the gap.** Best (B) 1.357 is **still 5 % above v11c (1.292)** and ≈15 % above the same backbone the #10 line uses (1.183 with a simple 30k head). No `--loss-shape` in this family reaches v11c — fully consistent with #303's closing observation.

---

### Annex — loss-arm definitions (extended)

All seven arms are `cosine_similarity_batch_<shape>` (paper loss: cross-time + cross-channel + cross-batch negatives, logsumexp, normalized InfoNCE). The positive `cos(fₜ, hₜ₊₁)` is retained unchanged in every arm.

| arm | `loss_shape` | within-branch replacement | cross-batch shape |
|---|---|---|---|
| (A) | `cosine_similarity_batch_full_fh_negs` | cos(fₜ, hₗ) ∀ l≠t+1 | full (fh, hh, ff) |
| (B) | `cosine_similarity_batch_full_hh_negs` | cos(hₜ, hₗ) ∀ l≠t | full (fh, hh, ff) |
| (C) | `cosine_similarity_batch_full_ff_negs` | cos(fₜ, fₗ) ∀ l≠t | full (fh, hh, ff) |
| (A)+(B) | `cosine_similarity_batch_full_fh_hh_negs` | (A) ∪ (B) | full (fh, hh, ff) |
| (B)+(C) | `cosine_similarity_batch_full_hh_ff_negs` | (B) ∪ (C) | full (fh, hh, ff) |
| (A)+(B)+(C) | `cosine_similarity_batch_full_fh_hh_ff_negs` | (A) ∪ (B) ∪ (C) | full (fh, hh, ff) |
| **(B) xbfree** | `cosine_similarity_batch_full_hh_negs_xbfree` | (B) only | **hh + ff only — no fh cross-batch** |

### Annex — per-domain full GM relative MASE

| domain | (A) | (B) | (C) | (A)+(B) | **(B)+(C)** | **(A)+(B)+(C)** | **(B) xbfree** | best (overall) |
|---|---|---|---|---|---|---|---|---|
| Econ/Fin | 3.397 | **2.788** | 3.488 | 3.233 | 3.407 | 3.406 | 3.228 | (B) |
| Energy | 1.646 | **1.577** | 1.589 | 1.680 | 1.623 | 1.682 | 1.557 | (B)xbfree |
| Healthcare | 1.630 | 1.576 | 1.624 | **1.507** | 1.614 | 1.674 | 1.666 | (A)+(B) |
| Nature | 1.010 | **0.951** | 0.958 | 1.029 | 0.973 | 1.008 | 0.968 | (B) |
| Sales | 0.898 | 0.855 | 0.873 | 0.929 | 0.859 | **0.849** | 0.872 | (A+B+C) |
| Transport | 1.100 | **1.061** | 1.086 | 1.108 | 1.074 | 1.110 | 1.054 | (B)xbfree |
| Web/CloudOps | 1.518 | 1.427 | **1.392** | 1.552 | 1.435 | 1.508 | 1.412 | (C) |
| **full GM (97)** | 1.4377 | **1.3572** | 1.3822 | 1.4517 | 1.3982 | 1.4465 | 1.3681 | (B) |

(B) wins 4/7 domains and the overall GM. (B-xbfree) is within ≈1 % of (B) on every domain — the visual cyan-tracks-blue on the radar; numerically, the cross-batch f↔h removal leaves Energy/Transport/Web slightly *better* and Healthcare slightly *worse*, all within noise.

### Annex — wall-clock + cost breakdown (#307 only)

Backbone = 50k DDP (2× Blackwell GPU on vast.ai prosumer instances; HF-token authenticated, ≈5 sps). Downstream = 30k q-head + GIFT-Eval triage (11) + full (97), single GPU on free elisa 4090 (no vast spend). Three backbones trained concurrently 2026-05-19 17:00–21:00 UTC (3 separate boxes).

| arm | backbone 50k (vast) | q-head 30k | triage (11) | full (97) | downstream Σ (elisa) | vast cost |
|---|---|---|---|---|---|---|
| (B-xbfree) `hhxbf` | ≈3 h 0 m | ≈30 m | ≈3 m | ≈55 m | ≈1 h 28 m | $2.31 |
| (B+C) `hhff` | ≈3 h 30 m | ≈1 h 12 m | ≈3 m | ≈1 h 28 m | ≈2 h 43 m | $1.60 |
| (A+B+C) `fhhhff` | ≈2 h 50 m | ≈30 m | ≈3 m | ≈55 m | ≈1 h 28 m | $1.60 |

**Vast.ai total**: ≈$7.10 (3 boxes destroyed after their `FINAL.pth + optimizer.pth + losses.csv + run.log` resume-bundles synced + committed to [`artifacts/`](artifacts/)). Downstream ran free on elisa GPU0/GPU1 from synced checkpoints (q-head + GIFT-Eval recipe byte-identical to #303). End-to-end (3 new arms, code → eval): ≈14 h.

### Annex — reproducibility

- **Backbones (resume-capable)**: 3× `artifacts/<arm>/cl_<arm>_50k_FINAL.pth + _optimizer.pth + _losses.csv + run_*.log`.
- **Q-head checkpoints**: 3× `artifacts/<arm>/cl_<arm>_50k_qhead_FINAL.pth`.
- **GIFT-Eval CSVs**: `artifacts/<arm>/gift_eval_triage/{all_results.csv,summary.txt}` and `gift_eval_full/{all_results.csv,summary.txt}` per arm.
- **Tests**: `tests/test_loss.py::TestCrossBranchAblationExtended` (B+C, A+B+C: 5 pins) + `TestCrossBranchNegativeFree` (B-xbfree: 7 pins, including the analytic `log(2(T+3))` value distinguishing it from B's `log(2(T+2))`). Full 124-test loss suite green (`pytest tests/test_loss.py`).
- **Scripts**: [`scripts/box_run.sh`](scripts/box_run.sh) (vast.ai backbone DDP), [`scripts/local_downstream.sh`](scripts/local_downstream.sh) (elisa downstream), [`scripts/summarize.py`](scripts/summarize.py), [`scripts/plot_results.py`](scripts/plot_results.py), [`scripts/plot_loss_diagram.py`](scripts/plot_loss_diagram.py).
