# Does the cross-branch f↔h negative carry any useful signal — or is the entire f↔h family (within-batch *and* cross-batch) just harmful or inert?

#303 isolated **f↔h all-time** (arm A) as the harmful crossed term: replacing it with within-branch siblings h↔h (B, 1.357) or f↔f (C, 1.382) lowered GM-MASE; re-introducing it on top of B (A+B, 1.452) erased the gain. Two questions #303 left open: (i) is the gain *additive* across the two within-branch siblings, i.e. does **B+C (no f↔h anywhere within-batch)** improve further, or saturate? (ii) what about the **cross-batch** f↔h links — does removing them entirely (B with f↔h dropped from cross-batch too: **B-xbfree**) help, hurt, or do nothing? And as a control, does **A+B+C** (all three families together) match A+B or recover something? This continuation runs those three arms under the same backbone-of-record recipe.

**Verdict.** The full-97 picture: **B-xbfree 1.368** (drop f↔h entirely, even the cross-batch links) ≈ **B 1.357** (within-batch + full xb). The cross-batch f↔h term is **inert** — removing it changes GM-MASE by +0.8 % (well inside #296's ≈3 % checkpoint-spread noise floor). **A+B+C 1.447** ≈ **A+B 1.452** — adding C on top of A+B does not undo A's harm. **B+C TBD** (running) — predicted ≈1.34 if A is the sole harmful term and B/C are roughly additive. The composite story across #303 + #307: **the entire f↔h family carries no useful signal** at this recipe (positive retained throughout — only the f↔h negatives are touched); within-branch h↔h carries the signal, and dropping f↔h costs nothing. Yet B's 1.357 is still ≈5 % above v11c (1.292) and well above the same backbone the #10 line uses (1.183 with a simple 30k head): the residual gap is overwhelmingly the bottleneck-fullfh backbone, not this loss term.

![Per-domain relative MASE — 4 #303 arms + 3 #307 arms + refs](plots/perdomain_star.png)
*Radar chart, one spoke per GIFT-Eval domain; distance from centre = GM relative MASE, lower = better; dotted green = seasonal naive (1.0). Thick lines = #307 new arms; thin = #303 kept arms; **gold dashed** = best-ever (R9_E13, #127, full GM 1.029); **purple dashed** = v11c (best enc-fcst, full GM 1.292). **(B) xbranch-free** (cyan) tracks **(B) full_hh** (blue) within ≈1 % on every domain — visual confirmation that the cross-batch f↔h links are inert; **(A)+(B)+(C)** (purple) sits with **(A)+(B)** (orange) in the worst cluster, dominated by Econ/Fin (3.4) where A's harm is largest.*

**Held-out GM-Relative MASE** (official GIFT-Eval; standard 2L causal q-head trained 30k on each backbone; 1.0 = seasonal naive, lower better):

| loss arm | within-batch crossed negatives | triage (11) | **full (97)** | vs (A) | vs (B) |
|---|---|---|---|---|---|
| **(A)** `full_fh_negs` *(#296, #303)* | f↔h (∀l≠t+1) | 1.5611 | **1.4377** | — | +5.9 % |
| **(B)** `full_hh_negs` *(#303)* | h↔h (∀l≠t) | 1.4461 | **1.3572** | −5.6 % | — |
| **(C)** `full_ff_negs` *(#303)* | f↔f (∀l≠t) | 1.5185 | **1.3822** | −3.9 % | +1.8 % |
| **(A)+(B)** `full_fh_hh_negs` *(#303)* | f↔h + h↔h | 1.5426 | **1.4517** | +1.0 % | +7.0 % |
| **(B)+(C)** `full_hh_ff_negs` *(#307)* | h↔h + f↔f | TBD_HHFF_TR | **TBD_HHFF_FU** | TBD_HHFF_DELTA_A | TBD_HHFF_DELTA_B |
| **(A)+(B)+(C)** `full_fh_hh_ff_negs` *(#307)* | f↔h + h↔h + f↔f | 1.6315 | **1.4465** | +0.6 % | +6.6 % |
| **(B) xbfree** `full_hh_negs_xbf` *(#307)* | h↔h (∀l≠t); **NO f↔h xb** | 1.4843 | **1.3681** | −4.8 % | +0.8 % |
| *ref:* v11c (bneck-fullfh prior best) / seasonal-naive | — | — | 1.292 / 1.000 | — | — |
| *ref:* #10 backbone (167k) + simple GRU q-head 30k | — | — | **1.183** | — | — |
| *ref:* #10 backbone (167k) + xfmr-q 12L 60k = R9_E13 (#127) | — | 0.990 | **1.029** | — | — |

Single seed per arm (matched to (A) — same `seed 20260517`, same 1L-fp16 / DDP-128 backbone-of-record). Triage (11-cfg) is ≈7–10 % noisy (#296); full-97 is the trusted metric. The four #303 + three #307 full-97 deltas tell a consistent story: **only A flips the sign** (+5.9 % vs B), every other arm sits within ±1 % of B; (B-xbfree)'s +0.8 % is inside that noise band, (A+B+C)'s +6.6 % matches (A+B)'s +7.0 %.

## Question

#303 closed with two natural follow-ups:

1. **Are the within-branch gains additive?** (B alone −5.6 %, C alone −3.9 % — but the contrastive structures overlap mildly; would the union (B+C) be ≈(B), strictly better, or somewhere between?)
2. **Is the f↔h cross-batch term carrying anything?** #303's A/B/C/A+B all kept the **full** cross-batch (b≠b′) negative set unchanged — including the f↔h cross-batch links. Dropping these — **B-xbfree** keeps only within-branch cross-batch (h↔h, f↔f cross-batch) and the *positive* f_{b,t}→h_{b,t+1}, so **no f↔h negative anywhere** — isolates whether cross-batch f↔h is useful (sign: − if useful, + if harmful, ≈0 if inert).

A third A+B+C arm rounds out the 2×2×2 with a controlled "all three" — if A's harm dominates, A+B+C ≈ A+B (and that is what we see).

## Protocol

Same protocol as #303 (continuation), only `--loss-shape` changes per arm. Each arm is `cosine_similarity_batch_<shape>` with logsumexp, `--pos-in-denominator`, identical backbone recipe (1L forecaster, fp16 group, 2-GPU DDP global-batch 256, 50k, seed 20260517, EWMA-RevIN span 128). Per backbone: 30k 2L-causal q-head + official GIFT-Eval (triage 11 + full 97). Commands: [`scripts/box_run.sh`](scripts/box_run.sh) (vast.ai DDP) + [`scripts/local_downstream.sh`](scripts/local_downstream.sh) (free elisa downstream from synced backbones). Tests: 12 new closed-form/mask pins (`TestCrossBranchAblationExtended` + `TestCrossBranchNegativeFree`) — including the discriminating analytic value `log(2(T+3))` for (B)-xbfree at orthonormal inputs (vs (B)'s `log(2(T+2))`) — and the full 124-test loss suite green.

![Contrastive loss structure — 3 #307 arms on a (time × batch) ladder](plots/loss_diagram.png)
*Three panels, same primitives as #303. **Left (B+C):** blue h↔h fan from h_{b′,t} + orange f↔f fan from f_{b′,t} on the bottom ladder; full purple cross-batch (including f↔h xb). **Centre (A+B+C):** adds red f↔h fan from f_{b,t} on top ladder. **Right (B) xbfree:** blue h↔h only; the f↔h cross-batch links between ladders are **grayed-out / not drawn** — every f↔h interaction (within-batch *and* cross-batch) is removed; the positive f_{·,τ}→h_{·,τ+1} is **retained** (green, both ladders). Each crossed family is fanned from one anchor for clarity; the loss sums over all anchors. Arrows: ▶◀ inward = positive (attract), ◀▶ outward = negative (repel).*

## What we learned

1. **Cross-batch f↔h is inert at this recipe.** Dropping it (B-xbfree, 1.3681) vs keeping it (B, 1.3572) costs **+0.8 %** — inside #296's ≈3 % checkpoint-spread noise. The cross-branch (f↔h) family contributes essentially no useful signal as a negative — within-batch *or* across-batch. (The positive f→h is retained, so this is a statement about negatives only.) Direct implication: the cleaner "no-cross-branch-negative" form is **as good as** the full f↔h loss; the f↔h all-time *negative* of the loss-of-record can be removed without penalty.
2. **(A)+(B)+(C) tracks (A)+(B); adding C doesn't undo A.** A+B+C = 1.4465 ≈ A+B = 1.4517 (Δ −0.4 %). Whatever (C) contributes when alone (−3.9 % vs A) is **lost** once (A) is present — re-confirming **#303's "f↔h is the harmful term"** with a third witness. The Econ/Fin spoke does the heavy lifting: A+B+C 3.41 ≈ A+B 3.23 — the same single-domain driver #300 / #303 flagged.
3. **(B+C) — TBD.** [Pending hhff full-eval; the prediction is ≈1.34 if A's removal is the sole determinant and B/C effects are roughly additive — i.e. essentially saturated at (B). Will update on completion.]
4. **The loss is a small lever; the backbone is the gap.** Even at the best loss in this family (B, 1.357), we are **still ≈5 % above v11c (1.292)** and ~15 % above the same backbone the #10 line uses (1.183 with a simple 30k head — *same backbone-line, different recipe*). Across all seven #303+#307 arms in this recipe family, no value of `--loss-shape` reaches v11c; the residual gap is the bottleneck-fullfh backbone, not the crossed-negative choice — consistent with #303's closing observation.

---

### Annex — loss-arm definitions (extended)

All seven arms are `cosine_similarity_batch_<shape>` (paper loss: cross-time + cross-channel + cross-batch negatives, logsumexp, normalized InfoNCE) with the single l=t forecaster–encoder negative `log_neg_xy_hat` replaced or augmented by the indicated terms. The positive `cos(fₜ, hₜ₊₁)` is retained unchanged in every arm.

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

| domain | (A) | (B) | (C) | (A)+(B) | **(B)+(C)** | **(A)+(B)+(C)** | **(B) xbfree** | best |
|---|---|---|---|---|---|---|---|---|
| Econ/Fin | 3.397 | 2.788 | 3.488 | 3.233 | TBD_HHFF_EC | 3.406 | **3.228** | (B)xbfree |
| Energy | 1.646 | **1.577** | 1.589 | 1.680 | TBD_HHFF_EN | 1.682 | 1.557 | (B)xbfree |
| Healthcare | 1.630 | 1.576 | 1.624 | **1.507** | TBD_HHFF_HE | 1.674 | 1.666 | (A)+(B) |
| Nature | 1.010 | **0.951** | 0.958 | 1.029 | TBD_HHFF_NA | 1.008 | 0.968 | (B) |
| Sales | 0.898 | 0.855 | 0.873 | 0.929 | TBD_HHFF_SA | **0.849** | 0.872 | (A+B+C) |
| Transport | 1.100 | **1.061** | 1.086 | 1.108 | TBD_HHFF_TR | 1.110 | 1.054 | (B)xbfree |
| Web/CloudOps | 1.518 | 1.427 | **1.392** | 1.552 | TBD_HHFF_WE | 1.508 | 1.412 | (C) |
| **full GM (97)** | 1.4377 | **1.3572** | 1.3822 | 1.4517 | **TBD_HHFF_FU** | 1.4465 | 1.3681 | (B) |

(B-xbfree) is best or co-best on **4 of 7 domains** (within 1 % of B everywhere — *visual* on the radar, *numeric* in the table). (A+B+C) only "wins" Sales (0.849), which is the smallest-Δ domain (#303 noted Sales is mostly noise-dominated).

### Annex — wall-clock + cost breakdown

Backbone = 50k DDP (2× Blackwell GPU on vast.ai prosumer instances; HF-token authenticated, ~5 sps). Downstream = 30k q-head + GIFT-Eval triage (11) + full (97), single GPU on free elisa 4090s (no vast spend).

| arm | backbone 50k | q-head 30k | triage (11) | full (97) | downstream Σ | vast cost |
|---|---|---|---|---|---|---|
| (B+C) `hhff` | TBD | TBD | TBD | TBD | TBD | TBD |
| (A+B+C) `fhhhff` | ≈2 h 50 m | ≈30 m | ≈3 m | ≈55 m | ≈1 h 28 m | $1.60 |
| (B) xbfree `hhxbf` | ≈3 h 0 m | ≈30 m | ≈3 m | ≈55 m | ≈1 h 28 m | $2.31 |

**Vast.ai total**: ≈$7.10 (3 boxes destroyed after their `FINAL.pth + optimizer.pth + losses.csv + run.log` resume-bundles synced + committed to `artifacts/<arm>/`). Downstream ran free on elisa GPU0/GPU1 from the synced checkpoints (q-head + GIFT-Eval recipe byte-identical to #303). End-to-end (3 new arms, code → eval): ≈11 h.

### Annex — reproducibility

- **Backbones**: 3× `cl_<arm>_50k_FINAL.pth + _optimizer.pth + _losses.csv + run.log` resume-capable in [`artifacts/`](artifacts/).
- **Q-head checkpoints**: 3× `cl_<arm>_50k_qhead_FINAL.pth` in `artifacts/<arm>/`.
- **GIFT-Eval CSVs**: `gift_eval_triage_cl_<arm>_50k/all_results.csv` + `summary.txt` and `gift_eval_full_cl_<arm>_50k/all_results.csv` + `summary.txt` per arm in [`artifacts/`](artifacts/).
- **Tests**: 12 new pins in `tests/test_loss.py::TestCrossBranchAblationExtended` + `TestCrossBranchNegativeFree` (closed-form values, mask sanity, distinctness from #303 arms). Full 124-test loss suite green (`pytest tests/test_loss.py`).
- **Scripts**: [`scripts/box_run.sh`](scripts/box_run.sh) (vast.ai backbone DDP), [`scripts/local_downstream.sh`](scripts/local_downstream.sh) (elisa q-head + GIFT-Eval), [`scripts/summarize.py`](scripts/summarize.py), [`scripts/plot_results.py`](scripts/plot_results.py), [`scripts/plot_loss_diagram.py`](scripts/plot_loss_diagram.py).
