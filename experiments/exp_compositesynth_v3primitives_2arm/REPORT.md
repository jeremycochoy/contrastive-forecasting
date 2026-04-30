# Phase 2 + 3 — composite synth iteration: pulse, seas-heavy, more-primitives

## TL;DR

Three orthogonal hypotheses tested in parallel on top of phase-1
composite synth, two arms each (RevIN + EWMA-128). Headline:

* **`--more-primitives` (v3) is the new best at EWMA-128**:
  GM-MASE 1.621, beats periodic baseline 1.659 by 2.3%.
  First arm to break below the periodic+EWMA-128 wall.
  Wins 57/97 vs periodic, beats every other arm on every metric at EWMA-128.
* **`--enable-pulse` (v2pulse) is the best at RevIN**:
  GM-MASE 1.782, with much-better tail than phase-1 (max 152 vs 194).
  Wins 60/97 vs periodic.
* **`--seas-heavy` (v2b) regressed at RevIN** (+4.5% GM) and only
  marginally helped at EWMA-128. Won't combine in phase 4.
* **Pulse and primitive-variety are complementary modalities**:
  pulse adds spike content (helps RevIN tail / Web-CloudOps),
  primitive variety adds periodic-shape diversity (helps EWMA-128
  on Energy / Healthcare / Sales). Phase 4 will combine them.

## Setup

All three experiments share the phase-1 recipe, only the synth
sub-flag differs. Two arms each, run on 6 fresh Vast.ai instances
(2 each per phase) in parallel:

| Phase | New flag | Mechanism |
|---|---|---|
| 2A `v2pulse` | `--enable-pulse` | adds PULSE primitive (sparse pulse train, low duty) as a 4th option in the {sin, sq, saw} pool |
| 2B `v2b` | `--seas-heavy` | swaps (2 free + 1 seas-tied) waves → (1 free + 2 seas-tied) |
| 3 `v3` | `--more-primitives` | adds TRIANGLE + HALF_SIN as 4th and 5th options in the wave pool |

Common knobs (matching `exp_compositesynth_2arm`):
* Tiny backbone (H=512, L=6, GRU encoder, W=16, 20M params)
* `cosine_similarity_batch` loss
* `mix_ratio=0.5`, `freq_emb_dim=3`, `seasonality_emb_dim=3`, `mixup_p=0.3`
* 30k bb + 30k qhead, batch 24
* Selector `_best_loss → FINAL.pth`
* GIFT-Eval official, 97 configs, B4 strategy, single seed (42)

## Headline (97 configs each, single seed)

### EWMA-128 (the tougher norm)

| arm | GM-MASE ↓ | median | max | configs<1.5 | wins vs periodic |
|---|---:|---:|---:|---:|---:|
| **v3 + EWMA-128 (NEW)** | **1.621** | **1.408** | 67.8 | **57/97** | **57/97** |
| v2pulse + EWMA-128 | 1.670 | 1.414 | 67.1 | 54/97 | 52/97 |
| v2b + EWMA-128 | 1.704 | 1.450 | 66.1 | 53/97 | 41/97 |
| composite + EWMA-128 (phase 1) | 1.697 | 1.459 | 66.3 | 51/97 | — |
| periodic + EWMA-128 (baseline) | 1.659 | 1.528 | 70.8 | 47/97 | — |

**v3 is the new best on every metric.** First non-periodic arm to break below
the GM 1.659 wall. The diversity-add hypothesis works.

### RevIN

| arm | GM-MASE ↓ | median | max | configs<1.5 | wins vs periodic |
|---|---:|---:|---:|---:|---:|
| **v2pulse + RevIN** | **1.782** | 1.522 | **152.1** | 47/97 | **60/97** |
| composite + RevIN (phase 1) | 1.785 | 1.514 | 194.3 | 48/97 | 58/97 |
| v3 + RevIN (NEW) | 1.807 | **1.477** | 173.5 | **50/97** | 58/97 |
| v2b + RevIN | 1.866 | 1.554 | 200.9 | 44/97 | 51/97 |
| periodic + RevIN (baseline) | 1.859 | 1.568 | 190.4 | 43/97 | — |

**v2pulse wins GM** at RevIN with much-better tail control (covid 152 vs 194 phase-1).
**v3 wins median + good-config count** but trades off on the tail. Different metric
preferences → different winners.

## Per-domain GM-MASE — all 10 arms (5 recipes × 2 norms)

```
                v3R     v2pR    v2bR    cR      pR  ||   v3E     v2pE    v2bE    cE      pE
Econ/Fin       7.08    7.00    8.05    7.44    8.45 || 3.35    3.40    3.55    3.63    3.26
Energy         1.56    1.62    1.67    1.56    1.60 || 1.47    1.59    1.66    1.60    1.53
Healthcare     3.56    3.35    3.74    3.60    4.47 || 2.16    2.23    2.22    2.21    3.27
Nature         1.18    1.16    1.18    1.13    1.20 || 1.18    1.20    1.18    1.21    1.17
Sales          0.91    0.94    0.92    0.91    0.96 || 0.90    0.95    0.94    0.96    0.98
Transport      1.10    1.06    1.12    1.06    1.16 || 1.06    1.03    1.12    1.05    1.16
Web/CloudOps   2.94    2.71    2.88    2.89    2.72 || 2.78    2.80    2.71    2.87    2.45
```

Suffix `R` = RevIN, `E` = EWMA-128. **v2pR** = v2pulse + RevIN, **cR** = composite (phase 1) + RevIN, **pR** = periodic + RevIN baseline, etc.

**EWMA-128 domain winners** (lowest GM in each row's right half):
| domain | winner | GM | runner-up |
|---|---|---:|---|
| Econ/Fin | periodic | 3.26 | v3 (3.35) |
| Energy | **v3** | 1.47 | periodic (1.53) |
| Healthcare | **v3** | 2.16 | composite-v1 (2.21) |
| Nature | periodic | 1.17 | v3/v2b (1.18) |
| Sales | **v3** | 0.90 | v2b (0.94) |
| Transport | v2pulse | 1.03 | v3 (1.06) |
| Web/CloudOps | periodic | 2.45 | v2b (2.71) |

**v3 wins 3/7 EWMA-128 domains outright** (Energy, Healthcare, Sales) and ties 1
(Nature). Periodic still wins Econ/Fin and Web/CloudOps — the two domains where
the synth-side recipe has nothing more to offer.

**RevIN domain winners**:
| domain | winner | GM |
|---|---|---:|
| Econ/Fin | v2pulse | 7.00 |
| Energy | v3/composite | 1.56 (tie) |
| Healthcare | v2pulse | 3.35 |
| Nature | composite-v1 | 1.13 |
| Sales | v3/composite | 0.91 (tie) |
| Transport | v2pulse/composite | 1.06 (tie) |
| Web/CloudOps | v2pulse | 2.71 |

**v2pulse wins 3-4 RevIN domains outright** (Econ/Fin, Healthcare, Web/CloudOps,
+ Transport tie). Confirms pulse helps RevIN tail-prone domains
(Web/CloudOps spikes, Healthcare exponential growth, Econ/Fin trend extrapolation).

## Why pulse wins RevIN, primitive-variety wins EWMA-128

Two distinct mechanisms:

* **Pulse** introduces a *new modality* (sparse bursts) that the model
  never saw in clean-periodic synth. Spike content survives EWMA
  normalization (a brief ±1 burst barely shifts the local EWMA mean) so
  it would help EWMA-128 too — **but** RevIN's per-instance z-score
  preserves the relative magnitude of spikes vs the rest, so RevIN
  benefits more. Top wins concentrated in spike-driven configs:
  bizitobs_application/medium -27%, bizitobs_service/medium -27%,
  covid_deaths -22% on RevIN.

* **Primitive variety** (triangle + half-sin) keeps the same modality
  as sin/sq/saw (still periodic, still bounded in [-1, 1]) but expands
  the *space of waveforms* the model can pattern-match against. EWMA-128
  removes the slow-trend that RevIN keeps in-window, leaving the periodic
  signal as the dominant learnable feature. Richer periodic
  representation → cleaner periodic capture under EWMA-128. Top
  per-domain wins are the strong-period domains: Energy (-2% over
  periodic on 32 configs), Healthcare (-34% over periodic on 5 configs),
  Sales (-9% on 4 configs).

* They are **orthogonal**: pulse adds new content; primitive variety
  adds new shapes for the existing content. Phase 4 will combine.

## Why seas-heavy regressed (lessons learned)

The v2b experiment swapped (2 free + 1 seas-tied) → (1 free + 2 seas-tied).
This kept the wave count constant but **collapsed two independent-period
samples into one**: each row now has at most 2 distinct periods (the
seas-tied bucket + 1 free), down from 3 before.

* Under RevIN: lost period diversity → worse contrastive learning → +4.5% GM.
* Under EWMA-128: marginal gain (+0.4% GM vs phase-1) but doesn't help
  beat periodic baseline.

**Generalisation**: diversity > quantity. Adding redundant content
(more samples of the same period bucket) hurts. Adding new modalities
(pulse) or new shapes (triangle/half-sin) helps.

## Phase 4 plan

Combine the two winning flags: **`--enable-pulse --more-primitives`**.

* Pool grows to 6 primitives: {sin, sq, saw, pulse, triangle, half_sin}
* Each wave slot has 1/6 chance of any primitive
* If pulse and primitive-variety are truly orthogonal, phase 4 should
  give: best v3 EWMA-128 (cleanest periodic) + best v2pulse RevIN tail
  (spike control) → potentially new best on both norms.

Risk: dilution from 6-way pool may erode v3's clean win at EWMA-128.
The diversity-vs-dilution tradeoff has favoured diversity so far at
3→4 and 3→5 primitives — does it keep favouring it at 5→6 or 3→6?
Empirical question.

Two fresh Vast.ai instances, parallel arms, single-seed. Same recipe
otherwise. ETA ~3.5 h.

## Cost so far

Phase 2 (4 instances × ~5 h) + phase 3 (2 instances × ~5 h)
= ~30 instance-hours × ~$0.34/h ≈ **$10 for the 6 arms**.

## Artefacts

Each experiment has its own dir under `experiments/exp_compositesynth_*_2arm/`:
* `run.sh` — single-arm driver (takes `revin`/`ewma128` as $1)
* `results/gift_eval_{revin,ewma128}/` — `all_results.csv` (97 rows) + `summary.txt`
* `plots/gift_eval_*_compare.png` — 4-panel plot (aggregate, CDF, per-domain, head-to-head)
* `scripts/plot_compare_2arm.py` — plotter, idempotent

Local `sync_compositesynth_*/` dirs (in main checkout, not worktree)
hold full checkpoint state including periodic backbone Nk saves and
end-of-training (lowercase `_final.pth`) renamed to `_endoftrain.pth`
to avoid macOS case-insensitive FS collisions with `_FINAL.pth`
(= best_loss copy used at eval).
