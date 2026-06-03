# Frequency-label embedding: does a cheap periodicity hint help?

The Tiny backbone predicts periodic GIFT-Eval datasets poorly. State-of-the-art time-series foundation models condition on a series' frequency via a token; this experiment tested a much cheaper version — a small learned **frequency-label embedding** (10 frequency classes × 3 dimensions, 30 parameters) concatenated to every patch of the encoder input.

Training data is a **50/50 mix of real HuggingFace bundles and on-the-fly periodic synthetic series**. The synthetic half carries its true frequency class (1–9); real rows are tagged class 0, "unknown". So the frequency hint is only ever present on the synthetic, periodic half — it is a pass-through on real, non-periodic data. A **mixup** variant additionally interpolates two samples' inputs and their frequency embeddings.

## Result

The embedding helps where the hint can apply — on periodic series — and leaves the rest unchanged.

![GM-MASE per arm, benchmark-wide (97 configs) vs the 6 periodic-focus configs. freq-emb is flat overall but ~5% better than control on the periodic subset.](plots/gm_mase_per_arm.png)

> *GM-MASE = geometric mean of MASE over the listed configs (lower is better).*

- **On the 6 periodic-focus configs** (m4_hourly/H, solar/10T ×2, solar/H, ett1/15T, ett2/W — series whose real data has the multi-period structure the hint targets), freq-emb reaches GM-MASE **2.49 vs control's 2.62 (−5%)**; adding mixup is a touch lower (2.48).
- **Benchmark-wide (97 configs) freq-emb is flat: 1.702 vs control 1.702.** This follows from the design — most configs are non-periodic and their real rows carry frequency "unknown", so the embedding is a pass-through there and cannot move the average.
- freq-emb + mixup does edge the full benchmark (1.669, −1.9% vs control); since freq-emb alone does not, that benchmark-wide gap is mixup's contribution.

## Arms

| Arm | freq-emb | mixup | GM-MASE (97 configs) | GM-MASE (periodic-6) |
|---|:---:|:---:|---:|---:|
| control (no emb) | – | – | 1.702 | 2.620 |
| **freq-emb** | dim 3 | – | 1.702 | **2.487** |
| **freq-emb + mixup** | dim 3 | p=0.3 | **1.669** | 2.480 |

Two longer-training variants — a 90k-step backbone (no embedding) and the fe+mu arm with a 90k-step head — reach 1.692 and 1.681 benchmark-wide; the longer head did not beat the 30k fe+mu (1.669).

## Protocol

Tiny backbone (C=4, H=512, W=16, GRU, 6 layers, RevEWMNorm span=32), `cosine_similarity_batch` loss (τ=0.07), AdamW lr 1e-4, 30k steps from scratch. Data: HF `base_mixed_v1` with `mix_ratio=0.5` (synthetic classes 1–9, real rows class 0). Frequency embedding: 10 classes × 3 dims, concatenated per-patch per-channel; mixup p=0.3, λ ~ Beta(0.2, 0.2) over both inputs and embeddings. R1 reconstruction head, evaluated on the full GIFT-Eval suite (97 configs). Per-arm result CSVs are in [results/](results/); design rationale is in [notes/DESIGN.md](notes/DESIGN.md).

## What we learned

A 30-parameter frequency label moves the periodic-focus configs by ~5%, so the hint carries usable signal where it is present. It does not change the benchmark-wide number because it reaches only the synthetic half — real rows are "unknown". The obvious next step is to give real rows their true frequency; the dual frequency + seasonality embedding it points to is tested in [dualemb_3arm](../2026-04-28_exp_dualemb_3arm/exp_dualemb_3arm.md).

This is a single seed on a 6-config subset — directional, not settled. The cross-cutting comparison across the whole sequence is in the [aggregate report](../2026-04-27__aggregate/aggregate.md); this directory is also the sequence's shared-script home ([notes/SEQUENCE.md](notes/SEQUENCE.md)).
