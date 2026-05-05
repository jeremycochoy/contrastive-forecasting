# qhead-improvements — candidate ledger

Backbone: **backbone beta** (`tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth`,
C=1, H=384, nhead=6, num_layers=6, T_RAW=4096, freq_emb_dim=3,
seasonality_emb_dim=3, rev_norm_kind=ewma span=128).

Baseline (#10 RESUME50k report): **GM-MASE 1.1828** with default GRU
quantile head + lr=3e-4 + no schedule + AdamW defaults (β2=0.999, wd=0.01).

Target: **GM-MASE ≈ 0.81** (Moirai). Each accepted change should compose with
the next.

## Round 1 (in flight on vast 36184338)

| ID | head | HP | schedule | total_steps | status | GM-MASE | Δ vs base |
|---|---|---|---|---|---|---|---|
| baseline | GRU-q | lr 3e-4, β2 0.999, wd 0.01 | constant | 30k | done | 1.183 | 0 |
| E1 | linear-q (55k params) | lr 3e-4, β2 0.999, wd 0.01 | constant | 30k | running | TBD | TBD |
| E2 | GRU-q (628k) | lr 1e-3, β2 0.98, wd 0.1 (Moirai) | WSD warmup 500, decay 24k→30k to 0.1*peak | 30k | queued | TBD | TBD |

E1 question: **does head capacity matter?** If linear ≈ baseline, schedule + HP
are the entire game; if linear ≪ baseline, head-arch sweeps are warranted.

E2 question: **do schedule + Moirai-HP together close most of the gap?** The
24k checkpoint is the WSD branchable point — usable for cooldown fan-outs in
Round 2 without retraining the prefix.

## Pending candidates (ordered by EV-per-compute)

These will be triaged after Round 1 results.

### A. Schedule
- A1. WSD with shorter / longer / earlier cooldown branched from E2's 24k.
- A2. Cosine warmup→decay (alt to WSD).
- A3. Two-stage: 5k constant high-lr exploration → 25k cosine.

### B. HP (composed with A)
- B1. Sweep peak lr ∈ {3e-4, 1e-3, 3e-3} at 5k steps each → pick best.
- B2. Larger batch 512 / 1024 (gradient accum if VRAM-bound).
- B3. β1=0.9 vs 0.95 (LLaMA style).

### C. Architecture (informed by E1)
- C1. Smaller GRU (1 layer, hidden=64) — if E1 says capacity isn't binding.
- C2. Deeper GRU (3 layers) or wider (hidden=256) — if E1 says capacity binds.
- C3. Self-attention pool over backbone tokens instead of bidir GRU.
- C4. Per-quantile output linears (shared trunk) — fixes quantile crossing.

### D. Loss / target
- D1. Gaussian-NLL head (μ, σ; closed-form CRPS).
- D2. Mixture-of-Gaussians.

### E. Data
- E1d. Mix synth into head training (mix_ratio > 0).
- E2d. Larger forecast_len at train time.
