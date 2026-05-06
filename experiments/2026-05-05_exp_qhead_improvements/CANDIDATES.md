# qhead-improvements — candidate ledger

Backbone: **backbone beta** (`tiny_full4096_moirai_hp_FRESH_RESUME50k_FINAL.pth`,
C=1, H=384, nhead=6, num_layers=6, T_RAW=4096, freq_emb_dim=3,
seasonality_emb_dim=3, rev_norm_kind=ewma span=128).

Baseline (#10 RESUME50k report): **GM-MASE 1.1828** on full 97-config eval
with default GRU quantile head + lr=3e-4 + no schedule + AdamW defaults
(β2=0.999, wd=0.01).

Target: **GM-MASE ≈ 0.81** (Moirai). Each accepted change should compose with
the next.

Triage proxy: 11-config small-test-set filter, biased ~0.06 below the full
97-config GM-MASE on the baseline (1.128 triage vs 1.183 full) but preserves
ranking. Useful for fast triage; full eval to confirm before claiming wins.

## Round 1 (done — vast 36184338, $3.64)

| ID | head | HP | schedule | total_steps | GM-MASE (triage 11) | Δ vs base |
|---|---|---|---|---|---|---|
| baseline | GRU-q | lr 3e-4, β2 0.999, wd 0.01 | constant | 30k | 1.128 | 0% |
| **E1** | **linear-q (55k)** | lr 3e-4, β2 0.999, wd 0.01 | constant | 30k | **1.066** | **−5.5%** |
| E2 | GRU-q (628k) | lr 1e-3, β2 0.98, wd 0.1 | WSD warmup 500, decay 24k→30k to 0.1×peak | 30k | 1.109 | −1.7% |

**Surprises**:
1. Linear probe **beats** GRU (1.066 < 1.128). Capacity isn't binding —
   the GRU was likely overfitting on this task. Backbone latents already
   well-structured.
2. WSD + Moirai HP helps modestly (−1.7%) but architecture dominates.

E1 full eval running on elisa GPU 1 to confirm the triage-vs-full delta.

## Round 2 — combining wins (done)

| ID | head | HP | schedule | total_steps | GM-MASE (triage) | Δ |
|---|---|---|---|---|---|---|
| R2_E3 | linear-q | Moirai (β2 0.98, wd 0.1, lr 1e-3) | WSD warmup 500, decay 24k→30k → 0.1×peak | 30k | 1.067 | −5.4% |

R2_E3 confirmed the linear head plateaus regardless of HP/schedule —
training-loss trajectory was bit-identical to R1_E1. Linear is at its
representational ceiling.

## Round 3 — transformer head (per user)

| ID | head | HP | schedule | total_steps | GM-MASE (triage) | Δ |
|---|---|---|---|---|---|---|
| **R3_E4** | xfmr-q (6L H=384 nhead=6, 10.7M) | Moirai | cosine warmup=1000 → 0.1×peak | 30k | **1.017** | **−9.8%** |

The user's hypothesis paid off: a 6-layer causal-decoder transformer
head matching the backbone's depth+width with Moirai HP and cosine LR
broke through the linear plateau. Loss was still dropping at step 30000
(final ema_loss=0.192) so longer training should help further.

## Round 4 — push the transformer

| ID | head | HP | schedule | total_steps | status | GM-MASE | Δ |
|---|---|---|---|---|---|---|---|
| **R4_E5** | same as R3_E4 | Moirai | cosine warmup=2000 → 0.1×peak | 60k | running on vast 36231634 | TBD | TBD |

## Pending candidates (re-ordered after R3_E4)

### Top priority — biggest expected wins

- **R4_E6**: deeper transformer (12 layers, same H=384). 21M params.
  R3_E4's 10.7M was clearly under-trained at 30k. After R4_E5 (longer
  training) lands, R4_E6 tests if depth on top of length helps.
- **R4_E7**: wider transformer (H=512, nhead=8, 6 layers). ~19M params.
  Different than depth — more representational power per layer. Needs
  a CLI flag for `--head-d-model` (currently inherits from backbone H).

### Schedule / HP refinements

- **R4_E8**: WSD 60k steps (stable to 48k, decay 48k→60k) — vs cosine
  60k. Tests whether WSD's late-cooldown recipe beats smooth cosine.
- **R4_E9**: longer warmup (5k vs 1k–2k) — gives the 10M-param
  transformer more time to escape initialization.

### Output structure / loss

- **R4_E10**: Gaussian-NLL head on the transformer trunk (predict μ, log σ²
  per step; closed-form CRPS). Smoother gradient than 9-bin pinball; might
  benefit the larger head.
- **R4_E11**: per-quantile output Linears on shared transformer trunk.

### Lower priority

- Mix synth into training data (mix_ratio > 0).
- Train forecast_len=128 (multi-step decoder).
- Larger batch (512 or 1024).
- Backbone fine-tune (last resort, breaks user's frozen-backbone assumption).
