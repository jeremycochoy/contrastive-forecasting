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

## Round 2 — combining wins

| ID | head | HP | schedule | total_steps | status | GM-MASE | Δ |
|---|---|---|---|---|---|---|---|
| **R2_E3** | linear-q | Moirai (β2 0.98, wd 0.1, lr 1e-3) | WSD warmup 500, decay 24k→30k → 0.1×peak | 30k | running on elisa GPU 0 | TBD | TBD |

R2_E3 hypothesis: stack the two wins. Expect <1.066.

## Pending Round 2/3 candidates (re-ordered after R1)

### Top priority — likely biggest wins

- **R2_E4**: linear-q + Moirai HP + WSD + **60k or 100k steps**. The user's
  big-idea principle: longer training under WSD often yields more, especially
  when small head + frozen backbone (no overfitting blowup risk).
- **R2_E5**: linear-q + Moirai HP + WSD + **lr 3e-3** (peak lr sweep). E1
  used lr=3e-4 and won; E2 used lr=1e-3 (smaller win). Maybe linear head
  benefits from yet-higher peak lr.
- **R2_E6**: linear-q + cosine warmup→decay (vs WSD). Simpler schedule,
  matches typical fine-tune recipes.

### Architecture (linear was best, GRU was worst — explore in between)

- **R2_E7**: 1-layer GRU hidden=32 (or simply 2-layer MLP no GRU): a touch
  of nonlinearity above pure linear. Tests whether the GRU's recurrent
  bias hurt or whether it's just over-parameterization.

### Output structure

- **R2_E8**: per-quantile output Linears (shared trunk: backbone latent →
  trunk → 9 separate Linears, one per quantile). Fixes potential quantile
  crossing pathology.
- **R2_E9**: Gaussian-NLL parametric head (μ, σ; closed-form CRPS for each
  position). 2 outputs vs 9; massively fewer params; closed-form quantiles.

### Cooldown branching from R2_E3's STABLE (after R2_E3 finishes)

- Cooldown variant A: 6k linear decay 1e-3 → 3e-5 (more aggressive)
- Cooldown variant B: 4k linear decay (shorter cooldown)
- Cooldown variant C: cosine cooldown shape (vs linear)

### Lower priority

- Mix synth into training data (mix_ratio > 0)
- Train forecast_len=128 (multi-step decoder)
- Larger batch (512 or 1024)
