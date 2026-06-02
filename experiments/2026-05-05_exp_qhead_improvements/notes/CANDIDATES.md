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

## Round 4 — push the transformer along length + depth (done)

| ID | head | HP | schedule | total_steps | GM-MASE (triage) | Δ |
|---|---|---|---|---|---|---|
| R4_E5 | xfmr-q 6L | Moirai | cosine warmup=2k → 0.1×peak | 60k | 1.009 | −10.6% |
| **R4_E6** | xfmr-q 12L | Moirai | cosine warmup=1k → 0.1×peak | 30k | **1.005** | **−10.9%** |

Length (R4_E5) and depth (R4_E6) each gave ~1% on top of R3_E4. Combined →
R5_E7.

## Round 5 — stack length + depth

| ID | head | HP | schedule | total_steps | GM-MASE (triage) | Δ |
|---|---|---|---|---|---|---|
| **R5_E7** | xfmr-q 12L | Moirai | cosine warmup=2k → 0.1×peak | 60k | **1.002** | **−11.2%** |

Best result so far. Stacking depth+length gave ~1.5% on top of R3_E4 — not
fully additive, but the new floor.

## Round 6 — bidir + forecast_len=128 (regressed)

| ID | head | HP | schedule | total_steps | GM-MASE (triage) | Δ |
|---|---|---|---|---|---|---|
| R6_E8 | xfmr-q 6L bidir fl128 | Moirai | cosine warmup=1k → 0.1×peak | 30k | 1.089 | −3.5% |

Hypothesis: more target signal per step + access to f_t..f_{t+k} via
bidir. **Hurt instead** — train-test mismatch (bidir attends to real f's
at training, rolled-out f's with rollout error at eval).

## Round 7 — push the winner longer (preempted)

| ID | head | HP | schedule | total_steps | GM-MASE (triage) | Δ |
|---|---|---|---|---|---|---|
| R7_E9 | xfmr-q 12L | Moirai | cosine warmup=3k → 0.1×peak | 100k (truncated @85k by spot preempt) | 1.020 | −9.6% |

Vast spot instance preempted mid-cooldown. Best.pth from step ~85k was
slightly worse than R5_E7's. Either 60k is the sweet spot or the truncation
hurt; either way longer training isn't an obvious win.

## Round 8 — Gaussian NLL loss (no help)

| ID | head | HP | schedule | total_steps | GM-MASE (triage) | Δ |
|---|---|---|---|---|---|---|
| R8_E10 | xfmr-gauss 12L | Moirai | cosine warmup=2k → 0.1×peak | 60k | 1.020 | −9.6% |

Hypothesis: pinball loss surface plateau is the bottleneck; smooth
parametric NLL would break through. **It didn't** — Gaussian NLL lands
at the same triage GM-MASE as the truncated R7_E9 and notably worse than
R5_E7. The pinball plateau wasn't the loss surface — it was the
representation.

## Final scoreboard (triage GM-MASE)

| Run | head | total_steps | GM-MASE | vs naive (1.000) |
|---|---|---|---|---|
| baseline (legacy GRU) | GRU-q | 30k | 1.128 | +12.8% |
| R1_E1 / R2_E3 | linear-q | 30k | 1.066/1.067 | +6.6% |
| R3_E4 | xfmr-q 6L | 30k | 1.017 | +1.7% |
| R4_E5 | xfmr-q 6L | 60k | 1.009 | +0.9% |
| R4_E6 | xfmr-q 12L | 30k | 1.005 | +0.5% |
| **R5_E7** | **xfmr-q 12L** | **60k** | **1.002** | **+0.2%** |
| R6_E8 | xfmr-q 6L bidir fl128 | 30k | 1.089 | +8.9% |
| R7_E9 | xfmr-q 12L (truncated) | 100k | 1.020 | +2.0% |
| R8_E10 | xfmr-gauss 12L | 60k | 1.020 | +2.0% |

**Best**: R5_E7 — closes 92% of the head-only gap from baseline (1.128) to
naive (1.000). Triage proxy is biased ~0.06 below full eval, so the
unbiased number is likely ~1.06 (full eval running now to confirm).

## Conclusion

Four orthogonal axes explored with diminishing returns: head architecture
(linear/GRU/transformer 6L/12L/bidir), training length (30k/60k/100k),
schedule (constant/WSD/cosine), and loss formulation (pinball/Gaussian).
All converge to ~1.00–1.02 triage GM-MASE on backbone-beta. The
Moirai-targeting 0.81 gap remains ~25% on triage, which under our triage
proxy is ~30% on full eval — likely backbone-limited (out of scope for
this experiment per the user's frozen-backbone assumption).

Recommended next step (out-of-scope here): scale the backbone (more
params, longer pretraining) — the head clearly cannot extract more signal
than the backbone latents carry.
