# Bottleneck + full-fh-negs (normalized InfoNCE), 2-GPU DDP — DIVERGED

**Q.** Does the v13 forecaster-bottleneck (6 encoder / 2 forecaster layers @ d128, 4 heads), dropkey 0.70 (per-step attention key-dropout, p=0.70), AdamW β2 0.95, with the all-(fₜ,hₗ)-negatives loss under normalized InfoNCE (`--pos-in-denominator`, loss ≥ 0) and 2-GPU DDP (256 global, full cross-rank negatives), train stably at **residual fp32 + attn/ffn/conv bf16**?

**A. No — it diverged.** Clean descent to a healthy **min loss 2.80 @ step 1104**, then collapse (loss > 6 by step 1413, ≈ 11 by 6k) onto a flat collapsed plateau for the remaining ~44k steps (contrastive top-1 — share of windows whose nearest-neighbour forecast is the true future — 1.00 → 0.02). The driver is the project's documented fresh-init residual-amplitude explosion: forecaster-L1 post-FFN residual max-abs goes 65 → 3.0e5 while attention QK ᵀ logits stay O(10–100), so the value/residual path overflows, not the softmax. This reproduces the earlier **v25** result — a fp32 residual stream alone does **not** stabilise a low-precision attn/ffn body at fresh init; adding the conv to bf16 did not change it.

![Divergence, log-log: loss (top) and forecaster-L1 amplitude (bottom)](plots/divergence_loglog.png)

| step | loss | top-1 | resid post-FFN (fcst L1) |
|---:|---:|---:|---:|
| 1104 | **2.80** (min) | 1.00 | ~65 (stable to ~1.2k) |
| 1413 | 6.1 | — | rising |
| 6000 | 10.7 | 0.02 | 1.6e5 |
| 50000 | 10.5 | 0.02 | peak 3.0e5 @ 8.6k |

**Takeaway.** Fresh-init partial-low-precision diverges even with the fp32 residual anchor → per the standing rule, go pure fp32. Follow-up (1L forecaster, fp16 → fp32 fallback): see [`RUN_PLAN.md`](RUN_PLAN.md).
