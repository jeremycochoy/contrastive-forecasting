# Tiny backbone: first long streaming-data training

The architecture search ([contrastive-arma](../2026-04-12_contrastive-arma/contrastive-arma.md)) fixed the Tiny backbone recipe on short synthetic runs. The goal here was the next step: take that recipe to its first long-form training on HuggingFace-streamed bundle data, and produce the backbone the downstream forecasting experiments would build on.

## Result

The backbone trained to a contrastive **gap** of around 0.40; its best checkpoint (`tiny_v2_best_gap`, annotated gap 0.428) became the backbone reused by [head / rollout](../2026-04-16_head-rollout-comparison/head-rollout-comparison.md) and [encoder-comparison](../2026-04-19_encoder-comparison/encoder-comparison.md).

> *Contrastive **gap** = FF − FP, where FF = cosine(forecast, future) and FP = cosine(forecast, present). A higher gap means the forecast embedding resembles its own future more than its present — the quantity the contrastive loss exists to grow.*

![Training dashboard: per-step loss, contrastive gap, the FF/FP components, and a zoom on the 30k resume region.](plots/training_dashboard.png)

Loss falls from ≈ 14 to near zero; the contrastive gap rises to a plateau around 0.40, with a mid-run dip near step 11k and a second, sharper dip at the step-30,000 resume — a resume artifact, not a property of the model (see the annex).

## Protocol

| | |
|---|---|
| Backbone | Tiny: GRU patch encoder, 6 transformer layers, 8 heads, FFN ×4, GELU, depthwise causal conv k=3, H=512, patch W=16, RevEWMNorm span=32 (~20M params) |
| Loss | `cosine_similarity_batch` (cross-batch + cross-channel negatives, no within-time negatives), temperature 0.07 |
| Data | HuggingFace streaming bundles (`tiny_mixed_v1` / `base_mixed_v1`) |
| Optimiser | AdamW, batch 24, lr 1e-4 (constant), no gradient clipping |
| Hardware | Vast.ai, then Elisa (2× RTX 4090), across resumes |

The exact launch command was not preserved, so the values above are the trainer defaults the run used. The dashboard shows the ≈ 0.40 plateau but stores no exact value or underlying CSV; the 0.428 is the annotation carried on the best saved checkpoint.

## What we learned

The contrastive-arma recipe trains stably on streamed bundle data over a single long run, reaching a gap around 0.40, and the resulting backbone was good enough to carry the next round of forecasting experiments.

## Annex: two issues surfaced during the run

The run did not complete on the first attempt; two infrastructure problems had to be fixed, both since closed. Full root-cause and fixes are in [notes/INCIDENT_NAN_AND_RESUME.md](notes/INCIDENT_NAN_AND_RESUME.md).

![NaN-crash forensic: loss decays normally, then spikes to NaN and the run stops.](plots/crash_analysis.png)

1. **An all-NaN data row crashed the run** (above): one row in the stream was entirely NaN and passed through the loader's forward-fill untouched, corrupting the weights until the next step produced NaN everywhere. Fixed by skipping rows still NaN after forward- then back-fill, plus a NaN/Inf guard that checkpoints and exits on contact.
2. **Resumes lost state:** the checkpoint saved weights, optimiser, and step counter, but not `best_loss`, the EMAs, the RNG state, or the true stream position — so each resume overwrote the best checkpoint and perturbed the metrics (the dip at step 30,000 above, which recovers over several thousand steps). Fixed by saving and restoring the full state.
