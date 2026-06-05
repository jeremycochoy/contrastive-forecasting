# Head / rollout comparison: is the forecasting head the ~1.27 MASE bottleneck?

The frozen Tiny backbone keeps getting better at its own job — its contrastive gap rises from 0.10 to 0.43 as it trains ([notes/DESIGN.md](notes/DESIGN.md); this is the v2 backbone, best gap 0.428, [notes/EXECUTION_PLAN.md](notes/EXECUTION_PLAN.md)) — yet its GIFT-Eval **GM-Relative MASE stays flat at ≈1.26 (1.256 @30k, 1.274 @60k)** ([../2026-04-13_gift-eval/gift-eval.md](../2026-04-13_gift-eval/gift-eval.md)). So the limit is not the backbone. This experiment asks whether the **forecasting head and its rollout strategy** are what hold the score on that plateau: we hold the backbone fixed and swap in 6 different ways of turning latents into a forecast.

> *GM-Relative MASE = geometric mean over the 97 GIFT-Eval configs of (model MASE ÷ seasonal-naive MASE), where MASE is Mean Absolute Scaled Error. 1.0 = seasonal-naive; lower is better.*
> *Contrastive gap = how much more a window's forecast resembles its own future than its present — the margin the backbone's contrastive loss grows.*

## Result

Every one of the 6 variants lands on the same plateau (1.258–1.288); none escapes it.

![GM-Relative MASE for all 6 head/rollout variants. Red = value-space rollout (decode latents to values, then slide in value space); blue = latent-space rollout (roll the backbone's own latents forward, decode once). Green dashed line is seasonal-naive (1.0); grey band marks the 1.258–1.288 spread the whole family occupies. Lower is better.](plots/rollout_comparison.png)

Latent-space rollout edges out value-space (best B = 1.258 vs the A1 baseline 1.275), and the value-space round-trip does look mildly harmful — but the entire effect is a ~0.03 swing within a tight cluster — it does not even reach seasonal-naive parity (~0.25 below the plateau), let alone the leaderboard band (≈0.6–0.83, ~0.45 below the plateau). Whatever strategy decodes the latents, the result is the same plateau.

| ID | Rollout space | Head output | Step | GM-Rel MASE | Source |
|----|---------------|-------------|------|------------:|--------|
| A1 | value | 128 values | slide by 128 | 1.275 | cited (v2 baseline) |
| A2 | value | W=16 values | slide by 16 | 1.262 | cited ([reconstruction-head.md](../2026-04-17_reconstruction-head/reconstruction-head.md), A2=1.2620) |
| B1 | latent | 128 values | decode at end | **1.258** | recomputed |
| B2 | latent | 128 → crop 16 | decode each step | **1.258** | recomputed |
| B3 | latent | 128 non-overlap | decode every 8 | **1.260** | recomputed |
| B4 | latent | W=16 values | decode each step | **1.288** | recomputed |

*Single run per variant, no variance estimated; GIFT-Eval scoring is deterministic, so the only noise is head-training init. Treat the sub-0.03 ordering as suggestive, not significant.*

## Protocol

- **Backbone (frozen):** Tiny v2 contrastive backbone (`tiny_v2_best_gap.pth`, gap 0.428; GRU patch encoder, 6 causal-transformer layers, ~20M). Identical across all 6 variants — only the head and the rollout strategy change.
- **Head:** bidirectional-GRU forecasting head (~0.6M), trained 30k steps (AdamW, lr 3e-4, batch 24) on the `tiny_mixed_v2` split. Three heads cover the six variants (W=16-real → A2; 128-mixed → B1/B2/B3; W=16-mixed → B4); A1 reuses the existing 128-real head. Rollout strategy and head-output length per variant are defined in [notes/DESIGN.md](notes/DESIGN.md) (with the infrastructure plan in [notes/EXECUTION_PLAN.md](notes/EXECUTION_PLAN.md)).
- **Eval:** official GIFT-Eval suite, 97 configs, deterministic point forecast scored against seasonal-naive — same harness as [../2026-04-13_gift-eval/gift-eval.md](../2026-04-13_gift-eval/gift-eval.md).
- **B1–B4 numbers are recomputed from committed data.** Each variant's per-config MASE[0.5] is in [results/B1](results/B1/all_results.csv)…[results/B4](results/B4/all_results.csv) (97 configs each) but the seasonal-naive baseline is not. Seasonal-naive is dataset-intrinsic and identical across experiments, so [scripts/plot_rollout_comparison.py](scripts/plot_rollout_comparison.py) joins SN_MASE in by config string from the v2 GIFT-Eval summary (`../2026-04-13_gift-eval/results/v2_pair_30k_summary.txt`) — all 97 configs join — and takes the geomean of MASE/SN_MASE. **A1 (1.275) and A2 (1.262) are value-space evals run earlier; their per-config outputs are not committed here, so they are cited (A1 is the v2 baseline, [notes/EXECUTION_PLAN.md](notes/EXECUTION_PLAN.md)), not recomputed.**

## What we learned

All six rollout strategies cluster on the same ~1.27 plateau, so **how the head decodes latents is not the bottleneck** — the value-space round-trip is at most a ~0.03 nuisance, nowhere near the ~0.45 that separates us from the leaderboard band (≈0.6–0.83), and not even the ~0.25 to seasonal-naive parity. The plateau pointed instead at a mis-alignment hypothesis: the backbone already places `f[t] ≈ e[t+1]`, yet every head here was trained to *predict the future* from a latent rather than *reconstruct the patch that latent already represents* — re-doing prediction the backbone had done. This experiment motivated that hypothesis; the follow-up [reconstruction-head experiment](../2026-04-17_reconstruction-head/reconstruction-head.md) confirmed it — a head trained to reconstruct (R1) reaches 1.121, finally breaking the plateau (a 12% improvement over the A1 baseline).
