# Small-model contrastive-loss ablation sweep — 29 arms at 40k steps

At backbone step 40k, every one of the 11 evaluated arms is worse than the seasonal-naive baseline on GM-Relative MASE (all values > 1.0). Among the 11, `arm6_v2_combab` reaches the lowest at `1.2025`; the next-best is `arm5_tr1` at `1.3254`.

![GM-Relative MASE at backbone step 40k, quantile-head trained 15k steps on top, GIFT-Eval B4 full-97 datasets. Red dashed line at 1.0 = seasonal-naive reference; every arm is above it. Error bars ±0.01 = borrowed seed-noise band (see caveat).](plots/eval_2L_gm_mase_bars.png)

## Definitions

- `ff` = mean `cos(f̂, h_{t+1})` between the forecaster's next-step prediction `f̂` and the encoder's next-step latent `h_{t+1}` (unit-normalised on the sphere). `1 − ff` is a distance in [0, 2], smaller = better forecast; behaves like a log perplexity of `f̂` under the true `h_{t+1}` von-Mises-Fisher.
- `h_t`, `e_t`: the encoder-output latent and the patch-embedding latent respectively, shape `[B, T, C, H]`.
- `u_batchtime` = `1/(d · off-diag Gram mean)` over `(B×T)` samples of the specified latent, clamped to `[0, 1]`. 1 = every H dim carries independent info; low = collapsed onto a subspace. Exactly what SIGReg regularises.
- **Latent drift** at checkpoint pair `(step_i, step_j)` on a fixed held-out batch (`torch.manual_seed(20260722)`, `B=8` ARMA-synthetic): `mean_{b,t,c} 1 − cos(h_t(model_j), h_t(model_i))` (and analogously for `e_t`). Range `[0, 2]`; low = the mapping learned by the model on this fixed input hasn't rotated between the two checkpoints. Referred to below as "drift".
- **GM-Relative MASE** = geometric mean over 97 GIFT-Eval configs of `MASE(model) / MASE(seasonal_naive)`, official B4 strategy, forecast horizon 16 (see `experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py`). Lower = better; > 1 = beaten by seasonal-naive on that geometric average.
- Loss recipes for the six base arms are inherited verbatim from #374: arm 1 (`L_pred + L_rep`), arm 3 (`L_pred_moco + L_rep`), arm 4 (`pooled + MoCo`), arm 5 (`L_align + L_rep`), arm 6 v2 (`L_align + L_rep_moco`), bimoco (`L_pred_moco + L_rep_moco`). `MoCo` = negatives drawn from an EMA teacher. Ablations: `tr1` = all InfoNCE temperatures `τ` set to 1.0; `nse` = `--sigreg-embedding-weight 0.0`; `ncpc` = `--cpc-infonce-weight 0.0`; `combab` = all three (nse only applied to arm 1/3/4 — the arms where nse reduces drift per §Latent-drift results).

## Backbone and training

`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, batch_size=64, seed=20260520`, dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`. All 29 configurations trained to step 40,000 (`save_every=10000`, extra snapshot at 2500 and 40000). Data behind every number in this report is under `results/`.

## Latent-drift results

Paired Wilcoxon signed-rank on end-of-40k mean drift, ablation vs base, N=6 arm pairs (arm 1/3/4/5/6_v2/bimoco). Underlying values: `results/wave_d_metrics.csv`.

Signed-rank test (`scipy.stats.wilcoxon(base, ablation, alternative='greater', zero_method='wilcox')`), recomputed from the committed CSV:

| Ablation | h_t drift lower vs base / 6 | h_t p | e_t drift lower vs base / 6 | e_t p | verdict |
|----------|-----------------------------|-------|-----------------------------|-------|---------|
| ncpc     | 3/6 (arm 5/6_v2/bimoco)     | 0.281 | **5/6**                     | **0.047** | reduces `e_t` drift; `h_t` effect loss-shape-dependent |
| nse      | 3/6 (arm 1/3/4)             | 0.344 | 3/6                         | 0.281 | mixed; helps arm 1/3/4, hurts arm 5/6_v2/bimoco |
| tr1      | 1/6                         | 0.922 | 3/6                         | 0.656 | no drift reduction |

![Latent drift per adjacent-checkpoint pair — rows = variant, columns = h_t / e_t. The ncpc row (`h_t` column) is not below the base row for arm 1/3/4 but is below it for arm 5/6_v2/bimoco. In the e_t column ncpc is below base for 5 of 6 arms. combab traces stay below base on both latents for arm 6 v2 and bimoco.](plots/latent_movement_per_arm.png)

## GM-Relative MASE at step 40k (11 arms)

15,000 head-training steps of a 2-layer transformer quantile head (`--head-arch transformer --head-num-layers 2 --head-nhead 8 --head-ffn-mult 4.0 --head-causal true --head-train-input e_then_f --forecast-len 16 --batch-size 256 --lr 1e-3`) on the frozen 40k backbone, then GIFT-Eval B4 on all 97 official configs. Per-cell summaries: `results/eval_gm_mase/`.

| Rank | Arm            | GM-Rel MASE | Δ from #1 |
|------|----------------|-------------|-----------|
| 1    | arm6_v2_combab | 1.2025      | —         |
| 2    | arm5_tr1       | 1.3254      | +0.1229   |
| 3    | arm3_combab    | 1.4056      | +0.2031   |
| 4    | arm4_tr1       | 1.4414      | +0.2389   |
| 5    | bimoco_combab  | 1.4420      | +0.2395   |
| 6    | arm3_tr1       | 1.4547      | +0.2522   |
| 7    | arm5_nse       | 1.4682      | +0.2657   |
| 8    | arm6_v2_tr1    | 1.4684      | +0.2659   |
| 9    | arm4_nse       | 1.4852      | +0.2827   |
| 10   | bimoco_tr1     | 1.4892      | +0.2867   |
| 11   | arm5_ncpc      | 1.5079      | +0.3054   |

Coverage caveat: 11 of the 29 arm configurations were evaluated at 40k. Candidates were selected from three separate criteria on Wave-D snapshots (best 40k `1−ff`, still-improving trajectory, lowest `h_t` drift), with researcher-added coverage for arm 3 and arm 4. Combab is over-sampled (3/11 vs 6/29 in the population); ncpc is under-sampled (1/11 arm5_ncpc only vs 6/29). The 18 non-evaluated arms may include values better than the current ranks 6–11.

**Caveat on the error bars.** The ±0.01 shown is not a measured seed-replicate CI for this experiment (each cell is `N=1`). It is a borrowed constant from the 2026-05-08 τ-sweep paired reruns via [LeJEPA-SIGReg-τ report annex F](../2026-06-21_lejepa_sigreg_tau098/lejepa_sigreg_tau098.md#f-seed-noise-band), on a different backbone (17M params, T=4096, τ=0.90) and a different recipe. It is here as a visual reference for "differences smaller than this in the past turned out to be within-seed noise", not as a confidence interval for this ranking. The top-vs-#2 gap of 0.1229 is 12× that borrowed band, which is why it is called out separately.

## Supporting figures

![1 − ff per arm across training steps, 3×2 grid by variant (base/tr1/nse/ncpc/combab), shared y. In every panel except tr1, the six arms cluster and separate similarly to base, meaning the variants change absolute perplexity more than they change which arm is "hardest" to fit. tr1 explodes arm 1 and pulls bimoco/arm 6 v2 to near-zero — a loss-shape rescaling effect visible before any encoder-quality claim.](plots/cos_error_per_arm.png)

![u_batchtime per arm (one panel per arm, h_t solid + e_t dashed). The two arm-groups from #374 hold at d_model=64: arm 3/4/5 keep `u_batchtime(h_t) ≥ 0.97` (nearly independent dimensions) while arm 6 v2 and bimoco compress `h_t` toward 0.5–0.8 by 40k. combab variants of arm 6 v2 and bimoco compress further into the low-`u_batchtime` regime.](plots/dim_usage_per_arm.png)

## Annex

### A. Candidate selection for the 11 evaluated cells

Three criteria applied to the 29-arm Wave-D snapshot:
- A. Lowest end-of-40k `1 − ff` (best perplexity so far).
- B. Trajectory still improving with least post-min rebound in the [20k, 40k] window.
- C. Lowest `h_t` drift.

Top ~3 per criterion, deduped, plus researcher-added `arm3_combab`, `arm4_tr1`, `arm4_nse` for arm 3 and arm 4 coverage. Selection favours arms already scoring well on training-side metrics; the ranking above is conditional on that selection.
