# Small-model contrastive-loss ablation sweep — 29 arms at 40k steps

At backbone step 40k, every one of the 11 evaluated arms is worse than the seasonal-naive baseline on GM-Relative MASE (all values > 1.0). Among the 11, `arm6_v2_combab` is the lowest.

![GM-Relative MASE at backbone step 40k, quantile-head trained 15k steps on top, GIFT-Eval B4 full-97 datasets. Red dashed line at 1.0 = seasonal-naive reference; every arm is above it. Error bars ±0.01 = borrowed seed-noise band (see caveat).](plots/eval_2L_gm_mase_bars.png)

## Definitions

- `ff` = mean `cos(f̂, h_{t+1})` between the forecaster's next-step prediction `f̂` and the encoder's next-step latent `h_{t+1}` (unit-normalised on the sphere). `1 − ff` is a distance in [0, 2]; smaller = better forecast.
- `h_t`, `e_t`: the encoder-output latent and the patch-embedding latent respectively, shape `[B, T, C, H]`.
- **InfoNCE**: the standard cross-entropy contrastive loss `−log(exp(pos/τ) / Σ_j exp(neg_j/τ))` used throughout, with temperature `τ`.
- **SIGReg**: the LeJEPA-style spectral regulariser that pushes each latent's Gram matrix toward a Gaussian off-diagonal; `--sigreg-embedding-weight` (`sigreg_e`) applies it to `e_t`, `--sigreg-encoding-weight` to `h_t`.
- **CPC** (`--cpc-infonce-weight`, "cpc"): the CPC-InfoNCE auxiliary of van den Oord et al. 2018 predicting `e_{t+1}` from a bilinear projection of `h_t`.
- **MoCo**: negatives drawn from an EMA teacher (momentum contrast).
- `u_batchtime` = `1/(d · off-diag Gram mean)` over `(B×T)` samples of the specified latent, clamped to `[0, 1]`. 1 = every H dim carries independent info; low = collapsed onto a subspace. Exactly what SIGReg regularises.
- **Latent drift** at checkpoint pair `(step_i, step_j)` on a fixed held-out batch (`torch.manual_seed(20260722)`, `B=8` ARMA-synthetic): `mean_{b,t,c} 1 − cos(h_t(model_j), h_t(model_i))` (and analogously for `e_t`). Range `[0, 2]`; low = the mapping learned by the model on this fixed input hasn't rotated between the two checkpoints. Referred to below as "drift".
- **GM-Relative MASE** = geometric mean over 97 GIFT-Eval configs of `MASE(model) / MASE(seasonal_naive)`, official GIFT-Eval **B4 strategy** (single-window in-context prediction with backbone context length matching the config's expected horizon), forecast horizon 16 (see `experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py`). Lower = better; > 1 = beaten by seasonal-naive on that geometric average.
- Ablations: `tr1` = all InfoNCE temperatures `τ` set to 1.0; `nse` = SIGReg on `e_t` disabled; `ncpc` = CPC auxiliary disabled; `combab` = all three, with `nse` applied only to arm 1/3/4 (the arms where `nse` reduces drift per §Latent-drift results).

## Backbone

`d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, batch_size=64, seed=20260520`, dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`. All 29 configurations have at least a step-40k checkpoint (`_40k.pth`); several arms have been extended past 40k via a separate scheduler and appear at their extended horizon in the training-side figures. The ranking uses each backbone's 40k checkpoint. Data behind every number in this report is under `results/`.

## GM-Relative MASE at step 40k (11 arms)

15,000 head-training steps of a 2-layer transformer quantile head (settings in Annex C) on the frozen 40k backbone, then GIFT-Eval B4 on all 97 official configs. Per-cell summaries: `results/eval_gm_mase/`.

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

Coverage caveat: 11 of the 29 configurations were evaluated at 40k. Combab is over-sampled and ncpc is under-sampled versus the arm-population distribution; non-evaluated arms may include values better than the current ranks 6–11. Selection procedure in Annex B.

**Caveat on the error bars.** The ±0.01 shown is not a measured seed-replicate CI for this experiment (each cell is `N=1`). It is a borrowed constant from the 2026-05-08 τ-sweep paired reruns via [LeJEPA-SIGReg-τ report annex F](../2026-06-21_lejepa_sigreg_tau098/lejepa_sigreg_tau098.md#f-seed-noise-band). It is here as a visual reference for "differences smaller than this in the past turned out to be within-seed noise", not as a confidence interval for this ranking.

## Latent-drift results

Paired Wilcoxon signed-rank on end-of-40k mean drift, ablation vs base, N=6 arm pairs (arm 1/3/4/5/6_v2/bimoco), one-sided base > ablation, `scipy.stats.wilcoxon(..., zero_method='wilcox')`. Underlying values: `results/wave_d_metrics.csv`.

| Ablation | h_t drift lower vs base / 6 | h_t p | e_t drift lower vs base / 6 | e_t p |
|----------|-----------------------------|-------|-----------------------------|-------|
| ncpc     | 3/6 (arm 5/6_v2/bimoco)     | 0.281 | 5/6 (all except arm 6_v2)   | 0.047 |
| nse      | 3/6 (arm 1/3/4)             | 0.344 | 3/6                         | 0.281 |
| tr1      | 1/6                         | 0.922 | 3/6                         | 0.656 |

![Latent drift per adjacent-checkpoint pair, rows = variant, columns = h_t / e_t.](plots/latent_movement_per_arm.png)

## Supporting figures

![1 − ff per arm across training steps, 3×2 grid by variant (base/tr1/nse/ncpc/combab), shared y axis.](plots/cos_error_per_arm.png)

![u_batchtime per arm, one panel per arm × variant, `h_t` solid + `e_t` dashed, x-axis log training step.](plots/dim_usage_per_arm.png)

## Annex

### A. Loss recipes for the six base arms

Inherited verbatim from the split-pred/rep sweep (`reports/2026-07-10_split_pred_rep/`):

| Arm     | Recipe                           |
|---------|----------------------------------|
| arm 1   | `L_pred + L_rep`                 |
| arm 3   | `L_pred_moco + L_rep`            |
| arm 4   | `pooled + MoCo`                  |
| arm 5   | `L_align + L_rep`                |
| arm 6 v2| `L_align + L_rep_moco`           |
| bimoco  | `L_pred_moco + L_rep_moco`       |

### B. Candidate selection for the 11 evaluated cells

Three criteria applied to the 29-arm Wave-D snapshot: (A) lowest end-of-40k `1 − ff`, (B) trajectory still improving with least post-min rebound in the [20k, 40k] window, (C) lowest `h_t` drift. Top ~3 per criterion, deduped, plus researcher-added `arm3_combab`, `arm4_tr1`, `arm4_nse` for arm 3 and arm 4 coverage.

### C. 2L quantile-head training settings

| Param                | Value          |
|----------------------|----------------|
| `--head-arch`        | `transformer`  |
| `--head-num-layers`  | 2              |
| `--head-nhead`       | 8              |
| `--head-ffn-mult`    | 4.0            |
| `--head-causal`      | true           |
| `--head-train-input` | `e_then_f`     |
| `--forecast-len`     | 16             |
| `--batch-size`       | 256            |
| `--lr`               | 1e-3           |
| `--total-steps`      | 15,000         |
