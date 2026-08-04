# The EMA-teacher alignment target lowers 2 of 5 measured cells and leaves 3 inside head-seed noise

Pointing `L_align` at the EMA teacher lowers 2 of the 5 measured cells, `arm5 base` and `arm5 ncpc`, by more than both their bootstrap interval and the largest measured head-seed range. It moves the other 3 by less than that range.

![Teacher minus student GM-Relative MASE, backbone 40k, two comparisons](plots/controlled_vs_cross_delta.png)

*Left: the controlled comparison, where only `--align-target` differs. Right: the cross-experiment comparison, where the flag and the code snapshot differ together. Whiskers on both panels are 95% dataset-cluster bootstrap intervals (`results/controlled_delta_40k.csv`, `results/eval_bootstrap_ci.csv`).*

The earlier small-model sweep computed `L_align` against the student encoder under stop-gradient rather than against the EMA teacher it was meant to target, and the ten `L_align` cells here are that retrain, with the teacher. The metric is **GM-Relative MASE**, the geometric mean over 97 GIFT-Eval configs of `MASE(model) / MASE(seasonal_naive)`, where lower is better and 1.0 is parity.

![GM-Relative MASE of all 30 cells at backbone 40k, seasonal-naive parity dashed](plots/eval_2L_gm_mase_bars.png)

*All 30 cells at backbone 40k against the dashed parity line at 1.0. Coloured = the 10 cells retrained with `--align-target teacher`.*

> **Cell** here is one loss recipe plus one setting, e.g. `arm5 combab`. The CSVs name it `arm_slug`. A cell measured under several head seeds is still one cell. The recipes and the settings are tabulated in [Protocol](#8-protocol).

## 1. What the flag does when nothing else changes

| Cell | teacher | student control (this branch) |
|---|---|---|
| `arm5 base` | 1.3515 | 1.4501 |
| `arm5 tr1` | 1.3396 | 1.3254 |
| `arm5 nse` | 1.4536 | 1.4682 |
| `arm5 ncpc` | 1.2923 | 1.5079 |
| `arm5 combab` | 1.2728 | 1.2868 |
| `arm6_v2 base` | 1.4322 | *pending* |
| `arm6_v2 tr1` | 1.5315 | *pending* |
| `arm6_v2 nse` | 1.3074 | *pending* |
| `arm6_v2 ncpc` | 1.3159 | *pending* |
| `arm6_v2 combab` | 1.2765 | *pending* |

*The differences and their intervals are the left panel above. Backbone 40 000 pretraining steps, quantile head 15 000 steps on the frozen backbone, 97 GIFT-Eval configs under the official **B4 strategy** (single-window in-context prediction), forecast horizon 16. Intervals are dataset-level paired cluster bootstraps over the 28 base datasets, 10 000 resamples (`results/controlled_delta_40k.csv`, `experiments/2026-08-01_lalign_teacher/scripts/controlled_delta.py`).*

Of the five finished controls, four reproduce the earlier sweep's number exactly and one, `arm5 base`, differs from it by 0.0977.

## 2. How far a cell moves under nothing but a head seed, on four frozen backbones

![Per-seed GM-Relative MASE on four frozen backbones](plots/head_seed_spread.png)

*Four frozen backbones, the quantile head retrained under extra seeds — its init and its data order — and the full 97-config eval re-run each time (`results/seed_spread.csv`). Only these four cells carry replicate head seeds.*

Of the 20 cross-experiment differences in section 5, 8 exceed the largest head-seed range in absolute value and 11 exceed it in relative value.

## 3. Where the 30 cells sit across backbone horizons

![GM-Relative MASE at backbone 40k, 100k and 200k for all 30 cells](plots/eval_2L_gm_mase_progression.png)

*All 30 cells; the 8 that improved over 40k→100k were extended to 200k.*

![GM-Relative MASE of the 8 cells extended to backbone 200k](plots/eval_2L_gm_mase_bars_200k.png)

*The 8 cells extended past 100k. Filled dot = backbone 200k, hollow = 100k, vertical whisker = measured head-seed range, on the 3 of 8 cells that carry replicate seeds.*

| Rank @100k | Cell | bb 40k (head 15k) | bb 100k (head 30k) | 40k→100k | bb 200k (head 30k) | 100k→200k |
|---|---|---|---|---|---|---|
| 1 | `arm6_v2 combab` ⟲ | 1.2765 | 1.2514 | −0.0251 | 1.1850 † | −0.0664 |
| 2 | `arm6_v2 ncpc` ⟲ | 1.3159 | 1.3012 | −0.0147 | 1.3325 † | +0.0313 |
| 3 | `arm4 combab` | 1.2748 | 1.3219 | +0.0471 | — | — |
| 4 | `arm6_v2 nse` ⟲ | 1.3074 | 1.3368 | +0.0294 | — | — |
| 5 | `arm5 ncpc` ⟲ | 1.2923 | 1.3419 | +0.0496 | — | — |
| 6 | `arm4 ncpc` | 1.2957 | 1.3441 | +0.0484 | — | — |
| 7 | `arm5 combab` ⟲ | 1.2728 | 1.3678 | +0.0950 | — | — |
| 8 | `arm5 tr1` ⟲ | 1.3396 | 1.3710 | +0.0314 | — | — |
| 9 | `bimoco ncpc` | 1.3739 | 1.3833 | +0.0094 | — | — |
| 10 | `arm1 base` | 1.3674 | 1.3909 | +0.0235 | — | — |
| 11 | `arm5 nse` ⟲ | 1.4536 | 1.4017 | −0.0519 | 1.8887 † | +0.4870 |
| 12 | `arm4 base` | 1.3537 | 1.4051 | +0.0514 | — | — |
| 13 | `bimoco base` | 1.5123 | 1.4144 | −0.0979 | 1.3993 | −0.0151 |
| 14 | `bimoco nse` | 1.3673 | 1.4234 | +0.0561 | — | — |
| 15 | `arm3 tr1` | 1.4547 | 1.4467 | −0.0080 | 1.4706 | +0.0239 |
| 16 | `arm4 tr1` | 1.4414 | 1.4469 | +0.0055 | — | — |
| 17 | `arm5 base` ⟲ | 1.3515 | 1.4510 | +0.0995 | — | — |
| 18 | `bimoco combab` | 1.4420 | 1.4517 | +0.0097 | — | — |
| 19 | `arm1 nse` | 1.5579 | 1.4548 | −0.1031 | 1.3308 | −0.1240 |
| 20 | `arm3 combab` | 1.4056 | 1.4921 | +0.0865 | — | — |
| 21 | `arm1 ncpc` | 1.5100 | 1.4963 | −0.0137 | 1.4041 | −0.0922 |
| 22 | `arm3 base` | 1.4545 | 1.5255 | +0.0710 | — | — |
| 23 | `arm4 nse` | 1.4852 | 1.5687 | +0.0835 | — | — |
| 24 | `bimoco tr1` | 1.4892 | 1.5823 | +0.0931 | — | — |
| 25 | `arm3 ncpc` | 1.4635 | 1.5973 | +0.1338 | — | — |
| 26 | `arm1 tr1` | 1.3725 | 1.6036 | +0.2311 | — | — |
| 27 | `arm6_v2 tr1` ⟲ | 1.5315 | 1.7064 | +0.1749 | — | — |
| 28 | `arm3 nse` | 1.4432 | 1.7372 | +0.2940 | — | — |
| 29 | `arm1 combab` | 3.1251 | 1.7595 | −1.3656 | 1.7107 | −0.0488 |
| 30 | `arm6_v2 base` ⟲ | 1.4322 | 1.9057 † | +0.4735 | — | — |

*Ranked by the 100k value. `⟲` marks a cell retrained with `--align-target teacher`; the other 20 carry no `L_align` term and are the earlier sweep's numbers, on the same seasonal-naive denominator. A dash means the cell was not extended. Values are head seed 20260722. † replicate head seeds (`results/seed_spread.csv`): `arm6_v2 base` @100k 1.8659 / 1.9057, mean 1.8858, range 0.0398 — the value shown is seed 20260722 of the two. `arm5 nse` @200k 1.7979 / 1.8655 / 1.8887, mean 1.8507, range 0.0908. `arm6_v2 combab` @200k mean 1.1851, range 0.0063. `arm6_v2 ncpc` @200k mean 1.3472, range 0.0252.*

## 4. Which domains the lowest cells win on

![Two radars of per-domain relative MASE](plots/eval_domain_radar.png)

*The headline geometric mean split by dataset domain. The left radar puts cells measured at backbone 100k and at 200k on one chart, so it moves the backbone step as well as the cell. Per-config source: `results/eval_gm_mase/<cell>/all_results.csv`, same seasonal-naive denominator as the aggregate.*

| Cell (bb 200k) | Energy (32) | Web/CloudOps (20) | Transport (15) | Nature (15) | Econ/Fin (6) | Healthcare (5) | Sales (4) | all 97 |
|---|---|---|---|---|---|---|---|---|
| `arm6_v2 combab` ⟲ | 1.388 | 1.283 | 1.021 | **0.867** | 1.489 | 1.261 | **0.830** | 1.1850 |
| `arm1 nse` | 1.594 | 1.368 | 1.226 | **0.954** | 1.823 | 1.225 | **0.898** | 1.3308 |
| `arm6_v2 ncpc` ⟲ | 1.553 | 1.470 | 1.133 | **0.943** | 2.070 | 1.230 | **0.917** | 1.3325 |
| `bimoco base` | 1.668 | 1.554 | 1.195 | **0.978** | 2.192 | 1.234 | **0.840** | 1.3993 |
| `arm1 ncpc` | 1.617 | 1.559 | 1.173 | **0.998** | 2.442 | 1.329 | **0.886** | 1.4041 |
| `arm3 tr1` | 1.744 | 1.693 | 1.202 | **0.980** | 2.469 | 1.386 | **0.897** | 1.4706 |
| `arm1 combab` | 1.985 | 1.799 | 1.379 | 1.208 | 3.934 | 1.572 | 1.068 | 1.7107 |
| `arm5 nse` ⟲ | 2.394 | 2.111 | 1.836 | 1.324 | 2.075 | 1.340 | **0.912** | 1.8887 |
| *cells below 1.0, of 8* | 0/8 | 0/8 | 0/8 | 6/8 | 0/8 | 0/8 | 7/8 | 0/8 |

*Configs per domain in brackets. Bold = below 1.0. Source: `results/eval_gm_mase/<cell>_bb200k_hd30000s/all_results.csv`.*

## 5. The retrained cells against the earlier sweep

Across the ten cells the direction is 6 lower at 40k and 4 lower at 100k, sign test p = 0.75 at both steps (`results/eval_paired_tests.csv`).

| Cell | bb steps | teacher | earlier sweep | ratio | CI 95% lo | CI 95% hi | CI excludes 1 |
|---|---|---|---|---|---|---|---|
| `arm5 base` | 40k | 1.3515 | 1.5478 | 0.8732 | 0.8411 | 0.9069 | yes |
| `arm5 tr1` | 40k | 1.3396 | 1.3254 | 1.0107 | 0.9827 | 1.0359 | no |
| `arm5 nse` | 40k | 1.4536 | 1.4682 | 0.9901 | 0.9288 | 1.0327 | no |
| `arm5 ncpc` | 40k | 1.2923 | 1.5079 | 0.8570 | 0.8294 | 0.8787 | yes |
| `arm5 combab` | 40k | 1.2728 | 1.2868 | 0.9891 | 0.9606 | 1.0160 | no |
| `arm6_v2 base` | 40k | 1.4322 | 1.3149 | 1.0892 | 1.0447 | 1.1295 | yes |
| `arm6_v2 tr1` | 40k | 1.5315 | 1.4684 | 1.0430 | 1.0041 | 1.0737 | yes |
| `arm6_v2 nse` | 40k | 1.3074 | 1.3791 | 0.9480 | 0.9109 | 0.9861 | yes |
| `arm6_v2 ncpc` | 40k | 1.3159 | 1.3623 | 0.9660 | 0.9356 | 0.9966 | yes |
| `arm6_v2 combab` | 40k | 1.2765 | 1.2025 | 1.0615 | 1.0249 | 1.1117 | yes |
| `arm5 base` | 100k | 1.4510 | 1.5579 | 0.9314 | 0.8881 | 0.9669 | yes |
| `arm5 tr1` | 100k | 1.3710 | 1.4249 | 0.9622 | 0.9061 | 1.0084 | no |
| `arm5 nse` | 100k | 1.4017 | 1.3980 | 1.0026 | 0.9657 | 1.0395 | no |
| `arm5 ncpc` | 100k | 1.3419 | 1.4459 | 0.9281 | 0.8658 | 0.9753 | yes |
| `arm5 combab` | 100k | 1.3678 | 1.2456 | 1.0981 | 1.0645 | 1.1346 | yes |
| `arm6_v2 base` | 100k | 1.9057 | 1.3449 | 1.4169 | 1.3030 | 1.5135 | yes |
| `arm6_v2 tr1` | 100k | 1.7064 | 1.5188 | 1.1236 | 1.0766 | 1.1705 | yes |
| `arm6_v2 nse` | 100k | 1.3368 | 1.3914 | 0.9608 | 0.9343 | 0.9864 | yes |
| `arm6_v2 ncpc` | 100k | 1.3012 | 1.2978 | 1.0027 | 0.9672 | 1.0452 | no |
| `arm6_v2 combab` | 100k | 1.2514 | 1.1616 | 1.0773 | 1.0439 | 1.1254 | yes |

*This comparison moves the flag and the code snapshot together, so it does not attribute a difference to the flag; section 1 is the one that does. Ratio = teacher ÷ earlier sweep, below 1.0 means the teacher run scored lower. Intervals are dataset-level paired cluster bootstraps, 10 000 resamples (`results/eval_bootstrap_ci.csv`).*

## 6. Does latent movement track the improvement

![Late h_t drift against the 100k to 200k change in GM-Relative MASE](plots/drift_vs_improvement.png)

*Panels 1 and 2: `h_t` drift between adjacent checkpoints, and GM-Relative MASE on the same axis, for the 5 of the 8 extended cells whose GM-Relative MASE fell from 100k to 200k (`arm6_v2 combab`, `arm1 nse`, `bimoco base`, `arm1 ncpc`, `arm1 combab`). Panel 3: mean late drift against the 100k→200k change, all 8 extended cells (`results/latent_movement_pairs.csv`, 250 pairs over the 30 arms).*

Panel 3 gives Spearman ρ = −0.33 on n = 8 cells at p = 0.42, so late drift does not separate the improvers at this sample size.

*A setting counts as* lower *when its mean `1 − cos` displacement over adjacent-checkpoint pairs falls below the base setting of the same recipe; denominator is the six recipes, p is a two-sided exact binomial test (`results/latent_movement_pairs.csv`).*

| Setting | `h_t` drift lower than base / 6 | `h_t` p | `e_t` drift lower than base / 6 | `e_t` p |
|---|---|---|---|---|
| `ncpc` | 1/6 (arm5) | 0.219 | 4/6 | 0.688 |
| `nse` | 3/6 (arm3 / arm4 / arm5) | 1.000 | 2/6 | 0.688 |
| `tr1` | 1/6 (arm5) | 0.219 | 2/6 | 0.688 |
| `combab` | 1/6 (arm5) | 0.219 | 4/6 | 0.688 |

## 7. Backbone loss and attention logit magnitude of the highest cell at each backbone step

![Backbone loss and attention logit magnitude of the highest cell at each backbone step](plots/per_run_loss.png)

*The three cells below are selected by one rule: the highest GM-Relative MASE of the field at backbone 40k, at 100k and at 200k (`results/gm_relative_mase.csv`). The selected run in colour, the other four settings of the same recipe in grey (`results/training_curves/`, `results/attn_amplitude/`). Lower panels: `qk_logit_maxabs` averaged over layers and blocks; the table column is the maximum, not this mean.*

*Each loss value is the mean over the first, or the last, 5% of the logged steps of that window, not the value at a single step (`results/anomaly_windows.csv`). **Rise-from-minimum** is the run's final loss minus the lowest loss it ever recorded. **Peak qk logit** is `qk_logit_maxabs`, the peak absolute pre-softmax attention logit, logged every 200 steps; the value is the maximum over layers, over the `enc` and `fcst` blocks, and over the run. Rank 1 = lowest of the 40 backbones. No non-finite loss or gap appears in any of the 40 (`results/anomaly_inspection.csv`).*

| Cell | GM-Rel MASE | loss, start of 0–40k | end of 0–40k | end of 40–100k | end of 100–200k | peak qk logit | rank of 40 | rise-from-min | rank of 40 |
|---|---|---|---|---|---|---|---|---|---|
| `arm1 combab` | 3.1251 @ 40k | 19.03 | 19.82 | 19.54 | 19.75 | 6.51 | 1 | 4.4648 | 40 |
| `arm5 nse` ⟲ | 1.8887 @ 200k | 19.80 | 14.11 | 13.95 | 13.97 | 259.94 | 40 | 0.0688 | 7 |
| `arm6_v2 base` ⟲ | 1.9057 @ 100k | 12.98 | 4.92 | 4.92 | — | 12.75 | 7 | 0.2342 | 16 |

*Over 0k–40k the `arm1 combab` loss window mean rises by +0.7939, against −3.10 to −7.09 for every other `arm1` setting over the same window.*

## 8. Protocol

### Loss recipes

| Recipe | Loss |
|---|---|
| `arm1` | `L_pred + L_rep` |
| `arm3` | `L_pred_moco + L_rep` |
| `arm4` | `pooled + MoCo` |
| `arm5` | `L_align + L_rep` ⟲ |
| `arm6_v2` | `L_align + L_rep_moco` ⟲ |
| `bimoco` | `L_pred_moco + L_rep_moco` |

### Settings applied to each recipe

| Setting | Change from base |
|---|---|
| `base` | `τ = 0.10`, `cpc = 1`, `sigreg_e = 1` |
| `tr1` | all InfoNCE temperatures set to `τ = 1.0` |
| `nse` | SIGReg (the LeJEPA sketched isotropic-Gaussian regulariser) on `e_t` disabled (`sigreg_e = 0`) |
| `ncpc` | CPC auxiliary disabled (`cpc = 0`) |
| `combab` | `τ = 1.0` and `cpc = 0`; additionally `sigreg_e = 0` for `arm1`/`arm3`/`arm4` |

### Metrics

- **GM-Relative MASE** — geometric mean over 97 GIFT-Eval configs of `MASE(model) / MASE(seasonal_naive)`, official GIFT-Eval **B4 strategy** (single-window in-context prediction, backbone context length matching the config's expected horizon), forecast horizon 16. Lower is better; above 1.0 means seasonal naive wins on that geometric average. All 30 cells share one seasonal-naive denominator file, byte-identical to the earlier sweep's, so all of them sit on one scale.
- **`h_t`, `e_t`** — the encoder-output latent and the patch-embedding latent, shape `[B, T, C, H]`.
- **`ff`** — mean `cos(f̂, h_{t+1})` between the forecaster's next-step prediction and the encoder's next-step latent, unit-normalised. `1 − ff` is a cosine distance in [0, 2]; smaller = closer forecast.
- **Latent drift** at checkpoint pair `(step_i, step_j)` — `mean_{b,t,c} 1 − cos(h_t(model_j), h_t(model_i))` on a fixed held-out batch (`torch.manual_seed(20260722)`, `B=8`, `T=4096`, `C=1`, ARMA-synthetic), and the same for `e_t`.
- **`u_batchtime`** — `1/(d · off-diagonal Gram mean)` over `(B×T)` samples of the given latent, clamped to [0, 1]. 1 = every `H` dimension carries independent information.
- **`L_align`** — `(2 − 2·cos(f_t, h_target_{t+1})).mean()`, pulling the forecaster output toward the next encoder latent, gradient through the forecaster only. `--align-target student` takes `h_target` from the student encoder under stop-gradient; `--align-target teacher` takes it from the EMA teacher. The ten `⟲` cells are the teacher setting.

### Training

Backbone: `d_model=64, n_heads=8, num_encoder_layers=3, num_layers=3, batch_size=64, seed=20260520`, dataset `jeremycochoy/gift-pretrain-full-4096 / small_v1`, EMA teacher throughout (`--ema-embedding --ema-encoder --ema-tau 0.9`). The 200k backbones continue the same run from its 100k checkpoint with the saved optimizer state, same seed and same flags.

Quantile head, trained on the frozen backbone, `--grad-clip 1.0` at every horizon, kept for comparability with the earlier sweep:

| Param | bb 40k | bb 100k | bb 200k |
|---|---|---|---|
| `--head-arch` | `transformer` | `transformer` | `transformer` |
| `--head-num-layers` | 2 | 2 | 2 |
| `--head-nhead` | 8 | 8 | 8 |
| `--head-ffn-mult` | 4.0 | 4.0 | 4.0 |
| `--head-causal` | true | true | true |
| `--head-train-input` | `e_then_f` | `e_then_f` | `e_then_f` |
| `--forecast-len` | 16 | 16 | 16 |
| `--batch-size` | 256 | 256 | 256 |
| `--lr` | 1e-3 | 1e-3 | 1e-3 |
| `--total-steps` | 15,000 | 30,000 | 30,000 |
| head seed | 20260722 | 20260722 | 20260722 |

### What a comparison here can and cannot carry

- The head budget changes with the backbone step (15k at 40k, 30k at 100k and 200k), so any within-arm statement across backbone steps moves two things at once.
- The 200k row compares a teacher-selected subset against a differently selected subset of the earlier sweep, so no like-for-like 200k claim is available.
- Section 5 changes the flag and the code snapshot together.

### Training-curve diagnostics

![1 − ff against training step, one panel per setting](plots/cos_error_per_arm.png)

*`1 − ff`, the cosine distance between the forecaster's next-step output and the encoder's next-step latent, per training step (`results/training_curves/`). Shared y-scale across panels.*

![u_batchtime against training step, one panel per cell](plots/dim_usage_per_arm.png)

*`u_batchtime` on `h_t`, the dimension-usage measure defined above, per training step (`results/training_curves/`). Per-panel y-scale.*

## 9. Data

Every CSV behind this report is in [`results/`](results/), documented in [`results/README.md`](results/README.md). The plot scripts are in [`plots/`](plots/); the analysis scripts are in `experiments/2026-08-01_lalign_teacher/scripts/`. The pre-teacher measurements of the same ten cells are in [`reports/2026-07-21_split_pred_rep_small/small_long.md`](../2026-07-21_split_pred_rep_small/small_long.md).
