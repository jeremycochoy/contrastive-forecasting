# LeJEPA spherical regulariser at half batch size

Does adding a spherical regulariser on the patch-embed and on the encoder output, with batch cut from 1024 to 512, keep the GIFT-Eval full-97 score within noise of the B=1024 reference and fill the latent sphere? GM-Rel MASE is within ≤0.006 of the EMA-target B=1024 reference in every cell (SIGReg below it in all four: Δ = −0.0004, −0.0059, −0.0033, −0.0041 at 2L/best, 2L/last, 6L/best, 6L/last). `h_t` fills the sphere to the same extent as the EMA-target reference (`u_batch` ends at 0.80 vs 0.82, `u_temporal` at 0.62 vs 0.57); `e_t` stays at ~17× `1/K` (`u_batch_e` 0.0438, `u_temporal_e` 0.0315).

## Result

![GM-Rel MASE on the GIFT-Eval full-97 benchmark, four (head-depth, backbone-checkpoint) cells per arm](plots/gm_rel_mase.png)

GM-Rel MASE (lower = better; 1.0 = seasonal-naive parity):

| head / checkpoint | enc3+CPC, B=1024 | EMA-target enc3+CPC, B=1024 | SIGReg + EMA-target, B=512 |
| --- | ---: | ---: | ---: |
| 2L / best | 1.1846 | 1.1614 | 1.1610 |
| 2L / last | 1.1531 | 1.1817 | 1.1758 |
| 6L / best | 1.1584 | 1.1576 | 1.1543 |
| 6L / last | 1.1436 | 1.1597 | 1.1556 |

GM-MASE (geometric mean of per-config `MASE[0.5]` across 97 configs; lower = better; not seasonal-naive-normalised):

| head / checkpoint | enc3+CPC, B=1024 | EMA-target enc3+CPC, B=1024 | SIGReg + EMA-target, B=512 |
| --- | ---: | ---: | ---: |
| 2L / best | 1.6559 | 1.6235 | 1.6229 |
| 2L / last | 1.6119 | 1.6519 | 1.6436 |
| 6L / best | 1.6193 | 1.6182 | 1.6135 |
| 6L / last | 1.5986 | 1.6211 | 1.6154 |

**Metric scope.** GM-Rel MASE and GM-MASE are both available from the wrapper artefacts; GM-MAPE_SN and GM-CRPS_SN are not, since the per-config seasonal-naive denominators for MAPE and CRPS are not written to `all_results.csv` (annex C).

**Reference-values provenance.** The enc3+CPC and EMA-target columns are hard-coded from prior arms (constants `REF_GM` / `EMA_GM` / `REF_GM_MASE` / `EMA_GM_MASE` in [`build_report.py`](../../experiments/2026-06-20_lejepa_sigreg/scripts/build_report.py)). The SIGReg column is freshly evaluated at this code revision and HF cache snapshot — GM-MASE is the geometric mean of `eval_metrics/MASE[0.5]` over the 97 configs in each `gift_eval_full_<tag>{,_last}_{2L,6L}/all_results.csv`.

## Per-domain split

![Per-domain GM-Rel MASE on GIFT-Eval full-97, 2-layer head (left) and 6-layer head (right), three arms × {best, last}](plots/perdomain_radar.png)

## Dimension usage

![Cross-batch (left) and cross-time (right) uniformity over training; h_t solid vs e_t dashed for the SIGReg arm, h_t overlays for the two reference arms](plots/uniformity.png)

## Training loss

![Training loss, 50-step rolling mean, three arms overlaid](plots/loss_curve.png)

## SIGReg term magnitudes

![SIGReg term trajectories on log scale (upper) and their ratio to total loss (lower), 50-step rolling means](plots/sigreg_e_inspection.png)

## Protocol

One arm, single seed `20260520`, 12 500 steps (N=1; no replicates run for this report or for the two reference arms it compares against). Launcher: [`experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh`](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh).

Backbone: GRU patch-embed → 3-layer transformer encoder (`K`=384, 6 heads). The arm changes exactly three flags vs the EMA-target enc3+CPC reference (B=1024):

| flag | reference | this arm |
| --- | --- | --- |
| `--batch-size` | 1024 | 512 |
| `--sigreg-embedding` | OFF | ON |
| `--sigreg-encoding` | OFF | ON |

Other flags from the reference (`--ema-embedding --ema-encoder --ema-tau 0.99`, `--cpc-infonce-weight 1.0`, `--encoder-dropkey 0.70`, `--mix-ratio 0.0078125`, dataset, dtypes) are kept verbatim. `--sigreg-post-normalization` is OFF; `--sigreg-weight 0.1`, `--sigreg-m 1024`, `--sigreg-t-knots 17`.

### Head-matched downstream

Each backbone checkpoint (`best` = best train-loss, `last` = step 12 500) trains a 2-layer and a 6-layer quantile head, then evaluates on GIFT-Eval full-97 via `scripts/run_gift_eval_full.sh`. Per-cell summaries live at `results/gift_eval_full_<tag>{,_last}_{2L,6L}/summary.txt`.

## Annex

### A. Cross-arm plot provenance

SIGReg arm training CSV: `runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv` (12 500 rows, seed `20260520`). `loss_curve.png`, `uniformity.png`, and `sigreg_e_inspection.png` read from this CSV; `loss_curve.png` and `uniformity.png` additionally overlay:

- EMA-target arm: `experiments/2026-06-19_ema_target_encoder/runs/bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc_losses.csv` (steps 1 → 10 000) concatenated with `…_r2_losses.csv` (steps 10 001 → 12 500)
- enc3+CPC arm: `experiments/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv`

`gm_rel_mase.png` and `perdomain_radar.png` read each arm's `gift_eval_full_<tag>{,_last}_{2L,6L}/{summary.txt,all_results.csv}`; the radar takes the geometric mean of per-config `Relative MASE` within each of the 7 GIFT-Eval domains. Reference-arm CSVs and eval directories are taken from those arms' own code revisions; no fresh re-training was run for this report. Colour mapping (bar plot + radar): grey = enc3+CPC B=1024, blue = EMA-target enc3+CPC B=1024, red = this arm.

### B. Attribution — three axes move together

The arm changes three things vs the EMA-target B=1024 reference: SIGReg on `e_t`, SIGReg on `h_t`, batch 1024 → 512. The four-cell GM-Rel MASE table measures the joint perturbation. No-SIGReg B=512 and SIGReg B=1024 controls were not run.

### C. Metric availability from the wrapper artefacts

`scripts/run_gift_eval_full.sh` writes `Aggregate GM-Relative MASE (97 configs)` + per-config `Config / MASE / SN_MASE / Relative` to `summary.txt`, and per-config `eval_metrics/MASE[0.5]`, `eval_metrics/MAPE[0.5]`, `eval_metrics/mean_weighted_sum_quantile_loss`, `domain` to `all_results.csv`. GM-Rel MASE is the wrapper's headline aggregate. GM-MAPE_SN and GM-CRPS_SN need the per-config seasonal-naive denominators for MAPE and the seasonal-naive weighted quantile loss for CRPS, neither of which is written to either file.

### D. Trajectory of SIGReg terms and `e_t` / `h_t` dimensionality

50-step rolling means:

| step | `L_SIGReg(e_t)` | `L_SIGReg(h_t)` | `u_batch_e` | `u_batch` (`h_t`) | `u_temporal_e` | `u_temporal` (`h_t`) | `loss` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 1.76e-3 | 2.59e-3 | 0.0100 | 0.4072 | 0.0095 | 0.2332 | 3.13 |
| 500 | 1.36e-3 | 1.91e-3 | 0.0110 | 0.5376 | 0.0103 | 0.2895 | 2.99 |
| 1 000 | 7.54e-4 | 1.23e-3 | 0.0126 | 0.6173 | 0.0115 | 0.3422 | 2.88 |
| 2 000 | 6.41e-4 | 8.58e-4 | 0.0151 | 0.7188 | 0.0131 | 0.4391 | 3.07 |
| 5 000 | 9.37e-4 | 8.40e-4 | 0.0240 | 0.7862 | 0.0199 | 0.6083 | 4.50 |
| 7 500 | 9.95e-4 | 5.18e-4 | 0.0341 | 0.7920 | 0.0243 | 0.6202 | 4.54 |
| 10 000 | 9.69e-4 | 4.12e-4 | 0.0395 | 0.7938 | 0.0277 | 0.6197 | 4.43 |
| 12 500 | 1.01e-3 | 3.81e-4 | 0.0438 | 0.8016 | 0.0315 | 0.6184 | 4.24 |

`1/K` = 1/384 ≈ 0.00260.

## Vocabulary

| term | definition |
| --- | --- |
| `enc3` | 3-layer transformer encoder (hidden size `K`=384, 6 heads — the codebase's depth used here). |
| `CPC` | InfoNCE auxiliary head on the encoder, `--cpc-infonce-weight 1.0`. |
| **EMA-target** | exponential-moving-average teacher on the encoder + patch-embed, `--ema-tau 0.99`. |
| `e_t` | output of the GRU patch-embed, per (batch, time, channel) position; `K`=384. |
| `h_t` | output of the 3-layer transformer encoder (the codebase's `original_latent`), same shape. |
| **SIGReg** | LeJEPA-style spherical regulariser. Epps–Pulley test statistic averaged over `M`=1024 random unit-direction 1-D projections of the pooled latent, trapezoidal-integrated on `[−6/√K, 6/√K]` against `N(0, 1/K)`. Drives the pooled marginal toward `Unif(S^{K-1})`. Two terms here: `L_SIGReg(e_t)` (`--sigreg-embedding`) and `L_SIGReg(h_t)` (`--sigreg-encoding`), both pre-`F.normalize` (`--sigreg-post-normalization` OFF). Each weighted by `λ`=0.1. |
| `u_batch` | cross-batch dimensionality usage of `h_t`, clipped to `[1/K, 1]`. `1/K` ≈ 0.00260 = one direction; 1 = uniform sphere coverage. `u_batch_e` is the same statistic on `e_t`. |
| `u_temporal` | cross-time analogue of `u_batch`; `u_temporal_e` is the `e_t` version. |
| **GM-Rel MASE** | GIFT-Eval full-97 aggregate: geometric mean over 97 configs of (model MASE ÷ seasonal-naive MASE). Lower = better; 1.0 = seasonal-naive parity. |
