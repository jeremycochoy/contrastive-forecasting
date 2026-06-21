# LeJEPA spherical regulariser at half batch size

## Question

Does adding a spherical regulariser on the patch-embed and on the encoder output, with batch cut from 1024 to 512, keep the GIFT-Eval full-97 score in the same neighbourhood as the B=1024 reference and fill the latent sphere?

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

## Result

![GM-Rel MASE on the GIFT-Eval full-97 benchmark, four (head-depth, backbone-checkpoint) cells per arm](plots/gm_rel_mase.png)

| head / checkpoint | enc3+CPC, B=1024 | EMA-target enc3+CPC, B=1024 | SIGReg + EMA-target, B=512 |
| --- | ---: | ---: | ---: |
| 2L / best | 1.1846 | 1.1614 | 1.1610 |
| 2L / last | 1.1531 | 1.1817 | 1.1758 |
| 6L / best | 1.1584 | 1.1576 | 1.1543 |
| 6L / last | 1.1436 | 1.1597 | 1.1556 |

![Per-domain GM-Rel MASE on GIFT-Eval full-97, 2-layer head (left) and 6-layer head (right), three arms × {best, last}](plots/perdomain_radar.png)

The four-cell head-to-head does not separate the SIGReg + B=512 arm from the EMA-target B=1024 reference in either direction.

**Metric scope.** Wrapper emits only GM-Rel MASE; SN-relative MASE / MAPE / CRPS are not produced (annex C).

**Reference-values provenance.** The enc3+CPC and EMA-target columns reproduce prior arms' published head-matched tables at their own code revisions, embedded as constants `REF_GM` and `EMA_GM` in [`build_report.py`](../../experiments/2026-06-20_lejepa_sigreg/scripts/build_report.py); the SIGReg column is the only fresh head-matched eval at this code revision and HF cache snapshot.

### What the two SIGReg terms did

![SIGReg term trajectories on log scale (upper) and their ratio to total loss (lower), 50-step rolling means](plots/sigreg_e_inspection.png)

Mean over the last 50 of 12 500 steps:

| quantity | value |
| --- | ---: |
| total `loss` | 4.2478 |
| `λ · L_SIGReg(e_t)` | 1.001e-4 |
| `λ · L_SIGReg(h_t)` | 3.805e-5 |
| `λ · L_SIGReg(e_t) / loss` | 2.36e-5 |
| `λ · L_SIGReg(h_t) / loss` | 8.96e-6 |

### Sphere coverage

![Cross-batch (left) and cross-time (right) uniformity over training; h_t solid vs e_t dashed for the SIGReg arm, h_t overlays for the two reference arms](plots/uniformity.png)

## Protocol

One arm, seed `20260520`, 12 500 steps. Launcher: [`experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh`](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh).

Backbone: GRU patch-embed → 3-layer transformer encoder (`K`=384, 6 heads). The arm changes exactly three flags vs the EMA-target enc3+CPC reference (B=1024):

| flag | reference | this arm |
| --- | --- | --- |
| `--batch-size` | 1024 | 512 |
| `--sigreg-embedding` | OFF | ON |
| `--sigreg-encoding` | OFF | ON |

Other flags from the reference (`--ema-embedding --ema-encoder --ema-tau 0.99`, `--cpc-infonce-weight 1.0`, `--encoder-dropkey 0.70`, `--mix-ratio 0.0078125`, dataset, dtypes) are kept verbatim. `--sigreg-post-normalization` is OFF; `--sigreg-weight 0.1`, `--sigreg-m 1024`, `--sigreg-t-knots 17`.

### Head-matched downstream

Each backbone checkpoint (`best` = best train-loss, `last` = step 12 500) trains a 2-layer and a 6-layer quantile head, then evaluates on GIFT-Eval full-97 via `scripts/run_gift_eval_full.sh`. Per-cell summaries live at `results/gift_eval_full_<tag>{,_last}_{2L,6L}/summary.txt`.

### Training loss

![Training loss, 50-step rolling mean, three arms overlaid](plots/loss_curve.png)

## Annex

### A. Cross-arm plot provenance

All plots embed the SIGReg arm's training CSV `runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv` (12 500 rows, seed `20260520`). `loss_curve.png` and `uniformity.png` overlay:

- EMA-target arm: `experiments/2026-06-19_ema_target_encoder/runs/bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc_losses.csv`
- enc3+CPC arm: `experiments/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv`

`perdomain_radar.png` reads `summary.txt` + `all_results.csv` from each arm's `gift_eval_full_<tag>{,_last}_{2L,6L}/` directory, computes the geometric mean of per-config `Relative MASE` within each domain, and overlays the three arms × {best, last}. Both overlay CSVs and the EMA-target / enc3+CPC eval directories are taken from those arms' own code revisions; no fresh re-training was run for this report. Colour mapping (bar plot + radar): grey = enc3+CPC B=1024, blue = EMA-target enc3+CPC B=1024, red = this arm.

### B. Attribution — three axes move together

The arm changes three things vs the EMA-target B=1024 reference: SIGReg on `e_t`, SIGReg on `h_t`, batch 1024 → 512. The four-cell GM-Rel MASE table measures the joint perturbation. The issue spec asks for the single B=512 arm; no-SIGReg B=512 and SIGReg B=1024 controls were not run.

### C. GIFT-Eval wrapper emits only GM-Rel MASE

`scripts/run_gift_eval_full.sh` writes `Aggregate GM-Relative MASE (97 configs)` to each `summary.txt`. The per-config `all_results.csv` carries raw `MASE[0.5]`, `MAPE[0.5]`, and `mean_weighted_sum_quantile_loss`, but not the seasonal-naive denominators needed to form GM-MASE / GM-MAPE_SN / GM-CRPS_SN.

### D. Trajectory of SIGReg terms and `e_t` dimensionality

50-step rolling means:

| step | `L_SIGReg(e_t)` | `L_SIGReg(h_t)` | `u_batch_e` | `u_batch` (`h_t`) | `loss` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 1.76e-3 | 2.59e-3 | 0.0100 | 0.4069 | 3.13 |
| 500 | 1.36e-3 | 1.92e-3 | 0.0110 | 0.5379 | 2.99 |
| 1 000 | 7.55e-4 | 1.23e-3 | 0.0126 | 0.6173 | 2.88 |
| 2 000 | 6.41e-4 | 8.57e-4 | 0.0151 | 0.7190 | 3.07 |
| 5 000 | 9.38e-4 | 8.40e-4 | 0.0240 | 0.7867 | 4.50 |
| 7 500 | 9.96e-4 | 5.17e-4 | 0.0341 | 0.7925 | 4.54 |
| 10 000 | 9.70e-4 | 4.12e-4 | 0.0395 | 0.7923 | 4.43 |
| 12 500 | 1.01e-3 | 3.81e-4 | 0.0438 | 0.8020 | 4.24 |

`1/K` = 1/384 ≈ 0.00260. Final-step `u_temporal` = 0.6194, `u_temporal_e` = 0.0315.
