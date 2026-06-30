# LeJEPA spherical regulariser: EMA-target window sweep

## Question

Sweep the EMA-target window `τ ∈ {0.99, 0.98, 0.90, 0.80}` plus a no-EMA (τ=0) arm at B=512, head-matched on GIFT-Eval full-97 GM-Rel MASE.

## Result

![GM-Rel MASE on the GIFT-Eval full-97 benchmark, four (head-depth, backbone-checkpoint) cells per arm](plots/gm_rel_mase.png)

| head / checkpoint | enc3+CPC, B=1024 | EMA-target enc3+CPC, B=1024, τ=0.99 | SIGReg + EMA-target, B=512, τ=0.99 | SIGReg + EMA-target, B=512, τ=0.98 | SIGReg + EMA-target, B=512, τ=0.90 | SIGReg + EMA-target, B=512, τ=0.80 | SIGReg (no EMA-target), B=512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2L / best | 1.1846 | 1.1614 | 1.1610 | 1.1807 ⁂ | **1.1569** | 1.1904 | 1.2119 |
| 2L / last | 1.1531 | 1.1817 | 1.1758 | 1.1662 | **1.1624** | 1.1667 | 1.2105 |
| 6L / best | 1.1584 | 1.1576 | 1.1543 | 1.1493 ⁂ | **1.1489** | 1.1867 | 1.2081 |
| 6L / last | 1.1436 | 1.1597 | 1.1556 | 1.1462 | **1.1373** | 1.1470 | 1.1999 |

Bold = column minimum within B=512 (the experimental axis); reference-row cells are never bolded. ⁂ τ=0.98 `*_best` cells use the `_10k.pth` periodic save (no `best_loss.pth` tracker), so cross-arm `*_best` deltas in that column are proxies.

![τ-sweep: mean GM-Rel MASE over the (2L/last, 6L/last) cells; no-EMA (τ=0) at left](plots/tau_sweep_last_avg.png)

### Sphere coverage

![Cross-batch and cross-time uniformity; top row h_t (shared 0–1), bottom row e_t (auto-scaled); five SIGReg + EMA-target arms (including no-EMA at τ=0) plus two reference overlays for h_t](plots/uniformity.png)

### Training loss

![Training loss, 50-step rolling mean, five SIGReg arms plus two B=1024 references overlaid](plots/loss_curve.png)

### SIGReg term trajectories

![SIGReg term trajectories on log scale (upper) and their ratio to total loss (lower), 50-step rolling means — τ=0.90 arm shown; other arms in `plots/sigreg_e_inspection*.png`](plots/sigreg_e_inspection_tau090.png)

| tail-50 mean at step | τ=0.99 (12 500) | τ=0.98 (12 400) | τ=0.90 (12 500) | τ=0.80 (12 500) | no-EMA (12 500) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `u_batch` (h_t) | 0.7964 | 0.7626 | 0.7537 | 0.7712 | 0.1491 |
| `u_batch_e` (e_t) | 0.0438 | 0.0333 | 0.0212 | 0.0287 | 0.00433 |
| `u_temporal` (h_t) | 0.6194 | 0.5735 | 0.6062 | 0.6451 | 0.1399 |
| `u_temporal_e` (e_t) | 0.0315 | 0.0251 | 0.0181 | 0.0238 | 0.00418 |
| `L_SIGReg(e_t)` | 1.001e-3 | 1.251e-3 | 1.989e-3 | 1.697e-3 | 5.68e-7 |
| `L_SIGReg(h_t)` | 3.80e-4 | 6.04e-4 | 4.58e-4 | 4.84e-4 | 7.81e-5 |
| total `loss` | 4.248 | 4.083 | 3.830 | 3.893 | 0.867 |

τ=0.90 is the column minimum among the swept B=512 arms; `*_last` deltas vs τ=0.99 exceed the ~0.01 seed band, `*_best` deltas sit inside.

## Protocol

Per-arm launchers: [τ=0.98](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh), [τ=0.90](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg_tau090.sh), [τ=0.80](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg_tau080.sh), [no-EMA](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg_noema.sh). Seed `20260520`, target 12 500 steps, dataset `gift-pretrain-full-4096` / `small_v1`. Eval wrapper emits only GM-Rel MASE (annex D); ~0.01 seed-noise band (annex F).

## Annex

### A. Per-arm head-matched cells vs the τ=0.99 reference

| head / checkpoint | τ=0.99 | τ=0.98 | Δ | τ=0.90 | Δ | τ=0.80 | Δ | no-EMA | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2L / best | 1.1610 | 1.1807 ⁂ | — | 1.1569 | −0.0041 | 1.1904 | +0.0294 | 1.2119 | +0.0509 |
| 2L / last | 1.1758 | 1.1662 | −0.0096 | 1.1624 | −0.0134 | 1.1667 | −0.0091 | 1.2105 | +0.0347 |
| 6L / best | 1.1543 | 1.1493 ⁂ | — | 1.1489 | −0.0054 | 1.1867 | +0.0324 | 1.2081 | +0.0538 |
| 6L / last | 1.1556 | 1.1462 | −0.0094 | 1.1373 | −0.0183 | 1.1470 | −0.0086 | 1.1999 | +0.0443 |

### B. Plot provenance

Legend colours, source CSVs, and the `gm_rel_mase` / `tau_sweep` / overlay-plot inputs are pinned in [`build_report_tau098.py`](../../experiments/2026-06-20_lejepa_sigreg/scripts/build_report_tau098.py) and [`plot_tau_sweep_last_avg.py`](../../experiments/2026-06-20_lejepa_sigreg/scripts/plot_tau_sweep_last_avg.py).

### C. Reference-values provenance

`enc3+CPC`, `EMA-target enc3+CPC, τ=0.99`, `SIGReg + EMA-target, τ=0.99` columns are imported from `experiments/2026-06-13_cpc_infonce_aux/`, `experiments/2026-06-19_ema_target_encoder/`, `reports/2026-06-20_lejepa_sigreg/` (pinned in `build_report_tau098.py`).

### D. GIFT-Eval wrapper emits only GM-Rel MASE

`scripts/run_gift_eval_full.sh` emits `Aggregate GM-Relative MASE (97 configs)` only; seasonal-naive denominators for GM-MASE / GM-MAPE_SN / GM-CRPS_SN are not produced.

### E. Trajectory of SIGReg terms and dimensionality (τ=0.98 arm)

50-step rolling means; `1/K` ≈ 0.00260.

| step | `L_SIGReg(e_t)` | `L_SIGReg(h_t)` | `u_batch_e` | `u_batch` (`h_t`) | `u_temporal_e` | `u_temporal` (`h_t`) | `loss` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 1.74e-3 | 2.74e-3 | 0.0102 | 0.366 | 0.0096 | 0.214 | 3.50 |
| 500 | 1.25e-3 | 1.85e-3 | 0.0114 | 0.517 | 0.0105 | 0.297 | 3.09 |
| 1 000 | 8.56e-4 | 1.26e-3 | 0.0131 | 0.613 | 0.0117 | 0.383 | 3.00 |
| 2 000 | 8.13e-4 | 1.00e-3 | 0.0141 | 0.721 | 0.0124 | 0.464 | 3.79 |
| 5 000 | 1.06e-3 | 6.99e-4 | 0.0256 | 0.783 | 0.0201 | 0.618 | 4.48 |
| 7 500 | 1.18e-3 | 6.01e-4 | 0.0302 | 0.765 | 0.0221 | 0.589 | 4.46 |
| 10 000 | 1.16e-3 | 5.64e-4 | 0.0317 | 0.773 | 0.0233 | 0.582 | 4.27 |
| 12 400 (tail-50) | 1.25e-3 | 6.04e-4 | 0.0333 | 0.763 | 0.0251 | 0.574 | 4.08 |

Source: `results/trajectory_table.csv`, `results/final_trajectories.txt`.

### F. Seed-noise band

`experiments/2026-05-08_exp_tau_sweep` paired re-runs: ~0.01 GM-Rel MASE band. Each arm here is one seed.

### G. Vocabulary

| term | definition |
| --- | --- |
| `enc3` | 3-layer transformer encoder (hidden size `K`=384, 6 heads). |
| `CPC` | InfoNCE auxiliary head on the encoder, `--cpc-infonce-weight 1.0`. |
| **EMA-target** | exponential-moving-average teacher on the encoder + patch-embed; `--ema-tau τ`, half-life ≈ ln(2)/(1−τ) steps. `τ=0.99` ≈69, `τ=0.98` ≈34, `τ=0.90` ≈7, `τ=0.80` ≈3. The no-EMA (τ=0) arm passes neither `--ema-embedding`, `--ema-encoder`, nor `--ema-tau`. |
| `e_t` | output of the GRU patch-embed, per (batch, time, channel) position; `K`=384. |
| `h_t` | output of the 3-layer transformer encoder (the codebase's `original_latent`), same shape. |
| **SIGReg** | LeJEPA-style spherical regulariser. Epps–Pulley test statistic averaged over `M`=1024 random unit-direction 1-D projections of the pooled latent, trapezoidal-integrated on `[−6/√K, 6/√K]` against `N(0, 1/K)`. Two terms: `L_SIGReg(e_t)` (`--sigreg-embedding`) and `L_SIGReg(h_t)` (`--sigreg-encoding`), both pre-`F.normalize` (`--sigreg-post-normalization` OFF), each weighted by `λ`=0.1. |
| `u_batch` | cross-batch dimensionality usage of `h_t`, clipped to `[1/K, 1]`. `1/K` ≈ 0.00260 = one direction; 1 = uniform sphere coverage. `u_batch_e` is the same on `e_t`. |
| `u_temporal` | cross-time analogue; `u_temporal_e` is `e_t`. |
| **GM-Rel MASE** | GIFT-Eval full-97 aggregate: geometric mean over 97 configs of (model MASE ÷ seasonal-naive MASE). Lower = better; 1.0 = seasonal-naive parity. |
| `*_best` | head trained on the lowest-train-loss backbone checkpoint of the arm. |
| `*_last` | head trained on the final-step backbone checkpoint. |
