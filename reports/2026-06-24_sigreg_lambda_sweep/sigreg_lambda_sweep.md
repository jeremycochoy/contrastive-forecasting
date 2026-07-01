# SIGReg λ-sweep — embedding-side weight pushed further, encoding-side weight varied

## Question

Sweep `(λ_e, λ_h)` on the SIGReg + EMA-target, B=512 recipe (six arms on a log grid), head-matched on GIFT-Eval full-97 GM-Rel MASE.

## Result

![GM-Rel MASE on the GIFT-Eval full-97 benchmark, four (q-head depth, backbone checkpoint) cells per arm; sweep-arm bars carry paired-bootstrap 95% CI whiskers vs the `λ_e=1.0, λ_h=0.1` anchor; horizontal lines = the four anchors](plots/gm_rel_mase.png)

| head / ckpt | `enc3+CPC`, B=1024 | `EMA enc3+CPC`, B=1024 | `λ_e=λ_h=0.1`, B=512 | `λ_e=1.0, λ_h=0.1`, B=512 | arm 1 (10.0, 0.1) | arm 2 (10.0, 1.0) | arm 3 (10.0, 10.0) | arm 4 (1.0, 1.0) | arm 5 (100.0, 0.1) | arm 6 (1000.0, 1.0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2L / best | 1.1846 | 1.1614 | 1.1610 | 1.1470 | 1.1474 | **1.1302** | 1.1540 | 1.1435 | 1.1554 | 1.1488 |
| 2L / last | 1.1531 | 1.1817 | 1.1758 | 1.1681 | 1.1610 | 1.1756 | 1.1682 | 1.1679 | 1.1828 | **1.1537** |
| 6L / best | 1.1584 | 1.1576 | 1.1543 | 1.1408 | 1.1447 | **1.1294** | 1.1465 | 1.1449 | 1.1462 | 1.1397 |
| 6L / last | 1.1436 | 1.1597 | 1.1556 | 1.1482 | 1.1473 | 1.1515 | 1.1538 | 1.1517 | 1.1682 | **1.1415** |

Bold = column minimum within the B=512 sweep family; B=1024 anchor cells are never bolded. Arm 2 (`λ_e=10.0, λ_h=1.0`) holds both `*_best` column minima; only its 2L/best Δ vs the `λ_e=1.0, λ_h=0.1` anchor has a paired-bootstrap 95% CI excluding zero (annex E). Arm 6 (`λ_e=1000.0, λ_h=1.0`) holds both `*_last` column minima but within the ~0.01 seed-noise band (annex K).

### Sphere coverage

![(1 − U) on `h_t` per pooling axis (`u_batch`, `u_temporal`, `u_batchtime`); log-y `[0.05, 1]`; `u_batchtime` panel: dotted line + `●` = retroactive per-checkpoint trajectory (arms 4 and 6 absent), `★` at the best-loss step (= `FINAL.pth`); HIGH on the plot = closer to rank-1 collapse (BAD), LOW = closer to isotropic (GOOD, all K dims used) — inverted from the U convention](plots/dim_usage_h.png)

![(1 − U) on `e_t` per pooling axis (`u_batch_e`, `u_temporal_e`, `u_batchtime_e`); log-y `[0.9, 1.0]`; `u_batchtime_e` panel: dotted line + `●` = retroactive per-checkpoint trajectory (arms 4 and 6 absent), `★` at the best-loss step (= `FINAL.pth`); HIGH on the plot = closer to rank-1 collapse (BAD), LOW = closer to isotropic (GOOD, all K dims used) — inverted from the U convention](plots/dim_usage_e.png)

### Training loss

![Total training loss, log-y, 50-step rolling mean from step 100 onwards; six sweep arms plus the two prior `λ_h=0.1` SIGReg anchors overlaid](plots/loss_curve.png)

### SIGReg term trajectories

![SIGReg-term and embedding-side dim-usage trajectories from step 100 onwards: `L_SIGReg(e_t)`, `L_SIGReg(h_t)`, `u_batch_e`, `u_temporal_e`; log-y, 50-step rolling mean; bottom-row dotted line = `1/K ≈ 0.0026` floor](plots/sigreg_e_inspection.png)

| tail-50 mean at step 12 500 | arm 1 (10.0, 0.1) | arm 2 (10.0, 1.0) | arm 3 (10.0, 10.0) | arm 4 (1.0, 1.0) | arm 5 (100.0, 0.1) | arm 6 (1000.0, 1.0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L_SIGReg(e_t)` | 1.90e-4 | 4.49e-4 | 4.46e-4 | 8.49e-4 | 7.10e-6 | 4.50e-6 |
| `L_SIGReg(h_t)` | 5.07e-4 | 3.31e-4 | 3.20e-4 | 3.41e-4 | 9.61e-5 | 1.64e-4 |
| `u_batch_e` | 0.0190 | 0.0506 | 0.0588 | 0.0363 | 0.0177 | 0.0318 |
| `u_temporal_e` | 0.0164 | 0.0340 | 0.0400 | 0.0259 | 0.0160 | 0.0267 |
| `u_batch` (`h_t`) | 0.7734 | 0.7964 | 0.7853 | 0.7822 | 0.6161 | 0.5930 |
| `u_temporal` (`h_t`) | 0.4825 | 0.6291 | 0.6138 | 0.5977 | 0.2705 | 0.2562 |
| total `loss` | 4.203 | 4.550 | 4.256 | 4.467 | 3.878 | 3.636 |

## Protocol

Per-arm launcher: [`scripts/train_backbone_sigreg.sh`](../../experiments/2026-06-24_sigreg_lambda_sweep/scripts/train_backbone_sigreg.sh). Seed `20260520`, 12 500 steps, dataset `gift-pretrain-full-4096` / `small_v1`. Only `λ_e` and `λ_h` change across arms; all other flags identical to the `λ_e=1.0, λ_h=0.1` anchor. The issue specified arms 1–3 plus arm 4 as an optional interior point; arms 5 and 6 extend the `λ_e` axis. Each arm produces two backbone checkpoints (`best` = best train-loss, `last` = step 12 500); each backbone trains a 2-layer and a 6-layer quantile head, evaluated on GIFT-Eval full-97 via `scripts/run_gift_eval_full.sh`. Eval wrapper emits only GM-Rel MASE (annex J); ~0.01 GM-Rel MASE seed-noise band (annex K).

| flag | arm 1 | arm 2 | arm 3 | arm 4 | arm 5 | arm 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `--sigreg-embedding-weight` (`λ_e`) | 10.0 | 10.0 | 10.0 | 1.0 | 100.0 | 1000.0 |
| `--sigreg-encoding-weight` (`λ_h`) | 0.1 | 1.0 | 10.0 | 1.0 | 0.1 | 1.0 |

## Annex

### A. Anchors

| label | recipe |
| --- | --- |
| `enc3+CPC, B=1024` (#344) | non-SIGReg, non-EMA baseline |
| `EMA enc3+CPC, B=1024` (#353) | EMA-target only, no SIGReg |
| `SIGReg λ_e=λ_h=0.1, B=512` (#355) | `λ_e=λ_h=0.1` (per-config rel-MASE re-read from `reports/2026-06-20_lejepa_sigreg/`) |
| `SIGReg λ_e=1.0, λ_h=0.1, B=512` (#359) | the `λ_e=1.0, λ_h=0.1` anchor (per-config rel-MASE re-read from `reports/2026-06-22_lejepa_sigreg_emb10/`) |

### B. (λ_e, λ_h) heatmap

![4-panel GM-Rel MASE heatmap over (λ_e, λ_h), one panel per (q-head depth, backbone checkpoint) cell; log axes; diverging colormap centred on the per-cell `λ_e=1.0, λ_h=0.1` anchor (red = worse, blue = better); hatched tiles = points not run](plots/heatmap.png)

### C. λ_e ladders

![Two λ_e ladders. Left: λ_h=0.1, λ_e ∈ {0.1, 1.0, 10.0 (arm 1), 100.0 (arm 5)}. Right: λ_h=1.0, λ_e ∈ {1.0 (arm 4), 10.0 (arm 2), 1000.0 (arm 6)}. Log axes; 4 curves per panel (2L/6L × best/last); shaded bands = paired-bootstrap 95% CI vs the `λ_e=1.0, λ_h=0.1` anchor](plots/lambda_e_ladder.png)

### D. Best-vs-last drift

![Drift = last − best GM-Rel MASE per arm, split by 2L vs 6L q-head; positive = `last` worse than `best`, negative = still improving at step 12 500](plots/best_vs_last_drift.png)

### E. Δ vs the `λ_e=1.0, λ_h=0.1` anchor with paired-bootstrap 95% CI

B=10 000 draws, n=97 configs, paired on per-config rel-MASE; Δ on absolute GM-Rel MASE scale via `GM_anchor · (exp(quantile) − 1)`. Bold = 95% CI excludes zero.

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --- | --- | --- | --- |
| arm 1 (10.0, 0.1) | +0.0004 `[−0.0081, +0.0106]` | −0.0071 `[−0.0230, +0.0087]` | +0.0039 `[−0.0036, +0.0130]` | −0.0010 `[−0.0156, +0.0175]` |
| arm 2 (10.0, 1.0) | **−0.0168 `[−0.0293, −0.0045]`** | +0.0075 `[−0.0061, +0.0220]` | −0.0114 `[−0.0258, +0.0067]` | +0.0032 `[−0.0110, +0.0185]` |
| arm 3 (10.0, 10.0) | +0.0070 `[−0.0035, +0.0193]` | +0.0002 `[−0.0143, +0.0142]` | +0.0057 `[−0.0042, +0.0173]` | +0.0056 `[−0.0105, +0.0221]` |
| arm 4 (1.0, 1.0) | −0.0035 `[−0.0113, +0.0048]` | −0.0002 `[−0.0145, +0.0155]` | +0.0042 `[−0.0038, +0.0134]` | +0.0035 `[−0.0114, +0.0213]` |
| arm 5 (100.0, 0.1) | +0.0084 `[−0.0010, +0.0188]` | +0.0148 `[−0.0055, +0.0363]` | +0.0055 `[−0.0034, +0.0147]` | +0.0199 `[−0.0045, +0.0458]` |
| arm 6 (1000.0, 1.0) | +0.0018 `[−0.0144, +0.0204]` | −0.0144 `[−0.0299, +0.0013]` | −0.0011 `[−0.0151, +0.0129]` | −0.0067 `[−0.0254, +0.0146]` |

### F. Δ vs arm 1

Arm 1 = `λ_e=10.0, λ_h=0.1`. Vs arm 1, the single-axis arms are 2 (`λ_h`: 0.1 → 1.0), 3 (`λ_h`: 0.1 → 10.0), 5 (`λ_e`: 10.0 → 100.0); arms 4 (`λ_e` 10.0 → 1.0 *and* `λ_h` 0.1 → 1.0) and 6 (`λ_e` 10.0 → 1000.0 *and* `λ_h` 0.1 → 1.0) move both axes. Bold = 95% CI excludes zero.

| arm | 2L / best | 2L / last | 6L / best | 6L / last |
| --- | --- | --- | --- | --- |
| arm 2 (10.0, 1.0) | **−0.0172 `[−0.0291, −0.0062]`** | +0.0146 `[−0.0009, +0.0306]` | **−0.0153 `[−0.0280, −0.0018]`** | +0.0042 `[−0.0121, +0.0207]` |
| arm 3 (10.0, 10.0) | +0.0066 `[−0.0065, +0.0200]` | +0.0072 `[−0.0072, +0.0215]` | +0.0018 `[−0.0073, +0.0106]` | +0.0066 `[−0.0089, +0.0219]` |
| arm 4 (1.0, 1.0) | −0.0039 `[−0.0156, +0.0068]` | +0.0069 `[−0.0061, +0.0201]` | +0.0003 `[−0.0081, +0.0095]` | +0.0045 `[−0.0099, +0.0202]` |
| arm 5 (100.0, 0.1) | +0.0080 `[−0.0022, +0.0179]` | **+0.0218 `[+0.0047, +0.0410]`** | +0.0016 `[−0.0100, +0.0107]` | **+0.0209 `[+0.0026, +0.0411]`** |
| arm 6 (1000.0, 1.0) | +0.0014 `[−0.0125, +0.0150]` | −0.0073 `[−0.0215, +0.0091]` | −0.0050 `[−0.0188, +0.0081]` | −0.0057 `[−0.0160, +0.0048]` |

### G. `u_batchtime` per-checkpoint retroactive table

`u_batchtime` is not in the losses CSVs; values come from saved backbone checkpoints over a fixed batch (B=512, seed 20260520). `FINAL.pth` = the best-train-loss checkpoint, copied byte-identical at training end; below = single-step retro on that snapshot. Retro set covers arms 1/2/3/5 plus the two SIGReg anchors; arms 4 and 6 are not in it.

| arm / anchor | recipe | `u_batchtime` (`h_t`) | `u_batchtime_e` (`e_t`) |
| --- | --- | ---: | ---: |
| `λ_e=0.1, λ_h=0.1` (#355) | `λ_e=0.1, λ_h=0.1` | 0.3897 | 0.0136 |
| `λ_e=1.0, λ_h=0.1` (#359) | `λ_e=1.0, λ_h=0.1` | 0.3535 | 0.0133 |
| arm 1 | `λ_e=10.0, λ_h=0.1` | 0.3527 | 0.0132 |
| arm 2 | `λ_e=10.0, λ_h=1.0` | 0.3535 | 0.0131 |
| arm 3 | `λ_e=10.0, λ_h=10.0` | 0.3663 | 0.0137 |
| arm 5 | `λ_e=100.0, λ_h=0.1` | 0.3144 | 0.0154 |

Sources: [`results/u_batchtime_retro.csv`](results/u_batchtime_retro.csv), [`results/u_batchtime_trajectory.csv`](results/u_batchtime_trajectory.csv); `scripts/compute_u_batchtime_retro.py`, `compute_u_batchtime_trajectory.py`.

### H. Plot and CI provenance

- **Training CSVs.** `experiments/2026-06-24_sigreg_lambda_sweep/runs/bb_<tag>_<arm>_losses.csv` (12 500 rows each).
- **Per-config rel-MASE.** `experiments/2026-06-24_sigreg_lambda_sweep/results/gift_eval_full_<tag>_<arm>{,_last}_{2L,6L}/summary.txt`.
- **CI computation.** [`scripts/compute_bootstrap.py`](scripts/compute_bootstrap.py) → `results/bootstrap_ci_vs_359.csv`, `results/bootstrap_ci_vs_arm1.csv`.
- **Plot scripts.** [`scripts/build_plots.py`](scripts/build_plots.py), [`scripts/build_heatmap.py`](scripts/build_heatmap.py). All trajectory panels pin `PLOT_START_STEP = 100` via `ax.set_xlim(100, 12500)`.
- **Bar-chart colour map.** grey = `enc3+CPC, B=1024`; blue = `EMA enc3+CPC, B=1024`; red = `λ_e=λ_h=0.1`; green = `λ_e=1.0, λ_h=0.1`; purple/brown/pink/cyan/olive/orange = arms 1/2/3/4/5/6.

### I. Reference-values provenance

Anchor GM-Rel MASE values are transcribed from `reports/2026-06-22_lejepa_sigreg_emb10/results/gm_table.csv`. The `λ_e=λ_h=0.1` anchor's per-config rel-MASE is re-read from `reports/2026-06-20_lejepa_sigreg/`; the `λ_e=1.0, λ_h=0.1` anchor's from `reports/2026-06-22_lejepa_sigreg_emb10/`.

### J. GIFT-Eval wrapper emits only GM-Rel MASE

`scripts/run_gift_eval_full.sh` emits `Aggregate GM-Relative MASE (97 configs)` only; seasonal-naive denominators for GM-MASE / GM-MAPE_SN / GM-CRPS_SN are not produced.

### K. Seed-noise band

`experiments/2026-05-08_exp_tau_sweep` paired re-runs: ~0.01 GM-Rel MASE band. Each arm here is one seed; the §E/F paired-bootstrap CIs cover sampling variability across the 97 GIFT-Eval configs, not run-to-run seed variability.

### L. Vocabulary

| term | definition |
| --- | --- |
| `K` | latent dimensionality, 384. `1/K ≈ 0.00260`. |
| `enc3` | 3-layer transformer encoder (hidden size `K`, 6 heads). |
| `CPC` | InfoNCE auxiliary head on the encoder, `--cpc-infonce-weight 1.0`. |
| **EMA-target** | exponential-moving-average teacher on encoder + patch-embed, `--ema-tau 0.99`. |
| `e_t` | GRU patch-embed output at position (batch, time, channel); dimension `K`. |
| `h_t` | 3-layer transformer encoder output at the same position. |
| **SIGReg** | LeJEPA spherical regulariser: Epps–Pulley statistic over `M`=1024 random 1-D projections of the pooled latent, against `N(0, 1/K)`. Two terms: `L_SIGReg(e_t)` weighted by `λ_e`, `L_SIGReg(h_t)` by `λ_h`. |
| `U`, `u_*` | **dimension usage** of the latent: `U = 1 / (K · E[cos²(z_i, z_j)])`, clipped to `[1/K, 1]`. `1/K` = rank-1 collapse; **higher = more dims in use**; `K · U ≈` effective dims (`U = 0.79` at `K=384` ≈ 303). Pooling axes: cross-batch (`u_batch`), cross-time (`u_temporal`), cross-(batch × time) (`u_batchtime`, same `(B·T, K)` sample axis SIGReg uses), each on `h_t` (no suffix) and `e_t` (`_e` suffix). Plotted as `1 − U` on log y-scale in §Sphere coverage — the log axis reveals late-training distances from the isotropic ceiling. Math check: [`docs/u_metric_check.md`](../../docs/u_metric_check.md). |
| **GM-Rel MASE** | GIFT-Eval full-97 aggregate: geometric mean over 97 configs of (model MASE ÷ seasonal-naive MASE). Lower = better; 1.0 = seasonal-naive parity. |
| **best-ckpt / last-ckpt** | `best` = backbone at lowest-train-loss step; `last` = backbone at step 12 500. |
| **paired bootstrap** | resample 97 per-config rel-MASE values with replacement (B=10 000 draws, seed 20260624), statistic `mean(log(rel_arm) − log(rel_baseline))`, 2.5/97.5 quantiles, back to GM scale via `GM_baseline · (exp(quantile) − 1)`. |
