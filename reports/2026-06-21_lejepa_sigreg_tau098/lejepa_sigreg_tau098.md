# LeJEPA spherical regulariser: shorter EMA window and no-EMA ablations

## Question

The reference arm at B=512 trains SIGReg on the patch-embed and on the encoder output against an EMA-target teacher at `τ=0.99` (half-life ≈69 steps). Two single-axis variants of that recipe are evaluated here against the head-matched downstream metric (GIFT-Eval full-97 GM-Rel MASE) and the cross-batch / cross-time sphere-coverage diagnostics:

1. **Shorter EMA window** — `--ema-tau 0.99 → 0.98` (half-life ≈34 steps).
2. **no-EMA arm** — `--ema-embedding`, `--ema-encoder`, `--ema-tau` all removed (`--sigreg-embedding` and `--sigreg-encoding` kept on).

## Vocabulary

| term | definition |
| --- | --- |
| `enc3` | 3-layer transformer encoder (hidden size `K`=384, 6 heads — the codebase's depth used here). |
| `CPC` | InfoNCE auxiliary head on the encoder, `--cpc-infonce-weight 1.0`. |
| **EMA-target** | exponential-moving-average teacher on the encoder + patch-embed; `--ema-tau τ`, half-life ≈ ln(2)/(1−τ) steps. `τ=0.99` ≈69 steps, `τ=0.98` ≈34 steps. The no-EMA arm removes the teacher entirely (`--ema-embedding`, `--ema-encoder`, `--ema-tau` not passed). |
| `e_t` | output of the GRU patch-embed, per (batch, time, channel) position; `K`=384. |
| `h_t` | output of the 3-layer transformer encoder (the codebase's `original_latent`), same shape. |
| **SIGReg** | LeJEPA-style spherical regulariser. Epps–Pulley test statistic averaged over `M`=1024 random unit-direction 1-D projections of the pooled latent, trapezoidal-integrated on `[−6/√K, 6/√K]` against `N(0, 1/K)`. Drives the pooled marginal toward `Unif(S^{K-1})`. Two terms here: `L_SIGReg(e_t)` (`--sigreg-embedding`) and `L_SIGReg(h_t)` (`--sigreg-encoding`), both pre-`F.normalize` (`--sigreg-post-normalization` OFF). Each weighted by `λ`=0.1. |
| `u_batch` | cross-batch dimensionality usage of `h_t`, clipped to `[1/K, 1]`. `1/K` ≈ 0.00260 = one direction; 1 = uniform sphere coverage. `u_batch_e` is the same statistic on `e_t`. |
| `u_temporal` | cross-time analogue of `u_batch`; `u_temporal_e` is the `e_t` version. |
| **GM-Rel MASE** | GIFT-Eval full-97 aggregate: geometric mean over 97 configs of (model MASE ÷ seasonal-naive MASE). Lower = better; 1.0 = seasonal-naive parity. |
| `*_best` | head-matched downstream cell where the head is trained on the lowest-train-loss backbone checkpoint of the arm. **In the τ=0.98 arm only**, that checkpoint is the `_10k.pth` periodic save (see Protocol), not a tracker-emitted `best_loss.pth` as in the other four arms (including the no-EMA arm here). |
| `*_last` | head-matched downstream cell where the head is trained on the final-step backbone checkpoint. Same convention across all five arms in the table below. |

## Result

![GM-Rel MASE on the GIFT-Eval full-97 benchmark, four (head-depth, backbone-checkpoint) cells per arm](plots/gm_rel_mase.png)

| head / checkpoint | enc3+CPC, B=1024 | EMA-target enc3+CPC, B=1024, τ=0.99 | SIGReg + EMA-target, B=512, τ=0.99 | SIGReg + EMA-target, B=512, τ=0.98 | SIGReg (no EMA-target), B=512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2L / best | 1.1846 | 1.1614 | 1.1610 | 1.1807 ⁂ | 1.2119 |
| 2L / last | 1.1531 | 1.1817 | 1.1758 | 1.1662 | 1.2105 |
| 6L / best | 1.1584 | 1.1576 | 1.1543 | 1.1493 ⁂ | 1.2081 |
| 6L / last | 1.1436 | 1.1597 | 1.1556 | 1.1462 | 1.1999 |

⁂ The τ=0.98 `*_best` row uses the `_10k.pth` periodic save as the "best" backbone (see Protocol); the four other arms (including the no-EMA arm) use a `best_loss.pth` tracker checkpoint. Cross-arm `*_best` deltas involving the τ=0.98 column are not directly comparable; the no-EMA `*_best` cell is on the same axis as the τ=0.99 reference.

Two head-matched comparisons against `SIGReg + EMA-target, B=512, τ=0.99` (the direct single-axis reference for both new arms):

- **τ=0.98**: 2L / last 1.1662 vs 1.1758 (Δ −0.0096); 6L / last 1.1462 vs 1.1556 (Δ −0.0094). Within the ~0.01 seed-noise band observed in the τ-sweep card (annex E); not separable from seed noise at one seed.
- **no-EMA**: every cell above the τ=0.99 reference by +0.035 to +0.054 (table below). Every no-EMA cell is also above every `enc3+CPC, B=1024` cell.

### Shorter EMA window (τ=0.98 vs τ=0.99)

Tail-50 means at the final logged step (annex D):

| quantity | τ=0.99 (final step 12 500) | τ=0.98 (final step 12 400) |
| --- | ---: | ---: |
| `u_batch` (h_t) | 0.7964 | 0.7626 |
| `u_batch_e` (e_t) | 0.0438 | 0.0333 |
| `u_temporal` (h_t) | 0.6194 | 0.5735 |
| `u_temporal_e` (e_t) | 0.0315 | 0.0251 |
| `L_SIGReg(e_t)` | 1.001e-3 | 1.251e-3 |
| `L_SIGReg(h_t)` | 3.80e-4 | 6.04e-4 |

### no-EMA arm

Head-matched cells against the τ=0.99 reference (both arms use a tracker-emitted `best_loss.pth` for `*_best` and a final-step save for `*_last`, so `*_best` and `*_last` are both directly comparable):

| head / checkpoint | τ=0.99 | no-EMA | Δ (no-EMA − τ=0.99) |
| --- | ---: | ---: | ---: |
| 2L / best | 1.1610 | 1.2119 | +0.0509 |
| 2L / last | 1.1758 | 1.2105 | +0.0347 |
| 6L / best | 1.1543 | 1.2081 | +0.0538 |
| 6L / last | 1.1556 | 1.1999 | +0.0443 |

The four deltas (+0.0347 to +0.0538) sit above the ~0.01 seed-noise band (annex E).

Tail-50 means at the final logged step (12 500):

| quantity | τ=0.99 | no-EMA |
| --- | ---: | ---: |
| `u_batch` (h_t) | 0.7964 | 0.1491 |
| `u_batch_e` (e_t) | 0.0438 | 0.00433 |
| `u_temporal` (h_t) | 0.6194 | 0.1399 |
| `u_temporal_e` (e_t) | 0.0315 | 0.00418 |
| `L_SIGReg(e_t)` | 1.001e-3 | 5.68e-7 |
| `L_SIGReg(h_t)` | 3.80e-4 | 7.81e-5 |
| total `loss` | 4.248 | 0.867 |

![SIGReg term trajectories on log scale (upper) and their ratio to total loss (lower), 50-step rolling means — no-EMA arm](plots/sigreg_e_inspection_noema.png)

### SIGReg term trajectories under EMA-target (τ=0.98 arm)

![SIGReg term trajectories on log scale (upper) and their ratio to total loss (lower), 50-step rolling means — τ=0.98 arm](plots/sigreg_e_inspection.png)

Mean over the last 50 of 12 400 logged steps:

| quantity | value |
| --- | ---: |
| total `loss` | 4.0834 |
| `λ · L_SIGReg(e_t)` | 1.251e-4 |
| `λ · L_SIGReg(h_t)` | 6.038e-5 |
| **`λ · L_SIGReg(e_t) / loss`** | **3.06e-5** (~33 000× smaller than total) |
| **`λ · L_SIGReg(h_t) / loss`** | **1.48e-5** (~68 000× smaller than total) |

Both SIGReg terms remained ~10⁴× smaller than the contrastive + CPC + EMA-target sum throughout training.

### Sphere coverage

![Cross-batch (left) and cross-time (right) uniformity over training; h_t solid vs e_t dashed for the three SIGReg arms; h_t overlays for the two reference arms](plots/uniformity.png)

The contrastive loss `cosine_similarity_batch_full_hh_negs_xshh_allt` directly contains an angular-separation term over `h_t` vectors (see [`src/loss.py`](../../src/loss.py) for the formula).

**Metric scope.** Only GM-Rel MASE is emitted here; the project's GM-MASE / GM-MAPE_SN / GM-CRPS_SN aggregates are not produced (see annex C).

**Reference-values provenance.** The `enc3+CPC`, `EMA-target enc3+CPC, τ=0.99`, and `SIGReg + EMA-target, τ=0.99` columns reproduce prior arms' published head-matched tables at their own code revisions — source experiments `experiments/2026-06-13_cpc_infonce_aux/`, `experiments/2026-06-19_ema_target_encoder/`, and `reports/2026-06-20_lejepa_sigreg/` respectively — embedded as constants `REF_GM`, `EMA_GM`, and `SIGREG_TAU099_GM` in [`build_report_tau098.py`](../../experiments/2026-06-20_lejepa_sigreg/scripts/build_report_tau098.py). The τ=0.98 and no-EMA columns are fresh head-matched evals at this code revision and HF cache snapshot.

## Protocol

Both new arms use seed `20260520`, target 12 500 steps.

**τ=0.98 arm.** The arm's CSV ends at step 12 400 and no `best_loss.pth` tracker checkpoint exists; the `bb_*_FINAL.pth` used as the `*_best` backbone is byte-identical to the periodic `bb_*_10k.pth` save (`md5sum` verified). The launcher [`experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh`](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh) prefers `best_loss.pth` when present, so a re-run that completes the full schedule and emits a tracker checkpoint would promote a tracker `best_loss.pth` instead.

**no-EMA arm.** Training ran to step 12 500. The trainer's `best_loss.pth` tracker emitted a checkpoint and the launcher [`experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg_noema.sh`](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg_noema.sh) promoted it to `bb_*_FINAL.pth` via its `if [ -f best_loss.pth ]; then cp best_loss.pth $BB` branch (`md5sum` confirms `FINAL.pth = best_loss.pth`). Both `*_best` and `*_last` are head-matched on the tracker convention.

Flag changes vs `SIGReg + EMA-target, B=512, τ=0.99`:

| flag | reference (τ=0.99) | τ=0.98 arm | no-EMA arm |
| --- | --- | --- | --- |
| `--ema-tau` | 0.99 | 0.98 | (not passed) |
| `--ema-embedding` | ON | ON | (not passed) |
| `--ema-encoder` | ON | ON | (not passed) |

`--batch-size 512`, `--sigreg-embedding`, `--sigreg-encoding`, `--sigreg-post-normalization` OFF, `--sigreg-weight 0.1`, `--sigreg-m 1024`, `--sigreg-t-knots 17`, `--cpc-infonce-weight 1.0`, `--encoder-dropkey 0.70`, `--mix-ratio 0.0078125`, dataset, dtypes, and the GRU patch-embed → 3-layer transformer encoder (`K`=384, 6 heads) backbone are kept verbatim from that reference.

### Head-matched downstream

Each backbone checkpoint trains a 2-layer and a 6-layer quantile head (`2L`/`6L`), then evaluates on GIFT-Eval full-97 via `scripts/run_gift_eval_full.sh`. The `*_best` cells use `bb_*_FINAL.pth` (= `bb_*_10k.pth` in the τ=0.98 arm; = `bb_*_best_loss.pth` in the no-EMA arm; see above); the `*_last` cells use `bb_*_final.pth` (the final-step trainer save). The 2-layer head trains 30 000 steps on the `*_best` backbone and 10 000 steps on the `*_last` backbone (same per-cell budgets as the τ=0.99 arm). Per-cell summaries live at `results/gift_eval_full_<tag>{,_last}_{2L,6L}/summary.txt`.

### Training loss

![Training loss, 50-step rolling mean, five arms overlaid](plots/loss_curve.png)

## Annex

### A. Cross-arm plot provenance, training-CSV truncation

All plots embed the τ=0.98 arm's training CSV `runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau098_losses.csv` (12 401 rows, steps 0–12 400). `loss_curve.png` and `uniformity.png` overlay:

- no-EMA SIGReg arm: `runs/bb_allt08_xftrip_nobn_enc3_sigreg_qk_aon_b512_cpc_noema_losses.csv` (12 501 rows, steps 0–12 500).
- τ=0.99 SIGReg arm: `reports/2026-06-20_lejepa_sigreg/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv` (12 500 rows).
- EMA-target arm: `experiments/2026-06-19_ema_target_encoder/runs/bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc_losses.csv`.
- enc3+CPC arm: `experiments/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv`.

The overlay CSVs are taken from those arms' own code revisions; no fresh re-training was run for this report. The `gm_rel_mase.png` legend mapping: grey = enc3+CPC B=1024, blue = EMA-target enc3+CPC B=1024 τ=0.99, red = SIGReg + EMA-target B=512 τ=0.99, green = SIGReg + EMA-target B=512 τ=0.98, purple = SIGReg no EMA-target B=512.

### B. Attribution

The τ=0.98 arm changes one flag vs the τ=0.99 reference (`--ema-tau` 0.99 → 0.98); its `*_last`-cell GM-Rel MASE deltas are head-matched on the same step (final-step backbone, same head training budget). Its `*_best` cell uses a different checkpoint-selection rule for this arm than for the τ=0.99 reference, so its number reflects both the EMA-tau change and the checkpoint-selection change.

The no-EMA arm removes three flags vs the τ=0.99 reference (`--ema-embedding`, `--ema-encoder`, `--ema-tau` all not passed); both `*_best` and `*_last` are head-matched on the same checkpoint-selection rule as the τ=0.99 reference (tracker `best_loss.pth` and final-step `final.pth`).

### C. GIFT-Eval wrapper emits only GM-Rel MASE

`scripts/run_gift_eval_full.sh` writes `Aggregate GM-Relative MASE (97 configs)` to each `summary.txt`. The per-config `all_results.csv` carries raw `MASE[0.5]`, `MAPE[0.5]`, and `mean_weighted_sum_quantile_loss`, but not the seasonal-naive denominators needed to form GM-MASE / GM-MAPE_SN / GM-CRPS_SN. Computing those would require re-running the wrapper with an additional emit flag — out of scope here.

### D. Trajectory of SIGReg terms and dimensionality

50-step rolling means at the listed step. `1/K` = 1/384 ≈ 0.00260.

τ=0.98 arm. The trajectory table built by `build_report_tau098.py` requests steps `(250, 500, 1000, 2000, 5000, 7500, 10000, 12500)`; step 12 500 is absent from this arm's CSV, so the table stops at step 10 000. A tail-50-row mean centred on the final logged step (12 400) is reported on the last line for alignment with the `Result` section.

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

no-EMA arm. Tail-50 mean at the final logged step (step 12 500):

| step | `L_SIGReg(e_t)` | `L_SIGReg(h_t)` | `u_batch_e` | `u_batch` (`h_t`) | `u_temporal_e` | `u_temporal` (`h_t`) | `loss` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 500 (tail-50) | 5.68e-7 | 7.81e-5 | 0.00433 | 0.149 | 0.00418 | 0.140 | 0.867 |

Source: `results/trajectory_table.csv` (τ=0.98 steps 250–10 000), `results/final_trajectories.txt` (τ=0.98 step 12 400), `results/final_trajectories_noema.txt` (no-EMA step 12 500).

### E. Seed noise for `~0.01` GM-Rel MASE deltas

In the τ-sweep card (`experiments/2026-05-08_exp_tau_sweep`), arms at distinct `--tau` values were observed to differ by `~0.01` on GM-Rel MASE while being indistinguishable at single-seed precision and matching the seed-noise band of paired re-runs. The τ=0.98 arm is one seed; its `*_last`-cell deltas vs τ=0.99 (2L: −0.0096, 6L: −0.0094) fall in that range and are not separable from seed noise without replicates. The no-EMA arm's deltas vs τ=0.99 (+0.0347 to +0.0538) sit above this band.
