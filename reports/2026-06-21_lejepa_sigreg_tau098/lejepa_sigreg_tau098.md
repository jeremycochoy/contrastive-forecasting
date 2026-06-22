# LeJEPA spherical regulariser at a tighter EMA teacher (τ=0.98)

## Question

The previous arm at B=512 with SIGReg on the patch-embed and on the encoder output used an EMA-target teacher at `τ=0.99` (half-life ≈69 steps). Halving the half-life by moving the EMA decay to `τ=0.98` (half-life ≈34 steps) tightens the teacher's tracking of the student. Does this single-axis change move GM-Rel MASE on the GIFT-Eval full-97 benchmark, and does it shift the cross-batch / cross-time sphere-coverage diagnostics?

## Vocabulary

| term | definition |
| --- | --- |
| `enc3` | 3-layer transformer encoder (hidden size `K`=384, 6 heads — the codebase's depth used here). |
| `CPC` | InfoNCE auxiliary head on the encoder, `--cpc-infonce-weight 1.0`. |
| **EMA-target** | exponential-moving-average teacher on the encoder + patch-embed; `--ema-tau τ`, half-life ≈ ln(2)/(1−τ) steps. `τ=0.99` ≈69 steps, `τ=0.98` ≈34 steps. |
| `e_t` | output of the GRU patch-embed, per (batch, time, channel) position; `K`=384. |
| `h_t` | output of the 3-layer transformer encoder (the codebase's `original_latent`), same shape. |
| **SIGReg** | LeJEPA-style spherical regulariser. Epps–Pulley test statistic averaged over `M`=1024 random unit-direction 1-D projections of the pooled latent, trapezoidal-integrated on `[−6/√K, 6/√K]` against `N(0, 1/K)`. Drives the pooled marginal toward `Unif(S^{K-1})`. Two terms here: `L_SIGReg(e_t)` (`--sigreg-embedding`) and `L_SIGReg(h_t)` (`--sigreg-encoding`), both pre-`F.normalize` (`--sigreg-post-normalization` OFF). Each weighted by `λ`=0.1. |
| `u_batch` | cross-batch dimensionality usage of `h_t`, clipped to `[1/K, 1]`. `1/K` ≈ 0.00260 = one direction; 1 = uniform sphere coverage. `u_batch_e` is the same statistic on `e_t`. |
| `u_temporal` | cross-time analogue of `u_batch`; `u_temporal_e` is the `e_t` version. |
| **GM-Rel MASE** | GIFT-Eval full-97 aggregate: geometric mean over 97 configs of (model MASE ÷ seasonal-naive MASE). Lower = better; 1.0 = seasonal-naive parity. |
| `*_best` | head-matched downstream cell where the head is trained on the lowest-train-loss backbone checkpoint of the arm. **In this arm only**, that checkpoint is the `_10k.pth` periodic save (operator-promoted by a one-time `cp _10k.pth _FINAL.pth`; see Protocol), not a tracker-emitted `best_loss.pth` as in `enc3+CPC`, `EMA-target enc3+CPC, τ=0.99`, and `SIGReg + EMA-target, τ=0.99`. |
| `*_last` | head-matched downstream cell where the head is trained on the final-step backbone checkpoint. Same convention across all four arms in the table below. |

## Result

![GM-Rel MASE on the GIFT-Eval full-97 benchmark, four (head-depth, backbone-checkpoint) cells per arm](plots/gm_rel_mase.png)

| head / checkpoint | enc3+CPC, B=1024 | EMA-target enc3+CPC, B=1024, τ=0.99 | SIGReg + EMA-target, B=512, τ=0.99 | SIGReg + EMA-target, B=512, τ=0.98 |
| --- | ---: | ---: | ---: | ---: |
| 2L / best ⁂ | 1.1846 | 1.1614 | 1.1610 | 1.1807 |
| 2L / last | 1.1531 | 1.1817 | 1.1758 | 1.1662 |
| 6L / best ⁂ | 1.1584 | 1.1576 | 1.1543 | 1.1493 |
| 6L / last | 1.1436 | 1.1597 | 1.1556 | 1.1462 |

⁂ The τ=0.98 `*_best` row uses the `_10k.pth` periodic save as the "best" backbone (see Protocol); the other three arms use a `best_loss.pth` tracker checkpoint. **The `*_best` row is therefore not on the same axis as the other arms' `*_best` cells and is not directly comparable.** Cross-arm `*_best` deltas involving the τ=0.98 column are not reported here for that reason.

Scoping the comparison vs `SIGReg + EMA-target, B=512, τ=0.99` to the head-matched `*_last` cells (same checkpoint convention across both arms):

- 2L / last: 1.1662 − 1.1758 = **−0.0096**
- 6L / last: 1.1462 − 1.1556 = **−0.0094**

Both `*_last` cells move ~0.01 in favour of the τ=0.98 arm. The arm is a single seed (`20260520`); paired-seed re-runs of similar arms in the τ-sweep card (see annex E) found ~0.01 GM-Rel MASE differences within seed noise, so this delta does not separate the arms on the head-matched downstream metric.

The backbone diagnostics move in the opposite direction (annex D, tail-50 means at the final logged step):

| quantity | τ=0.99 (final step 12 500) | τ=0.98 (final step 12 400) |
| --- | ---: | ---: |
| `u_batch` (h_t) | 0.7964 | 0.7626 |
| `u_batch_e` (e_t) | 0.0438 | 0.0333 |
| `u_temporal` (h_t) | 0.6194 | 0.5735 |
| `u_temporal_e` (e_t) | 0.0315 | 0.0251 |
| `L_SIGReg(e_t)` | 1.001e-3 | 1.251e-3 |
| `L_SIGReg(h_t)` | 3.80e-4 | 6.04e-4 |

All four sphere-coverage statistics are lower (further from uniform) under τ=0.98 than under τ=0.99, and both SIGReg term values are larger (further from `Unif(S^{K-1})` by the Epps–Pulley statistic).

**Metric scope.** This GIFT-Eval wrapper emits only GM-Rel MASE per `summary.txt`. The project's preferred aggregates are GM-MASE / GM-MAPE_SN / GM-CRPS_SN; those are not produced here — the per-config `all_results.csv` carries raw `MASE[0.5]`, `MAPE[0.5]`, and `mean_weighted_sum_quantile_loss`, but not the seasonal-naive denominators needed to form the SN-relative versions (see annex C).

**Reference-values provenance.** The `enc3+CPC`, `EMA-target enc3+CPC, τ=0.99`, and `SIGReg + EMA-target, τ=0.99` columns reproduce prior arms' published head-matched tables at their own code revisions, embedded as constants `REF_GM`, `EMA_GM`, and `SIGREG_TAU099_GM` in [`build_report_tau098.py`](../../experiments/2026-06-20_lejepa_sigreg/scripts/build_report_tau098.py); the τ=0.98 column is the only fresh head-matched eval at this code revision and HF cache snapshot.

### What the two SIGReg terms did

![SIGReg term trajectories on log scale (upper) and their ratio to total loss (lower), 50-step rolling means](plots/sigreg_e_inspection.png)

Mean over the last 50 of 12 400 logged steps:

| quantity | value |
| --- | ---: |
| total `loss` | 4.0834 |
| `λ · L_SIGReg(e_t)` | 1.251e-4 |
| `λ · L_SIGReg(h_t)` | 6.038e-5 |
| **`λ · L_SIGReg(e_t) / loss`** | **3.06e-5** (~33 000× smaller than total) |
| **`λ · L_SIGReg(h_t) / loss`** | **1.48e-5** (~68 000× smaller than total) |

Both SIGReg terms remained ~10⁴× smaller than the contrastive + CPC + EMA-target sum throughout training, the same regime as the τ=0.99 arm.

### Sphere coverage

![Cross-batch (left) and cross-time (right) uniformity over training; h_t solid vs e_t dashed for both SIGReg arms (τ=0.98 green, τ=0.99 red); h_t overlays for the two reference arms](plots/uniformity.png)

The contrastive loss `cosine_similarity_batch_full_hh_negs_xshh_allt` directly rewards angular separation between `h_t` vectors, so `h_t`'s sphere coverage is not attributable to the SIGReg term alone.

## Protocol

One arm, seed `20260520`, target 12 500 steps; the training process was terminated at step 12 400 by the operator after the periodic `_10k.pth` save and a later `_12k.pth` save were already on disk, so no `best_loss.pth` tracker checkpoint was emitted by the trainer. The `bb_*_FINAL.pth` file used as the `*_best` backbone for the downstream chain is byte-identical to `bb_*_10k.pth` (verified by `md5sum`), promoted by a one-time operator `cp _10k.pth _FINAL.pth`. **The launcher [`experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh`](../../experiments/2026-06-20_lejepa_sigreg/scripts/train_backbone_sigreg.sh) does not reproduce this `*_best` mapping**: in a re-run that completes 12 500 steps without termination, the launcher's `if [ -f best_loss.pth ]; then cp best_loss.pth $BB` branch would promote a tracker `best_loss.pth` instead.

This arm changes exactly one flag vs `SIGReg + EMA-target, B=512, τ=0.99`:

| flag | reference (τ=0.99) | this arm |
| --- | --- | --- |
| `--ema-tau` | 0.99 | 0.98 |

`--batch-size 512`, `--sigreg-embedding`, `--sigreg-encoding`, `--sigreg-post-normalization` OFF, `--sigreg-weight 0.1`, `--sigreg-m 1024`, `--sigreg-t-knots 17`, `--cpc-infonce-weight 1.0`, `--encoder-dropkey 0.70`, `--mix-ratio 0.0078125`, dataset, dtypes, and the GRU patch-embed → 3-layer transformer encoder (`K`=384, 6 heads) backbone are kept verbatim from that reference.

### Head-matched downstream

Each backbone checkpoint trains a 2-layer and a 6-layer quantile head (`2L`/`6L`), then evaluates on GIFT-Eval full-97 via `scripts/run_gift_eval_full.sh`. The `*_best` cells use `bb_*_FINAL.pth` (= `bb_*_10k.pth` in this arm; see above); the `*_last` cells use `bb_*_final.pth` (the step-12 400 trainer save). The 2-layer head trains 30 000 steps on the `*_best` backbone and 10 000 steps on the `*_last` backbone (same per-cell budgets as the τ=0.99 arm). Per-cell summaries live at `results/gift_eval_full_<tag>{,_last}_{2L,6L}/summary.txt`.

### Training loss

![Training loss, 50-step rolling mean, four arms overlaid](plots/loss_curve.png)

This report does not isolate the cause of the shape. The τ=0.98 curve ends at step 12 400 (annex A).

## Annex

### A. Cross-arm plot provenance, training-CSV truncation

All plots embed the τ=0.98 arm's training CSV `runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_tau098_losses.csv`. The CSV holds 12 401 rows (steps 0–12 400); the last 100 steps of the originally targeted 12 500-step schedule did not flush to disk because the file handle's tail-buffer state was recovered post-hoc rather than from a normal trainer exit. `loss_curve.png` and `uniformity.png` overlay:

- τ=0.99 SIGReg arm: `reports/2026-06-20_lejepa_sigreg/runs/bb_allt08_xftrip_nobn_enc3_emateach_sigreg_qk_aon_b512_cpc_losses.csv` (12 500 rows).
- EMA-target arm: `experiments/2026-06-19_ema_target_encoder/runs/bb_allt08_xftrip_nobn_enc3_emateach_qk_aon_b1024_cpc_losses.csv`.
- enc3+CPC arm: `experiments/2026-06-13_cpc_infonce_aux/runs/bb_allt08_xftrip_nobn_enc3_sgpos_qk_aon_b1024_cpc_losses.csv`.

The overlay CSVs are taken from those arms' own code revisions; no fresh re-training was run for this report. The `gm_rel_mase.png` legend mapping: grey = enc3+CPC B=1024, blue = EMA-target enc3+CPC B=1024 τ=0.99, red = SIGReg + EMA-target B=512 τ=0.99, green = this arm.

### B. Attribution

This arm changes one flag vs the τ=0.99 reference (`--ema-tau` 0.99 → 0.98). The `*_last`-cell GM-Rel MASE deltas are head-matched on the same step (final-step backbone, same head training budget). The `*_best` cell uses a different checkpoint-selection rule for this arm than for the τ=0.99 reference, so its number reflects both the EMA-tau change and the checkpoint-selection change.

### C. GIFT-Eval wrapper emits only GM-Rel MASE

`scripts/run_gift_eval_full.sh` writes `Aggregate GM-Relative MASE (97 configs)` to each `summary.txt`. The per-config `all_results.csv` carries raw `MASE[0.5]`, `MAPE[0.5]`, and `mean_weighted_sum_quantile_loss`, but not the seasonal-naive denominators needed to form GM-MASE / GM-MAPE_SN / GM-CRPS_SN. Computing those would require re-running the wrapper with an additional emit flag — out of scope here.

### D. Trajectory of SIGReg terms and dimensionality (τ=0.98)

50-step rolling means at the listed step. The trajectory table built by `build_report_tau098.py` requests steps `(250, 500, 1000, 2000, 5000, 7500, 10000, 12500)`; step 12 500 is absent from the CSV, so the table stops at step 10 000. A tail-50-row mean centred on the final logged step (12 400) is reported on the last line for alignment with the `Result` section.

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

`1/K` = 1/384 ≈ 0.00260. Source: `results/trajectory_table.csv` (steps 250–10 000) plus `results/final_trajectories.txt` for the step-12 400 row.

### E. Seed noise for `~0.01` GM-Rel MASE deltas

In the τ-sweep card (`experiments/2026-05-08_exp_tau_sweep`), arms at distinct `--tau` values were observed to differ by `~0.01` on GM-Rel MASE while being indistinguishable at single-seed precision and matching the seed-noise band of paired re-runs. This card is one seed; the `*_last`-cell deltas vs τ=0.99 (2L: −0.0096, 6L: −0.0094) fall in that range and are not separable from seed noise without replicates.
