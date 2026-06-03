# GIFT-Eval Experiment Status

## Date: 2026-04-15 (final update)

## Summary

Trained a Tiny contrastive forecasting backbone (20M params) on the
`tiny_mixed_v1` dataset and evaluated on GIFT-Eval. The run was stopped
at step 262k (~2.1 epochs) after identifying that the dataset ordering
is the root cause of training instability: the data was not shuffled
properly, and all GiftEval pretrain data is concentrated in the first
few shards, making each epoch contain the same distribution shift at
the same point.

**A new dataset (`tiny_mixed_v2`) with proper shuffling is being prepared.**

## Training Curve

![Training curve](../plots/fig_500k_training_curve.png)

- **Best EMA gap: 0.677** at step 261,149
- The model improves across epochs despite data-driven dips
- Repeating dip pattern at steps ~24k, ~35k, ~40k in each epoch
  (same shards cause the same instability each time)
- Each epoch recovers to a higher peak: epoch 1 peak ~0.60, epoch 2 peak ~0.68

## GIFT-Eval Results

Evaluated with the official gift-eval library using gluonts metrics.
Results are directly comparable to the GIFT-Eval leaderboard.

| Pair | Backbone | Head | GM-Relative MASE |
|---|---|---|---|
| 20k | `tiny_fresh_best_gap.pth` | `head_bb20k_20k_best.pth` | **1.257** |
| 50k | `tiny_150k_50k.pth` | `head_bb50k_50k_r3_best.pth` | **1.277** |
| 80k | `tiny_full_80k.pth` | `head_bb80k_80k_best.pth` | **1.268** |

Leaderboard reference (GM-relative MASE, lower = better):

| Model | MASE |
|---|---|
| Sundial | 0.673 |
| TimesFM | 0.680 |
| PatchTST | 0.762 |
| Chronos | 0.786 |
| Moirai | 0.809 |
| Seasonal Naive | 1.000 |
| **Ours (Tiny, 20k-80k)** | **~1.26** |

The scaling curve is **flat** (1.257 → 1.277 → 1.268): more training steps
do not improve GIFT-Eval performance. This is likely because the data
ordering means the model never sees a representative sample of all
time series types during early training.

## Key Investigations

### 1. NaN Crash (step 24,970)
- **Root cause**: All-NaN row in the HF dataset silently passed through `_forward_fill_nan`
- **Fix**: Skip rows that remain NaN after ffill + bfill
- **Documented in**: `../2026-04-12_tiny-training/notes/INCIDENT_NAN_AND_RESUME.md`

### 2. Training Instability (steps 24k, 35k, 40k)
- Three crash-recovery episodes per epoch at the same data positions
- **Not caused by**: LR (identical dip at 1e-4, 5e-5, 1e-5), gradient explosion,
  RevEWMNorm artifacts, extreme values in the data
- **Caused by**: Data distribution shift — specific shard regions trigger
  FP (first-part similarity) inflation and representation collapse
- **Documented in**: `LR_SWEEP_EXPERIMENT.md`, `../2026-04-12_tiny-training/REVEWMNORM_CLAMP.md`

### 3. RevEWMNorm Analysis
- Max |x_norm| = 4.06 across 49M values — normalization handles all data correctly
- The vectorized cumsum EMA adjusts stdev instantaneously at constant→change transitions
- Proposed clamp (±10) proven unnecessary and removed
- **Documented in**: `../2026-04-12_tiny-training/REVEWMNORM_CLAMP.md`

### 4. Gradient Analysis
- Per-batch gradient norms identical across all shards (L2 ~0.5-2.2)
- **AdamW effective updates are identical** (~0.089 L2) regardless of data
- No single-batch gradient spike — instability accumulates over many steps

### 5. Data Ordering Problem (root cause of flat scaling curve)
- The `tiny_mixed_v1` dataset was not shuffled: GiftEval pretrain data
  concentrated in early shards, synthetic data in later shards
- Each epoch encounters the same distribution shift at the same step
- The model never sees a representative mix early in training
- **Solution**: New dataset `tiny_mixed_v2` with proper inter-shard shuffling

## Artifacts

### Checkpoints (local)

| Directory | Contents |
|---|---|
| `checkpoints/vast_tiny_500k/` | 500k run: 70k-260k periodic + best_gap (step 261k) |
| `checkpoints/vast_tiny_clean/` | Clean run: 10k-60k periodic + best (step 68k) |
| `checkpoints/vast_tiny_fresh/` | Original run (crashed at 25k) |
| `checkpoints/vast_tiny_full/` | Flawed continuation (60k-83k) |
| `checkpoints/elisa_tiny_150k/` | Elisa flawed run (20k-63k) |

### Plots

| File | Description |
|---|---|
| `fig_500k_training_curve.png` | Full training curve 0→262k with best gap marker |
| `fig_training_curve_clean_lr1e-4.png` | Clean run 0→69k (baseline for LR sweep) |
| `fig_lr_sweep_final.png` | LR sweep comparison (1e-4, 5e-5, 1e-5) |

### Results

| File | Description |
|---|---|
| `results/pair_20k_clean_all_results.csv` | 20k pair: 97-config GIFT-Eval results |
| `results/pair_50k_all_results.csv` | 50k pair: 97-config GIFT-Eval results |
| `results/pair_80k/all_results.csv` | 80k pair: 97-config GIFT-Eval results (on Elisa) |

### Code Changes (PRs merged to `experiments`)

| PR | Description |
|---|---|
| #13 | Complete checkpoint state (best_loss, EMA, RNG, hf_rows) |
| #14 | Optional gradient clipping |
| #15 | Multi-epoch training (restart stream on exhaustion) |
| #16 | RNG state restore across PyTorch versions |

## Next Steps

1. **Wait for `tiny_mixed_v2` dataset** at
   `jeremycochoy/contrastive-training-tiny-bundles/tiny_mixed_v2`
   (1088 shards + manifest + eval directory, properly shuffled)
2. **Train single-epoch** on `v2` from step 0 (no resume, clean start)
3. **Re-evaluate** with matched heads at 20k, 50k, 100k milestones
4. Expect a smoother training curve without the data-driven dips
