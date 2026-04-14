# RevEWMNorm Output Clamping

## Problem

RevEWMNorm computes `x_norm = (x - ema_mean) / max(ema_stdev, eps)` where
`eps = 1e-5` (absolute). When a time series has a constant segment followed
by a value change, the EMA stdev is ≈ 0 while the residual (x - mean) is
non-zero. The ratio can reach 1e14+, producing extreme activations and
destabilizing training.

Constant segments arise from:
- **NaN forward-fill**: missing values are filled with the last valid value,
  creating runs of identical values before a genuine change.
- **Genuinely constant data**: e.g., Bitcoin mining difficulty (unchanged for
  ~2 weeks between adjustments), stored at scale 1e11-1e13.

## Evidence

Measured on 49M values from the training set (500 batches):
- Mean |x_norm|: 0.82
- p99.9: 3.51
- p99.999: 4.06
- **Max: 4.06** — zero values above 5

The training instability at step ~24.5k correlated with shard 235 containing
3 rows with constant segments at scale 1e11-1e13 (Bitcoin difficulty, market
cap). The normalized values at the constant→change transition exceeded 1e14.

## Fix

Clamp normalized output to `[-C, C]` after division by stdev:
```python
x = x / self.stdev.clamp(min=self.eps)
x = x.clamp(-self.norm_clamp, self.norm_clamp)  # default C=10
```

### Why C = 10

- Normal data never exceeds |x_norm| = 4.06 → C=10 gives **2.5x headroom**
- Worst-case gradient amplification: 10x (vs 1e14x without clamp)
- AdamW handles 10x gradient spikes easily (within normal SGD noise)
- Values beyond 10 sigma are artifacts, not signal

### Why not alternative fixes

| Alternative | Issue |
|---|---|
| Relative eps (`eps * |mean|`) | Circular (mean used to set stdev floor). Can still reach ~150x. |
| Larger absolute eps | Would affect ALL data, not just artifacts |
| Better EMA init | Constant patches have range=0 too, same problem |
| Gradient clipping | Masks the symptom, doesn't fix the activation |

### Effect on denormalization

None. Denormalization uses the mean/stdev at the LAST timestep, which by
then has fully converged to the true scale. The clamp only affects artifact
timesteps during the forward pass.

### Effect on inference (GIFT-Eval)

None. Inference uses the final EMA statistics for denormalization. The
forecasting head produces normalized values that are denormalized with
converged stdev. Clamping doesn't affect this path.
