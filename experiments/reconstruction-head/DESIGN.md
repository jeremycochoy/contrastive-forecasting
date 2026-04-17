# Reconstruction Head Experiment

## Motivation

The current forecasting head **predicts the future** — given f[t], it outputs values at (t+1)*W to (t+1)*W+128. But the backbone already did the temporal prediction: contrastive training makes f[t] ≈ e[t+1]. The head should just **reconstruct** the patch that f[t] represents, not re-predict the future.

This misalignment explains why latent rollout (B-variants) performed worse than value-space rollout (A-variants) in the head/rollout comparison:

| Strategy | MASE |
|----------|------|
| A2 (value, W=16) | 1.262 |
| A1 (value, 128) | 1.275 |
| B1 (latent) | 1.310 |
| B3 (latent) | 1.421 |

The head was never trained to decode rolled latents — it was trained to predict the future from context latents. For latent rollout to work, the head must learn to **reconstruct** values from a latent that represents them.

## Key Insight

```
Backbone:   f[t] ≈ e[t+1]           (temporal prediction, already done)
Old head:   f[t] → 128 future vals  (predicts future AGAIN — redundant + misaligned)
New head:   f[t] → W=16 values of patch t+1  (reconstructs what f[t] represents)
```

At latent rollout time:
- f[T] ≈ e[T+1] → head reconstructs patch T+1 values
- f[T+1] (rolled) ≈ e[T+2] → head reconstructs patch T+2 values
- Each rolled latent decodes exactly the patch it represents. No extrapolation.

## Variants

### R1: Forecaster reconstruction (W=16)
- **Input:** f[t] from full forward pass (encoder → transformer)
- **Target:** x_norm[(t+1)*W : (t+1)*W + W] — the W=16 values of patch t+1
- **Rationale:** Most natural. f[t] ≈ e[t+1], reconstruct what it encodes.
- **Eval rollout:** Roll out f[T+1..T+N] in latent space, decode each → W values.

### R2: Encoder reconstruction (W=16)
- **Input:** e[t] from encoder only (before transformer)
- **Target:** x_norm[t*W : t*W + W] — the W=16 values of patch t itself
- **Rationale:** Encoder latent directly represents its own patch. Train head to decode e[t] → patch t.
- **Eval rollout:** Use rolled f[t] ≈ e[t+1]; head trained on e[t]→patch t, so f[T+k]≈e[T+k+1]→patch T+k+1.

### R3: Rolled reconstruction (W=16)
- **Input:** [f_ctx, f_rolled] — mixed real + rolled latents
- **Target:** x_norm values at each position, **loss only on rolled positions**
- **Rationale:** Train specifically on what the head will see at eval time. Handles distribution shift from rolled latents.
- **Eval rollout:** Same as R1 but head is trained on rolled latents.

### R4: Forecaster reconstruction (W=128)
- **Input:** f[t] from full forward pass
- **Target:** x_norm[(t+1)*W : (t+1)*W + 128] — 128 values (8 patches)
- **Rationale:** Wider output. The GRU's bidirectional context from the full sequence helps decode beyond the single patch that f[t] represents.
- **Eval rollout:** Same as R1 but output 128 values per position.

## Architecture

Same GRU head architecture as before (h=128, 2 layers, bidirectional), just with **time-aligned targets**:

- R1/R2/R3: `ForecastingHead(forecast_len=16)` — output W=16
- R4: `ForecastingHead(forecast_len=128)` — output 128

The GRU processes the full latent sequence, so each position has context from all other positions.

## Training

- **Backbone:** frozen `tiny_v2_best_gap.pth`
- **Data:** HF `jeremycochoy/contrastive-training-tiny-bundles`, `tiny_mixed_v2`
- **Steps:** 30k (quick comparison)
- **Optimizer:** AdamW, lr=3e-4
- **Batch size:** 24

### Target computation (the key change)

**R1 (forecaster, W=16):**
```python
# f[t] → patch t+1 values
target[t] = x_norm[(t+1)*W : (t+1)*W + W]  # Just W=16 values
```

**R2 (encoder, W=16):**
```python
# e[t] → patch t values
target[t] = x_norm[t*W : t*W + W]  # The patch e[t] encodes
```

**R3 (rolled, W=16):**
Same targets as R1, but only compute loss on rolled positions.

**R4 (forecaster, 128):**
Same as current training (but this time we understand it's reconstruction, not prediction).

## Eval

Run GIFT-Eval on each variant using latent rollout (strategy B4 for W=16, B1/B3 for W=128).

Compare to:
- A1 (value, 128): 1.275
- A2 (value, W=16): 1.262
- B1 (latent, old head): 1.310
- B2, B3, B4: pending from current experiment

## Expected Outcome

R1 and R2 should perform significantly better than the current B-variants because the head is properly aligned — it reconstructs what the latent represents instead of predicting the future.

If R1 beats A2, latent rollout IS viable and the previous failure was due to head misalignment, not a fundamental backbone limitation.
