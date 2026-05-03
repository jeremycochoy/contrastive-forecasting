# Head / Rollout Architecture Comparison

## Motivation

The V2 scaling curve is **flat**: GM-Relative MASE stays at ~1.27 from 30k to 112k backbone training steps, even though the backbone's contrastive gap improves from 0.10 to 0.43. The bottleneck is the **head + rollout architecture**, not the backbone.

Currently, we use a GRU head that decodes 128 normalized values per patch, then rolls out in value space (decode 128 → denormalize → append to context → re-run backbone). This experiment tests whether alternative strategies can close the gap.

## Background: Encoder / Forecaster Structure

The backbone has two key components:
- **Encoder** (`input_to_latent`): maps raw patches `(B, T, C, W)` → encoder latents `e[t]` of shape `(B*C, T, H)`
- **Causal Transformer** (6 layers): takes `e[0..T]` → produces forecaster latents `f[0..T]`

The contrastive loss trains `cos(f[t], e[t+1])` to be high — so `f[t] ≈ e[t+1]` in the latent space. This means **the backbone is already a latent-space next-token predictor**: `f[T]` can be fed back as `e[T+1]` into the transformer to generate `f[T+1]`, enabling rollout in latent space without any additional model.

## Variants

### A-variants: Value-space rollout

| ID | Head output | Rollout step | Description |
|----|------------|-------------|-------------|
| **A1** | 128 values | slide by 128 | **Current baseline**. Head decodes f_lat → 128 normalized values. Denormalize, append to raw context, drop oldest 128, re-run full backbone. |
| **A2** | W=16 values | slide by 16 | Same architecture but head outputs only 16 values (one patch). Smaller steps = less error accumulation per step, but 8x more rollout iterations for same horizon. |

### B-variants: Latent-space rollout

All B-variants share the same latent rollout mechanism:
1. Encode context → `e[0..T]`, run transformer → `f[0..T]`
2. Append `f[T]` to the latent sequence as `e[T+1]`
3. Run transformer on `[e[0..T], f[T]]` → last position gives `f[T+1]`
4. Repeat to generate `f[T+1], f[T+2], ...`
5. Decode accumulated latents to values using the head

No additional model needed — the backbone's own causal transformer does the transition.

| ID | Head output | Decode strategy | Description |
|----|------------|----------------|-------------|
| **B1** | 128 values | Decode at end | Generate N future latent tokens, then decode all at once. Each decoded output covers 128 values, but only the first W=16 are non-overlapping with the next decode. Use them all for the final chunk, crop others to 16. |
| **B2** | 128 → crop to 16 | Decode each step, keep W | At each latent step, decode with 128-head but keep only first W=16 values. Effectively uses the head as a "next-patch predictor" with extra context. |
| **B3** | 128 non-overlapping | Decode every 8 tokens | Latent at position t predicts 128 values = 8 patches worth. Roll forward 8 latent tokens, decode once, then roll another 8, etc. Aligned: no overlap between decoded chunks. |
| **B4** | W=16 values | Decode each step | Latent rollout + small head (output dim = W). One patch decoded per latent step. Cleanest alignment. |

## Implementation Details

### Latent rollout function (`rollout_latent`)

```python
def rollout_latent(backbone, encoder_latents, n_future_tokens, device):
    """Generate future forecaster latents via autoregressive rollout.
    
    Args:
        backbone: frozen ConfigurableModel
        encoder_latents: (B*C, T, H) encoder latents from context
        n_future_tokens: how many future latent tokens to generate
        device: torch device
    
    Returns:
        future_latents: (B*C, n_future_tokens, H) generated forecaster latents
    """
    # Start with encoder latents, iteratively append f[-1] and re-run transformer
    seq = encoder_latents  # (B*C, T, H)
    generated = []
    
    for _ in range(n_future_tokens):
        # Run transformer on current sequence (bypass encoder)
        causal_mask = generate_causal_mask(seq.size(1), device)
        x = seq
        for layer in backbone.transformer.layers:
            x = layer(x, tgt_mask=causal_mask, tgt_is_causal=True)
        # x[:, -1, :] is f[-1] ≈ e[next]
        new_token = x[:, -1:, :]  # (B*C, 1, H)
        generated.append(new_token)
        # Append as next encoder latent
        seq = torch.cat([seq, new_token], dim=1)
    
    return torch.cat(generated, dim=1)  # (B*C, n_future_tokens, H)
```

Note: This is O(T * n_steps) per step — acceptable for short horizons. KV caching can optimize later if needed.

### Head variants

- **128-head**: existing `ForecastingHead(forecast_len=128)` — reuse as-is
- **16-head**: `ForecastingHead(forecast_len=16)` — same architecture, just smaller output

Both trained with the same loss (MSE on normalized targets).

### Training

All variants use the **same training setup**:
- Backbone: frozen `tiny_v2_best_gap.pth`
- Data: HF `jeremycochoy/contrastive-training-tiny-bundles`, split `tiny_mixed_v2`
- Optimizer: AdamW, lr=3e-4
- Batch size: 24
- Steps: 20-30k (enough to compare; ~2-3 hours per variant)
- Loss: MSE on normalized targets

For W=16 heads (A2, B4): targets are `x_norm[(t+1)*W : (t+1)*W + 16]` (only the next patch).

For 128-heads (A1, B1, B2, B3): targets are `x_norm[(t+1)*W : (t+1)*W + 128]` (existing).

### Evaluation

Run GIFT-Eval on each variant using the same eval script with a `--rollout-strategy` flag. Each strategy implements `forecast_autoregressive()` differently:

- **A1**: current code (decode 128, slide in value space)
- **A2**: decode 16, slide by 16 in value space
- **B1**: generate ceil(horizon/128) latent groups of 8, decode, concat
- **B2**: generate ceil(horizon/16) latents, decode each with 128-head, crop to 16
- **B3**: generate latents in groups of 8, decode 128 non-overlapping per group
- **B4**: generate ceil(horizon/16) latents, decode each with 16-head

## Expected Outcomes

- B-variants may win because error doesn't accumulate through denormalize→renormalize roundtrips
- A2/B4 (W=16) may produce more coherent short-range predictions but struggle at long horizons
- B3 may be optimal: latent rollout avoids value-space error, 128-value decode is efficient, non-overlapping alignment is clean
- If all B-variants beat A-variants, it confirms the value-space roundtrip is the problem

## Files

- `experiments/head-rollout-comparison/DESIGN.md` — this document
- `src/forecasting_head.py` — modified with rollout strategies
- `experiments/head-rollout-comparison/scripts/train_head_variant.py` — training script
- `experiments/head-rollout-comparison/scripts/eval_variant.py` — evaluation wrapper
- `experiments/head-rollout-comparison/RESULTS.md` — comparison table (after experiments)
