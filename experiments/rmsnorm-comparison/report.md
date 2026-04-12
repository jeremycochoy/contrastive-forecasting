# LayerNorm vs RMSNorm Comparison

## Objective

Determine whether replacing Pre-LayerNorm with Pre-RMSNorm improves contrastive gap or training speed on the Tiny backbone with synthetic ARIMA data.

## Setup

| Parameter | Value |
|-----------|-------|
| Backbone | Tiny (H=512, L=6, W=16, GRU encoder, 8 heads) |
| Parameters | 19,952,384 |
| Data | Synthetic composite ARIMA (TimesFM recipe, integrated) |
| RevEWMNorm | span=32 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| T_raw | 1024 |
| Training steps | 3,000 |
| GPUs | 2x RTX 4090, one experiment per GPU |
| RMSNorm | `torch.nn.RMSNorm` (fused CUDA kernel, PyTorch 2.8) |

Three independent runs per configuration (different random seeds from data generation). Normalization is Pre-Norm in both cases (`norm_first=True`).

## Results

| Run | LayerNorm gap | LayerNorm sps | RMSNorm gap | RMSNorm sps |
|-----|-------------|--------------|------------|------------|
| 1 | 0.442 | 23.5 | 0.421 | 21.9 |
| 2 | 0.419 | 21.0 | 0.419 | 22.7 |
| 3 | 0.420 | 23.4 | 0.423 | 22.4 |
| **Mean** | **0.427** | **22.6** | **0.421** | **22.3** |
| Std | 0.013 | 1.4 | 0.002 | 0.4 |

Note: LayerNorm run 1 appears to be an outlier (0.442 vs 0.419-0.420 for the other two). Excluding it, the LayerNorm mean is 0.420, essentially identical to RMSNorm.

## Analysis

### Gap

No meaningful difference. Both converge to ~0.420 best gap at 3k steps. The variance between runs (std ~0.01) is larger than any systematic difference between the two normalizations.

### Speed

LayerNorm is marginally faster (~22.6 vs 22.3 sps, ~1% difference), likely due to PyTorch's highly optimized fused LayerNorm kernel. Both use fused CUDA implementations.

## Conclusion

**No reason to switch from LayerNorm to RMSNorm.** At the Tiny scale on synthetic ARIMA data:
- Gap is identical within noise
- Speed is comparable (LayerNorm slightly faster)
- LayerNorm is the safer default (better PyTorch ecosystem support)

RMSNorm support has been added to the codebase (`norm_type='rmsnorm'` in `ConfigurableModel`) for future experiments at larger scales where the difference may become meaningful.
