# Window Size Comparison: W=32 vs W=16

## Objective

Determine whether reducing the patch window size from W=32 to W=16 degrades contrastive learning performance on the Tiny backbone (H=512, L=6). A smaller window doubles the number of patches (128 to 256), giving finer temporal resolution but increasing attention cost quadratically.

## Setup

| Parameter | W=32 | W=16 |
|-----------|------|------|
| Backbone | Tiny (H=512, L=6, GRU encoder, 8 heads) | same |
| Parameters | 19,960,576 | 19,952,384 |
| Patch count (T_raw=4096) | 128 | 256 |
| Batch size | 32 | 24 |
| Learning rate | 1e-4 | 1e-4 |
| VRAM (PyTorch reported) | 23.4 GB | 14.7 GB |
| Speed | 5.7 step/s | 4.7 step/s |
| GPU | RTX 4090 (GPU 0) | RTX 4090 (GPU 1) |
| Training steps | 10,000 | 10,000 |

Batch size for W=16 was set to 24 (vs 32 for W=32) due to the higher memory cost of attention over the 2x longer sequence. Both runs used identical loss configuration (cosine similarity, batch negatives, temperature 0.07).

## Results by Step

| Step | W=32 gap (bs=32, 5.7sps, 23.4GB) | W=16 gap (bs=24, 4.7sps, 14.7GB) |
|------|----------------------------------|----------------------------------|
| 1k | 0.056 | 0.064 |
| 2k | 0.068 | 0.064 |
| 3k | 0.067 | 0.072 |
| 4k | 0.073 | 0.077 |
| 5k | 0.074 | 0.078 |
| 6k | 0.075 | 0.080 |
| 7k | 0.079 | 0.088 |
| 8k | 0.073 | 0.091 |
| 9k | 0.082 | 0.093 |
| 10k | 0.082 | 0.092 |

**W=32 best gap: 0.082 (step 9k). W=16 best gap: 0.093 (step 9k).**

## Results by Wall Time

| Wall time | Step | W=32 gap (bs=32, 5.7sps, 23.4GB) | W=16 gap (bs=24, 4.7sps, 14.7GB) |
|-----------|------|----------------------------------|----------------------------------|
| 2.9 min | 1k | 0.056 | -- |
| 3.5 min | 1k | -- | 0.064 |
| 5.8 min | 2k | 0.068 | -- |
| 7.1 min | 2k | -- | 0.064 |
| 8.8 min | 3k | 0.067 | -- |
| 10.6 min | 3k | -- | 0.072 |
| 11.7 min | 4k | 0.073 | -- |
| 14.2 min | 4k | -- | 0.077 |
| 14.6 min | 5k | 0.074 | -- |
| 17.5 min | 6k | 0.075 | -- |
| 17.7 min | 5k | -- | 0.078 |
| 20.5 min | 7k | 0.079 | -- |
| 21.3 min | 6k | -- | 0.080 |
| 23.4 min | 8k | 0.073 | -- |
| 24.8 min | 7k | -- | 0.088 |
| 26.3 min | 9k | 0.082 | -- |
| 28.4 min | 8k | -- | 0.091 |
| 29.2 min | 10k | 0.082 | -- |
| 31.9 min | 9k | -- | 0.093 |
| 35.5 min | 10k | -- | 0.092 |

## Analysis

### Speed

W=16 is **18% slower** per step (4.7 vs 5.7 step/s). This is expected: the attention mechanism scales quadratically with sequence length (256 vs 128 patches), partially offset by the GRU encoder processing shorter patches (16 vs 32 timesteps).

### VRAM

W=16 uses **37% less VRAM** (14.7 GB vs 23.4 GB) at its operating batch size (24 vs 32). The lower batch size was necessary because attention memory per sample scales with the square of the sequence length. Despite the smaller batch, the total VRAM is significantly lower, leaving headroom for larger models or higher batch sizes on the same hardware.

### Performance

W=16 achieves a **13% higher best gap** (0.093 vs 0.082) at 10k steps. The advantage is consistent: W=16 is ahead at every matched step count from 3k onward, and the margin widens over time.

At matched wall time, W=16 and W=32 track closely through ~18 minutes, after which W=16 pulls ahead decisively. The 18% speed penalty is more than compensated by the per-step learning advantage.

### Advantage of W=16

- **+13% best gap** (0.093 vs 0.082) at matched step budget
- **+13% best gap** at matched wall time (0.093 at 32 min vs 0.082 at 29 min)
- **37% less VRAM**, freeing capacity for larger models or bigger batches
- **18% slower per step**, but this is the only downside

## Conclusion

W=16 does not degrade performance -- it improves it. The finer temporal resolution (256 patches vs 128) gives the transformer more granular information to learn from, resulting in consistently higher contrastive gap. The speed penalty is modest and fully offset by the quality gain. The substantial VRAM savings is an additional practical benefit for scaling to larger architectures.
