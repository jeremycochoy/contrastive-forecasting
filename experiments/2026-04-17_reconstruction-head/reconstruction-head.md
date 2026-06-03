# Reconstruction Head Experiment Report

## Motivation

The contrastive backbone is trained so that f[t] approximates e[t+1] -- the forecaster
latent at position t encodes information about patch t+1.  The original prediction head
was trained to **predict the future** from a latent, outputting 128 values starting
after the latent's position.  This meant the head was re-doing temporal prediction that
the backbone had already performed, and it was never trained to decode rolled latents.

When we tried latent rollout with the prediction head, it underperformed value-space
rollout (A1 = 1.275 GM-Relative MASE).  The hypothesis: the head should
**reconstruct** the patch each latent represents, not predict the future again.

## Method

### Head architecture

All reconstruction heads use the same architecture as prior work:

- Bidirectional GRU, hidden 512 -> 128, 2 layers (~626K params)
- Frozen backbone: tiny_v2 (GRU patch encoder, 6 transformer layers, ~20M params)
- Training: 30k steps, AdamW lr=3e-4, batch size 24, HF streaming data

### Reconstruction targets

The key change is in what the head is trained to output:

**R1 -- Forecaster reconstruction (W=16).**
Trained on f[t] (forecaster latents).  Target = the W=16 values of patch t+1.
Since f[t] ~ e[t+1], the head learns to decode the patch that f[t] represents.

```
f[t] -> x_norm[(t+1)*W : (t+1)*W + W]
```

**R2 -- Encoder reconstruction (W=16).**
Trained on e[t] (encoder latents).  Target = the W=16 values of patch t itself.
The encoder latent directly represents its own patch.

```
e[t] -> x_norm[t*W : t*W + W]
```

**R4 -- Encoder reconstruction (W=128).**
Same as R2 but outputs 128 values (8 patches) per position.  The bidirectional GRU
provides cross-position context to support wider output.

```
e[t] -> x_norm[t*W : t*W + 128]
```

### Decode alignment (PR #33)

All results below use the corrected decode path from PR #33, which unified the latent
sequence as [e[0]..e[k], f[k+1]..f[k+m]].  The old code had a one-patch misalignment
caused by duplicating the first rolled token with f_ctx; see notes/FAILED_EXPERIMENTS.md for
pre-fix numbers.

## Evaluation

Each head was evaluated on GIFT-Eval (97 dataset configurations) using latent rollout.
The metric is GM-Relative MASE: the geometric mean of per-config MASE ratios relative
to a seasonal-naive baseline, aggregated across all 97 configs.  Lower is better.

### Rollout strategies

- **B4:** Each rolled position outputs W=16 values.  Used for R1 and R2.
- **B3R:** First position in each group of 8 rolled tokens provides 128 values per
  block.  Used for R4.
- **Value-space rollout (A1/A2):** Head runs once on context latents, outputs a full
  forecast in value space.  No latent rollout.  Used as baseline.

## Results

| Head | Type | Output | Strategy | GM-Rel MASE |
|------|------|--------|----------|-------------|
| R1   | Forecaster recon | W=16  | B4  | **1.1208** |
| R2   | Encoder recon    | W=16  | B4  | 1.1650 |
| R4   | Encoder recon    | W=128 | B3R | 1.1912 |
| A2   | Value-space      | W=16  | --  | 1.2620 |
| A1   | Value-space      | W=128 | --  | 1.2751 |

All three reconstruction heads outperform both value-space baselines.

## Analysis

### Reconstruction validates latent rollout

The original prediction heads gave GM-Relative MASE between 1.258 and 1.288 with latent
rollout, which was comparable to or slightly worse than value-space rollout (A1 = 1.275).
Reconstruction heads flip this: R1 at 1.121 is a **12% improvement** over the A1
value-space baseline.

This confirms the hypothesis.  The backbone's contrastive training already produces
latents where f[t] ~ e[t+1].  The head's job is to decode what a latent represents, not
to predict the future again.  Once the head is properly aligned, latent rollout
substantially outperforms value-space rollout.

### R1 > R2: forecaster latents are better targets

R1 (forecaster recon, 1.121) outperforms R2 (encoder recon, 1.165) by 3.8%.

At inference time, the rolled latents come from the forecaster (transformer output).
R1 trains directly on forecaster latents, so there is no distribution gap between
training and inference.  R2 trains on encoder latents e[t] and relies on f[t] ~ e[t+1]
at inference -- the approximation introduces a small mismatch.

### W=16 > W=128 for reconstruction

R2 (encoder recon W=16, 1.165) outperforms R4 (encoder recon W=128, 1.191).

Each latent represents a single W=16 patch.  Asking the head to output 128 values
(8 patches) from one latent forces it to extrapolate beyond what the latent encodes.
The GRU's bidirectional context helps, but the W=16 head that matches the backbone's
patch size performs better.

## Conclusion

Reconstruction heads dramatically improve forecasting quality over both prediction heads
and value-space rollout.  The best variant (R1, forecaster reconstruction W=16) achieves
a GM-Relative MASE of 1.121, a 12% improvement over the value-space baseline (A1 =
1.275).

The key insight is confirmed: the backbone's contrastive objective already handles
temporal prediction (f[t] ~ e[t+1]).  The head should reconstruct the patch each latent
represents, aligning training and inference so that latent rollout works as intended.
