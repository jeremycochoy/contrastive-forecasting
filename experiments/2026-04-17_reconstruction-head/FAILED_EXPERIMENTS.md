# Failed Experiments

## fR4: Forecaster reconstruction W=128

- **Head:** forecaster reconstruction, forecast_len=128, trained on f[t] → patches t+1 to t+8
- **Training:** `--reconstruction forecaster --forecast-len 128`, 30k steps. Training itself was valid.
- **Checkpoint:** `fR4_forecaster_recon_w128_best.pth`

### fR4 eval attempt 1: B1 strategy (wrong)
- **Result:** GM-Relative MASE = 1.170
- **Why wrong:** B1 asks each single latent to independently output 128 values. Each latent represents one patch (W=16) — the other 112 values are extrapolation without block context.

### fR4 eval attempt 2: B3 strategy (wrong position)
- **Result:** Killed after 1 config (MASE=0.933 on loop_seattle/5T/short)
- **Why wrong:** B3 takes output from the LAST position in each group of 8 rolled tokens. With forecaster reconstruction, position t outputs patches (t+1) to (t+8). The last position (T_ctx+7) outputs patches T_ctx+8 to T_ctx+15, but we need T_ctx+1 to T_ctx+8. The forecast is shifted by 7 patches.

## fR5: Encoder reconstruction W=128

- **Head:** encoder reconstruction, forecast_len=128, trained on e[t] → patches t to t+7
- **Training:** `--reconstruction encoder --forecast-len 128`, 30k steps. Training itself was valid.
- **Checkpoint:** `fR5_encoder_recon_w128_best.pth`

### fR5 eval attempt 1: B4 strategy (wrong)
- **Result:** Killed before completion
- **Why wrong:** B4 crops output to W=16 per token, discarding the W=128 capability. Pointless — equivalent to running a W=16 head.

### fR5 eval attempt 2: B3 strategy (wrong position)
- **Result:** Killed during setup (deps installing)
- **Why wrong:** Same B3 position issue as fR4. B3 takes the last position in each group, producing shifted forecasts.

## Root Cause

The B3 strategy was designed for old prediction heads, not reconstruction heads. For reconstruction, the position-to-output alignment is different:
- **Forecaster recon:** position t → patches t+1 to t+8 (need output from last CONTEXT position, not rolled)
- **Encoder recon:** position t → patches t to t+7 (need output from FIRST rolled position, not last)

B3 always takes the LAST position in each group, which is wrong for both reconstruction modes.

## Lesson

The rollout strategy must be designed specifically for reconstruction heads. A new R4/R5 experiment should:
1. Fix the B3 position selection for reconstruction alignment
2. Retrain heads if needed for the corrected setup
3. Verify alignment by checking that predicted time range matches ground truth

---

## B1-B4: Prediction heads (wrong hypothesis)

The B-variant heads were trained to **predict the future** from a latent: given f[t],
output values starting after position t.  This is redundant with the backbone's
contrastive objective, which already makes f[t] ~ e[t+1].  The head re-predicts what
the backbone already encoded, and was never trained to decode rolled latents.

| Head | Strategy | GM-Rel MASE | Description |
|------|----------|-------------|-------------|
| B1   | B1       | 1.2581      | Prediction W=128, each position outputs full forecast_len |
| B2   | B2       | 1.2581      | Prediction W=128, crop 128->W per position |
| B3   | B3       | 1.2602      | Prediction W=128, last position in each group |
| B4   | B4       | 1.2882      | Prediction W=16, per-position output |

All B-variants perform near or slightly worse than the A1 value-space baseline (1.275).
The prediction head cannot benefit from latent rollout because it was trained to predict
the future from context latents, not to reconstruct what rolled latents represent.

For comparison, the best reconstruction head (R1) achieves 1.121 -- a 12% improvement
over B1/B2 and an even larger gap over B4.

## R3: Rolled reconstruction (design mistake)

- **Head:** forecaster reconstruction W=16, same architecture as R1
- **Training:** loss computed **only on rolled positions**, not on context positions
- **GM-Relative MASE:** 1.3916

R3 was intended to handle distribution shift between real and rolled latents by training
exclusively on rolled latents.  However, this starved the head of supervision: most of
the sequence consists of context positions, and the loss was zeroed out on all of them.
The head received gradients from only a few rolled positions per sample.

Since f[t] ~ e[t+1], rolled latents should be close to real encoder latents.  R1 (which
trains on all positions) achieves 1.121, confirming that distribution shift is not a
significant problem.  R3's poor result (1.392) is explained entirely by insufficient
supervision, not by any fundamental issue with reconstruction from rolled latents.

**Lesson:** Train the reconstruction head on all positions.  The distribution gap
between real and rolled latents is small enough that full-sequence supervision is far
more important than specializing for rolled inputs.

## Pre-fix alignment results (before PR #33)

All latent rollout evaluations before PR #33 had a one-patch (W=16 values) misalignment
in the decode path.  The old code used f_ctx (forecaster latents over the context) as
the context portion of the decode sequence, which duplicated the first rolled token.
A `skip_first_rolled` flag shifted the forecast by one patch to compensate, but the
underlying alignment was still wrong.

PR #33 fixed this by unifying the decode sequence as [e[0]..e[k], f[k+1]..f[k+m]] for
all head types, eliminating the duplicate and the shift.

Pre-fix GM-Relative MASE values (broken alignment):

| Head | Pre-fix GM-Rel MASE | Post-fix GM-Rel MASE |
|------|---------------------|----------------------|
| R1   | 1.168               | 1.121                |
| R2   | 1.164               | 1.165                |

R2 was least affected by the fix because the encoder reconstruction path was already
correctly aligned: e[t] -> patch t does not depend on the forecaster context sequence.
The misalignment only affected how the forecaster latents were stitched into the decode
sequence, which matters more for R1 (trained on f[t] targets).

R1 improved substantially after the fix (1.168 -> 1.121), confirming that the alignment
correction was important for forecaster-based reconstruction.
