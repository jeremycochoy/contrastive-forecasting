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
