# Tiny Backbone Training

Training infrastructure and incident documentation for the Tiny v2 backbone (~20M params, H=512, L=6, W=16, GRU encoder) trained on HuggingFace streaming data.

## Key Result

Successful backbone training to gap 0.428, but uncovered critical issues in checkpoint completeness and data handling. Four PRs (#13-#16) merged to fix: complete training state in checkpoints, gradient clipping, multi-epoch data restart, and RNG state restore across PyTorch versions.

## Documents

| File | Description |
|---|---|
| [INCIDENT_NAN_AND_RESUME.md](INCIDENT_NAN_AND_RESUME.md) | Root cause analysis of NaN crash at step 24,970 (all-NaN row in HF data) and discovery of incomplete checkpoint resume state. Documents the harmful zero-fill intermediate fix and the correct skip-row solution. |

## Training Dashboard

![Training dashboard](training_dashboard.png)

## NaN Crash Analysis

![Crash analysis](crash_analysis.png)

## Code Changes

| PR | Description |
|---|---|
| #13 | Save complete training state: best_loss, EMA, RNG, hf_rows |
| #14 | Optional gradient clipping |
| #15 | Restart data stream on exhaustion for multi-epoch training |
| #16 | RNG state restore across PyTorch versions |
