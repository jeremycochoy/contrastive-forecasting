# Claude Code Instructions

## Remote Server
- **Elisa**: `jupyter@elisa`, workdir: `~/workspaces/contrastive-forecasting/`
- Two RTX 4090 GPUs (24GB each). Pick the most free GPU at runtime.

## Remote Machine Monitoring
- When a training run is launched on a remote/cloud machine (Vast.ai, etc.), always set up a periodic sync loop that copies checkpoints, loss CSV, and logs to a local directory every 15 minutes.
- This protects against preemption or unexpected instance termination.
- Sync at minimum: `*_best_gap.pth`, `*_best_loss.pth`, `*_losses.csv`, and the training log. Also sync periodic saves (`*_Nk.pth`) as they appear.
- Use `rsync -avz --timeout=60 -e "ssh ..."` for robustness.
- The sync loop should also watch for NaN, process death, and completion, and alert accordingly.

## Checkpoint Safety Rules
1. **After any long training run completes**, immediately copy the best checkpoint to a clearly named permanent file (e.g., `20L_H1024_2M_final.pth`). Never rely on `_best.pth` or periodic saves as the only copy.
2. **Never reuse `--save-path` when resuming training.** Always use a new, distinct path. The code has `safe_save_path()` to catch conflicts, but don't rely on it alone — be explicit.
3. **Before launching a continuation run**, verify the save path doesn't overlap with any existing checkpoint or its companions (`_best.pth`, `_optimizer.pth`).

## Training Conventions
- Contrastive backbone: `train_contrastive_v2.py`
- Recovery head: `train_parameter_recovery_v2.py` (frozen backbone)
- Recovery metric: MSE-based improvement ratio (Nx over zero-baseline), 4 AR + 4 MA coefficients
- Best recovery head: GRU h=128, 2 layers, MSE loss (~676K params)
- LR scaling for depth: multiply base LR by `sqrt(base_layers / new_layers)` when increasing depth

## Git Workflow
- `master`: base code
- `experiments`: main working branch (merged results)
- Feature branches from `experiments` for new experiment sets
- Always create PR, review, then merge
