# Claude Code Instructions

## Remote Server
- **Elisa**: `jupyter@elisa`, workdir: `~/workspaces/contrastive-forecasting/`
- Two RTX 4090 GPUs (24GB each). Pick the most free GPU at runtime.

## Remote Machine Monitoring
- **Assume the machine can crash at any time.** Every sync must protect against this.
- When a training run is launched on a remote/cloud machine (Vast.ai, etc.), always set up a periodic sync loop that copies checkpoints, loss CSV, and logs to a local directory.
- **Sync frequency:** Every 5 minutes for the first hour, then every 15 minutes.
- **Atomic writes:** Always download to a `.tmp` file first, verify file size (checkpoints must be >70MB), then `mv` over the old copy. A crash mid-transfer must never corrupt existing local copies.
- Sync at minimum: `*_best_gap.pth`, `*_best_gap_optimizer.pth`, `*_best_loss.pth`, `*_best_loss_optimizer.pth`, `*_losses.csv`, the training log, and periodic saves (`*_Nk.pth` + `*_Nk_optimizer.pth`) as they appear. **Always sync optimizer files** — without them, resume loses step counter, RNG state, and AdamW momentum.
- Use `scp` (not `rsync` on macOS — it's v2.6.9 and unreliable through vast.ai proxy).
- The sync loop should also watch for NaN, process death, and completion, and alert accordingly.
- **After launching any background process, ALWAYS verify the first output before reporting it as running.** No exceptions — wait for the first cycle, check the log, confirm it produced correct results. Do not assume it works because similar scripts worked before; the environment may differ.

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
- `master`: stable base code
- `experiments`: main working branch (merged results)
- **Never commit directly to `experiments` or `master`.** Always create a feature branch from `experiments`, do the work there, then PR into `experiments`.
- Always create PR, review, then merge
