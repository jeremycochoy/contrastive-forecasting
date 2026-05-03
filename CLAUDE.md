# Claude Code Instructions

## Project intent

Contrastive-forecasting trains small transformer backbones on financial-style
time series via contrastive prediction (forecast vs future cosine similarity,
with cross-batch and cross-channel negatives). Goal: backbone that beats
GIFT-Eval baselines on GM-MASE / GM-MAPE_SN / GM-CRPS_SN. Current focus: full-
epoch retrain on `jeremycochoy/gift-pretrain-full-4096` (path `small_v1`,
~42.57M windows) with MOIRAI optimizer hyperparameters.

## Empirical learnings (don't relearn these)

- **MOIRAI HP wins.** lr=1e-3, wd=0.1, β2=0.98, no warmup, no cosine beats
  default HP (lr=1e-4, wd=0.01, β2=0.999) on 30k steps (#9 vs #6: GM-MASE
  1.6391 vs 1.8043, GM-MAPE_SN 1.1850 vs 1.3698, GM-CRPS_SN 1.0155 vs 1.1000,
  PR #104). Use MOIRAI HP for any new run unless otherwise specified.
- **Resumed runs corrupt loss-std.** Fresh runs at bs=96 do NOT show a per-
  batch loss-std jump. Resumed runs DO (+52% std at the resume boundary,
  persists for 130k+ steps). Cause unknown after exhaustive investigation
  (RNG state, optimizer binding, hidden buffers, PR #106 port — all refuted).
  One remaining suspect: CUDA RNG state (we save CPU torch RNG and numpy RNG,
  not CUDA RNG). **Practical: cannot safely resume mid-flight; if interrupted
  the model trains with worse statistics post-resume even though it loads
  cleanly.** Restart fresh.
- **Throughput sweet spot bs=256.** bs=96 prefetch=2 is data-bound (~738
  samples/s). bs=256 prefetch=16 is ~614 samples/s — slower because the
  contrastive loss has B² cross-batch term that dominates at large B. bs=384
  OOM'd in backward at 32GB VRAM, bs=512 OOM'd in cross-batch tensor at 23GB.
  bs=256 is the practical sweet spot for 32GB VRAM (75% used).
- **HF dataloader resume fix (PR #112).** Pre-sweeping all 4274 parquet
  shards' metadata to find the resume start-shard takes 18 min on cold cache.
  Now reads shard 0's metadata only and assumes uniform sizing — works for
  the gift-pretrain bundles whose shards are uniformly 10000 rows each at
  upload. Don't revert.
- **PrefetchIterator doesn't actually parallelize.** The Python thread is
  CPU-bound and holds the GIL during HF/parquet decode. To get >2×
  throughput we'd need multiprocessing workers (PyTorch `DataLoader(
  num_workers>0)` pattern). Not landed yet.
- **Never use grad-clip in this project.** The fix for any divergence is to
  fix the data / normalization, not clip gradients.

## Code style

- Scripts under `scripts/` (top level for plotting), `experiments/<date>_exp_*/`
  for per-experiment launchers and reports.
- Plot scripts use a consistent color scheme: blue / green / red / orange
  across all panels for the four-arm comparison.
- `B<batch>` convention in plot labels for batch-size-dependent runs.
- Run-name convention: `tiny_<dataset>_<arm>_<suffix>` for backbones,
  `R1q_<dataset>_<arm>_<suffix>` for quantile heads.

## How the user works

- Direct, terse. Wants short responses. Doesn't want "would you like me
  to..." for trivial follow-ups.
- Skeptical of plots — frequently catches glitches (missing curves, hidden
  curves under same-color overlays, scale issues). Always self-check rendered
  plots before delivering.
- Approves destructive Vast.ai operations explicitly (`vastrun-destroy
  --force`) only when asked.
- Prefers PRs reviewed and merged in small focused units.
- Uses sub-agents liberally for parallelizable work; expects same from Claude.

## Memory rule

> **Do not write project-specific memory to `~/.claude/projects/.../memory/`.**
> Anything project-relevant goes in this `CLAUDE.md` or in
> `docs/contrastive-forecasting-handoff.md`. Memory written to the user's
> laptop is not portable and will be lost when migrating.

## Remote Server
- **Elisa**: `jupyter@elisa`, workdir: `~/workspaces/contrastive-forecasting/`
- Two RTX 4090 GPUs (24GB each). Pick the most free GPU at runtime.

## HuggingFace token

Every vast.ai run that streams from HF datasets **must** authenticate, or
HF's anonymous rate-limit will throttle the stream and idle the GPU
(observed: 0.5–1.5 sps with GPU util ~20% vs 5–9 sps with token).

The read-only token lives at `experiments/hf_token.txt` (gitignored so
GitHub secret scanning doesn't reject pushes). Put your token there
manually; in run scripts:

```bash
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

## Remote Machine Monitoring
- **Assume the machine can crash at any time.** Every sync must protect against this.
- When a training run is launched on a remote/cloud machine (Vast.ai, etc.), always set up a periodic sync loop that copies checkpoints, loss CSV, and logs to a local directory.
- **Sync frequency:** Every 15 minutes (fixed). A ramped-up schedule was tried and offered no net benefit — a single tick itself can take 2–5 minutes over the vast.ai scp proxy for large checkpoints.
- **Atomic writes:** Always download to a `.tmp` file first, verify file size (checkpoints must be >70MB), then `mv` over the old copy. A crash mid-transfer must never corrupt existing local copies.
- Sync at minimum: `*_best_gap.pth`, `*_best_gap_optimizer.pth`, `*_best_loss.pth`, `*_best_loss_optimizer.pth`, `*_losses.csv`, the training log, and periodic saves (`*_Nk.pth` + `*_Nk_optimizer.pth`) as they appear. **Always sync optimizer files** — without them, resume loses step counter, RNG state, and AdamW momentum.
- Use `scp` (not `rsync` on macOS — it's v2.6.9 and unreliable through vast.ai proxy).
- The sync loop should also watch for NaN, process death, and completion, and alert accordingly.
- **After launching any background process, ALWAYS verify the first output before reporting it as running.** No exceptions — wait for the first cycle, check the log, confirm it produced correct results. Do not assume it works because similar scripts worked before; the environment may differ.
- **Always dry-run / test the sync loop before leaving a training unattended**, and **every time the sync code changes**. One full tick with at least one `✓ <file>` line per expected pattern (log, backbone, head, losses CSV) is the acceptance gate. Without this, you cannot rely on crash recovery — and you may silently drop files due to wrong min-size thresholds, wrong patterns, or wrong host/port. Learned the hard way (PR #45): a 70 MB min-size floor applied blanketly to `*.pth` silently dropped every 2.4 MB head checkpoint without logging a recognisable warning.
- **Verifying the sync means manually checking the files are there — all of them you expect.** Reading `sync.log` alone is insufficient; a missing `✗` line can just mean the pattern didn't match (silent bug), not that the file wasn't needed. After at least one full tick, open `<LOCAL_DIR>/checkpoints/` and confirm *by name and by size* that every file class that exists on the remote also exists locally: backbone `.pth`, backbone `_optimizer.pth`, head `.pth`, head `_optimizer.pth`, losses CSVs, run log. A missing class = broken sync regardless of what the log says.
- **Size thresholds are per-file-class, not per-extension.** Backbone ~80 MB, optimizer ~150 MB, head ~2.4 MB, losses CSV a few KB — never one floor for everything.
- **EVERY remote training run must have a sync_loop running for the duration of the run.** Not just long ones — short runs lose just as much when SSH drops on the final pull. The sync_loop pulls periodic snapshots throughout, so when the instance dies you still have the most recent ≥5k-step checkpoint as a fallback. The PR #45 RevIN run learned this the painful way: no sync_loop + manual final scp + SSH drop = unrecoverable.
- **NEVER use raw `scp` to pull a checkpoint from a remote.** It writes directly to the destination, so a connection drop mid-transfer leaves a partial/corrupt file in place of the previous good copy. Use `experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh <host> <port> <remote> <local> [min_bytes]` instead — it scp's to `.tmp`, size-checks, rotates the existing file to `.prev`, then atomic-mv's. The previous good copy survives a partial transfer.
- **The sync_loop also rotates one-deep**: when it pulls a fresh `<file>.pth`, the existing one moves to `<file>.pth.prev` before the new one is dropped in. That backup survives even if a future tick fetches a corrupt-but-large-enough file.

## Checkpoint Safety Rules
1. **After any long training run completes**, immediately copy the best checkpoint to a clearly named permanent file (e.g., `20L_H1024_2M_final.pth`). Never rely on `_best.pth` or periodic saves as the only copy.
2. **Never reuse `--save-path` when resuming training.** Always use a new, distinct path. The code has `safe_save_path()` to catch conflicts, but don't rely on it alone — be explicit.
3. **Before launching a continuation run**, verify the save path doesn't overlap with any existing checkpoint or its companions (`_best.pth`, `_optimizer.pth`).
4. **Sync directories live in the main checkout, never in an auxiliary worktree.** The local sync target for a remote training run (and any local `--save-dir` if training locally) must be an absolute path under the main repo checkout, e.g. `/Users/.../trading/contrastive-forecasting/sync_<run_name>/`. Never under `.claude/worktrees/<name>/`, never under a sibling worktree like `contrastive-forecasting-<feature>/`. Reason: `git worktree remove [--force]` deletes ALL untracked files in the worktree directory. Learned the hard way (Apr 2026): tearing down a code-review worktree with `--force` deleted the only local copy of an 80 MB span=512 backbone after the corresponding remote vast.ai instance had already been destroyed — forced retraining and a missing visual for the report. Code work (refactors, PRs) can happen in a worktree per the Git Workflow section, but the worktree tree must stay free of valuable untracked state. Anything irreplaceable goes back to the main checkout immediately.
5. **Before `git worktree remove [--force]`**, run `git -C <worktree> status -uall` (or `ls <worktree>/`) and verify nothing irreplaceable lives there outside git. If anything does, move it to the main checkout first. If in doubt, ask before removing.

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
- **Use a git worktree for any multi-file refactor or non-trivial change.** Don't mutate the user's checked-out tree mid-work; create a worktree (e.g. via `EnterWorktree`), do the work there, push the branch, open the PR. The user often has uncommitted state on `experiments` (training scripts, sync logs, in-flight runs) and the worktree keeps that untouched. If the worktree branch is created from a stale HEAD, `git reset --hard refs/heads/experiments` it forward before starting.
