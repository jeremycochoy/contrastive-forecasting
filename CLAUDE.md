# Claude Code Instructions

## Project intent

Contrastive-forecasting trains small transformer backbones on time series via contrastive prediction (forecast vs future cosine similarity, with cross-batch and cross-channel negatives). Goal: backbone that beats GIFT-Eval baselines on GM-MASE / GM-MAPE_SN / GM-CRPS_SN. Every experiment should maximise *useful information gained* per unit time / compute / $ — before any action, ask "does this increase the information we have?".

## Empirical learnings
- **Never use grad-clip in this project.** Fix divergence via data / normalization, not clipping.

## Code style

- Each experiment lives under `experiments/<YYYY-MM-DD>_<name>/` with its own `results/`, `plots/`, `sync/`. No stray results/plots at repo root.
- Shared utilities in top-level `scripts/`; experiment-specific launchers go in the experiment dir as `run.sh`.

## How the user works

- Agent is autonomous; troubleshoot independently. Escalation has cost — weigh it against the cost of inaction.
- Direct, terse. Short responses. No "would you like me to..." for trivial follow-ups.
- Prefers PRs reviewed and merged in small focused units.
- Uses sub-agents liberally for parallelizable work; expects same from Claude.

## Reporting

See [`experiments/REPORT_STANDARD.md`](experiments/REPORT_STANDARD.md) — the checklist a sub-agent runs against every report. Information that doesn't fit the report can be recorded elsewhere in the experiment directory (scripts, doc, execution log).

## Operational rules from prior incidents

- **Pause before any `vastrun-destroy`.** Check whether a resume bundle (`model.pth`, `optimizer.pth`, losses CSV, run.log) is on the instance — if yes, edit the continuation script in-place and re-launch on the live instance instead. May 3 2026: destroyed an instance holding #10's resume bundle, lost ~$2.94 + ~46k steps.
- **Resume bundle = `scripts/push_resume_bundle.sh`** (PR #96), not "four-files-from-memory". Bundles `<run>.pth + <run>_optimizer.pth + <run>_losses.csv + run_<arm>.log` atomically with a preflight gate.
- **vastrun-kit only — never raw `vastai`.** Kit at `~/Desktop/workspace/trading/vastrun-kit/` (laptop) or `~/vastrun-kit/` (elisa). If a kit command is missing/broken, file a `vastrun-kit` issue and use `vastrun-forward` as audited fallback.
- **Vast.ai is a SHARED account across concurrent agent sessions.** Never destroy/stop an instance unless its contract ID was returned by *your own* `vastrun-provision` AND its label + image/GPU class match what you set/requested.

## Memory rule

> **Do not write project-specific memory to `~/.claude/projects/.../memory/`.** Project-relevant content goes in local md docs.

## Remote Server
- **Elisa**: `jupyter@elisa`, workdir: `~/workspaces/contrastive-forecasting/`
- Two RTX 4090 GPUs (24GB each). Pick the most free GPU at runtime.

## HuggingFace token

Every vast.ai run that streams from HF datasets **must** authenticate, or HF's anonymous rate-limit throttles the stream and idles the GPU (0.5–1.5 sps with util ~20% vs 5–9 sps with token).

Read-only token at `experiments/hf_token.txt` (gitignored). In run scripts:

```bash
export HF_TOKEN=$(cat experiments/hf_token.txt)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

## Remote Machine Monitoring

**Before any remote launch, walk through [`experiments/REMOTE_LAUNCH_CHECKLIST.md`](experiments/REMOTE_LAUNCH_CHECKLIST.md) — every box must be checked, including a verified `sync_loop.sh` tick on elisa.**

- **Assume the machine can crash at any time.** EVERY remote training run (short or long) must have a `sync_loop` running for its full duration — short runs lose just as much when SSH drops on the final pull (PR #45 RevIN run: no sync_loop + manual final scp + SSH drop = unrecoverable).
- **Sync frequency: 15 min fixed.** A single tick takes 2–5 min over the vast.ai scp proxy.
- **Atomic writes only.** Download to `.tmp`, size-check per file class, `mv` over old copy. Sync_loop also rotates one-deep (`.pth` → `.pth.prev` before new file lands), so a corrupt-but-large-enough fetch still leaves the prior good copy.
- **Per-class size thresholds** (never one floor for everything): backbone ~80 MB, optimizer ~150 MB, head ~2.4 MB, losses CSV a few KB. Blanket 70 MB floor on `*.pth` silently drops 2.4 MB head checkpoints (PR #45).
- **Sync at minimum**: `*_best_gap.pth`, `*_best_gap_optimizer.pth`, `*_best_loss.pth`, `*_best_loss_optimizer.pth`, `*_losses.csv`, training log, periodic `*_Nk.pth` + `*_Nk_optimizer.pth`. **Always sync optimizer files** — without them, resume loses step counter, RNG state, AdamW momentum.
- Use `scp`, not `rsync` (macOS rsync v2.6.9 is unreliable through the vast.ai proxy). For one-off pulls, **use `experiments/2026-04-27_periodic-synth-mix/scripts/safe_pull.sh <host> <port> <remote> <local> [min_bytes]`**, never raw `scp` — raw scp writes directly to the destination, so a mid-transfer drop corrupts the prior good copy.
- Sync loop also watches for NaN, process death, completion.
- **After launching ANY background process, verify the first output before reporting it as running** — wait for the first cycle, check the log, confirm correct results. Don't assume similar scripts work in this environment.
- **Verify sync by `ls`, not by reading `sync.log`.** A missing `✗` line can mean the pattern didn't match (silent bug). After at least one full tick, confirm by name and size that every remote file class exists locally (backbone `.pth`, backbone `_optimizer.pth`, head `.pth`, head `_optimizer.pth`, losses CSV, run log). Re-test every time the sync code changes.

## Checkpoint Safety Rules
1. **After any long training run completes**, immediately copy the best checkpoint to a permanent name (e.g. `20L_H1024_2M_final.pth`). Don't rely on `_best.pth` or periodic saves alone.
2. **Never reuse `--save-path` when resuming.** Always a new, distinct path. `safe_save_path()` catches conflicts but be explicit.
3. **Before launching a continuation run**, verify the save path doesn't overlap with any existing checkpoint or its companions (`_best.pth`, `_optimizer.pth`).
4. **Sync directories live in the main checkout, never in an auxiliary worktree.** Local sync target (and any local `--save-dir`) must be an absolute path under the main repo, e.g. `/Users/.../trading/contrastive-forecasting/sync_<run_name>/`. `git worktree remove [--force]` deletes ALL untracked files in the worktree directory — Apr 2026: tearing down a code-review worktree with `--force` deleted the only local copy of an 80 MB span=512 backbone after the remote was already destroyed. Code work in worktrees is fine; valuable untracked state is not.
5. **Before `git worktree remove [--force]`**, run `git -C <worktree> status -uall` (or `ls`) and move anything irreplaceable back to the main checkout. If in doubt, ask.

## Git Workflow
- `master`: stable base code.
- `experiments`: main working branch (merged results).
- **Never commit directly to `experiments` or `master`.** Feature branch from `experiments`, do the work there, PR into `experiments`. Always create PR, review, then merge.
- **Use a git worktree for any multi-file refactor or non-trivial change** (e.g. via `EnterWorktree`) — keeps the user's uncommitted state on `experiments` untouched. If the worktree branch is from a stale HEAD, `git reset --hard refs/heads/experiments` it forward before starting.

### Pre-PR checklist

- [ ] Working in a worktree, not the main `experiments` checkout.
- [ ] Feature branch from `experiments`; PR targets `experiments` (never `master`).
- [ ] PR body opens with `«Agent <model> writing»` if agent-authored.
