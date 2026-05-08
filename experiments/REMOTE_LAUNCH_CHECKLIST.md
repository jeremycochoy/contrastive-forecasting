# Remote-launch preflight

Before launching ANY training on a remote (vast.ai) instance, walk through
this checklist. **Every box must be checked.** Don't launch until they are.

- [ ] Provisioned via `vastrun-provision` (never raw `vastai`). Contract
      ID, label, and GPU class match what you set. Vast.ai is a shared
      account — only touch instances you provisioned yourself.
- [ ] Code rsynced to `/workspace/app/` on the remote; HF token exported
      in the remote launcher (`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`).
- [ ] **`sync_loop.sh` running on elisa**, pulling FROM the remote,
      logging to `sync_<run>/sync.log`. *Started before training
      launches, not after.* Use the canonical pattern (atomic-mv,
      per-class size thresholds, append-regression guard).
- [ ] **First sync tick verified by `ls`** — every file class you expect
      is in `sync_<run>/checkpoints/` (backbone `.pth`, optimizer
      `.pth`, head `.pth` if applicable, losses CSV, run log). A missing
      class means a broken pattern, not "we'll catch it next tick".
      Reading `sync.log` alone is insufficient (a missing `✗` line can
      just mean the pattern didn't match).
- [ ] Local sync target is under the main repo checkout
      (`/home/jupyter/contrastive-forecasting/sync_<run>/` on elisa),
      never a worktree. `git worktree remove [--force]` would delete
      everything untracked there.
- [ ] One-shot DONE-marker scp-backs are NOT a substitute for a
      sync_loop. Use them only on top of a running sync_loop.

If you can't tick any box, fix it or destroy the instance and retry.
Don't launch and "fix it later" — the sweep arms that lose data are
the ones with no sync_loop running.

## Why this exists

**2026-05-08 τ-sweep arm 5 incident.** τ=0.20 was launched on a vast
spot in parallel with the elisa arms. Instead of `sync_loop.sh`, a
one-shot `scp_back_arm5.sh` was wired up to fire on the DONE marker
and pull `FINAL.pth`. The launcher reached DONE successfully and the
`FINAL.pth` was retrieved. But:

- `losses.csv` was not in the scp-back's file list — only the FINAL
  weights and the optimizer.
- The vast spot then auto-stopped after training, wiping the disk.
- The per-step trajectory CSV (R²/U/AUC/Top-1 by step) was lost.

A `sync_loop.sh` would have pulled the CSV every 15 min throughout the
run, and the data would have been preserved on elisa regardless of how
the instance ended. The CLAUDE.md rule "EVERY remote training run must
have a sync_loop running for the duration of the run" already said
this — the gap was a checklist that gates *when* the rule applies.
