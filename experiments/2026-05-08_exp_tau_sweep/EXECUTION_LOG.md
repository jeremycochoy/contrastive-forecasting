# τ-sweep — execution log

Operational notes about how the runs were carried out. Anything in this file
would NOT belong in `RESULTS.md` if the experiment were re-run cleanly —
it documents the path we actually took, not what we learned from it.

## τ=0.20 — original arm5 (vast 36356043)

- Launched on a vast.ai 5090 spot via
  [`scripts/run_tau_sweep_vast_arm5.sh`](scripts/run_tau_sweep_vast_arm5.sh)
  on Fri 8 May 2026, ~17:30 BST.
- Auto-DONE marker fired at 19:17:54 BST (per
  `sync_tau_sweep/scp_back_arm5.log`); marker is only printed by the launcher
  *after* the python training process returns rc=0 (which only happens once
  `--total-steps 15000` is reached) AND the post-training
  `cp best_loss.pth → FINAL.pth` step succeeds.
  **Original arm trained the full 15,000 steps.**
- The watcher
  ([`scripts/scp_back_arm5.sh`](scripts/scp_back_arm5.sh))
  pulled `tau_sweep_0_20_FINAL.pth` (45,743,947 bytes, md5
  `28ab99d8c6ef5e2226fb4f6a4c939d00`) successfully.
- The watcher's six other `atomic_scp` calls
  (`_FINAL_optimizer.pth`, `_best_loss.pth`, `_best_loss_optimizer.pth`,
  `_best_gap.pth`, `_best_gap_optimizer.pth`, `_losses.csv`,
  `run_tau_0_20.log`) all printed nothing further. With `|| true` swallowing
  errors, this is consistent with the spot instance being terminated mid-pull
  after the first file. Result: **per-step trajectory CSV and optimizer
  state are not on local disk**; only `_FINAL.pth` survived.
- Postmortem and the resulting addition (`sync_loop` always-on requirement
  even for short runs) are recorded in
  [`experiments/REMOTE_LAUNCH_CHECKLIST.md`](../REMOTE_LAUNCH_CHECKLIST.md).

## τ=0.20 — resync attempt (vast 36363148, abandoned)

- Started Fri 8 May 21:15 BST as a from-scratch retrain on a fresh instance
  to recover the lost trajectory CSV. Sync loop wired up properly; reached
  step 4300 before the instance was abandoned in favour of a v2 launch with
  a distinct run-name (so partial artefacts wouldn't collide with the
  surviving original FINAL.pth).
- Local artefacts under `sync_tau_sweep_arm5_resync/`. Not used in the
  reported results.

## τ=0.20 — v2 first attempt (vast 14946f6ce654, partial)

- Started Fri 8 May 22:40 BST under run-name `tau_sweep_0_20_v2`. Python
  process died at step 7800 (process absent from `ps`, no traceback in log;
  cause not investigated).
- `tau_sweep_0_20_v2_FINAL.pth` was copied from the step ~7700 best_loss
  snapshot. An eval-CSV row for `tau_sweep_0_20_v2` was generated against
  that partial-training snapshot.
- Per-step trajectory CSV up to step 7800 is on disk
  (`sync_tau_sweep_arm5_v2/checkpoints/tau_sweep_0_20_v2_losses.csv`,
  ~3 MB). The earlier `RESULTS.md` reported held-out eval values from this
  partial separately, but it is operational noise (under-trained snapshot)
  and is not reported in the cleaned-up RESULTS.

## τ=0.20 — v2 full-15k retrain (in flight, elisa GPU 1)

- Launched Sat 9 May 07:48 BST locally on elisa GPU 1, PID 294991. As of the
  fact-check (~08:48 BST), at step ~11,800 / 15,000 with sps ≈ 3.3 (ETA
  ~16 min). Local artefacts in `sync_tau_sweep_arm5_v2/`.
- This run will produce the per-step trajectory CSV the original arm5 lost.
  When it lands, the trajectory plot can be regenerated to include τ=0.20
  alongside the other arms; the held-out eval row in `RESULTS.md` is
  expected to move only marginally (it is computed against an independent
  full-15k snapshot).

## Eval CSV note

`results/tau_sweep_metrics_v2.csv` still contains a `tau_sweep_0_20_v2` row
(the 7800-step partial). The CSV is the raw artefact and is intentionally
not edited; the cleaned-up `RESULTS.md` simply does not surface that row.
The Exp-2 `exp2_residual_silu_tau_0_10` row is also in the CSV and likewise
not part of this sweep.
