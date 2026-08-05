# #393 execution log

Operational events only. The report carries the science; this carries what
happened to the machines, so the report does not have to.

Times are elisa's. The vast.ai containers run one hour behind, so a log
line stamped `03:12` on a vast box is `04:12` here. Nothing depends on the
skew, but do not diff timestamps across machines without correcting it.

## 2026-08-05 00:49 — arm5_combab_alignT died at step 0 on vast A

`torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or
unavailable`, raised inside `.to(device)` at model init. The box is in
`Exclusive_Process` compute mode and `arm5_combab_alignS` already held the
GPU, so the second CUDA context could not be created. The cell was dead
one second after launch and stayed dead for three hours: it consumed no
GPU time, so nothing downstream looked wrong.

Fixed by `scripts/gpu_gate.sh`, which every CUDA process in `run_leg.sh`
and `eval_stop.sh` now passes through. A queued cell waits for the device
instead of dying on it. The cell was restarted on its own box, cf393-c.

No other cell died this way. `arm6_v2_combab_alignT` also exited rc=1 at
01:30, but for an unrelated reason — `--align-target teacher` needed the
trainer change in `adde0cc` — and was relaunched at 01:34.

## 2026-08-05 03:57 — scp corrupted a running eval_stop.sh on vast A

Deploying the gate with `scp` overwrote `eval_stop.sh` **through its
existing inode** while a copy of it was running. bash reads a script
lazily, holding a byte offset into that inode; the insertions shifted
everything after them, so when the 15,000-step head finished and bash read
its next command it landed mid-word inside `--head-num-layers`:

    eval_stop.sh: line 97: ad-num-layers: command not found
    [03:12:58] head-train rc=127

`ladder.py` raises `SystemExit` on a non-zero worker, so the
`arm5_combab_alignS` driver died. The head itself had trained cleanly
(`rc=0`, `_final.pth` written), so the loss was the GIFT-Eval only, not
the 38 minutes of head training.

Three consequences, all handled:

- **vast B was armed with the same fault** and 90 seconds from hitting it.
  Restoring the original bytes *through the same inode* (`cat >`, not
  `mv`) put the running bash's saved offset back over the text it was
  parsed from, and its eval continued. The new version went on after.
- **vast A started a duplicate cell.** The 02:22 session's
  `queue_remote.sh` had been parked in its `while pgrep ladder.py` wait
  since 00:49; the driver's death released it and it began
  `arm5_combab_alignT` — a cell already running on cf393-c. Killed at
  ~1,400 steps, before any checkpoint, along with the queue. Its partial
  directory on vast A was removed so the box holds no stale copy.
- **elisa was untouched.** The local Edit tool writes a new inode and
  renames over the old one, so the three running cells kept reading the
  bytes they started with. That contrast is the fix: `scripts/deploy_scripts.sh`
  uploads to a temporary name and `mv`s it into place, so a remote deploy
  behaves the way the local edit already did.

## 2026-08-05 04:00 — four boxes added, and why

The 02:22 layout had two cells queued per vast box and two more behind a
waiter on elisa GPU 0 that was watching for the #390 session to release
the device. The spend order on the PR is "all ten cells to bb100k first",
and that layout could not deliver it: two cells had not started at all,
and the waiter's condition was outside this experiment's control.

cf393-c/d/e/f took `arm5_combab_alignT`, `arm6_v2_nse_alignT`, `arm1_nse`
and `arm4_combab`, one cell each. The waiter was killed so it could not
double-launch the two cells it was holding. `arm4_combab` moved off elisa
because it was queued *inside* `--cells arm6_v2_nse_alignS,arm4_combab`,
and that driver predates `HOLD_ABOVE`: it will be refused its 200k leg by
`run_leg.sh` and exit, taking the queued cell with it.
