# Execution log

Operational events of this card. The report holds the science. This file
holds what happened to the machines, so a later reader can tell a measurement
from an incident.

## Two agent sessions share one checkout

Two ExperimentRunner sessions drove this card at the same time, on one box
with two RTX 4090 cards. Other projects held 4 to 6 GB on each card
throughout. The sessions split the work by lane and agreed the split by
message. One session drove the rate sweep, the report and the PR comment. The
other drove the decay arm, the heartbeat and the reference tables.

## An edit under a running shell killed three lanes

Bash reads a script by byte offset and re-reads the file after each command.
A shell that is running a script resumes inside the NEW text after an edit,
and dies on the first syntax error it meets there.

`run_arm.sh` changed at 21:28 and `phase1.sh` at 21:44 on 2026-09-06, under
three lanes. Each lane died with `cf412_auc_warmup: command not found`, then a
syntax error. The three ended three different ways.

- `k3_r100_09b`, at 21:33:03. Its leg finished and its checkpoint WAS scored.
  `lane_b_relay.sh` waits on the pid and not on the exit code, so it started a
  fresh `phase1.sh`. That one found the checkpoint on disk, made the leg a
  no-op and ran the head. The relay absorbed the failure by accident.
- `k3_r100_09` at 100,000 steps, at 22:14:28. Its leg finished and its
  checkpoint sat unscored. A head started by hand at 22:15.
- `k32_r100_09_dec`, at 01:03:05. Its leg did NOT finish. The AUC gate stopped
  it at step 18,634, and its `leg_40k` holds no `_40k.pth`.

So one checkpoint of the three sat unscored. No training was lost.

### `phase1.log` is unreliable in both directions

`run_arm.sh` should have exited `CF412_RC_COLLAPSED`, which is 4, and
`phase1.sh` should have logged "LOST the contrastive task". It exited 2
instead. Two lines of `results/phase1.log` carry that exit code, and they hide
opposite things.

```
line 21  [09-06 22:14:28] backbone k3_r100_09 stop 100000 FAILED rc=2
line 65  [09-07 01:03:05] backbone k32_r100_09_dec stop 40000 FAILED rc=2
```

Line 65 is a real result written as a failure. That leg lost the contrastive
task at step 18,634, which is this card's clearest finding.

Line 21 is the reverse error. That leg SUCCEEDED. It wrote its 100,000-step
checkpoint, and that checkpoint scored 1.3395.

So read neither from `phase1.log`. The verdicts are in
`results/collapsed_<arm>.txt`, `results/auc_verdicts.tsv` and
`results/scores.csv`.

### The doubled gate line is the fingerprint

The corrupted shells did not simply die. Each one RE-RAN a command it had
already run. Every one of the three logged its AUC gate line twice, from one
process (`results/arms.log`):

```
[09-06 15:15:52] arm k3_r100_09b     AUC gate pid 364747 - ... warmup 1000
[09-06 21:33:03] arm k3_r100_09b     AUC gate pid 364747 - ... warmup
[09-06 15:34:23] arm k3_r100_09      AUC gate pid 380882 - ... warmup 1000
[09-06 22:14:28] arm k3_r100_09      AUC gate pid 380882 - ... warmup
[09-06 20:12:52] arm k32_r100_09_dec AUC gate pid 663584 - ... warmup 1000
[09-07 01:03:05] arm k32_r100_09_dec AUC gate pid 663584 - ... warmup
```

Same pid, same line, hours apart. The second of each pair has an empty warmup,
because `cf412_auc_warmup` did not exist in the `study.sh` that shell sourced
when it started. Bash resumed at a stale byte offset, re-entered a block it had
finished, and fell into text that was not there when it began.

### Two guards came out of it

- `run_snapshot/` is a frozen copy of `scripts/`. A lane launched from it
  cannot be edited under itself. It is gitignored, because it is a copy of
  committed files.
- `scripts/missing_heads.sh` sweeps for any checkpoint that has no score and
  no running head, and starts one. It is the net under every lane.

A file that must change while a shell runs it is replaced through `mv`, never
edited in place. The running shell keeps the old inode.

## Four shells died at about 04:20, cause unknown

One session lost four background shells inside one window: its decay-arm lane
driver, the lane C driver, the sweep loop and a file watcher. Its heartbeat
survived.

The launch method does not explain the cull. The surviving heartbeat used
`setsid nohup`, and so did two of the four that died. Lane C used `nohup` with
no `setsid`, and the watcher was a harness background task. No file under
`scripts/` or `run_snapshot/` changed after 22:16, so an edit under a running
shell is not the cause either.

No training was lost. All four trainers and both `run_arm.sh` shells stayed
alive. What died was the scoring above them.

Recovery, at 04:48 and 04:51:

- The sweep loop restarted from the frozen copy, on gpu 0, period 300 s.
- A waiter armed on the 200,000-step checkpoint. It starts that head itself,
  after the same three checks the sweep makes, so the two cannot both start
  it.

The cause is not known and this log does not guess at one.

## `results/lane_d.log` is not committed

Two sessions opened that file at once, one with `>`, so its text interleaves
at two offsets. Every line it held is also in `results/arms.log` and
`results/phase1.log`, which append.

## The frozen copy is load-bearing

The sweep reads `run_snapshot/arms.tsv`. An arm added to `scripts/arms.tsv`
after the freeze is invisible to it, and that arm's checkpoint would sit
unscored with no warning. Add an arm, then re-copy `scripts/` to
`run_snapshot/` and restart the loop between ticks.

`head_eval.sh` calls `head_eval_bb.sh` of the 2026-08-08 study, which is
outside the snapshot.
