«Agent ExperimentRunner claude-opus-5 writing»

Follow-up to the round-3 comment. Items 2 and 7 wait on card 1, so the code
that fires them carried the whole risk. I ran that code instead of reading
it.

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

A sandbox runs `band_queue.sh` against a stub `replicate_heads.sh` that
records its arguments. Eight tests cover the path.

| case | result |
|---|---|
| stage 1 fires seed 20260722 at 200k | `200000 20260722` |
| stage 1 waits while a band holds card 1 | no launch |
| stage 2 waits for stage 1 | no launch |
| stage 2 fires on the 450k checkpoint | `450000` |
| stage 2 waits without a checkpoint | no launch |
| a half-scored re-draw does not read as done | re-fires |
| the fire cap stops a runaway | 2 launches, then it gives up |
| a drained band ends the queue | no launch, clean exit |

## One defect, found and fixed

`replicate_alive` matched `*/replicate_heads.sh` from **any** checkout. The
band running now was launched by a relative path, so a basename test was the
only guard. A second worktree of this repo would have read as "a band is up",
and this queue would have waited for the whole card.

It now resolves `argv[1]` against the launching process's own working
directory and demands this checkout's copy. I checked both directions: the
tightened test still sees the live 200k band, and a sandbox run no longer
sees it.

The queue also gained `QUEUE_MAX_FIRES`, default 4. A draw that died at once
would otherwise have re-launched every 300 seconds for 40 hours.

The live queue restarted on the verified script at 19:42 UTC. It had fired
nothing, so the restart cost no work. One queue runs, PID 3575010.

## State

Items 1, 3, 4, 5, 6, 8 and 9 are closed on disk. Items 2 and 7 are armed and
now verified. Both are time-gated on card 1: the re-draw fires near 01:00
UTC, and the 450k band fires on its checkpoint at about 17:27 UTC on
2026-08-21. Neither can finish in this session, and neither blocks the 300k
stop.

The leg on card 0 is at step 218,000 of 665,000. I did not touch it.

Tests: 181 in `test_407_full_pass.py`, 2,078 in the suite.
