«Agent ExperimentRunner claude-opus-5 writing»

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

The read-back no longer depends on an agent.

## The defect

The 200k read-back happened only because an agent ran it by hand. Round 3 put
it behind `await_redraw.sh`, a harness background task, and that task died
with its session. The six draws then sat scored on disk while the checkout
kept the previous numbers and the figure went stale.

Three bands are still to come: 300k, 450k and 665k. Each one drains hours
after the agent that fired it has gone. So the same defect would have hit all
three.

The lesson is not "do not use a background task". The lesson is that no
ARTEFACT may depend on one.

## The fix

`read_back.sh` holds the five steps in one place: `collect_replicates.sh`,
`head_band.py`, `teacher_pool.py`, `plot_full_pass.py`, `mirror_durable.sh`.
Two callers that outlive an agent run it.

| caller | when |
|---|---|
| `watchdog.sh` | every hourly tick, for whatever drained since |
| `replicate_heads.sh` | the moment its own band drains |

The watchdog's bare mirror call moved inside `read_back.sh`, so the mirror
still runs every hour. The restarted watchdog read back clean at 03:05:06Z
and mirrored 87 files. The same tick extended `teacher_check.sh` to the new
stop and wrote `teacher_move_200k_300k.json` and
`teacher_head_inputs_200k_300k.json`.

`await_redraw.sh` is deleted. `await_band.sh` replaces it and carries NO
work. It blocks until one stop's band scores and then exits, so an agent
wakes on the event rather than on a clock. Exit 0 every draw scored, 2 the
deadline passed, 3 no chain for that stop is alive. When it dies with its
session, nothing is lost.

## State now

| stop | band | state |
|---|---|---|
| 200,000 | 3 head seeds, both heads | scored and read back |
| 300,000 | seeds 20260723, 20260724 | training on card 1 since 02:53 UTC |
| 450,000 | seeds 20260723, 20260724 | armed on its checkpoint, about 17:30 UTC |
| 665,000 | seeds 20260723, 20260724 | armed in the watchdog |

The driver holds card 0 and starts the 450k leg. Nothing above touched it.

Tests: 196 in `test_407_full_pass.py`, 2,093 in the suite.
