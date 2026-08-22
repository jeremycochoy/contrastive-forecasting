«Agent ExperimentRunner claude-opus-5 writing»

Item 2 is running. It no longer waits for the queue.

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

## Why it moved forward

The band's four draws run one at a time, because `head_vram_gate` holds one
flock per card. At 19:55 UTC draw one of four was 65 minutes into its
68-minute head. So the band's GPU work ends near 00:20 UTC, but the band
PROCESS ends near 01:30 UTC, after its last 72-minute eval on the CPU.

`band_queue.sh` waits for the process. It would have started the re-draw at
01:30 UTC and left card 1's GPU idle for about 70 minutes.

So the re-draw started at 19:56 UTC instead, with the same command the queue
would use.

```
[2026-08-20T19:56:47Z] [rep200k] seeds 20260722  heads student teacher  30000 steps  gpu 1
[2026-08-20T19:56:47Z] [rep200k] DRAW A4_k3_bb200k_student_s20260722 start
```

It holds no GPU. `fuser` on `/tmp/cf373_head_gpu1.lock` shows the band's head
holding the lock and the re-draw waiting behind it. Both cards read compute
mode `Default`, so `gpu_gate` returns at once and the flock is the only
queue. The re-draw takes the card the moment a band head releases it.

The queue did not double-fire. A `QUEUE_ONCE` probe against the live machine
reads the re-draw as up and launches nothing. The queue now marks stage 1
done when both re-draw heads score, and it keeps stage 2 for the 450k band.

## Where the nine items stand

| item | state |
|---|---|
| 1, 3, 4, 5, 6, 8, 9 | closed on disk |
| 2, the protocol re-draw at 200k | **running**, queued on the card-1 flock since 19:56 UTC |
| 7, the band at 450k | armed and tested. It cannot start before its checkpoint, due about 17:27 UTC on 2026-08-21 |

Item 7 is the only one that cannot move. Its checkpoint does not exist yet,
and the only way to make it exist sooner is to take card 0 from the driver.
I did not do that.

The leg on card 0 is at step 220,000 of 665,000. Nothing above touched it.

Tests: 183 in `test_407_full_pass.py`, 2,080 in the suite.
