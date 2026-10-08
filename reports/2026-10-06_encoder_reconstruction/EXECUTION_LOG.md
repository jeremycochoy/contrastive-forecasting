# Execution log: #425, the reconstruction head on the encoder output

All times are UTC, on 2026-10-07 unless a date is given.

## Where the files are

- Box: vast.ai 51431200, `ssh -p 31200 root@ssh5.vast.ai`. The queue code is `/workspace/cf-425` at 366c246f. The results are in `/workspace/results/cf-425`, and the heads in `/workspace/ckpt/cf-425`.
- elisa: the heads are in `~/checkpoints_backup/cf-412/vast_lr100x/cf-425`. The box results are in `~/checkpoints_backup/cf-425/box_results`. The sync log is `~/checkpoints_backup/cf-425/sync.log`.
- To fill this folder from elisa: `python3 scripts/collect.py && python3 scripts/check_scores.py && python3 scripts/plot_recon.py`.

## Timeline

| Time | Event |
|---|---|
| 01:08 | `forecast_scores.sh` ends: the 5 B4 forecast scores. |
| 05:56 | `queue.sh` starts on the box: the old-data wave (9 jobs) and wave 1 on GiftEvalPretrain (12 jobs). |
| 09:13 | elisa starts again. The sync loop starts again at 09:23. |
| 10:23 | The old-data wave ends (rc 0). Its 9 R scores exist at 11:26. |
| 11:43 | `snapshot_score.sh`: the earlier snapshot of 2 old-data heads, and 1 control. They end at 12:08. |
| 12:19 | Wave 1 ends (rc 0), and wave 2 starts. The 12 R scores of wave 1 exist at 13:36. |
| 13:38 | `snapshot_score.sh`: the earlier snapshot of 5 heads of wave 1. They end at 14:16. |
| 13:46 | The sync loop starts again with the fix of 9955742b. |
| 16:09 | Wave 2 ends (rc 0), and wave 3 starts. The 12 R scores of wave 2 exist at 17:27. |
| 19:55 | Wave 3 ends (rc 0), and wave 4 starts (11 jobs, the last wave). The 12 R scores of wave 3 exist at 21:13. |
| 20:54 | The linear-head queue starts on elisa (`queue_elisa.sh`, code 557cb48e, 4 lanes on 2 GPUs). |
| 23:17 | Wave 4 ends (rc 0). The box trains no more head. |
| 10-08 00:29 | `QUEUE_END: 56 scores` on the box, with no failed job. |
| 10-08 00:40 | `verify_mirror.sh`: elisa holds each box file at the same byte size, and no `.pth` stays on the box (`results/mirror_check.txt`). `check_scores.py`: 56 of 56 jobs pass. |

## Events

- **Sync gap, fixed.** The tick after a wave brings about 5 GB and takes 24 minutes. When a score came during that tick, the prune of the tick deleted the final head from the box, and no later tick listed the folder. So elisa had the head and the score of OCB 40k and OMB 166k, but not their per-config table. `check_scores.py` found the gap. Commit 9955742b lists the folder of each scored job. The first tick of the new loop brought the files.
- **Snapshot scores.** They are not jobs of the queue. They use the eval slots of the queue, and they do not write in its folders. While a score runs, a wave trains about 35% slower, because the box has 8 CPU cores.
- **Linear heads.** The owner added a linear head for the 56 checkpoints on 10-07. Its queue runs on the two GPUs of elisa, in `~/checkpoints_backup/cf-425-lin`, and it needs no sync. The figures draw its scores as dashed curves.
- **Head loss of the mean/std runs.** The training loss of a mean/std head has a median near 0.25 and steps up to 160. The loss of an EWMA head is 0.009 to 0.04. All mean/std heads of wave 1 have their best loss at step 25,000.
