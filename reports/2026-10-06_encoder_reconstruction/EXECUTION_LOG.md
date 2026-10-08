# Execution log: #425, the reconstruction head on the encoder output

All times are UTC, on 2026-10-07 unless a date is given.

## Where the files are

- Box: vast.ai 51431200, `ssh -p 31200 root@ssh5.vast.ai`. The queue code is `/workspace/cf-425` at 366c246f. Since 10-08 08:34, its `queue.sh` and its `jobs.tsv` are those of 2758c699, and `/workspace/cf-425-r3` keeps the code of the 56 scores. The results are in `/workspace/results/cf-425`, and the heads in `/workspace/ckpt/cf-425`.
- elisa: the heads are in `~/checkpoints_backup/cf-412/vast_lr100x/cf-425`. The box results are in `~/checkpoints_backup/cf-425/box_results`. The sync log is `~/checkpoints_backup/cf-425/sync.log`.
- elisa, the snapshot scores of 10-08: `~/checkpoints_backup/cf-425-snap`. `code/` is the code at a2bbd7ac, `ckpt/eval/<tag>/` holds the copy of each head and the files of its eval, and `results/` holds the scores and `run.log`.
- ABC inputs: the external disk of elisa holds the checkpoints of ABC, in `/media/jupyter/KINGSTON/checkpoints_backup/cf-412/vast_all/k3_r100_09_lr56_fix09_dec10k_lr10x`. The mirror and the box hold a copy of its 10 scored stops, at the paths of `scripts/jobs.tsv`.
- To fill this folder from elisa: `python3 scripts/collect.py && python3 scripts/check_scores.py && CF425_HEAD_ARCH=linear python3 scripts/check_scores.py && python3 scripts/plot_recon.py`.

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
| 10-08 00:42 | The sync loop of the box stops: it has no more file to bring. |
| 10-08 05:06 | `QUEUE_END: 56 scores` of the linear-head queue on elisa, with no failed job. `CF425_HEAD_ARCH=linear check_scores.py`: 56 of 56 jobs pass (`results/checks_lin.tsv`). |
| 10-08 05:21 | `snapshot_score.sh` with `CF425_SNAP_GPU`, on elisa: the earlier snapshot of 4 mean/std heads (BMS 100k, BMS 140k, OCF 40k, OCF 400k), and the final head of each as a control. Each GPU runs 3 scores at a time. |
| 10-08 05:36 | The 4 snapshot scores and 2 controls exist. |
| 10-08 05:49 | The 2 last controls exist. Each of the 8 evals ends with rc 0. `check_scores.py`: 56 of 56 jobs and 16 of 16 snapshot scores pass (`results/snapshots/checks.tsv`). |
| 10-08 06:08 | `snapshot_score.sh` with `CF425_SNAP_GPU`, on elisa: the `_best.pth` snapshot of the first head of TWN 100k and of TWN 420k, one on each GPU. |
| 10-08 06:22 | The 2 evals end with rc 0. `check_scores.py`: 56 of 56 jobs and 18 of 18 snapshot scores pass. |
| 10-08 09:30 | Report stage. New rules of the owner: the report holds the title and the figures only, no figure mixes the two heads, and no legend or key uses the word "seed". `plot_recon.py` draws each graph two times (transformer head, linear head), as R and as the overlay, with one shared y range for the two versions of a graph. The TWN markers of the final head and their key line are gone. The graphs now hold the Moirai group, for the MPM and MPE scores that come later. The tables stay in `results/`, and the protocol lives in the docstrings of the scripts. |
| 10-08 08:04 | The checkpoints of the 10 scored stops of ABC (40k to 460k) go from the external disk of elisa to the mirror. Each copy has the md5 of its source. |
| 10-08 08:15 | `stage_inputs.sh` copies the 10 ABC inputs to the box. Each box copy has the md5 of its source. |
| 10-08 08:33 | `deploy_elisa.sh` puts 2758c699 in the code folder of the linear queue. The trainer, the runner and the eval are the files of 557cb48e. |
| 10-08 08:34 | The box code folder gets `queue.sh` and `jobs.tsv` of 2758c699. The dry run of each queue plans one old-data wave of the 10 ABC jobs, and no other job. |
| 10-08 08:38 | `probe.sh` on the box, in test folders: the wave of the 10 ABC jobs trains 500 steps at 25.3 job steps/s, with 7,112 MiB of GPU memory (`results/probe_abc.txt`). No queue runs. |

## Events

- **Sync gap, fixed.** The tick after a wave brings about 5 GB and takes 24 minutes. When a score came during that tick, the prune of the tick deleted the final head from the box, and no later tick listed the folder. So elisa had the head and the score of OCB 40k and OMB 166k, but not their per-config table. `check_scores.py` found the gap. Commit 9955742b lists the folder of each scored job. The first tick of the new loop brought the files.
- **Snapshot scores on the box (10-07).** They are not jobs of the queue. They use the eval slots of the queue, and they do not write in its folders. While a score runs, a wave trains about 35% slower, because the box has 8 CPU cores.
- **Snapshot scores on elisa (10-08).** The 8 scores ran on the two GPUs of elisa. The box did not run them, and they trained no head. The eval code is the code of the two queues. For each of the 4 heads, the score of the final head on elisa is the score of the box to 4 decimals (0.2097, 0.4873, 0.3988 and 0.2727). So a snapshot score of elisa compares with a queue score of the box. `results/snapshots/scores.tsv` gives each snapshot score beside the score of the final head.
- **Snapshot scores of TWN (10-08).** The best training loss of each TWN head is at head step 30,000, so its `_best.pth` is its final head (the two files have the same SHA-256). Elisa gives 0.0351 for TWN 100k and 0.0543 for TWN 420k, the scores of the box. The MASE of each of the 97 configs agrees with the box to 0.002%. So the two scores are a control of the eval path of elisa for an old-data EWMA head. They do not measure the head noise of TWN, and no earlier snapshot of a TWN head exists. The figures draw the two scores as hollow markers on their dots, and one line of the key names them.
- **Linear heads.** The owner added a linear head for the 56 checkpoints on 10-07. Its queue runs on the two GPUs of elisa, in `~/checkpoints_backup/cf-425-lin`, and it needs no sync. The figures draw its scores as dashed curves.
- **Head loss of the mean/std runs.** The training loss of a mean/std head has a median near 0.25 and steps up to 160. The loss of an EWMA head is 0.009 to 0.04. All mean/std heads of wave 1 have their best loss at step 25,000.
- **Tests.** The two test files of the card give 173 passed. `test_two_lanes_of_one_stream_train_two_waves_at_one_time` failed in 1 of 3 full runs on 10-08, and it passed 3 times alone: it compares the times of two stub waves.
- **Report rewrite (10-08).** The draft report had text and tables. The final report holds the title and the figures. The verdict table and the head-noise table of the run stage are in the PR comments of 06:01 and 07:35 UTC, and their numbers are in `results/job_scores.tsv` and `results/snapshots/scores.tsv`.
