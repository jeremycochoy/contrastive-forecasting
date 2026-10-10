# Execution log: the frequency family of forecast decoders

This work has no issue card. The owner asked for it in chat on 2026-10-10.
All times are UTC, on 2026-10-10. The log files of elisa show British Summer Time, one hour later.

## Where the files are

- elisa: `/home/jupyter/cf_runs/freq_family`. A restart of elisa keeps it.
  - `code/`: the code that the waves read, from `scripts/deploy.sh`. `code/DEPLOYED_COMMIT` names the commit.
  - `code_v2/`: the code of fix round 1 (see below). `code/` holds ad1984a6 for the wave of BLK 200k that runs.
  - `heads/eval/<tag>/`: the head of an arm, its loss CSV and the files of its eval.
  - `results/`: `scores.tsv`, `config_mase.tsv`, `arms.tsv`, the scores and the logs.
  - `smoke/`: the test wave of 500 steps. No wave reads it.
  - `timing/`: the B4 score on GPU 1, and the two probes.
- This folder: `results/smoke/` and `results/b4_gpu_check/` hold the small files of the checks below.

## The checks before the waves

### A family of one member is the standard head

`--freq-family-members 16` sends each row and each config to the member 16.

| Check | Where | Result |
|---|---|---|
| Loss rows and final weights, 4 steps, each body and each rule | `tests/test_freq_family.py`, CPU | identical, bit for bit |
| Forecast of the eval, each body | `tests/test_freq_family.py`, CPU | identical, bit for bit |
| Loss rows of 500 steps, in one wave with the control | `smoke.sh`, GPU 1, BLK 200k | 500 of 500 rows identical for `shared_strict_m16` and for `heads_strict_m16` |
| Final weights after 500 steps | `smoke.sh` | 28 of 28 tensors identical, for each of the two |
| B4 MASE of 6 configs | `smoke.sh`, CPU | 6 of 6 identical (1.2653 for the control and for the two) |

### The member of each config

`tests/test_freq_family.py` checks the member of each of the 97 configs two times. The first test reads the frequency in the config name. The second test reads the frequency that each GIFT-Eval dataset holds on elisa. The counts are 64: 30, 32: 31, 16: 30 and 128: 6. The yearly and the quarterly config get the member 16.

The eval log names the member of each config. From the test score of `shared_draw`:

```
[eval] bizitobs_service/10S/short: frequency 10S, family member 128
[eval] ett1/15T/short: frequency 15T, family member 64
[eval] ett1/H/short: frequency H, family member 32
[eval] m4_weekly/W/short: frequency W-SUN, family member 16
[eval] m4_yearly/A/short: frequency A-DEC, family member 16
[eval] us_births/D/short: frequency D, family member 16
```

`collect_scores.py` reads these lines. It gives a family arm no row when a config has another member than the member of its frequency.

### The test wave of 500 steps

`smoke.sh` at 09:53, code 56780425, GPU 1, backbone BLK 200k. One process trained 7 arms on one data stream: the control, the 4 family arms and the two families of one member. It ended with code 0 at 09:58.

Rows of each member, of 128,000 rows (500 steps of 256 rows):

| Arm | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| `shared_strict` | 26,335 (20.6%) | 18,233 (14.2%) | 81,590 (63.7%) | 1,842 (1.4%) |
| `heads_strict` | 26,335 (20.6%) | 18,233 (14.2%) | 81,590 (63.7%) | 1,842 (1.4%) |
| `shared_draw` | 13,136 (10.3%) | 49,525 (38.7%) | 37,254 (29.1%) | 28,085 (21.9%) |
| `heads_draw` | 13,193 (10.3%) | 49,173 (38.4%) | 37,132 (29.0%) | 28,502 (22.3%) |

The two strict arms read the same batches, so they have the same counts. Each draw arm makes its own draws. The draw counts agree with the strict counts. Half of the rows of 16 stay at 16. The member 128 gets a third of the rows of 64 and half of the rows of 128: 28,118 rows expected.

Under `strict`, the member 128 gets 1.4% of the rows. If this share holds, that is about 110,000 of the 7.68 million rows of a wave of 30,000 steps.

Speed of the 7 arms: 2.07 to 2.27 steps/s in 4 of the last 5 reports. That is 14.5 to 15.9 job steps/s. The stream took 0% to 4% of the time in these reports, so the GPU set the rate. Memory: the process had a peak of 5.75 GiB allocated and 6.74 GiB reserved. `nvidia-smi` gave 7,308 MiB for GPU 1. At 15.6 job steps/s, the 150,000 job steps of a wave of 5 arms take 2.7 hours.

Parameters: 3,605,136 for the control, 3,771,456 for a `shared` family and 14,420,544 for a `heads` family.

### One B4 score on GPU 1

`b4_gpu_check.sh` scored the standard head of BLK 200k on GPU 1 with 4 shards. The code was 88f967ca: the eval path of a standard head did not change after it. The box scored the same head on the CPU on 10-05: 1.1262.

| Fact | Value |
|---|---|
| Time of the score, 97 configs, 4 shards | 37 min 24 s (09:12:56 to 09:50:20) |
| Peak GPU memory, 4 shards | 2,703 MiB |
| GM-Relative MASE, GPU of elisa | 1.126275 (the eval prints 1.1263) |
| GM-Relative MASE, CPU of the box | 1.126227 (the eval prints 1.1262) |
| Largest MASE difference of one config | 0.058% (`m4_quarterly/Q/short`) |
| Median MASE difference of one config | 0.0016% |
| Configs with a difference above 0.01% | 17 of 97 |

Two probes (`b4_probes.sh`):

- CPU of elisa against the CPU of the box, 6 configs: the MASE agrees to 0.00002%. So the CPU score does not change with the machine.
- GPU with TF32 off, 4 configs: the MASE moves by 0.0004% at most, and stays 0.02% from the CPU value. So TF32 is not the cause of the difference. The cause is not known.

Time of a CPU score on elisa: two sets of 6 configs ran on one CPU core. The core took 5.3 and 6.7 times the time of one GPU shard. The 4 GPU shards took 143 minutes together. So one CPU score needs 13 to 16 core-hours, which is 3.2 to 4 hours with 4 shards. The box took 3 h 17 min.

One GPU process alone ran the 4 probe configs 9% faster than a shard beside 3 other shards. So more than 4 shards can make a GPU score shorter. No run measured this.

**Decision.** The GPU does not give the same score as the CPU. So `run_wave.sh` scores on the CPU, as the standard score does. `FF_EVAL_DEVICE=cuda` scores on the GPU: 37 minutes for one score, with a GM that is 0.004% higher for this head. Score each arm of one wave on one device.

**Decision of the Orchestrator, 10:33.** Each wave of this work scores on the GPU (`FF_EVAL_DEVICE=cuda`). The CPU stays the default of the script. `scores.tsv` names the device of each row.

## Tests

`CUDA_VISIBLE_DEVICES= python3 -m pytest tests`, one process for each file, 8 at a time:

| Code | Passed | Failed | Skipped |
|---|---|---|---|
| 73d29629 | 4,477 | 0 | 3 |
| 6a27ba82 | 4,477 | 1 | 3 |

The failed test is `test_390_launcher_shape.py::test_monitor_does_not_quit_before_the_first_arm_is_up`. The monitor of #390 looks for its runs in the process table of the machine, where the other test processes ran. The test passed in the first run, and it passed 3 of 3 times alone. This work changes no file of #390.

`tests/test_freq_family.py`: 151 passed with 6a27ba82.

The tests of #412, #417, #421 and #425: 666 passed with 6a27ba82, and 666 passed on the base commit 88f967ca.

## Fix round 1: the scripts

The review and the audit of ad1984a6 found no bug in the family training or in the B4 path. They found gaps in the scripts. This round changes the scripts of this folder and `tests/test_freq_family.py`. It does not change the head trainer, the eval, `head_eval_bb.sh`, `eval_local.sh` or `src/`. So the heads and the scores of the code of ad1984a6 stay valid.

| Gap | Fix |
|---|---|
| Two starts of one wave trained the same arms two times. | A start makes its list of arms again after its wait for the stream. |
| Two starts scored one tag at the same time. | One start scores a tag at a time (`locks/score_<tag>.lock`). The wait ends after `FF_SCORE_WAIT` (one day). |
| A start kept a score of another config filter or of another device. It also went on from the shard tables of another device. | `run_wave.sh` writes the protocol of a score to `heads/eval/<tag>/gift/protocol.txt` before the eval: the device, the config count and the filter. A start with another protocol refuses the arm. `FF_RESCORE=1` moves the score and the eval folder to `old_eval_<time>/` and scores again. |
| The ratio compared two scores with other configs or another device. | `scores.tsv` has the column `device`. An arm gets a ratio only when its control has the same configs and the same device. The collect script names each arm with no ratio. |
| The collect script did not check the config count. | An eval table with another count than its protocol file gets no row, and the script ends with code 1. |
| The wait for the table lock had no end. | It ends after `FF_TABLES_WAIT` (600 s). |
| A wave with no GPU memory kept the stream lock during its scores. | Each way out of the training frees the stream. |
| The follower took the checkpoint of an older start while the trainer wrote the file of a new start (`_r2_`), or after one load that failed. | The follower looks again when a newer file does not load. A file that did not load at each look of `FF_SETTLE` (300 s) is cut. The follower then names it in its log and takes the older start. |
| A second start of `b4_gpu_check.sh` timed a skip. | The script refuses a second start in a folder. |
| `family_row_members` gets no vocabulary, and no guard refused a v1 backbone. | A test shows that each v1 id gives the member of its name. `run_wave.sh` refuses a family arm on a backbone that is not v2. |
| No test had more than 1 channel. | A test with 2 channels shows that each channel of a sample trains the member of the sample. No code change. |

The wait for an eval slot (`eval_slot.sh`) has an end: `CF393_EVAL_SLOT_TIMEOUT`, one day unless the caller sets it. The score then fails with no score file, and a later start scores the arm. A test shows it.

A score of the code of ad1984a6 has no protocol file. For such a score, `run_wave.sh` and `collect_scores.py` read the device in the `eval start` lines of `heads/eval/<tag>/stop.log`, and the config count in the eval table. These files name no filter. So a start with a filter refuses such a score.

**Two code folders.** `FF_CODE=<folder> bash deploy.sh` puts the code in another folder. The deploy replaces its target, so it refuses a folder with files that no deploy made. A deployed script reads the code folder that holds it (`code_folder.sh`). So `follow_abc_gift.sh` of `code_v2` starts waves with the code of `code_v2`.

An independent review of this round found 6 more gaps in the new code. Each one has a fix and a test:

- The deploy target above.
- A test score of ad1984a6 with another filter of the same count.
- The order of the two moves of `FF_RESCORE=1`: the score goes first.
- A protocol file with no text.
- One load that fails in the follower.
- The code folder of `b4_gpu_check.sh`, `b4_probes.sh` and `smoke.sh`.

**Caution.** The wave of BLK 200k runs with the code of ad1984a6, which holds no score lock. Do not start `blk 200` again before the line `WAVE_END` in `results/wave_blk200.log`.

## How to start the waves

On elisa, after `FF_CODE=/home/jupyter/cf_runs/freq_family/code_v2 bash scripts/deploy.sh` in a checkout of the branch. Each start gets the score device of this work:

```bash
S=/home/jupyter/cf_runs/freq_family/code_v2/reports/2026-10-10_frequency_family_head/scripts
R=/home/jupyter/cf_runs/freq_family/results
mkdir -p $R
export FF_EVAL_DEVICE=cuda
# BLK at 200k: the control and the 4 family arms.
nohup setsid bash $S/run_wave.sh blk 200 \
  ~/checkpoints_backup/cf-412/vast_lr100x/cf-419c/cos200k/leg_665k/cf419_cos200k_r2_200k.pth \
  >>$R/wave_blk200.log 2>&1 </dev/null &
# abc_gift: one wave for each stop, when its checkpoint exists.
nohup setsid bash $S/follow_abc_gift.sh control shared_strict >>$R/follow.log 2>&1 </dev/null &
# The tables again, from the scores on disk.
PYTHONPATH=${S%/reports/*} python3 $S/collect_scores.py
```

One wave trains at a time: the second waits for the stream lock. On the CPU, the 5 scores of a wave use 20 cores. The follower starts the scores of a stop and then trains the next stop. A second follower with other arms can run at the same time.

`bash $S/smoke.sh` runs the test wave again, and `bash $S/b4_gpu_check.sh` the B4 score on the GPU.
