# Execution log: the paper ablations (#482)

This file is the plan of the runs and the record of the checks. The report
file comes later.

## The code

| What | Commit | Folder on elisa |
|---|---|---|
| Run code of the 13 rows other than 4 and 6: 038ea26c (the code of the reference) plus the flags of #482 | `c886f88e`, branch `run/cf-482-ablations` | `/home/jupyter/cf_runs/cf-482/code_c886f88e` |
| Run code of the rows 4 and 6: c886f88e plus the two fixes of the review (round 2) | `3d7b8c2d`, branch `run/cf-482-ablations` | `/home/jupyter/cf_runs/cf-482/code_3d7b8c2d` |
| Score code of the row 8: the score code of #425 plus the patch of #482 | `f541cf17`, branch `run/cf-482-heads` | `/home/jupyter/cf_runs/cf-482/ff_code_f541cf17` |
| Score code of the row 6: f541cf17 plus the TimesFM block with no layer normalisation | `c0124ad0`, branch `run/cf-482-heads` | `/home/jupyter/cf_runs/cf-482/ff_code_c0124ad0` |
| Score code of the other rows | the folder of #425, unchanged | `/home/jupyter/cf_runs/hp64/ff/code` |

The score code of #425 is a416b3ae with three local changes on elisa: the
exact eval speedups of 11-24 and the stream module of PR #480 (11-25). The
commit `2736f022` records these files. The reference heads used this folder.

Each code folder holds the read token of the card
(`cf-482/secrets/hf_token_agent.txt`), not the shared token.

## The rows

Each run is the reference with the change of its row only
(`scripts/make_runs.py`). The run folders are in
`/home/jupyter/cf_runs/cf-482/box_train/<run>/`.

| Row | Run | Flag change | Flag |
|---|---|---|---|
| 1 | `g166b_w320e6f6_ab01_no_synth` | `--mix-ratio 0` | existing |
| 2 | `g166b_w320e6f6_ab02_no_crossfade` | `--crossfade-triplets 0` | existing |
| 3 | `g166b_w320e6f6_ab03_no_mixup` | `--mixup-p 0` | existing |
| 4 | `g166b_w320e6f6_ab04_instnorm` | `--rev-norm-kind meanstd --meanstd-no-leak --meanstd-z-max 0` | new |
| 5 | `g166b_w320e6f6_ab05_no_timescale` | `--zero-time-scale-embedding` | new |
| 6 | `g166b_w320e6f6_ab06_timesfm_patch` | `--encoder-type timesfm` | new |
| 7 | `g166b_w320e6f6_ab07_no_sphere` | `--latent-sphere` removed | existing |
| 8 | `g166b_w320e6f6_ab08_no_conv` | `--depthwise-conv 0` | existing |
| 9 | `g166b_w320e6f6_ab09_no_dropkey` | `--encoder-dropkey 0` | existing |
| 10 | `g166b_w320e6f6_ab10_no_dropout` | `--dropout 0` | new |
| 11 | `g166b_w320e6f6_ab11_no_ema` | `--ema-tau 0`, `--ema-tau-end` and `--ema-tau-ramp-steps` removed | new (0 was refused) |
| 12 | `g166b_w320e6f6_ab12_no_sigreg_e` | `--sigreg-embedding` removed | existing |
| 13 | `g166b_w320e6f6_ab13_no_sigreg_h` | `--sigreg-encoding` removed | existing |
| 14 | `g166b_w320e6f6_ab14_no_lrep` | `--rep-loss-weight 0` | existing |
| 15 | `g166b_w320e6f6_ab15_align_d1` | `--train-rollout-depth 0` | existing |

`docs/paper_ablation_flags.md` describes the new flags. Notes on the rows:

- Row 1. The 2 forked ARMA rows become real rows (256 real rows, 3 crossfade
  rows). The crossfade needs `--synth-kind forked-arma`, which stays.
- Row 3. The sign flip of each series stays: in the code it is not a part of
  the mixup.
- Row 4. A full window keeps the anchor t with probability min(1, (t+1)/32):
  about 75 % of its anchors. A window with fewer than two whole patches of
  real values has no target after its context, and no term reads it (round
  2). The head of a meanstd backbone trains with the uni2ts split (the score
  code does this for each meanstd checkpoint).
- Row 5. `--freq-emb-dim 0 --seasonality-emb-dim 0` also stops the mixup
  (`maybe_mixup` skips a model with no embedding), and the score code cannot
  load such a backbone. The tables at 0 remove the time scale with no other
  change. The GRU reads 22 numbers, and the last 6 are 0.
- Row 6. The block is the ResidualBlock of TimesFM v1, with no layer
  normalisation (round 2). It reads the input of the GRU, with no padding
  mask.
- Row 9. The two DropKey share flags stay. At p = 0 they do nothing.
- Row 11. The teacher takes the student weights after each step. It still
  reads the full context with no DropKey and no dropout (the SimSiam form).
  The other reading, sg(h) of the same forward, makes the positive of
  L_rep the constant cos(h, sg h) = 1, so that row would change two things.
- Row 14. L_rep stays in the forward at the weight 0, so the compute is the
  same.
- Row 15. Only L_align reads the rollout depth in the reference loss.

## The checks of this stage

| Check | Result | Record |
|---|---|---|
| Parity: the reference script on code_038ea26c and on code_c886f88e, CPU, batch 128, 3 steps, the corpus of the #419 tests | equal: 35 columns of the losses CSV at each step | `results/parity_cpu.txt` |
| Each row script, CPU, batch 128, 2 steps | 15 of 15 ran with exit 0, and each flag acts (one sign per row) | `results/rows_cpu.txt` |
| Score path: the head trainer of the score code, 3 CPU steps on the backbone of each row (rows 6 and 8: the patched score code) | 15 of 15 trained with exit 0. Rows 6 and 8 get the TimesFM and the no-conv backbone | `results/heads_cpu.txt` |
| Score code: the patched folder builds the same backbone as the folder of #425 for the reference checkpoint at 100k (config and weights, head trainer and eval) | same | `results/check_head_code_reference_100k.txt` |
| Tests: `tests/test_482_*.py` and the full suite | 103 passed. The full suite: 99 fail, as on `origin/experiments` | the PR |

The checks of round 2 (the fixes of the review, PR #484):

| Check | Result | Record |
|---|---|---|
| Row path: the reference script on code_3d7b8c2d, as it is and with `--skip-nan-samples` (the step trains the contrastive objective by row, the path of row 4), CPU, batch 128, 3 steps | equal: 35 columns at each step, and equal to `results/parity_cpu.txt` | `results/rowpath_cpu.txt` |
| Rows 4 and 6 on code_3d7b8c2d, CPU, batch 128, 2 steps | 2 of 2 ran with exit 0. Row 4 prints the no-target rule, and row 6 has no `encoder.layer_norm` key | `results/rows_cpu_r2.txt` |
| Score path: the head trainer, 3 CPU steps on these two backbones (row 6: ff_code_c0124ad0) | 2 of 2 trained with exit 0 (strict load) | `results/heads_cpu_r2.txt` |
| Score code: ff_code_c0124ad0 builds the same backbone as the folder of #425 for the reference checkpoint at 100k | same config and weights | `results/check_head_code_reference_100k_c0124ad0.txt` |
| The S3 copies from the box, from end to end, on the S3 of the runs, with a local folder as the box (`scripts/check_box_copy_s3.sh`): copy, the newest pair only, a save that is not whole, the seed of a new box, no URL in an output | 6 of 6 steps pass, S3 clean at the end | `results/box_copy_s3_check.txt` |

## The score plan

The same score method as the reference: at each stop 10, 25, 50, 75, 100,
125, 150 and 166k, one control head of 30,000 steps (seed 20260722, the TF32
trainer `train_heads_tf32.py`, md5 6d3a59a1), then GM-MASE over the 97
GIFT-Eval configurations (`device=cuda configs=97 filter= fbs=64`). The
follower of #425 does it on the box, beside the training
(`start_follow_w320e6f6_box.sh`).

The flow of one box. The box scripts are in `scripts/`. A new box uses
their copy in `/home/jupyter/cf_runs/cf-482/tools_r3/`. The boxes of round 1
keep `tools_r1/`, and rows 4 and 6 started with `tools_r2/`. `tools_r3` is
`tools_r2` with one fix of the seed (a failed move of a file of the pair
stops the seed). The copy to S3 is the same in both. The box keeps the
layout `/workspace/hp64` of #425.

1. `rent_box.sh` of #425 (1 x RTX 5090).
2. `setup_box.sh`: the setup of #425, the run code, the run folder, the
   follower, the TF32 head trainer and the card token, with md5 checks. Rows
   6 and 8 need `HEAD_CODE` (row 6 `ff_code_c0124ad0`, row 8
   `ff_code_f541cf17`), and the script refuses them without it. For row 6 it
   also refuses a score code whose TimesFM block is not the block of the run
   code. It refuses a box with no `curl`. At a failed copy or an md5
   mismatch it stops with exit code 2. Its last step writes
   `/workspace/hp64/setup_done_<run>` on the box.
3. `start_box.sh`: it refuses a box with no `setup_done_<run>`. Then the
   supervisor, the follower, then `pull_box.sh` on elisa.
4. `pull_box.sh`, every 10 min:
   - to elisa: the scores, the eval folders, the losses CSV and the logs.
     No `.pth` file of the run comes to elisa.
   - from the box to S3: `s3_box_copy.py copy` (below).
5. `hb_check.sh`: the health check of #425 for the boxes in
   `cf-482/boxes.tsv`.

Elisa disk for 15 runs: the losses CSV of the reference has about 90 MB, so
about 1.4 GB. The eval folders with their heads, about 0.4 GB.

## The S3 copies: the box sends them

The uplink of elisa sends about 0.16 MB/s (11-27, 16:03 UTC), too slow for
the checkpoints. Thus elisa sends no `.pth` bytes: the box copies its files
to `s3://model.artifacts/manual/cf_runs/cf-482/<run>/ckpt/` itself. At each
poll, `s3_box_copy.py copy`:

1. lists the ckpt folder of the run on the box. A save is whole when its
   optimizer file is there (the trainer writes it after the backbone).
2. for the backbone of each stop, then the optimizer file of the newest
   stop: the box computes the ETag of the file (a whole zip file only), elisa
   makes a presigned PUT URL (15 min), and the box runs `curl -T` with it.
   The URL reaches curl on its input: no command line and no log holds it.
3. checks the object with `head_object`: its ETag must equal the ETag of the
   box file. `<run>/s3_box_record.tsv` names each file that passes.
4. deletes the older optimizer files of the run in S3, when the record names
   the two files of the newest stop. The stop backbones stay.

Thus S3 holds each stop backbone and one resume pair: the backbone and the
optimizer file of the newest stop. A key that S3 holds with other bytes gets
no copy (a log line). A trainer crash resumes from the last save on the box.
### Resume on a new box

When the box of a run stops for good:

1. Stop the elisa puller of the old box: `touch $CF/logs/stop_pull_<short>`.
   Wait for its log line `end`, then delete the stop file. Else the new
   puller ends at its start.
2. Find the pair: the newest `_optimizer.pth` of the run in
   `<run>/s3_box_record.tsv`.
3. Find the stops with no score on elisa. The stop k has a score when
   `$CF/ff/results/score_<run>_bb<k>k_h30k_control.txt` exists. Keep the
   stops at or after the step of the pair: the new box holds no backbone of
   an earlier stop. S3 holds the backbone of each stop in `<run>/ckpt/`.
4. Read the credit, then rent a new box as for a first start.
5. `setup_box.sh` of `tools_r3` with the same `CODE`, `RUN` and
   `HEAD_CODE`.
6. `python3 tools_r3/s3_box_copy.py seed <run> <box> <port>` (the rnd venv):
   the box downloads the pair from S3 with presigned GET URLs into
   `<run>/ckpt/`, and the ETag of each file must equal the S3 ETag. It
   refuses a box with no `setup_done_<run>` or with a `.pth` file of the
   run. At a failure it removes the pair from the box and ends with exit
   code 1. Go to step 7 only after exit code 0: with a half pair the run
   script starts from step 0.
7. `STOPS="<the stops of step 3>" start_box.sh` of `tools_r3`. The run script
   resumes from the pair (`--resume`) and writes `<run>_r2_<k>k.pth`. The
   follower waits for each stop in the order of `STOPS`.
8. Write the new box in `cf-482/boxes.tsv`.

Destroy the old box only under the rules of `CLAUDE.md`.

The pullers of `tools_r1` (the boxes of round 1) send each stop backbone
through elisa and no optimizer file. To move a running row to the puller of
`tools_r3`: stop its puller as in step 1, then start `tools_r3/pull_box.sh`
with the same variables. This changes no file on the box. The new puller
accepts the backbones that the old puller sent in parts: it compares the
multipart ETag.

## Facts about the reference for the report

- Its box ran the stream module of PR #480 from 35k to 45k ("the same
  windows, fewer Hub requests"). It resumed at 35k, 40k, 45k and 50k (a box
  move), and each resume starts the stream from a new seed. The ablations
  use the module of 038ea26c from 0 to 166k.
- A step has 259 rows: 254 real, 2 forked ARMA and 3 crossfade rows. At
  038ea26c, `--batch-size` does not count the crossfade rows (#476 changed
  this later).
- A window has 1,024 values, so 64 patches. The GIFT stream does not read
  `--t-raw 4096`.

## Risks

- Row 4 has no window filter: a short context can give values of |z| in the
  hundreds after the context. If the run stops with NaN, a fallback is the
  filter of Moirai 2.0 (`--meanstd-z-max 100`), as a stated change.
- Each run is one seed. The reference curve moves by up to 0.015 between
  two stops (0.9412 at 125k, 0.9260 at 150k), so a row change smaller than
  this is within the noise of one run.
- Row 4 trains on the row path of the trainer (`contrastive_objective`), as
  each meanstd run does. On the reference flags this path gives the losses
  of `forward_step` (`results/rowpath_cpu.txt`). Its GPU memory and speed on
  the box are not known yet.
- Row 6 has no layer normalisation after the patch encoder. At the start,
  SIGReg on e is about 110, against 12.5 for row 4
  (`results/rows_cpu_r2.txt`). Watch its first steps after the warmup.
