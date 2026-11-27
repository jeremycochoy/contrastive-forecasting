# Execution log: the paper ablations (#482)

This file is the plan of the runs and the record of the checks. The report
file comes later.

## The code

| What | Commit | Folder on elisa |
|---|---|---|
| Run code: 038ea26c (the code of the reference) plus the flags of #482 | `c886f88e`, branch `run/cf-482-ablations` | `/home/jupyter/cf_runs/cf-482/code_c886f88e` |
| Score code of the rows 6 and 8: the score code of #425 plus the patch of #482 | `f541cf17`, branch `run/cf-482-heads` | `/home/jupyter/cf_runs/cf-482/ff_code_f541cf17` |
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
  about 75 % of its anchors. The head of a meanstd backbone trains with the
  uni2ts split (the score code does this for each meanstd checkpoint).
- Row 5. `--freq-emb-dim 0 --seasonality-emb-dim 0` also stops the mixup
  (`maybe_mixup` skips a model with no embedding), and the score code cannot
  load such a backbone. The tables at 0 remove the time scale with no other
  change. The GRU reads 22 numbers, and the last 6 are 0.
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
| Tests: `tests/test_482_*.py` and the full suite | to fill | the PR |

## The score plan

The same score method as the reference: at each stop 10, 25, 50, 75, 100,
125, 150 and 166k, one control head of 30,000 steps (seed 20260722, the TF32
trainer `train_heads_tf32.py`, md5 6d3a59a1), then GM-MASE over the 97
GIFT-Eval configurations (`device=cuda configs=97 filter= fbs=64`). The
follower of #425 does it on the box, beside the training
(`start_follow_w320e6f6_box.sh`).

The flow of one box (the scripts are in `scripts/`, and the box keeps the
layout `/workspace/hp64` of #425):

1. `rent_box.sh` of #425 (1 x RTX 5090).
2. `setup_box.sh`: the setup of #425, the run code, the run folder, the
   follower, the TF32 head trainer and the card token, with md5 checks. Rows
   6 and 8 need `HEAD_CODE=ff_code_f541cf17`, and the script refuses them
   without it.
3. `start_box.sh`: the supervisor, the follower, then `pull_box.sh` on elisa.
4. `pull_box.sh`, every 10 min:
   - to elisa: the scores, the eval folders, the losses CSV, the logs, the
     backbones of the stops (no optimizer file).
   - each backbone that loads goes to
     `s3://model.artifacts/manual/cf_runs/cf-482/<run>/ckpt/`
     (`upload_and_delete_pth.py`, ETag check), then leaves elisa.
5. `hb_check.sh`: the health check of #425 for the boxes in
   `cf-482/boxes.tsv`.

Elisa disk for 15 runs: the losses CSV of the reference has about 90 MB, so
about 1.4 GB. The eval folders with their heads, about 0.4 GB. A backbone
(93 MB) stays on elisa only until its upload.

The optimizer files stay on the box. A crash of the trainer resumes from the
last save on the box. If the box itself stops, the run starts again from 0.

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
  each meanstd run does. Its GPU memory and speed on the box are not known
  yet.
