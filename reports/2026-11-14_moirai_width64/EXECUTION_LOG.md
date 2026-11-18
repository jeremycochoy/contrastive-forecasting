# Execution log of #462

## The MPE recipe in the code of today (11-14)

- The MPE command is line 1 of `run_cf421ew_moirai_k3.log` in
  `s3://model.artifacts/manual/checkpoints_backup/cf-412/vast_lr100x_results/cf-421ew/`.
  `results/mpe_command.txt` holds a copy.
- No file records the commit of MPE. Its first try (09-29) and the try that
  trained (10-01) ran `3d57bf9a` (#421) or `5e258487` (#412, the branch of
  the #412 runs on 10-01). Their value paths are the same: `5e258487` only
  moves a part of `value_objective` into `inert_inputs`. The checks below
  use `5e258487`.
- Later commits that reach the MPE flags:
  - `289b2fc8` (10-06) drops each value target in the first window of the
    EWMA. This acts at each step. `--keep-start-window-targets` keeps these
    targets, as MPE did.
  - `289b2fc8` also changes the row search of a step with a gradient spike.
    MPE dropped no row in its legs 0 to 50k and 150k to 166k, so this change
    did not act on MPE.
  - The step diagnostics of today read only the rows with no padding. So
    their random draw changes, and the random stream of the run changes from
    step 2. This does not change the recipe.
- A/B test on GPU 1: the MPE command at width 64, `5e258487` against today
  with the flag. Step 1 is equal bit for bit: loss, the 4 depth terms and
  gradient norm. With the diagnostics on every row (a patch in a scratch
  copy), steps 1 to 6 are equal bit for bit. Without the flag, the loss of
  step 1 changes (2.66139 against 2.66129).

## The score path

- The A2V eval of today (`eval_local.sh`, CPU) scored the MPE checkpoint of
  166k on 6 configs. Each MASE agrees with the MPE table to 5.3e-6
  (relative) or less.
- `src/gift_scores.py` on the MPE tables gives 1.0247 at 10k and 0.9362 at
  166k, the values of `summary.txt`.
- The eval reads the number of encoder layers from the keys of the save
  (`experiments/2026-04-13_gift-eval/scripts/eval_gift_eval_official.py:887-900`)
  and loads the weights with `strict=True` (line 1004). Each of the 8 shard
  logs of the smoke eval shows `auto-detected num_encoder_layers=1`.

## The smoke (`run.sh smoke`)

- 300 steps on GPU 0, which other runs share: 1.0 step/s (994 ms: data 184,
  forward 334, backward 475). So 166k steps take about 46 h.
- Peak GPU memory of the trainer: 5,652 MiB with the memory knobs of
  `train.sh`. Without the knobs, two of these runs did not fit in the
  9.8 GB that GPU 1 had free.
- The save of step 300 has its optimizer file. The A2V score on 2 configs and
  the loss figure ran.

## Notes for the run stage

- MPE trained in 8 legs. Each leg seeds the data stream with the rows read
  before it. This run trains in one process, so its data order differs from
  MPE after 10k.
- `run.sh train` resumes after a crash. `FOLLOW=1 run.sh score` scores each
  stop when its save and its optimizer file load (`scripts/checkpoints.sh`).
- The orchestrator moves the first-try `$R/code` and `$R/logs` aside. Then it
  launches the run from a frozen copy of the reviewed head. The audit fixes
  of `05cb7c6d` do not change the training.

## The run

- The run started from step 0 on GPU 1 at 01:56 UTC on 11-15, from the
  frozen copy of `c80d7309`.
- Elisa rebooted at 21:55 UTC on 11-15, at step ~98,400. At 10:21 UTC on
  11-16, the run resumed from its 95k save and its optimizer file (the
  files `cf462_mpe_w64_r2_*`). So the run has 2 legs, and steps 95k to
  ~98.4k trained two times.
- The run ended at 00:20 UTC on 11-17 with code 0. Each leg ran at
  1.38 step/s. `results/logs/supervise.log` and `results/logs/score.log`
  record each start and each score.
- `s3://model.artifacts/cf462_mpe_w64/` holds 76 files: the 34 saves with
  their optimizer files, 6 CSV logs, `train.log` and `CODE_COMMIT`.
