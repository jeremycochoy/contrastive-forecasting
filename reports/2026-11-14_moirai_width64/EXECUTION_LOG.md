# Execution log of #462

## The MPE recipe in the code of today (11-14)

- The MPE command is line 1 of `run_cf421ew_moirai_k3.log` in
  `s3://model.artifacts/manual/checkpoints_backup/cf-412/vast_lr100x_results/cf-421ew/`.
  `results/mpe_command.txt` holds a copy.
- No file records the commit of MPE. Its 10k leg started on 10-01. Then the
  branch of the #412 runs, `feature/contrastive-forecasting-412-ours-moirai`,
  was at `5e258487`, the first code with each flag of the command.
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
  stop when its optimizer file is on disk.
