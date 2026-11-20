# The width-64 search (from 2026-11-08)

Small runs that test one change at a time on the same recipe. Each run has width 64, 8 heads, 1 encoder layer, 3 forecaster layers and the latent sphere, except the width-384 runs. Each run trains 166k steps at batch 256 with a constant learning rate of 1e-3. The EMA teacher ramps to 1.0 at 166k.

The scores are the forecast GM-Relative MASE on the 97 GIFT-Eval configs (lower is better). A forecast head trains on each stop save: 0 (the 250-step save), 10, 25, 35, 45, 50, 60, 75, 90, 100, 115, 125, 150 and 166k.

## The runs

Each run name is `g166b_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere<suffix>_rollfix`.

| Code name | Suffix | L_align | Distances (weight) | SIGReg on f |
|---|---|---|---|---|
| base | (none) | cosine | 1, 2, 3, 4 (x1) | no |
| SIGReg-F | `_sigf` | cosine | 1, 2, 3, 4 (x1) | yes |
| EMA-EXP | `_emalog` | cosine | 1, 2, 3, 4 (x1). EMA tau linear in ln(1 + step) | no |
| 10-L-ALIGN | `_align10` | cosine | 1, 2, 3, 4 (x10) | no |
| L2-B-NORM | `_l2raw` | L2 of the raw vectors | 1, 2, 3, 4 (x1) | no |
| red | `_sigf_align10pw2` | cosine | 1, 2, 4, 8, 16, 32 (x10) | yes |
| red without SIGReg-F | `_align10pw2` | cosine | 1, 2, 4, 8, 16, 32 (x10) | no |
| L2 run + SIGReg-F | `_sigf_align10pw2_l2raw` | L2 of the raw vectors | 1, 2, 4, 8, 16, 32 (x10) | yes |
| L2 run | `_align10pw2_l2raw` | L2 of the raw vectors | 1, 2, 4, 8, 16, 32 (x10) | no |
| 1/d run + SIGReg-F | `_sigf_align10harm` | cosine | 1 to 32, weight 1/d (x10 on the sum) | yes |
| 1/d run | `_align10harm` | cosine | 1 to 32, weight 1/d (x10 on the sum) | no |

The L2 of the raw vectors is $\frac{1}{K}\sum_k (f_k - h_k)^2$. Here $f$ is the forecaster output before the sphere and $h$ is the raw teacher latent ($K = 64$).

The user names three runs by their colour in the score figure. Green is 10-L-ALIGN, grey is the 1/d run and brown is 1/d run + SIGReg-F.

The cooldowns start from the 135k save of their run. The first 5 runs have the two cooldowns, and the 1/d run has the exponential cooldown. The exponential cooldown has the suffix `_cd135`: $lr = 10^{-3} \cdot 0.5^{(step - 135000)/5000}$. The 1-sqrt cooldown has the suffix `_cdsqrt135`: $lr = 10^{-3} \cdot (1 - \sqrt{(step - 135000)/31000})$.

## The width-384 runs (from 2026-11-18)

Each run is a width-64 run at width 384, with 1 encoder layer and 5 forecaster layers. The muP base stays at width 64 with 3 and 3 layers, so the learning rate of 1e-3 transfers. The attention, the FFN and the conv use bf16 in place of fp16. Every other flag of the width-64 run stays. `make_w384.py` writes the run folders. The scores use the stops of MPE: 10, 25, 50, 75, 100, 125, 150 and 166k.

| Code name | Width-64 run | Run name |
|---|---|---|
| green-384 | 10-L-ALIGN | `g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10_rollfix` |
| grey-384 | 1/d run | `g166b_w384e1f5_lr1em3_tau99995_bs256_rep01_sigfix_ss_sphere_align10harm_rollfix` |

## The joint teacher runs (from 2026-11-18)

Each run is 10-L-ALIGN at width 64 with `--loss-shape cosine_similarity_batch_joint_teacher` (PR #470, [`docs/joint_teacher_loss.md`](../../docs/joint_teacher_loss.md)) in place of L_align and L_rep. One InfoNCE fraction for each rollout depth holds the forecast pair and the representation pair, and every h key comes from the EMA teacher. The runs set `--align-loss-weight 0.0` and have no `--moco-rep-keys`, `--tau-rep`, `--rep-loss-weight` or `--align-target`. `--tau` is the one temperature of the fraction. `make_joint.py` writes the run folders.

| Code name | `--tau` | Run name |
|---|---|---|
| joint, tau 1.0 | 1.0 | `g166b_lr1em3_tau99995_bs256_sigfix_ss_sphere_joint_tau1_rollfix` |
| joint, tau 0.3 | 0.3 | `g166b_lr1em3_tau99995_bs256_sigfix_ss_sphere_joint_tau03_rollfix` |

## The reference lines

- MPE: our Moirai copy at width 384 (run `cf421ew_moirai_native`, 166k steps at batch 256). Its per-config tables are in [`reports/2026-09-06_moirai_small_size`](../../reports/2026-09-06_moirai_small_size).
- MPE recipe at width 64 (cf#462): the MPE recipe with width 64, 1 encoder layer and 3 forecaster layers. It trains 166k steps at batch 256. Its own value heads score it. Its per-config tables and score files are in [`reports/2026-11-14_moirai_width64`](../../reports/2026-11-14_moirai_width64).

## Where the files are on elisa

The scripts read the root folder from `HP64_ROOT` (default `/home/jupyter/cf_runs/hp64`):

- `<run>/`: the run script, the supervisor script, `ckpt/` (saves, optimizer files, the losses CSV) and `logs/`.
- `box_train/<run>/`: the run script and the supervisor script of a run that trains on a rented box, with the paths of the box. The saves of these runs come back to `<run>/ckpt/` on elisa.
- `code_<commit>/`: a frozen copy of the training code for the runs.
- `ff/`: the head work (`code/` is the deployed head code, `results/` has the score files and the follower logs, `heads/eval/<tag>/` has the heads and the GIFT-Eval tables).
- `figures/`: the figures.

Other paths have an environment variable. `SEASONAL_NAIVE_CSV` has the elisa default. `MPE_TABLES` and `W64_TABLES` default to the tables of the reference lines in `reports/`.

## scripts/

| Script | What it does |
|---|---|
| `make_run.py <name> <code> <total steps> <why> [flag value ...]` | Writes `<run>/` with the run script and the supervisor script of a 166k run. |
| `make_cooldown.py`, `make_cooldown_sqrt.py` | Write the folder of a cooldown that starts from a save with its optimizer. |
| `make_w384.py <run> <new name> <new run folder> <new code folder> <out dir>` | Writes `<out dir>/<new name>/` with the run script and the supervisor script of a width-384 copy of a width-64 run. |
| `make_joint.py <new run folder> <new code folder> <out dir> [tau]` | Writes the folder of a joint teacher run: 10-L-ALIGN with the joint teacher fraction and the temperature tau (default 1.0). The tau gives the run name. |
| `pause_after_save.sh <run dir> <pid> <k> [max wait] [stop text]` | Stops a trainer right after its save `<k>k` loads, then writes `<run dir>/stop`. The supervisor then ends. A resume branch writes `<run>_r<n>_<k>k.pth`: give the full path of that save in place of `<k>`. |
| `stop_at_repeated_nan.sh <run dir> <name> <supervisor pid> ...` | After a NaN and a restart from a save, writes the stop file when the NaN comes back before the next save. |
| `start_follow_elisa_runs.sh` | Starts a head follower (`follow_multi.sh`) for some runs on one GPU: `E_RUNS`, `E_STOPS`, `E_GPU`, `E_LOG`. `E_SHAPE` gives the backbone shape flags (default width 64). For the width-384 runs: `E_SHAPE="--d-model 384 --n-heads 8 --num-layers 5"`. |
| `wait_score_files.sh <follower pid> <score file> ...` | Waits for score files. It returns 0 at the scores, 2 at a failure line or at the end of the follower, 3 when its polls are used. `ANY=1` returns at the first score. |
| `wait_any_pid_end.sh <pid> ...` | Waits until one of the processes ends. |
| `rows_check.py` | Prints the health of the runs from their losses files: the last step, the means of the last 400 rows, finite values, the EMA formula and the largest SIGReg spike. It skips the lines with NUL bytes, and a resume replaces the rows of the dead branch. |
| `opt_lr.py <optimizer file>` | Prints the learning rate and the schedule in an optimizer file. |
| `plot_hp64_scores.py <png> [late]` | Draws the GM-Relative MASE (all configs, short, medium, long) of the runs and their cooldowns against the step. MPE and the MPE recipe at width 64 are the reference lines. `HP64_RUNS=<code name>,...` draws only these runs and their cooldowns. `late` draws the stops from 90k. |
| `plot_cooldowns.py <png>` | Draws each run at 125k and at 166k with no cooldown, the exponential cooldown and the 1-sqrt cooldown. |
| `plot_latent_cos.py <png>` | Draws cos(forecast latent, true latent) at 1 to 32 patches ahead, over the last 500 finite steps of each run. The runs can be at different steps. This is a training check only. |
| `plot_w384_scores.py <png>` | Draws the GM-Relative MASE (all configs, short, medium, long) of green-384 and grey-384 against the step, with MPE. Prints one table per stop. |
| `plot_gm_overview.py <png>` | Draws the GM-Relative MASE of the current runs as curves on a linear step axis. Row 1: green-384, grey-384 and MPE. Row 2: the joint teacher run with tau 1.0, green, grey and the MPE recipe at width 64. |
| `plot_loss_stack_384.py <png>` | Draws the parts of the training loss of green-384 and grey-384. The parts are SIGReg on e and on h, L_rep, and L_align at depth 0 and at the deeper depths. Row 1 has a stack plot per run, row 2 a panel per part. |
| `probe_raw_forecast_norm.py <png> <tsv> [--redraw]` | Loads each save of green and grey at widths 64 and 384 on the CPU. On a fixed batch, it writes the mean norm of the raw forecast f (before the latent sphere) at each rollout depth to the tsv. A value of 1 is the sphere radius. Then it draws the png. `--redraw` loads no save and draws the png from the tsv. |

## diag_variant_nan/

The study of the NaN of "red without SIGReg-F" (pass 1 at step 12,325, the restarts at step 13,756):

- `start_replay.sh` and `diag_replay.py` run the steps again from the 10k save with `CF_NAN_DEBUG=1` and read-only probes. `compare_rows.py` shows that the replay equals the run.
- `analyze_episode.py`, `plot_burst.py`, `plot_align_terms.py` and `plot_variant_nan.py` show the episodes: the gradient bursts, the L_align terms and the loss parts.
- `start_rerun*.sh`, `rerun_step.py` and `rerun_gain.py` run one step again on the CPU in fp32. `plot_rerun_compare.py` and `plot_call_gain.py` draw the gradient at each of the 32 forecaster calls.
- `weight_change.py` prints the weight change between two saves.

What the study found:

- The first non-finite value was a gradient of step 13,755, at the fp16 output of the encoder attention.
- The same step in fp32 on the CPU also gives the burst (gradient norm 625,664), so fp16 only turned it into inf.
- At each of the 32 forecaster calls, the forecaster layers multiply the gradient by 1.2 to 2.5 (red on the same batch: 1.01 to 1.25). The sphere and the feedback do not multiply it.
- Why this gain grew without SIGReg on f is not known.
