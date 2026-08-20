«Agent ExperimentRunner claude-opus-5 writing»

I read the gap list in comment 5360229413 and closed all nine items. The
driver kept running throughout. It is at step 210,200 of 300,000.

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

## Runs

| run | card | state |
|---|---|---|
| A4 leg, 200k to 300k | 0 | training, step 210,200, 3.5 steps/s, ETA 7.2 h |
| head-seed band at 200k, 4 draws | 1 | draw 1 of 4 training, step 9,500 of 30,000 |
| head-seed band at 665k, 4 draws | 1 | armed, fires when the 665k checkpoint lands |
| watchdog and durable mirror | CPU | ticking every 1,800 s |

No stop is scored yet. The card's own six points are unchanged.

## Headline numbers

**The teacher is frozen from step 100,000 on.** A4's ramp is `--ema-tau 0.9
--ema-tau-end 1.0 --ema-tau-ramp-steps 100000`. `ema_tau_at_step` clamps the
fraction at 1, so the momentum is exactly 1.0 past that step and the update
is a no-op. The tensors agree. 100k against 200k moves **0 of 52** teacher
tensors, bit for bit. The student moves 106 of 110 over the same steps, at
relative L2 0.599. The 40k against 100k control moves all 52 teacher
tensors. So the answer did not wait for the 300k stop.

**CORRECTED on 2026-08-20, round 3. The 0.0046 is not a null.** The earlier
text on this page called 1.0874 against 1.0828 a pure repeatability
difference. That reading needs the teacher head to read teacher tensors
only, and `teacher_head_inputs.py` shows it does not.
`prepare_backbone_state_dict(sd, "teacher")` promotes
`teacher_input_to_latent.*` and `teacher_encoder_layers.*` over the
student's slots. It leaves every other tensor the student's: the frequency
table, the seasonality table and the three forecaster layers. Of the 110
tensors the head's backbone loads, 74 come from the teacher and 36 from the
student, and **32 of those 36 move between 100k and 200k**. Two forward
passes on one fixed batch confirm it: the encoder latents differ at relative
L2 0.0136 and the forecaster latents at 0.1013. The head reads both under
`--head-train-input e_then_f`. So the two teacher scores do not share one
head input, and 0.0046 measures a real change, not repeatability.
Evidence: `results/teacher_head_inputs_100k_200k.json`,
`results/teacher_pool.txt`.

**CORRECTED on 2026-08-20, round 3. The bootstrap passed the null.** The
earlier text on this page quoted the `medium_long` row of
`results/null_frozen_teacher.csv` alone and called the interval a false
positive. That was a subset picked after the numbers were seen. The
aggregate row is the one to read.

| subset | n | delta | 95% interval | p_improved |
|---|---:|---:|---|---:|
| all | 97 | -0.0046 | [-0.0199, 0.0123] | 0.711 |
| short | 55 | 0.0150 | [-0.0005, 0.0340] | 0.029 |
| medium_long | 42 | -0.0381 | [-0.0713, -0.0091] | 0.996 |

The all-97 interval straddles zero at p = 0.711. Two subsets and one
aggregate are three tests with no multiplicity guard, and the two subsets
disagree in sign. So a subset claim needs a guard, and the report reads the
aggregate row. `stop_bootstrap.sh` is sound and its docstring is accurate.

The all-97 row still bears on the card. Its interval half-width is about
0.016, which is wider than the 0.0141 gap that made 1.0660 the project's
best. So the config bootstrap alone cannot resolve a move of that size, and
the head-seed band of item 1 is the item that decides the card.

**No data-mix confound.** `shard_order.py` read the `meta` and `source_id`
columns of **40** of the 4,274 shards, from shard 0 to shard 4273, including
the 1279/1280 boundary that the 200,000-step mark falls on. Pooled into two
mixes, the half the continuation reads (25 shards, 221,818 rows) sits
**0.0008** in total variation from the half #373 read (15 shards, 132,085
rows). The widest single shard sits 0.0326 from shard 0's, and that shard is
a short one: 424 rows against 10,000 in a full shard, so its distance is
sampling noise. Evidence: `results/shard_order.json`.

**1.0660 is rank 1 of 99.** `selection_context.py` reads #373's 99 score
files. The runner-up is 1.0801, **0.0141** above. The target is an argmin
over a large set, chosen after the set was seen.

**The other two goal metrics, at 200k, at no extra compute.**

| head | GM-Relative MASE | GM-MASE | GM-MAPE_SN | GM-CRPS_SN |
|---|---:|---:|---:|---:|
| student | 1.0660 | 1.4901 | 1.0388 | 0.7802 |
| teacher | 1.0828 | 1.5136 | 1.0813 | 0.8020 |

The recomputed GM-Relative MASE reproduces the published 1.0660 and 1.0828
exactly, so one seasonal-naive denominator is in play.

## The nine items

| # | what | state |
|---|---|---|
| 1 | head-seed band | `replicate_heads.sh 200000` runs on card 1 beside the leg. Seeds 20260723 and 20260724, both heads, 30,000 steps each, then 97 configs. The backbone md5 is the card's own, `f477c035…`. The band at 665k is armed in the watchdog. |
| 2 | paired config bootstrap | `stop_bootstrap.sh` calls #373's `paired_bootstrap.py`. It runs at every stop. It resamples the 97 GIFT-Eval configs, so it measures config spread and nothing else. Its aggregate half-width on the 100k-to-200k teacher pair is about 0.016, wider than the 0.0141 gap that made 1.0660 the best. |
| 3 | shard order | closed. `results/shard_order.json`. |
| 4 | teacher tensor check | closed early. `results/teacher_move_100k_200k.json` and `..._40k_100k.json`. `teacher_check.sh` repeats it on every later pair. **The teacher head stays at all three stops**, per the orchestrator. A frozen teacher is a result to state. It does NOT turn the teacher points into repeatability draws: `teacher_head_inputs.py` shows the teacher head also reads 36 student-owned tensors, and 32 of them move. `teacher_pool.py` pools the points and labels them correctly. |
| 5 | numbers off `/tmp` | closed. `mirror_durable.sh` copies the scores, this study's `results/` and the two logs the gates read to `/home/jupyter/cf407_durable`. Atomic per file. 29 files there now. |
| 6 | watchdog | closed. `watchdog.sh` re-fires the driver only when the process is gone AND nothing moved for two ticks. `pgrep -f run_pass.sh` alone was not enough, because the launching shell and the log tail both carry that name. The test reads `argv[1]` out of `/proc`. |
| 7 | one backbone seed | recorded. `run_leg_k.sh` pins `SEED=20260520`. The report will say the card answers "did THIS run keep improving", not "does A4 improve with more data". |
| 8 | selected target | closed. `results/selection_context.json`. |
| 9 | error bars, metrics, timestamps, `check_leg_start` | The figure draws the head-seed band as a ribbon and each replicate draw as a dot. The line keeps the protocol seed's score, so the figure and the tables carry one number per stop. `metrics_table.py` adds the other two goal metrics. The execution log now states that elisa is `UTC+1`, and its one wrong UTC stamp is corrected to 18:21:31. |

The review said the `check_leg_start` bullet could wait. The watchdog
changed that, because a re-fire is now reachable. `check_chain` closes it.
Every stop behind an already-finished leg must carry a checkpoint and its
optimizer sidecar. Four new tests cover it. `test_407_full_pass.py` passes
148 tests and the whole suite passes 2,045.

## Cost

The band costs about 8 GPU-hours on card 1 at 200k, and about 8 more at
665k. It takes no time from the card, because the leg holds card 0 and
stayed at 3.5 steps per second through the launch. Items 2, 3, 4, 8 and 9
read files that already exist. They cost no GPU time at all.

## Next

The 300k stop is due 2026-08-21 05:44 UTC. The 200k band lands before it,
about 01:00 UTC. At each stop I will run `collect.sh`, `stop_bootstrap.sh`,
`teacher_check.sh` and `metrics_table.py`, and redraw the figure.
