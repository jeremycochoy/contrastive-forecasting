«Agent ExperimentRunner claude-opus-5 writing»

Round 3, nine items. Seven are closed on disk. Two need card 1 and are armed
to fire without a poll. Nothing here touched the driver.

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

## Runs

| run | card | state |
|---|---|---|
| the leg to 300k | 0 | alive at step **215,600** of 665,000, 3.4 steps per second |
| the 200k head-seed band, seeds 20260723 and 20260724 | 1 | 4 draws, in progress. One head trains, the others hold the flock |
| the 200k protocol re-draw, seed 20260722 | 1 | armed. It fires when the band drains, near 01:00 UTC |
| the 450k band, two seeds | 1 | armed. It fires on the 450k checkpoint |
| `teacher_head_inputs.py`, two pairs | CPU | done |
| `shard_order.py`, 40 shards | CPU | done |

Tests: 173 in `test_407_full_pass.py`, 2,070 in the suite.

## Item 1. Claim 3, corrected

My round-2 reply quoted the `medium_long` row of `null_frozen_teacher.csv`
alone and called the interval a false positive. That was a subset picked
after the numbers were seen. The correct reading is the aggregate row.

| subset | n | delta | 95% interval | p_improved |
|---|---:|---:|---|---:|
| all | 97 | -0.0046 | [-0.0199, 0.0123] | 0.711 |
| short | 55 | 0.0150 | [-0.0005, 0.0340] | 0.029 |
| medium_long | 42 | -0.0381 | [-0.0713, -0.0091] | 0.996 |

The all-97 interval straddles zero at p = 0.711, so the aggregate bootstrap
passed. Two subsets and one aggregate are three tests with no multiplicity
guard, and the two subsets disagree in sign. `stop_bootstrap.sh` is sound
and its docstring is accurate. The correction is in
`results/pr_comment_20260820_gaps.md` and in the execution log.

The all-97 row still bears on the card. Its half-width is about 0.016, which
is wider than the 0.0141 gap that made 1.0660 the best. So the config
bootstrap alone cannot resolve a move of that size.

## Item 3. The teacher head does not read teacher tensors only

You asked me to print the tensor names the teacher head consumes and to show
that all of them sit in the frozen 52. They do not. `teacher_head_inputs.py`
builds the state dict the head trainer loads and compares it across two
checkpoints. Then it runs both backbones over one fixed batch.

| pair | loaded | from teacher | from student | teacher moved | student moved |
|---|---:|---:|---:|---:|---:|
| 100k to 200k | 110 | 74 | 36 | **0** | **32** |
| 40k to 100k | 110 | 74 | 36 | 74 | 32 |

`prepare_backbone_state_dict(sd, "teacher")` promotes
`teacher_input_to_latent.*` and `teacher_encoder_layers.*` over the
student's slots. Every other tensor stays the student's: the frequency
table, the seasonality table and the three forecaster layers. Those keep
training.

The forward passes confirm it. Encoder latents differ at relative L2
**0.0136**. Forecaster latents differ at relative L2 **0.1013**. The card's
head reads both, under `--head-train-input e_then_f`.

**So 0.0046 is not a null.** My claim 2 arithmetic holds, but its label does
not. The two teacher scores do not share one head input.
`results/teacher_head_inputs_100k_200k.json`.

## Item 4. The pool, with the right label

`teacher_pool.py` gives the pool you asked for. It also labels it, because
item 3 removes the premise. The teacher points share one encoder stack. They
do not share one head input.

At n = 2 the range is 0.0046. It grows to n = 5 as the stops land, and
`teacher_check.sh` refreshes it every watchdog tick. Read it as an upper
bound on the frozen-encoder contribution, not as a repeatability band.

## Item 6. The band comparison

`head_band.py` now prints this block. #393's 0.0384 is the largest range
over **every** cell. This cell's own rows are the closer comparison, and
`noise_band.py` gives them.

| number | value | source |
|---|---:|---|
| #393 pooled range, every cell | 0.0384 | `noise_band.py` |
| same cell, bb40k teacher | 0.0118 | #393, `arm6_v2_combab_alignS` |
| same cell, bb100k teacher | 0.0080 | same |
| same cell, bb40k student | 0.0049 | same |
| same cell, bb100k student | 0.0047 | same |
| gap that made 1.0660 the best | 0.0141 | `selection_context.json` |

This cell's published rows all sit below the 0.0141 gap. So the card may
resolve the difference. The measured band decides it, not the published one,
and the script prints the verdict once the band lands.

## Item 9. 40 shards, and the two halves pooled

A per-shard distance mixes a real mix change with one shard's sampling
noise, and `small_v1` holds short shards. So the survey also pools each half
of the run into one mix.

| half | shards | rows | total variation |
|---|---:|---:|---:|
| below shard 1280, which #373 read | 15 | 132,085 | reference |
| shard 1280 and up, which this card reads | 25 | 221,818 | **0.0008** |

The widest single shard sits 0.0326 from shard 0's. That shard holds 424
rows against 10,000 in a full shard, so its distance is sampling noise.

## Items 2 and 7. Armed on card 1

`band_queue.sh` runs detached with a 300-second period. It waits on card 1
rather than on a clock, because `head_vram_gate` serialises every head on
one flock.

Stage 1 is item 2. Head seed 20260722 draws again at 200,000 steps, here and
on this code. Its tag carries `_s20260722`, so it cannot overwrite the card's
own number. `head_band.py` then reports the re-draw delta against #373's
anchor, and it builds the band from the three draws measured here.

Stage 2 is item 7. Two more seeds at 450,000 steps, fired on the checkpoint.
The watchdog keeps 665k. So 200k, 450k and 665k carry error bars, and 300k
carries one draw per head. The report will say that beside the 300k point.

## Items 5 and 8. The caption

`plot_full_pass.py` writes `results/figure_caption.txt` and draws the same
text under the axes. It states four things: the ribbon is one number pooled
over both heads, it gives the measured range and the draw count beside the
standard deviation, it names the stops that carry draws and calls the rest
an extrapolation, and it records that 40k comes from `cf373_r2` while 100k
and 200k come from `cf373_r3`.

## Next

The 300k stop is due 2026-08-21 05:44 UTC. Nothing above blocks it.
