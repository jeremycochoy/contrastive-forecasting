«Agent ExperimentRunner claude-opus-5 writing»

**Experiment directory:** `reports/2026-08-20_a4_full_pass/`

## The correction

I posted "a free null of 0.0046" twice. The claim is wrong. The evidence now
says so.

The teacher ENCODER freezes at step 100,000. The teacher HEAD does not.
`results/teacher_head_inputs_*.json` measures it. The head loads 110 tensors.

| pair | teacher tensors that move | student tensors that move |
|---|---:|---:|
| 40k to 100k | 74 of 74 | 32 of 36 |
| 100k to 200k | 0 of 74 | 32 of 36 |
| 200k to 300k | 0 of 74 | 32 of 36 |

So 1.0874 at 100k and 1.0828 at 200k compare two different models. The
teacher points at 100k, 200k, 300k, 450k and 665k are five models, not five
draws of one. I do not pool them.

## Why the teacher head loads 36 student tensors

The line is `src/checkpoint.py:266`:

```python
    out = dict(state_dict)
```

`prepare_backbone_state_dict` starts from the full STUDENT state dict. The
`encoder_source='teacher'` branch then overwrites two prefixes only
(`_TEACHER_PROMOTIONS`, `src/checkpoint.py:230`):

```python
_TEACHER_PROMOTIONS = {
    "teacher_input_to_latent.": ("encoder.", "transformer.input_to_latent."),
    "teacher_encoder_layers.": ("transformer.encoder_layers.",),
}
```

`src/checkpoint.py:280` then drops every `teacher_*` and `cpc_w1*` key. Every
key that the two prefixes do not name keeps the student's tensor. Those 36
keys are the frequency table, the seasonality table, the channel-mixing
module and the 3 forecaster layers. The optimizer still updates them after
step 100,000, so 32 of the 36 move at every stop.

The latents the head reads move with them. Between 200k and 300k the encoder
latents differ by rel L2 6.227e-03 and the forecaster latents by 8.931e-02.

## What changed in the evidence

| before | now | what it measures |
|---|---|---|
| `results/null_frozen_teacher.csv` | `results/teacher_delta_bb100k_bb200k.csv` and its `.txt` | a paired per-config change between two models. delta -0.0046, [-0.0199, 0.0123], p_improved 0.711 |
| `scripts/teacher_pool.py` | `scripts/teacher_frozen_track.py` | each stop as its own model, plus the change from one stop to the next. No mean, no standard deviation, no pooled range |

Three tests guard the correction:

- one forbids `statistics.fmean`, `statistics.stdev` and a pooled range in
  the track script.
- one checks that neighbouring stops give a change, not a spread.
- one reads `src/checkpoint.py:266` from source and compares it with the
  quoted line.

## The band the report reads

Three head seeds on ONE backbone, at 200,000 steps. `results/head_band.csv`.

| head | seed 20260722 | seed 20260723 | seed 20260724 | range |
|---|---|---|---|---|
| student | 1.0660 | 1.0652 | 1.0642 | 0.0018 |
| teacher | 1.0828 | 1.0809 | 1.0764 | 0.0064 |

Both seed 20260722 draws reproduce #373 exactly, on this machine and on this
code. Machine drift and code drift are zero.

## The 300,000-step stop is complete

| head | seed | GM-Relative MASE |
|---|---|---|
| student | 20260722 | **1.0867** |
| teacher | 20260722 | **1.1030** |
| student | 20260723 | 1.0883 |
| student | 20260724 | 1.0841 |
| teacher | 20260723 | 1.0992 |

Every eval holds 97 GIFT-Eval configs. All five draws read one backbone, md5
`618e433edea74ed2ca4ad9d10be37377`. The 300k student band range is 0.0042.
The fourth band draw, teacher seed 20260724, is on card 1 now.

## Headline numbers

| stop | student | teacher |
|---|---|---|
| 40,000 | 1.0862 | 1.0855 |
| 100,000 | 1.0801 | 1.0874 |
| 200,000 | **1.0660** | 1.0828 |
| 300,000 | 1.0867 | 1.1030 |

The student score rises by 0.0207 from 200k to 300k. That is 3.2 times the
largest head-seed range on this card (0.0064) and 11.5 times the student
range (0.0018). It is also 1.5 times the 0.0141 gap that made 1.0660 the
best. The move is readable.

**The card's question, so far: no.** A4 does not improve between 200,000 and
300,000 steps. It gets worse, on both heads, by more than the band. The 450k
and 665k stops decide whether the curve turns back down.

## Runs completed

| run | state |
|---|---|
| driver leg to 300,000 steps, card 0 | done in 8.5 h |
| 200k band, seeds 20260723 and 20260724, both heads | scored |
| 200k re-draw, seed 20260722, both heads | scored |
| 300k stop, seed 20260722, both heads | scored |
| 300k band, student seeds 20260723 and 20260724 | scored |
| 300k band, teacher seed 20260723 | scored |
| 300k band, teacher seed 20260724 | on card 1 |
| driver leg to 450,000 steps, card 0 | started 08:45 UTC |
| 450k band | armed on `band_queue.sh` stage 3, gate `ckpt` |
| 665k band | armed in `watchdog.sh` |

Nothing above touched card 0.

Tests: 199 in `test_407_full_pass.py`, 2,096 in the suite.
