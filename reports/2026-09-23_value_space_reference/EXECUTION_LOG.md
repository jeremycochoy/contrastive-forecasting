# Execution log — #415

Operational facts. The science lives in
[`value_space_reference.md`](value_space_reference.md).

## Preflight, 2026-09-23, elisa GPU 1 (RTX 4090)

The leg is not started. These runs exist to size it and to check that the
Moirai rate holds at batch 64. All artefacts went to `/tmp` and are gone.

### Step time, at the card's shape

30 steps each, `d_model` 384, 3 + 3 layers, `T_raw` 4096, C = 1, batch 64,
`--residual-dtype fp32 --attn-dtype fp16 --ffn-dtype fp16`. The number is the
trainer's own steady-state `timing: total`.

| objective | depth | ms / step |
|---|---|---|
| value space (#415) | 3 | **357** |
| value space (#415) | 0 | 72 |
| #414's cell (contrastive, teacher, CPC, SIGReg) | 3 | 464 |

The value objective at depth 3 is **cheaper** than the cell it replaces. Each
depth costs one full pass over the body, because the rollout re-enters the
input head; #414's latent depth re-applies the forecaster stack alone but
pays for the Gram, the teacher and SIGReg on top.

### The full leg

3,000 steps on the real corpus (`gift-pretrain-full-4096`, `small_v1`), the
runner's command line, lr 1e-3.

- **4.8 steps/s** steady state → 665,000 steps is about **38 GPU-hours**.
- **5.4 GB** VRAM. One 4090 holds two of these legs.
- Loss falls 2.01 → 1.12 over 1,100 steps and keeps falling. **No divergence
  at the Moirai rate**, which is what #412's pass 1 found for the contrastive
  objective at this width.

### One reading to carry forward

The `gap` column (ff − fp: how much better the forecast latent matches the
future than the past does) goes **more negative** as the value loss falls:
−0.21 at step 400, −0.35 at step 1,100.

The latent is not being shaped to separate futures from pasts, which is what
dropping `L_rep` and `L_align` removes. The eval strategy B4 rolls the
forecast out in LATENT space — it feeds f back as the next encoder latent —
while this objective trains the rollout in VALUE space. So B4 composes an
operator this run never trains. Watch `gap` and `auc` over the leg: if B4
scores badly while the depth-0 value error is good, that mismatch is the
first thing to check, and strategy A2 (value-space rollout, already in
`src/forecasting_head.py`) is the matched read.
