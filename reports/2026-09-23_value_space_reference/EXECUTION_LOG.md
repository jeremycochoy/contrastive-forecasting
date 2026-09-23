# Execution log — #415

Operational facts. The science lives in
[`value_space_reference.md`](value_space_reference.md).

## Preflight, 2026-09-23, elisa (RTX 4090)

The leg is not started. These runs exist to size it and to check that the
Moirai rate holds at batch 64. All artefacts went to `/tmp` and are gone.

### The leg, 3,000 steps on the real corpus

The runner's own command line, `gift-pretrain-full-4096` / `small_v1`, batch
64, depth 3, lr 1e-3.

- **5.0 steps/s**, 151 ms of measured step work (`fwd` 74 + `bwd` 73).
  665,000 steps is then about **37 GPU-hours**.
- **5.4 GB** VRAM. One 4090 holds two of these legs.
- The loss falls 2.01 → 1.10 over 2,000 steps and keeps falling. **No
  divergence at the Moirai rate**, which is what #412's pass 1 found for the
  contrastive objective at this width.

### Cost against #414's cell

#414's own production log for the same cell at the same depth reads
**3.6 steps/s** on its own box. That is a different machine, so the pair is
not controlled. Back to back on one 4090 over 250 steps of synthetic data,
the value objective at depth 3 measured 352 ms a step against the cell's
313 ms, and the value run shared the box with a second job. Read these as
**the same cost class**, not as a ranking.

The DEPTH is the dominant cost of this objective: 67 ms a step at depth 0
against 352 ms at depth 3 on one probe, because each depth re-enters the
input head. #414's latent depth re-applies the forecaster stack alone, and
pays for the Gram, the teacher and SIGReg instead.

### One reading to carry forward

The `gap` column (ff − fp: how much better the forecast latent matches the
future than the past does) goes **more negative** as the value loss falls:
−0.21 at step 400, −0.35 at step 1,100. Dimension usage `U_t` sits near
0.01.

The latent is not being shaped to separate futures from pasts, which is what
dropping `L_rep`, `L_align` and SIGReg removes. Strategy B4 rolls the
forecast out in LATENT space — it feeds f back as the next encoder latent —
while this objective trains the rollout in VALUE space. So B4 composes an
operator this run never trains.

Watch `gap` and `auc` over the leg. If B4 scores badly while `val_err_d0` is
good, that mismatch is the first thing to check, and strategy A2 (value-space
rollout, already in `src/forecasting_head.py`) is the matched read.
