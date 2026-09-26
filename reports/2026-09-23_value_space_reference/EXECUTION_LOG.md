# Execution log — #415

Operational facts. The science lives in
[`value_space_reference.md`](value_space_reference.md).

## Preflight, 2026-09-23, elisa (RTX 4090)

The leg is not started. These runs exist to size it. All artefacts went to
`/tmp` and are gone. They ran at the first rate, 1e-3 flat. The leg now runs
5e-4 with a cosine anneal, after the review of PR #416.

### The leg, 3,000 steps on the real corpus

The runner's own command line, `gift-pretrain-full-4096` / `small_v1`, batch
64, depth 3, lr 1e-3.

- **5.0 steps/s**, 151 ms of measured step work (`fwd` 74 + `bwd` 73).
  665,000 steps is then about **37 GPU-hours**.
- **5.4 GB** VRAM. One 4090 holds two of these legs.
- The loss falls 2.01 → 1.10 over 2,000 steps and keeps falling, with no
  divergence at 1e-3.

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
0.01. The latent is not shaped to separate futures from pasts, because the
card drops `L_rep`, `L_align` and SIGReg. This is why every stop scores A2
beside B4.

## A2 against B4, 2026-09-24, elisa, one CPU core

`scripts/a2_cost.py`: one series, a random backbone and head at this card's
shape. The cost of a forward does not depend on the weights.

| horizon | B4 (s) | A2 (s) | A2 / B4 |
|---|---|---|---|
| 48 | 0.21 | 0.47 | 2.2 |
| 480 | 1.12 | 4.40 | 3.9 |
| 720 | 1.72 | 6.76 | 3.9 |

A2 re-runs the whole cell once per patch of 16 values. B4 spends 37 % of
its eval time on short-term configs and 63 % on medium and long ones
(`reports/2026-08-08_rollout_depth/results/config_costs.csv`). So an A2 eval
of a stop takes about 3.3 times its B4 eval. `run.sh` therefore trains in one
lane and scores in a second.

The real `eval_local.sh` ran under both strategies on two configs,
`ett1/D/short` and `bizitobs_l2c/H/long`, with a mirrored #414 backbone at
40,000 steps and its head: rc 0 each, B4 0.8625 in 10 s, A2 0.8573 in 20 s.
Both wrote quantile metrics per config. The two-config numbers rank nothing.

## The box, 2026-09-24, instance 51431200

One read-only `nvidia-smi`, no launch:

- Compute mode **Default**. `gpu_gate` returns at once, so the leg starts
  beside `cos200k`.
- 21,758 of 32,607 MiB in use by three processes, so 10,849 MiB free. The
  leg at batch 64 needs 5.4 GB and fits. The head's gate then waits for
  9,000 MiB free, for up to 4 h, and a stop whose head times out scores on
  the next run of `run.sh`.
- 8 cores. The score lane runs one eval at a time, 4 shards.
