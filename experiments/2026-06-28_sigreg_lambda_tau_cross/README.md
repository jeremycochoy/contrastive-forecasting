# SIGReg λ × EMA-τ cross (#366)

Cross the two single-axis winners on the #355 base recipe. The two arms
pair #363's per-checkpoint λ winners with #357's best τ:

- **Arm A** — `λ_e=ARM_A_LAMBDA_E, λ_h=ARM_A_LAMBDA_H, τ=BEST_TAU`,
  i.e. #363 best-at-best (lowest *best-checkpoint* GM-Rel MASE) crossed
  with the #357 best τ.
- **Arm B** — `λ_e=ARM_B_LAMBDA_E, λ_h=ARM_B_LAMBDA_H, τ=BEST_TAU`,
  i.e. #363 best-at-last (lowest *last-checkpoint* GM-Rel MASE) crossed
  with the same #357 best τ.

The concrete values for `ARM_A_LAMBDA_*`, `ARM_B_LAMBDA_*`, and `BEST_TAU`
are NOT baked into the launchers; they are read at launch time from a
manifest file (`winners.sh`) that the user creates by re-verifying the
final #363 / #357 winners. See [Launch-time gate](#launch-time-gate) below.

Both arms keep the #355 recipe fixed (SIGReg + EMA-target, B=512, enc3+CPC,
GRU patch-embed, 12,500 steps, seed `20260520`, dataset
`gift-pretrain-full-4096 / small_v1`). Only `--ema-tau`,
`--sigreg-embedding-weight`, `--sigreg-encoding-weight` change between arms.

## Current expected values

Pending launch-time re-verification, the working assumption based on the
scaffold-time state of #363 / #357 is:

| arm | λ_e | λ_h | τ | source of λ pair | source of τ |
| --- | ---: | ---: | ---: | --- | --- |
| Arm A (`lA_emb100_enc10_tau090`) | 10.0 | 1.0 | 0.90 | #363 best-at-best (arm 2) | #357 best (τ=0.90) |
| Arm B (`lB_emb10000_enc10_tau090`) | 1000.0 | 1.0 | 0.90 | #363 best-at-last (arm 6) | #357 best (τ=0.90) |

Suffix encoding: `lX_emb<10·λ_e>_enc<10·λ_h>_tau<100·τ>`. The launcher
derives the suffix from the manifest values; stale values change the
suffix, so wrong values do not silently overwrite a prior run's files.

If the manifest filled in at launch time differs from this table, update
the table to match before merging — `tests/test_366_launcher_shape.py`
asserts the table and the manifest agree.

## Launch-time gate

`launch_arms.sh` refuses to start without a winners manifest at
`$OUT/winners.sh`. Procedure:

```bash
# On elisa, in a worktree:
export WT=/home/jupyter/workspaces/contrastive-forecasting
export OUT=$WT/experiments/2026-06-28_sigreg_lambda_tau_cross

# 1. Confirm #363 is CLOSED on GitHub and read its final GM-Rel MASE
#    table from the merged report.
# 2. Confirm #357's final τ winner.
# 3. Copy + edit the manifest:
cp $OUT/scripts/winners.sh.example $OUT/winners.sh
$EDITOR $OUT/winners.sh    # fill ARM_*_LAMBDA_*, BEST_TAU, verifier
# 4. Update the "Current expected values" table above if values shifted.
bash $OUT/scripts/launch_arms.sh
# ONLY="lA_..." launch_arms.sh   to run just Arm A
```

Per-arm logs land in `$OUT/results/sweep_bb_*.log` and `$OUT/results/sweep_dl_*.log`.

## Layout

```
experiments/2026-06-28_sigreg_lambda_tau_cross/
  README.md                              this file
  winners.sh                             local-only manifest (gitignored), one per launch
  scripts/
    winners.sh.example                   template + re-verify procedure
    launch_arms.sh                       cross driver (sequential arms, sources winners.sh)
    launch_downstream.sh                 parallel 2L+6L dispatcher per arm
    train_backbone_sigreg.sh             backbone trainer (parameterised λ × τ)
    downstream_sigreg.sh                 q-head + GIFT-Eval per cell
  runs/                                  per-arm checkpoints, optimizer, losses csv
  results/                               per-arm logs, GIFT-Eval outputs
```

## Remote-machine wiring

This experiment runs locally on elisa — outputs land directly under the
repo checkout, so no `sync_loop` is needed (the remote-launch checklist's
sync-loop boxes don't apply). If this is ever re-run on vast.ai, follow
[`../REMOTE_LAUNCH_CHECKLIST.md`](../REMOTE_LAUNCH_CHECKLIST.md) and add a
sync_loop before launching.

## Anchors

The forthcoming report compares the two cross arms against the five
single-axis published anchors:

- **#344** enc3+CPC, B=1024 (baseline)
- **#353** EMA-target enc3+CPC, B=1024, τ=0.99
- **#355** SIGReg + EMA-target, B=512, τ=0.99 (`λ_e=λ_h=0.1`)
- **#357** SIGReg + EMA-target, B=512, **τ=0.90** (`λ_e=λ_h=0.1`) — best-τ axis
- **#363 arm 2** SIGReg + EMA-target, B=512, τ=0.99 (**`λ_e=10.0, λ_h=1.0`**) — best-at-best λ
- **#363 arm 6** SIGReg + EMA-target, B=512, τ=0.99 (**`λ_e=1000.0, λ_h=1.0`**) — best-at-last λ
