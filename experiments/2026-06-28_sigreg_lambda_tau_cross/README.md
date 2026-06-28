# SIGReg λ × EMA-τ cross (#366)

Cross the two single-axis winners on the #355 base recipe:

| arm | λ_e | λ_h | τ | source of λ pair | source of τ |
| --- | ---: | ---: | ---: | --- | --- |
| Arm A (`lA_emb100_enc10_tau090`) | 10.0 | 1.0 | 0.90 | #363 best-at-best (arm 2) | #357 best (τ=0.90) |
| Arm B (`lB_emb10000_enc10_tau090`) | 1000.0 | 1.0 | 0.90 | #363 best-at-last (arm 6) | #357 best (τ=0.90) |

Both arms keep the #355 recipe fixed (SIGReg + EMA-target, B=512, enc3+CPC,
GRU patch-embed, 12,500 steps, seed `20260520`, dataset
`gift-pretrain-full-4096 / small_v1`). Only `--ema-tau`,
`--sigreg-embedding-weight`, `--sigreg-encoding-weight` change between arms.

The λ-pair identity is read from the closed #363 GM table at launch time
and #357's τ winner from its canonical report; both were locked at scaffold
time on 2026-06-28.

## Running on elisa (local)

```bash
# On elisa, from a worktree checked out at $WT:
export WT=/home/jupyter/workspaces/contrastive-forecasting
export OUT=$WT/experiments/2026-06-28_sigreg_lambda_tau_cross
bash $WT/experiments/2026-06-28_sigreg_lambda_tau_cross/scripts/launch_arms.sh
# ONLY="lA_emb100_enc10_tau090" launch_arms.sh   to run just Arm A
```

Per-arm logs land in `$OUT/results/sweep_bb_*.log` and `$OUT/results/sweep_dl_*.log`.

## Layout

```
experiments/2026-06-28_sigreg_lambda_tau_cross/
  README.md                              this file
  scripts/
    launch_arms.sh                       cross driver (sequential arms)
    launch_downstream.sh                 parallel 2L+6L dispatcher per arm
    train_backbone_sigreg.sh             backbone trainer (parameterised λ × τ)
    downstream_sigreg.sh                 q-head + GIFT-Eval per cell
  runs/                                  per-arm checkpoints, optimizer, losses csv
  results/                               per-arm logs, GIFT-Eval outputs
```

## Anchors

The forthcoming report compares the two cross arms against the five
single-axis published anchors:

- **#344** enc3+CPC, B=1024 (baseline)
- **#353** EMA-target enc3+CPC, B=1024, τ=0.99
- **#355** SIGReg + EMA-target, B=512, τ=0.99 (`λ_e=λ_h=0.1`)
- **#357** SIGReg + EMA-target, B=512, **τ=0.90** (`λ_e=λ_h=0.1`) — best-τ axis
- **#363 arm 2** SIGReg + EMA-target, B=512, τ=0.99 (**`λ_e=10.0, λ_h=1.0`**) — best-at-best λ
- **#363 arm 6** SIGReg + EMA-target, B=512, τ=0.99 (**`λ_e=1000.0, λ_h=1.0`**) — best-at-last λ
