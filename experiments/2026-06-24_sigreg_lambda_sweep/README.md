# SIGReg λ-sweep (#363)

Sweep three (`λ_e`, `λ_h`) combinations on the #359 recipe (SIGReg + EMA-target,
B=512, enc3+CPC, GRU patch-embed, 12,500 steps, seed `20260520`, dataset
`gift-pretrain-full-4096 / small_v1`). Only `--sigreg-embedding-weight` and
`--sigreg-encoding-weight` vary across arms.

| suffix | λ_e | λ_h | required? |
| --- | ---: | ---: | --- |
| `emb100_enc01`  | 10.0 |  0.1 | yes (issue order 1) |
| `emb100_enc10`  | 10.0 |  1.0 | yes (issue order 2) |
| `emb100_enc100` | 10.0 | 10.0 | yes (issue order 3) |
| `emb10_enc10`   |  1.0 |  1.0 | optional 4th (`RUN_OPTIONAL=1`) |

## Running on elisa (local)

elisa is the long-lived GCP host (workdir `~/workspaces/contrastive-forecasting/`,
two RTX 4090s). The launchers run **directly on elisa**, writing checkpoints
to local NVMe — there is no vast.ai instance and **no `sync_loop` pull is
required** (the prior 2026-06-20 and 2026-06-22 SIGReg launchers follow the
same pattern). The REMOTE_LAUNCH_CHECKLIST applies to vast.ai launches; for
elisa, the equivalent guarantees come from the host itself (persistent disk,
no preemption window).

`WT` and `OUT` are now **required** environment variables — the launchers
abort with `exit 2` if either is unset or `WT` does not resolve to a
directory. This prevents the silent-throttle failure mode where an unset
`WT` resolved to a non-existent path on elisa, an empty `HF_TOKEN` was
warned (not errored), and the GPU then idled on anonymous HF rate-limits.

```bash
# On elisa, from a worktree checked out at $WT:
export WT=/home/jupyter/workspaces/contrastive-forecasting
export OUT=$WT/experiments/2026-06-24_sigreg_lambda_sweep
bash $WT/experiments/2026-06-24_sigreg_lambda_sweep/scripts/launch_arms.sh
# RUN_OPTIONAL=1 to add the emb10_enc10 arm
# ONLY="emb100_enc10 emb100_enc100" to pick a subset
```

The driver runs each arm sequentially (backbone, then downstream 2L on GPU 0
+ 6L on GPU 1 in parallel). Per-arm logs land in `$OUT/results/sweep_bb_*.log`
and `$OUT/results/sweep_dl_*.log`.

## Layout

```
experiments/2026-06-24_sigreg_lambda_sweep/
  README.md                              this file
  scripts/
    launch_arms.sh                       sweep driver (sequential arms)
    launch_downstream.sh                 parallel 2L+6L dispatcher per arm
    train_backbone_sigreg.sh             backbone trainer (parameterised λ)
    downstream_sigreg.sh                 q-head + GIFT-Eval per cell
    build_report.py                      gm_table.csv + plots
  runs/                                  per-arm checkpoints, optimizer, losses csv
  results/                               per-arm logs, GIFT-Eval outputs, gm_table.csv
  plots/                                 loss_curve, sigreg_e_inspection, uniformity, gm_rel_mase
```

## Anchors

The report compares the sweep arms against four published anchors (per-cell
GM-Rel MASE transcribed from each arm's `gm_table.csv` — see `ANCHOR_GM` in
`scripts/build_report.py` for the source paths):

- **#344** enc3+CPC, B=1024 (baseline)
- **#353** EMA-target enc3+CPC, B=1024
- **#355** SIGReg + EMA-target, B=512 (`λ_e=λ_h=0.1`)
- **#359** SIGReg + EMA-target, B=512 (`λ_e=1.0, λ_h=0.1`)
