# Execution log — #407, A4 to one full pass

Operational record. The report holds the science. This file holds the events.

## 2026-08-20, launch

Machine: elisa. Card 0 of two RTX 4090 cards. Other projects hold both cards.

### Preflight

| check | result |
|---|---|
| `full_pass.py --check-resume /home/jupyter/cf373_r3/sync` | pass |
| md5 `..._r2_200k.pth` | `f477c03525bf5e169704715511f1c6d7`, equal to the card |
| md5 `..._r2_200k_optimizer.pth` | `740891276637ff7bce744b1d9109d57a`, equal to the card |
| `full_pass.py --check-leg 300000` | pass, resumes `..._r2_200k.pth` |
| `experiments/hf_token.txt` | present, 37 bytes |
| `cell_claims.txt`, `HOLD_ABOVE` | absent, so no refusal |
| GIFT-Eval data at `~/workspaces/gift-eval-data` | present |
| GPU 0 free memory | 7522 MiB, above the 6500 MiB gate |
| disk free | 124 GB |

No artefact from an earlier attempt exists. No `run_pass.sh` process runs.
The `leg_300k`, `leg_450k` and `leg_665k` directories do not exist yet.

### Command

```
WT=/tmp/contrastive-forecasting-407 BB_GPU=0 nohup setsid bash \
  reports/2026-08-20_a4_full_pass/scripts/run_pass.sh \
  > reports/2026-08-20_a4_full_pass/results/run_pass.out 2>&1 &
```

Start 19:21:31 UTC.

### Continuity, confirmed

`train.py` printed these three lines:

```
[checkpoint] Restored optimizer from .../cf393_..._r2_200k_optimizer.pth (step=200000, best_ff=0.2490)
Resumed from .../cf393_..._r2_200k.pth at step 200000
  [dataloader] 4274 shards, 42740000 total rows, target skip 12800000
  [dataloader] Fast-skip: starting at shard 1280/4274, then skipping 0 rows within it
```

The weights, the optimizer state, the step counter and the data pointer all
continue. The run does not start at step 0.

The leg writes into a new directory, `leg_300k`, so it does not overwrite the
200k checkpoint set. Checkpoint safety rule 2 holds.

## Notes

The driver runs three legs in series, and each leg trains two heads and runs
two GIFT-Eval passes. One leg plus its two stops must finish before the next
leg starts. A dead head costs one point, not the stops behind it, because the
driver records the gap and continues.

## Known, and not a fault

`collect.sh` prints `downsample failed` while a leg still runs. The losses CSV
logs one row per step, and `downsample_curve.py` refuses a file that gives it
fewer than a few points at `--stride 200`. A leg of 100,000 steps gives 500
points, so the guard passes at the stop. The driver calls `collect.sh` after
each stop, which is when the CSV is complete.
