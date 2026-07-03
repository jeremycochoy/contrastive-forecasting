# 2026-07-03 — B=1024 retrain of the τ=0.90 last-ckpt winner, with trajectory checkpoints (#369)

Retrains the arm that minimises GM-Rel MASE on both `2L / last-ckpt` and
`6L / last-ckpt` in the #366 τ=0.90 grid, under one change only:
`--batch-size 1024` instead of `--batch-size 512`. Seed, steps, τ, λ pair,
optimiser, dataset all unchanged.

Two extras vs the #366 launchers:

  1. Backbone saves a fine-grained trajectory checkpoint every 500 steps
     (`--traj-save-every 500`) so the head can be trained from any step
     up to 12,500 — in particular the parent's `best-loss` step and the
     retrained `last` step.
  2. Downstream trains the same 2L and 6L quantile heads from **two**
     retrained-backbone checkpoints:
       - `_step<PARENT_BEST_LOSS_STEP>.pth` (the retrained backbone at
         the step number the parent's `best-loss` head landed on),
       - `_step12500.pth` (the retrained `last` at the same step budget).

## Launch-time gate

The winner arm (λ_e, λ_h) and the parent's best-loss step are NOT baked
into the scripts — they are read from a `winners.sh` manifest that the
operator writes at launch time, once #366 arms A–I have all finished
and the last-ckpt winner is confirmed. See `scripts/winners.sh.example`
for the format and the re-verify procedure.

`PARENT_BEST_LOSS_STEP` in the manifest must be a positive multiple of
`TRAJ_SAVE_EVERY` (default 500) — trajectory checkpoints land on those
step boundaries and nowhere else. The parent's raw `best_loss_step`
from the losses CSV has no reason to align, so snap it to the nearest
multiple of 500 when stamping the manifest. The launcher validates
this invariant and hard-aborts with a clear error if it's violated.

The launch procedure:

```bash
cp experiments/2026-07-03_b1024_traj_ckpts/scripts/winners.sh.example \
   experiments/2026-07-03_b1024_traj_ckpts/winners.sh
$EDITOR experiments/2026-07-03_b1024_traj_ckpts/winners.sh
# fill LAMBDA_E, LAMBDA_H, TAU, PARENT_BEST_LOSS_STEP, WINNERS_VERIFIED_BY, WINNERS_VERIFIED_AT.
# Update `winners.sh.example` in-tree too so pytest keeps README ↔ manifest ↔ suffix in sync.
bash experiments/2026-07-03_b1024_traj_ckpts/scripts/launch_experiment.sh
```

## Arms

The launch-time winner from #366 (A–I set) is filled in the manifest;
the launcher derives a run-name suffix `l_emb<10·λ_e>_enc<10·λ_h>_tau<100·τ>_b1024`.

| Arm (`suffix`) | λ_e | λ_h | τ | batch |
| --- | --: | --: | --: | --: |
| Arm 1 (`l_emb10_enc10_tau090_b1024`) | 1.0 | 1.0 | 0.90 | 1024 |

The single row above is a placeholder mirroring the current provisional
winner (arm C, λ_e=λ_h=1) from #366 while arms G/H/I are still in flight;
the operator re-verifies and overwrites the manifest (and this table) at
launch time. `tests/test_369_launcher_shape.py` guards
README ↔ `winners.sh.example` ↔ launcher-suffix consistency.

## Success criteria (from #369)

For each head depth (2L, 6L), compare the retrained cell against the
parent cell.

- If **both** retrained cells beat their parents' `last-ckpt` on
  GM-Rel MASE, extend backbone training to **~2×** the parent budget
  (≈ 25,000 steps) from the current 12,500 checkpoint, keeping trajectory
  saves, and re-evaluate heads at the same two loci.
- If neither retrained cell beats its parent, stop and report the
  difference.

## Layout

- `scripts/train_backbone_b1024.sh` — one-arm backbone trainer at B=1024
  with `--traj-save-every 500`. Parameterised on λ_e, λ_h, τ, suffix.
- `scripts/downstream_b1024.sh` — trains 2L or 6L q-head from BOTH the
  step-tagged `_step<PARENT_BEST_LOSS_STEP>.pth` and `_step12500.pth`
  backbones, then runs GIFT-Eval full-97 per cell.
- `scripts/launch_experiment.sh` — reads the manifest, runs backbone
  then downstream on GPU 0/1 in parallel.
- `scripts/build_gm_table.sh` — emits `results/gm_table.csv` combining
  the two new B=1024 cells with the parent B=512 cells (fetched from
  #366's branch by `git show`).
- `scripts/_compute_gm.py` — GM aggregation helper (identical to #366).
- `scripts/winners.sh.example` — committed manifest template.
