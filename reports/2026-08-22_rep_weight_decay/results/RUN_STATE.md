# #409 run state — the L_rep weight decay at k = 32

- updated: 2026-08-23 02:28:43
- note: starting
- cell: `arm6_v2_combab_alignT`, k = 32, reduce `mean`, target `teacher`
- decay: 1.0 to 0.0 at step 10000. No control arm: the references are 1.1507 (seed 20260520) and 1.1491 (seed 20260524), from `reports/2026-08-19_ema_momentum_k32/`.
- arms: dec_s20 dec_s24 dec_s22 dec_s23 dec_s25 dec_s26
- cards: 1 1 1, launcher pid 264855
- root: `/home/jupyter/checkpoints_backup/cf-409`
- artefacts: elisa holds them all, and no sync loop runs. See `notes/artefacts.md`.

## Scores

```
(none yet)
```

## Contrastive AUC

```
(none yet)
```

## Backbones on disk

```
(none yet)
```
