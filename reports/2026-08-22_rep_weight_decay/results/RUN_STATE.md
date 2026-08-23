# #409 run state — the L_rep weight decay at k = 32

- updated: 2026-08-23 03:34:45
- note: 3 lane(s) running
- cell: `arm6_v2_combab_alignT`, k = 32, reduce `mean`, target `teacher`
- decay: 1.0 to 0.0 at step 10000. No control arm: the references are 1.1507 (seed 20260520) and 1.1491 (seed 20260524), from `reports/2026-08-19_ema_momentum_k32/`.
- arms: dec_s20 dec_s24 dec_s22 dec_s23 dec_s25 dec_s26
- cards: 1 1 1, launcher pid 264855
- root: `/home/jupyter/checkpoints_backup/cf-409`
- artefacts: elisa holds them all, and no sync loop runs. See `notes/artefacts.md`.

## Scores

```
arm,seed,rep_end,ramp,rep_w_at_stop,align_target,stop,head_steps,encoder,score
```

## Contrastive AUC

```
run	verdict	lost_at	floor	floor_step	last	last_step	note
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_losses.csv	held	-	0.9578	5957	0.9584	6100	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_losses.csv	held	-	0.9436	3548	0.9531	5900	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_losses.csv	held	-	0.9567	2797	0.9608	5800	held
```

## Backbones on disk

```
(none yet)
```
