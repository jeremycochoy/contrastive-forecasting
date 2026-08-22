# #409 run state — the L_rep weight decay at k = 3

- updated: 2026-08-22 23:11:54
- note: starting
- cell: `arm6_v2_combab_alignS`, k = 3, reduce `sum`
- arms: ctrl_s20 dec0_s20 flr05_s20 flr02_s20
- cards: 1, launcher pid 4064271
- root: `/home/jupyter/checkpoints_backup/cf-409`
- artefacts: elisa holds them all, and no sync loop runs. See `notes/artefacts.md`.

## Scores

```
arm,rep_end,ramp,seed,align_target,rep_w_at_stop,stop,head_steps,encoder,score
```

## Contrastive AUC

```
run	verdict	lost_at	floor	floor_step	last	last_step	note
cf393_arm6_v2_combab_alignS_cf373k3_cf409_ctrl_s20_losses.csv	error	-	-	-	-	-	/home/jupyter/checkpoints_backup/cf-409/ctrl_s20/arm6_v2_combab_alignS/leg_40k/cf393_arm6_v2_combab_alignS_cf373k3_cf409_ctrl_s20_losses.csv: no readable `auc` row above step 1000
```

## Backbones on disk

```
(none yet)
```
