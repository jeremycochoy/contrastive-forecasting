# #409 run state — the L_rep weight decay at k = 32

- updated: 2026-08-23 09:52:34
- note: 1 lane(s) running
- cell: `arm6_v2_combab_alignT`, k = 32, reduce `mean`, target `teacher`
- decay: 1.0 to 0.0 at step 10000. No control arm: the references are 1.1507 (seed 20260520) and 1.1491 (seed 20260524), from `reports/2026-08-19_ema_momentum_k32/`.
- arms: dec_s25
- cards: 1, launcher pid 689385
- root: `/home/jupyter/checkpoints_backup/cf-409`
- artefacts: elisa holds them all, and no sync loop runs. See `notes/artefacts.md`.

## Scores

```
arm,seed,rep_end,ramp,rep_w_at_stop,align_target,stop,head_steps,encoder,score
```

## Contrastive AUC

```
run	verdict	lost_at	floor	floor_step	last	last_step	note
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_losses.csv	held	-	0.8638	13123	0.9842	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_losses.csv	held	-	0.8680	12323	0.9587	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_losses.csv	held	-	0.8944	11688	0.9841	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s23_losses.csv	error	-	-	-	-	-	/home/jupyter/checkpoints_backup/cf-409/dec_s23/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s23_losses.csv: no readable `auc` row above step 1000
```

## Backbones on disk

```
dec_s20/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_20k.pth
dec_s20/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_40k.pth
dec_s22/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_20k.pth
dec_s22/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_40k.pth
dec_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_20k.pth
dec_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_40k.pth
```
