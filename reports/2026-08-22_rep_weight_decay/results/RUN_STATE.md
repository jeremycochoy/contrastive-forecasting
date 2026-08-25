# #409 run state — the L_rep weight decay at k = 32

- updated: 2026-08-23 19:11:18
- note: done — 2 lane(s) failed
- cell: `arm6_v2_combab_alignT`, k = 32, reduce `mean`, target `teacher`
- decay: 1.0 to 0.0 at step 10000. Fixed on every arm.
- axis: the EMA schedule. No control arm: the sweep scored these schedules with no decay, in `reports/2026-08-19_ema_momentum_k32/`.
- arms: dec_m099_fix dec_m090_r60 dec_m095_fix dec_m090_fix dec_m080_r200
- cards: 1 1, launcher pid 1019407
- root: `/home/jupyter/checkpoints_backup/cf-409`
- artefacts: elisa holds them all, and no sync loop runs. See `notes/artefacts.md`.

## The schedules

```
dec_m099_fix   0.99 fixed             reaches 0.990 at 40000   seed 20260520
dec_m090_r60   0.9 to 1.0 at 60k      reaches 0.967 at 40000   seed 20260520
dec_m095_fix   0.95 fixed             reaches 0.950 at 40000   seed 20260520
dec_m090_fix   0.9 fixed              reaches 0.900 at 40000   seed 20260520
dec_m080_r200  0.8 to 1.0 at 200k     reaches 0.840 at 40000   seed 20260520
```

## Scores

```
arm,ema_tau,ema_end,ema_ramp,ema_at_stop,seed,rep_end,ramp,rep_w_at_stop,align_target,stop,head_steps,encoder,score
dec_s20,0.9,1.0,100000,0.940,20260520,0.0,10000,0.000,teacher,40000,30000,student,1.2670
dec_s22,0.9,1.0,100000,0.940,20260522,0.0,10000,0.000,teacher,40000,30000,student,1.2593
dec_s24,0.9,1.0,100000,0.940,20260524,0.0,10000,0.000,teacher,40000,30000,student,1.2812
```

## Contrastive AUC

```
run	verdict	lost_at	floor	floor_step	last	last_step	note
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_losses.csv	held	-	0.8638	13123	0.9842	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_losses.csv	held	-	0.8944	11688	0.9841	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_losses.csv	held	-	0.8680	12323	0.9587	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090_r60_losses.csv	error	-	-	-	-	-	/home/jupyter/checkpoints_backup/cf-409/dec_m090_r60/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090_r60_losses.csv: no readable `auc` row above step 1000
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m095_fix_losses.csv	error	-	-	-	-	-	/home/jupyter/checkpoints_backup/cf-409/dec_m095_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m095_fix_losses.csv: no readable `auc` row above step 1000
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_losses.csv	held	-	0.9049	19755	0.9055	19900	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_losses.csv	error	-	-	-	-	-	/home/jupyter/checkpoints_backup/cf-409/dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_losses.csv: no readable `auc` row above step 1000
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_losses.csv	held	-	0.7685	11213	0.8768	19900	held
```

## Backbones on disk

```
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_10k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_15k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_5k.pth
dec_s20/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_20k.pth
dec_s20/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_40k.pth
dec_s22/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_20k.pth
dec_s22/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_40k.pth
dec_s23/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s23_20k.pth
dec_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_20k.pth
dec_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_40k.pth
dec_s25/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s25_20k.pth
```
