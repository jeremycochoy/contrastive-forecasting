# #409 run state — the L_rep weight decay at k = 32

- updated: 2026-08-31 08:17:04
- note: the stop is part of the key of scores.csv, 15 rows over two stops
- cell: `arm6_v2_combab_alignT`, k = 32, reduce `mean`, target `teacher`
- decay: 1.0 to 0.0 at the arm's ramp, which is column 5 of `scripts/arms.tsv`.
- axes: the EMA schedule and the decay ramp. No control arm: the sweep scored these schedules with no decay, in `reports/2026-08-19_ema_momentum_k32/`.
- arms: dec_s20 dec_s22 dec_s23 dec_s24 dec_s25 dec_m090_fix dec_m090_r60 dec_m095_fix dec_m099_fix dec_m090_r200 dec_m080_r200 dec_m095_r100 dec_m070_fix dec_m050_fix dec_ramp30k_m080 dec_ramp20k_m080 dec_ramp5k_m080 dec_m080_r200_s24 dec_m090r100_ramp5k dec_m090r100_ramp2k dec_m090r100_ramp1k
- cards: -, launcher pid -
- root: `/home/jupyter/checkpoints_backup/cf-409`
- artefacts: elisa holds them all, and no sync loop runs. See `notes/artefacts.md`.

## The arms, and what each one reached

```
arm                schedule               reaches  ramp    seed      reached  score
dec_s20            0.9 to 1.0 at 100k     0.940    10000   20260520  40000    1.2670
dec_s22            0.9 to 1.0 at 100k     0.940    10000   20260522  40000    1.2593
dec_s23            0.9 to 1.0 at 100k     0.940    10000   20260523  22900    
dec_s24            0.9 to 1.0 at 100k     0.940    10000   20260524  40000    1.2812
dec_s25            0.9 to 1.0 at 100k     0.940    10000   20260525  22700    
dec_m090_fix       0.9 fixed              0.900    10000   20260520  0        
dec_m090_r60       0.9 to 1.0 at 60k      0.967    10000   20260520  100      
dec_m095_fix       0.95 fixed             0.950    10000   20260520  0        
dec_m099_fix       0.99 fixed             0.990    10000   20260520  40000    1.2849
dec_m090_r200      0.9 to 1.0 at 200k     0.920    10000   20260520  0        
dec_m080_r200      0.8 to 1.0 at 200k     0.840    10000   20260520  40000    1.2352
dec_m095_r100      0.95 to 1.0 at 100k    0.970    10000   20260520  0        
dec_m070_fix       0.7 fixed              0.700    10000   20260520  40000    1.3534
dec_m050_fix       0.5 fixed              0.500    10000   20260520  10600    
dec_ramp30k_m080   0.8 to 1.0 at 200k     0.840    30000   20260520  40000    1.3623
dec_ramp20k_m080   0.8 to 1.0 at 200k     0.840    20000   20260520  40000    1.3178
dec_ramp5k_m080    0.8 to 1.0 at 200k     0.840    5000    20260520  40000    1.2727
dec_m080_r200_s24  0.8 to 1.0 at 200k     0.840    10000   20260524  40000    1.2823
dec_m090r100_ramp5k 0.9 to 1.0 at 100k     0.940    5000    20260520  40000    1.2537
dec_m090r100_ramp2k 0.9 to 1.0 at 100k     0.940    2000    20260520  80000    1.2295
dec_m090r100_ramp1k 0.9 to 1.0 at 100k     0.940    1000    20260520  80000    1.2322
```

`reached` is the last step that arm's losses CSVs hold. 0 means the arm never wrote a step. `score` is the score at the 40000-step stop. The Scores section below holds every stop.

## Scores

```
arm,ema_tau,ema_end,ema_ramp,ema_at_stop,seed,rep_end,ramp,rep_w_at_stop,align_target,stop,head_steps,encoder,score
dec_m070_fix,0.7,-,-,0.700,20260520,0.0,10000,0.000,teacher,40000,30000,student,1.3534
dec_m080_r200,0.8,1.0,200000,0.840,20260520,0.0,10000,0.000,teacher,40000,30000,student,1.2352
dec_m080_r200_s24,0.8,1.0,200000,0.840,20260524,0.0,10000,0.000,teacher,40000,30000,student,1.2823
dec_m090r100_ramp1k,0.9,1.0,100000,0.940,20260520,0.0,1000,0.000,teacher,40000,30000,student,1.2322
dec_m090r100_ramp1k,0.9,1.0,100000,0.980,20260520,0.0,1000,0.000,teacher,80000,30000,student,1.2381
dec_m090r100_ramp2k,0.9,1.0,100000,0.940,20260520,0.0,2000,0.000,teacher,40000,30000,student,1.2295
dec_m090r100_ramp2k,0.9,1.0,100000,0.980,20260520,0.0,2000,0.000,teacher,80000,30000,student,1.2473
dec_m090r100_ramp5k,0.9,1.0,100000,0.940,20260520,0.0,5000,0.000,teacher,40000,30000,student,1.2537
dec_m099_fix,0.99,-,-,0.990,20260520,0.0,10000,0.000,teacher,40000,30000,student,1.2849
dec_ramp20k_m080,0.8,1.0,200000,0.840,20260520,0.0,20000,0.000,teacher,40000,30000,student,1.3178
dec_ramp30k_m080,0.8,1.0,200000,0.840,20260520,0.0,30000,0.000,teacher,40000,30000,student,1.3623
dec_ramp5k_m080,0.8,1.0,200000,0.840,20260520,0.0,5000,0.000,teacher,40000,30000,student,1.2727
dec_s20,0.9,1.0,100000,0.940,20260520,0.0,10000,0.000,teacher,40000,30000,student,1.2670
dec_s22,0.9,1.0,100000,0.940,20260522,0.0,10000,0.000,teacher,40000,30000,student,1.2593
dec_s24,0.9,1.0,100000,0.940,20260524,0.0,10000,0.000,teacher,40000,30000,student,1.2812
```

## Contrastive AUC

```
run	verdict	lost_at	floor	floor_step	last	last_step	note
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_losses.csv	held	-	0.8638	13123	0.9842	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_losses.csv	held	-	0.8944	11688	0.9841	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s23_losses.csv	held	-	0.9041	22249	0.9371	22900	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s23_r2_losses.csv	held	-	0.9208	20023	0.9261	20100	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_losses.csv	held	-	0.8680	12323	0.9587	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s25_losses.csv	held	-	0.7434	19846	0.8147	22700	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s25_r2_losses.csv	held	-	0.7447	20011	0.7654	20100	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090_r60_losses.csv	error	-	-	-	-	-	/home/jupyter/checkpoints_backup/cf-409/dec_m090_r60/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090_r60_losses.csv: no readable `auc` row above step 1000
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m095_fix_losses.csv	error	-	-	-	-	-	/home/jupyter/checkpoints_backup/cf-409/dec_m095_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m095_fix_losses.csv: no readable `auc` row above step 1000
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_losses.csv	held	-	0.9049	19755	0.9055	19900	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_losses.csv	held	-	0.9210	20785	0.9695	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_losses.csv	held	-	0.7685	11213	0.9924	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_losses.csv	held	-	0.5732	11452	0.9166	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m050_fix_losses.csv	lost	10162	0.5221	10348	0.5233	10600	at step 10162
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_losses.csv	held	-	0.8522	30725	0.8921	32400	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_r2_losses.csv	held	-	0.5428	34826	0.9310	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_losses.csv	held	-	0.8353	27830	0.9659	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_losses.csv	held	-	0.7157	6630	0.9928	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_losses.csv	held	-	0.7780	11270	0.8790	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_losses.csv	held	-	0.8688	7851	0.9797	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_losses.csv	held	-	0.9092	3209	0.9833	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_losses.csv	held	-	0.9536	44762	0.9846	80000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_losses.csv	held	-	0.9075	2062	0.9788	40000	held
cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_losses.csv	held	-	0.9543	68542	0.9848	80000	held
```

One row per losses CSV, not per arm. A leg re-fired after a crash resumes under a `_rN` name and writes a second file, and the AUC gate reads a file.

## Backbones on disk

```
dec_m050_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m050_fix_10k.pth
dec_m050_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m050_fix_5k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_10k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_15k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_20k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_25k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_30k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_35k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_40k.pth
dec_m070_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m070_fix_5k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_10k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_15k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_20k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_25k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_30k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_35k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_40k.pth
dec_m080_r200/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_5k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_10k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_15k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_20k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_25k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_30k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_35k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_40k.pth
dec_m080_r200_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m080_r200_s24_5k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_10k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_15k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_20k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_25k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_30k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_35k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_40k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_5k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_45k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_50k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_55k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_60k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_65k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_70k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_75k.pth
dec_m090r100_ramp1k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp1k_80k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_10k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_15k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_20k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_25k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_30k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_35k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_40k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_5k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_45k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_50k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_55k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_60k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_65k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_70k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_75k.pth
dec_m090r100_ramp2k/arm6_v2_combab_alignT/leg_80k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp2k_80k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_10k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_15k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_20k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_25k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_30k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_35k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_40k.pth
dec_m090r100_ramp5k/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m090r100_ramp5k_5k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_10k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_15k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_5k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_20k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_25k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_30k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_35k.pth
dec_m099_fix/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_m099_fix_r2_40k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_10k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_15k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_20k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_25k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_30k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_35k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_40k.pth
dec_ramp20k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp20k_m080_5k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_10k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_15k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_20k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_25k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_30k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_5k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_r2_35k.pth
dec_ramp30k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp30k_m080_r2_40k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_10k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_15k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_20k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_25k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_30k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_35k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_40k.pth
dec_ramp5k_m080/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_ramp5k_m080_5k.pth
dec_s20/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_20k.pth
dec_s20/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s20_40k.pth
dec_s22/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_20k.pth
dec_s22/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s22_40k.pth
dec_s23/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s23_20k.pth
dec_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_20k.pth
dec_s24/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s24_40k.pth
dec_s25/arm6_v2_combab_alignT/leg_40k/cf393_arm6_v2_combab_alignT_cf373k32_cf409_dec_s25_20k.pth
```
