# #401 run state — the mean objective

- updated: 2026-08-19 19:12:14
- note: rebuilt for the partial-fill report
- objective: `--train-rollout-reduce mean`, depths 8 32
- root (the sync loop lands the box's tree here): `/home/jupyter/cf401_sync/box_a/sync`
- results: `/tmp/contrastive-forecasting-401/reports/2026-08-15_rollout_depth_k16_8_32/results/mean`

## Scores so far

```
phase,k,variant,stop,head_steps,encoder,score
1,0,parent,100000,30000,student,1.1945
1,32,base,100000,30000,student,1.1803
1,32,base,200000,30000,student_s20260723,1.1676
1,32,base,200000,30000,student_s20260724,1.1576
1,32,base,200000,30000,student,1.1637
1,32,base,40000,30000,student,1.2082
2,32,base,40000,40000,student,1.2093
1,32,ema30k,40000,30000,student,1.2385
2,8,base,100000,100000,student,1.3270
1,8,base,100000,30000,student,1.2857
1,8,base,200000,30000,student,1.2898
1,8,base,40000,30000,student,1.2433
2,8,base,40000,40000,student,1.2543
```

## Cells the eval finished

```
k8_bb40k_h30k_student  1.2433
k32_bb40k_h30k_student  1.2082
k8_bb100k_h30k_student  1.2857
k32_bb100k_h30k_student  1.1803
k8_bb200k_h30k_student  1.2898
k32_ema30k_bb40k_h30k_student  1.2385
k32_bb200k_h30k_student  1.1637
k8_bb40k_h40k_student  1.2543
k8_bb100k_h100k_student  1.3270
k32_bb40k_h40k_student  1.2093
k32_bb200k_h30k_student_s20260724  1.1576
k0_parent_bb100k_h30k_student  1.1945
```

## Running now

```
head-train  qhead_k32_bb100k_h100k_student_s20260722
head-train  qhead_k8_bb200k_h200k_student_s20260722
```

## Backbone stops on this side

```
k32/arm6_v2_combab_alignS/leg_100k/cf393_arm6_v2_combab_alignS_cf373k32_mean_100k.pth
k32/arm6_v2_combab_alignS/leg_100k/cf393_arm6_v2_combab_alignS_cf373k32_mean_60k.pth
k32/arm6_v2_combab_alignS/leg_100k/cf393_arm6_v2_combab_alignS_cf373k32_mean_80k.pth
k32/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k32_mean_120k.pth
k32/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k32_mean_140k.pth
k32/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k32_mean_160k.pth
k32/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k32_mean_180k.pth
k32/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k32_mean_200k.pth
k32/arm6_v2_combab_alignS/leg_20k/cf393_arm6_v2_combab_alignS_cf373k32_mean_20k.pth
k32/arm6_v2_combab_alignS/leg_40k/cf393_arm6_v2_combab_alignS_cf373k32_mean_40k.pth
k8/arm6_v2_combab_alignS/leg_100k/cf393_arm6_v2_combab_alignS_cf373k8_mean_100k.pth
k8/arm6_v2_combab_alignS/leg_100k/cf393_arm6_v2_combab_alignS_cf373k8_mean_60k.pth
k8/arm6_v2_combab_alignS/leg_100k/cf393_arm6_v2_combab_alignS_cf373k8_mean_80k.pth
k8/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k8_mean_120k.pth
k8/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k8_mean_140k.pth
k8/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k8_mean_160k.pth
k8/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k8_mean_180k.pth
k8/arm6_v2_combab_alignS/leg_200k/cf393_arm6_v2_combab_alignS_cf373k8_mean_200k.pth
k8/arm6_v2_combab_alignS/leg_20k/cf393_arm6_v2_combab_alignS_cf373k8_mean_20k.pth
k8/arm6_v2_combab_alignS/leg_40k/cf393_arm6_v2_combab_alignS_cf373k8_mean_40k.pth
```
