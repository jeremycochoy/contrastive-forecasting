# #401 run state

- updated: 2026-08-17 16:35
- stage: phase 1, arm k = 32, 200k leg, near step 115,000, ETA about 14.5 h
- launcher pid: 542935 (`scripts/launch_elisa.sh`) — RUNS. Do not stop it.
- diag watcher pid: 2601931 (`scripts/watch_k32_200k.sh`) — CPU only, adds
  the k = 32 200k checkpoints to the collapse table as they land
- gpu: 0, arm slots: 2 (edit `results/SLOTS`)
- root: `/home/jupyter/checkpoints_backup/cf-401`

## Scores so far

```
phase,k,stop,head_steps,encoder,score
1,16,100000,30000,student,12.4827
1,16,200000,30000,student,2.9331
1,16,40000,30000,student,4.5297
1,32,100000,30000,student,1.7939
1,32,40000,30000,student,7.9575
1,8,100000,30000,student,7.9344
1,8,200000,30000,student,2.4755
1,8,40000,30000,student,2.0357
```

Controls, `results/diag/`: c1 pathbound 1.2910 against #373's published
1.2751. c2 k = 0 anchor 1.1600.

Open cell: `k = 32` at bb200k. The leg trains now. The launcher trains the
head and runs the eval on its own when the leg ends.

## Diagnosis state

`results/diag/DIAGNOSIS.md` holds it. All 25 `k > 0` checkpoints are
collapsed, all 6 `k = 0` checkpoints are not. See
`results/diag/collapse_vs_score.md` for the 31-row table.

## Checkpoints on disk

```
k16/arm6_v2_combab_alignS/leg_100k/  60k 80k 100k
k16/arm6_v2_combab_alignS/leg_200k/  120k 140k 160k 180k 200k
k16/arm6_v2_combab_alignS/leg_40k/   20k 40k
k32/arm6_v2_combab_alignS/leg_100k/  60k 80k 100k
k32/arm6_v2_combab_alignS/leg_200k/  (empty, leg runs)
k32/arm6_v2_combab_alignS/leg_40k/   20k 40k
k8/arm6_v2_combab_alignS/leg_100k/   60k 80k 100k
k8/arm6_v2_combab_alignS/leg_200k/   120k 140k 160k 180k 200k
k8/arm6_v2_combab_alignS/leg_40k/    20k 40k
```

The `k = 0` parent is `cf-393/arm6_v2_combab_alignS`, legs 40k and 100k.
