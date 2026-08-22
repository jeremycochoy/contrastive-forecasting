# One full pass over small_v1 does not improve on 200,000 steps

More backbone training past 200,000 steps makes the score worse. One full pass over the training dataset `small_v1` does not beat the 200,000-step stop, on either head (GM-Relative MASE, defined in [the rollout-depth study](../2026-08-08_rollout_depth/rollout_depth.md)).

![GM-Relative MASE against backbone train step](plots/full_pass.png)

*GM-Relative MASE against backbone train step, student and teacher heads.*

Every student draw after 200,000 steps scores above every student draw at 200,000 steps, with the closest pair at 1.0691 against 1.0660. The teacher draws show the same shape. The rise at 300,000 steps, +0.021 on both heads, is larger than the largest measured head-seed range, 0.0087.

## The teacher head keeps changing after step 100,000

The teacher encoder freezes at step 100,000. The teacher head also loads 36 student-owned tensors, and 32 of them keep training. So each teacher point after 100,000 steps is a different model.

## Reproduction check

The head-seed 20260722 re-draw at 200,000 steps returns the published values on both heads.

## The 450,000 and 665,000 stops are not separated

The 665,000-step stop has one draw for each head, and its +0.0040 student-mean gap to the 450,000-step stop is smaller than the largest measured head-seed range, 0.0087.

## Tables

**Table 1: scores**

| stop | head | s20260722 | s20260723 | s20260724 | mean | range |
|---:|:---|---:|---:|---:|---:|---:|
| 200k | student | 1.0660 | 1.0652 | 1.0642 | 1.0651 | 0.0018 |
| 200k | teacher | 1.0828 | 1.0809 | 1.0764 | 1.0800 | 0.0064 |
| 300k | student | 1.0867 | 1.0883 | 1.0841 | 1.0864 | 0.0042 |
| 300k | teacher | 1.1030 | 1.0992 | 1.1004 | 1.1009 | 0.0038 |
| 450k | student | 1.0691 | 1.0761 | 1.0778 | 1.0743 | 0.0087 |
| 450k | teacher | 1.0986 | 1.0924 | 1.0945 | 1.0952 | 0.0062 |
| 665k | student | 1.0783 | — | — | 1.0783 | — |
| 665k | teacher | 1.1038 | — | — | 1.1038 | — |

**Table 2: provenance**

| item | value |
|:---|:---|
| start checkpoint | `cf393_arm6_v2_combab_alignS_cf373k3_r2_200k.pth`, md5 `f477c03525bf5e169704715511f1c6d7` |
| its optimizer state | `cf393_arm6_v2_combab_alignS_cf373k3_r2_200k_optimizer.pth`, md5 `740891276637ff7bce744b1d9109d57a` |
| 300k checkpoint | md5 `618e433edea74ed2ca4ad9d10be37377` |
| 450k checkpoint | md5 `f505688b3168e32b72eb45dad0a897e0` |
| 665k checkpoint | md5 `ec1f64a5d4bc1e12d830a625b89cad84` |
| launcher | `scripts/run_pass.sh`, which calls the rollout-depth study's `run_leg_k.sh` and `stop_k.sh` |

## Protocol

The rollout-depth study stopped its best backbone, A4 (the `arm6_v2_combab_alignS` cell at rollout depth k = 3), at 200,000 steps, and this card gives the same run one full pass. The run continues from that 200,000-step checkpoint, with its saved optimizer state, the same flags (constant lr 1e-3, no schedule) and the same data seed. 665,000 steps is 99.97% of one pass over `small_v1` (42,571,692 rows, 665,182 steps for one pass). Each stop trains a fresh head for 30,000 steps, with seeds 20260722 to 20260724 at 200k, 300k and 450k, and seed 20260722 at 665k. Each head scores on 97 GIFT-Eval configs with the B4 strategy at horizon 16. The arm6_v2 head recipe and the GM-Relative MASE metric are defined in the rollout-depth study.
