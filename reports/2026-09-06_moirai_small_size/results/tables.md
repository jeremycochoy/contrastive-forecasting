### The scores

The band is **0.0568** (this card measured it at 11.4M). Two numbers closer than that are not ranked. The `twin minus 11.4M` column subtracts the row's 11.4M score from its 1.1M twin, and positive favours the 11.4M arm.

| arm | k | reduce | seed | lr | L_rep decay | stop | 11.4M | 1.1M twin (parent seed range) | twin minus 11.4M | head-matched |
|---|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09_lr56 | 3 | sum | 20260520 | 5.6e-4 | no | 40,000 | 1.1820 | never run | — | — |
| k3_r100_09b_lr56 | 3 | sum | 20260525 | 5.6e-4 | no | 40,000 | 1.2469 | never run | — | — |
| k3_r100_09_lr33 | 3 | sum | 20260520 | 3.3e-4 | no | 40,000 | 1.2483 | never run | — | — |
| k3_r100_09_lr17 | 3 | sum | 20260520 | 1.67e-4 | no | 40,000 | 1.2612 | never run | — | — |
| k3_r100_09b | 3 | sum | 20260525 | 1e-3 | no | 40,000 | 1.2927 | 1.3618 (not seed-matched) | +0.0691 | no, 15,000-step head |
| k3_r100_09_lr56 | 3 | sum | 20260520 | 5.6e-4 | no | 100,000 | 1.3170 | never run | — | — |
| k3_r100_09_lr56 | 3 | sum | 20260520 | 5.6e-4 | no | 200,000 | 1.3189 | never run | — | — |
| k3_r100_09_dec | 3 | sum | 20260520 | 1e-3 | yes | 40,000 | 1.3236 | never run | — | — |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 100,000 | 1.3395 | 1.3010 | -0.0385 | yes |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 40,000 | 1.3495 | 1.3618 | +0.0123 | no, 15,000-step head |
| k3_r100_09 | 3 | sum | 20260520 | 1e-3 | no | 200,000 | 1.3910 | 1.3998 | +0.0088 | yes |
| k32_r100_09_lr56 | 32 | mean | 20260520 | 5.6e-4 | no | 40,000 | 1.4404 | never run | — | — |
| k8_r100_09 | 8 | mean | 20260520 | 1e-3 | no | 40,000 | 1.4537 | never run | — | — |
| k32_r100_09 | 32 | mean | 20260520 | 1e-3 | no | 40,000 | 1.4629 | 1.1507 (1.1491 to 1.1507) | -0.3122 | yes |

### The contrastive AUC, step by step

Lower is worse. A run at 0.5 has lost the task. The last column is what the same cell, seed and stop reached at 1.1M parameters.

| arm | k | EMA momentum | L_rep decay | 2,000 | 5,000 | 8,000 | 10,000 | 12,000 | 15,000 | 18,600 | 25,000 | 28,000 | 40,000 | verdict | 1.1M twin at 40,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09 | 3 | 0.9 to 1.0 at 100k | no | 0.993 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.999 | 0.999 | 0.999 | held | — |
| k3_r100_09b | 3 | 0.9 to 1.0 at 100k | no | 0.993 | 0.995 | 0.998 | 0.999 | 0.999 | 0.997 | 0.998 | 0.999 | 0.999 | 0.998 | held | — |
| k3_r100_09_lr33 | 3 | 0.9 to 1.0 at 100k | no | 0.997 | 0.992 | 0.997 | 0.998 | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 | 0.998 | held | — |
| k3_r100_09_lr17 | 3 | 0.9 to 1.0 at 100k | no | 0.998 | 0.996 | 0.991 | 0.996 | 0.997 | 0.998 | 0.998 | 0.998 | 0.999 | 0.999 | held | — |
| k3_r100_09_lr56 | 3 | 0.9 to 1.0 at 100k | no | 0.996 | 0.998 | 0.998 | 0.998 | 0.997 | 0.998 | 0.998 | 0.997 | 0.998 | 0.996 | held | — |
| k3_r100_09_mean | 3 | 0.9 to 1.0 at 100k | no | 0.976 | 0.976 | 0.853 | 0.863 | 0.682 | 0.828 | 0.807 | 0.632 | — | — | lost at 26,413 | — |
| k3_r100_09b_lr56 | 3 | 0.9 to 1.0 at 100k | no | 0.997 | 0.998 | 0.998 | 0.997 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | held | — |
| k3_r100_09_lr45 | 3 | 0.9 to 1.0 at 100k | no | — | — | — | — | — | — | — | — | — | — | — | — |
| k3_r100_09_dec | 3 | 0.9 to 1.0 at 100k | yes | 0.996 | 0.992 | 0.986 | 0.980 | 0.984 | 0.992 | 0.992 | 0.997 | 0.997 | 0.997 | held | — |
| k8_r100_09 | 8 | 0.9 to 1.0 at 100k | no | 0.971 | 0.961 | 0.920 | 0.881 | 0.812 | 0.717 | 0.769 | 0.769 | 0.772 | 0.730 | held | — |
| k32_r100_09 | 32 | 0.9 to 1.0 at 100k | no | 0.968 | 0.937 | 0.889 | 0.892 | 0.840 | 0.793 | 0.763 | 0.843 | 0.789 | 0.773 | held | 0.978 (#404) |
| k32_r200_08 | 32 | 0.8 to 1.0 at 200k | no | 0.877 | 0.814 | 0.788 | 0.749 | 0.739 | 0.751 | 0.796 | 0.568 | 0.555 | — | lost at 28,152 | 0.957 (#404) |
| k32_r100_09_lr56 | 32 | 0.9 to 1.0 at 100k | no | 0.973 | 0.964 | 0.932 | 0.925 | 0.945 | 0.936 | 0.934 | 0.929 | 0.887 | 0.730 | held | — |
| k32_r100_09_sum | 32 | 0.9 to 1.0 at 100k | no | — | — | — | — | — | — | — | — | — | — | — | — |
| k32_r100_09_dec | 32 | 0.9 to 1.0 at 100k | yes | 0.978 | 0.746 | 0.758 | 0.742 | 0.642 | 0.747 | 0.718 | — | — | — | lost at 18,634 | 0.983 (#409) |

**Read this table by row and by column, never on the diagonal.** Two rows are comparable only when they differ in ONE column. These are the pairs, and there are no others:

- `k32_r100_09_lr56` against `k32_r100_09_sum`, which moves the reduction
- `k32_r100_09_sum` against `k3_r100_09_lr56`, which moves the rollout depth
- `k32_r100_09` against `k32_r100_09_dec`, which moves the L_rep decay
- `k32_r100_09` against `k32_r100_09_lr56`, which moves the learning rate
- `k32_r100_09` against `k32_r200_08`, which moves the EMA momentum
- `k32_r100_09` against `k3_r100_09_mean`, which moves the rollout depth
- `k32_r100_09` against `k8_r100_09`, which moves the rollout depth
- `k3_r100_09_lr17` against `k3_r100_09_lr33`, which moves the learning rate
- `k3_r100_09_lr17` against `k3_r100_09_lr45`, which moves the learning rate
- `k3_r100_09_lr17` against `k3_r100_09_lr56`, which moves the learning rate
- `k3_r100_09_lr33` against `k3_r100_09_lr45`, which moves the learning rate
- `k3_r100_09_lr33` against `k3_r100_09_lr56`, which moves the learning rate
- `k3_r100_09_lr45` against `k3_r100_09_lr56`, which moves the learning rate
- `k3_r100_09_lr56` against `k3_r100_09b_lr56`, which moves the seed
- `k3_r100_09_mean` against `k8_r100_09`, which moves the rollout depth
- `k3_r100_09` against `k3_r100_09_dec`, which moves the L_rep decay
- `k3_r100_09` against `k3_r100_09_lr17`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09_lr33`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09_lr45`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09_lr56`, which moves the learning rate
- `k3_r100_09` against `k3_r100_09_mean`, which moves the reduction
- `k3_r100_09` against `k3_r100_09b`, which moves the seed
- `k3_r100_09b` against `k3_r100_09b_lr56`, which moves the learning rate

### The loss by term

| arm | stop | last step | total loss | L_rep | L_align | L_rep weight | EMA momentum | AUC |
|---|---|---|---|---|---|---|---|---|
| k32_r100_09 | 40,000 | 40,000 | 13.4947 | 11.6270 | 1.7583 | 1.00 | 0.9400 | 0.7629 |
| k32_r100_09_dec | 40,000 | 19,100 | 1.8013 | — | 1.8219 | 0.00 | 0.9191 | 0.5010 |
| k32_r100_09_lr56 | 40,000 | 40,000 | 13.5159 | 11.6087 | 1.7892 | 1.00 | 0.9400 | 0.7098 |
| k32_r200_08 | 40,000 | 28,500 | 13.6105 | 11.5968 | 2.0018 | 1.00 | 0.8285 | 0.5254 |
| k3_r100_09 | 100,000 | 100,000 | 12.3238 | 11.7174 | 0.1472 | 1.00 | 1.0000 | 0.9997 |
| k3_r100_09 | 200,000 | 200,000 | 12.2424 | 11.7181 | 0.1319 | 1.00 | 1.0000 | 0.9938 |
| k3_r100_09 | 40,000 | 40,000 | 12.5323 | 11.7822 | 0.1852 | 1.00 | 0.9400 | 0.9993 |
| k3_r100_09_dec | 40,000 | 40,000 | 1.0579 | — | 0.2487 | 0.00 | 0.9400 | 0.9980 |
| k3_r100_09_lr17 | 40,000 | 40,000 | 12.8115 | 11.7678 | 0.2532 | 1.00 | 0.9400 | 0.9992 |
| k3_r100_09_lr33 | 40,000 | 40,000 | 12.5972 | 11.7772 | 0.2020 | 1.00 | 0.9400 | 0.9983 |
| k3_r100_09_lr56 | 100,000 | 100,000 | 12.5741 | 11.7491 | 0.1995 | 1.00 | 1.0000 | 0.9995 |
| k3_r100_09_lr56 | 200,000 | 200,000 | 12.5101 | 11.7601 | 0.1870 | 1.00 | 1.0000 | 0.9945 |
| k3_r100_09_lr56 | 40,000 | 40,000 | 12.6520 | 11.6891 | 0.2118 | 1.00 | 0.9400 | 0.9973 |
| k3_r100_09_mean | 40,000 | 27,400 | 13.5760 | 11.5998 | 1.9446 | 1.00 | 0.9274 | 0.5278 |
| k3_r100_09b | 40,000 | 40,000 | 12.9777 | 11.7567 | 0.2832 | 1.00 | 0.9400 | 0.9984 |
| k3_r100_09b_lr56 | 40,000 | 40,000 | 12.7730 | 11.7634 | 0.2463 | 1.00 | 0.9400 | 0.9990 |
| k8_r100_09 | 40,000 | 40,000 | 13.3835 | 11.6129 | 1.7117 | 1.00 | 0.9400 | 0.7563 |

### The cost

| run | stage | steps | hours |
|---|---|---|---|
| k3_r100_09 | backbone | 40,000 | 4.6 |
| k3_r100_09_bb40k_h30k_student | head | 30,000 | 1.9 |
| k3_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 2.8 |
| k32_r100_09 | backbone | 40,000 | 9.6 |
| k32_r100_09_bb40k_h30k_student | head | 30,000 | 1.8 |
| k32_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 2.9 |
| k3_r100_09b_bb40k_h30k_student | head | 30,000 | 1.4 |
| k3_r100_09_bb100k_h30k_student | head | 30,000 | 1.5 |
| k3_r100_09b_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.1 |
| k3_r100_09_bb100k_h30k_student | GIFT-Eval, 97 configs | — | 2.9 |
| k3_r100_09_dec | backbone | 40,000 | 4.2 |
| k3_r100_09_lr33 | backbone | 40,000 | 4.2 |
| k3_r100_09_lr56 | backbone | 40,000 | 4.8 |
| k3_r100_09_lr33_bb40k_h30k_student | head | 30,000 | 1.7 |
| k3_r100_09_dec_bb40k_h30k_student | head | 30,000 | 1.8 |
| k3_r100_09_lr56_bb40k_h30k_student | head | 30,000 | 1.7 |
| k3_r100_09_lr33_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.9 |
| k3_r100_09 | backbone | 200,000 | 13.4 |
| k3_r100_09_lr17 | backbone | 40,000 | 5.1 |
| k3_r100_09_bb200k_h30k_student | head | 30,000 | 1.8 |
| k3_r100_09_dec_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 4.2 |
| k3_r100_09_lr17_bb40k_h30k_student | head | 30,000 | 1.8 |
| k3_r100_09_lr56_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 4.2 |
| k8_r100_09 | backbone | 40,000 | 5.3 |
| k3_r100_09_bb200k_h30k_student | GIFT-Eval, 97 configs | — | 4.0 |
| k8_r100_09_bb40k_h30k_student | head | 30,000 | 1.6 |
| k3_r100_09_lr17_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.9 |
| k8_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 4.0 |
| k3_r100_09_lr56 | backbone | 100,000 | 5.5 |
| k3_r100_09_lr56_bb100k_h30k_student | head | 30,000 | 1.6 |
| k32_r100_09_lr56 | backbone | 40,000 | 7.7 |
| k32_r100_09_lr56_bb40k_h30k_student | head | 30,000 | 1.6 |
| k3_r100_09_lr56_bb100k_h30k_student | GIFT-Eval, 97 configs | — | 3.7 |
| k3_r100_09b_lr56 | backbone | 40,000 | 3.9 |
| k32_r100_09_lr56_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.7 |
| k3_r100_09b_lr56_bb40k_h30k_student | head | 30,000 | 1.8 |
| k3_r100_09_lr56 | backbone | 200,000 | 11.3 |
| k3_r100_09b_lr56_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 3.6 |
| k3_r100_09_lr56_bb200k_h30k_student | head | 30,000 | 1.6 |
| k3_r100_09_lr56_bb200k_h30k_student | GIFT-Eval, 97 configs | — | 3.5 |
