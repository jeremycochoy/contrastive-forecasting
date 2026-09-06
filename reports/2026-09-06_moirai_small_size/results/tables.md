### The scores

| arm | k | reduce | seed | L_rep decay | stop | 11.4M | 1.1M twin (parent seed range) | gap | head-matched |
|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09 | 3 | sum | 20260520 | no | 40,000 | 1.3495 | 1.3618 | -0.0123 | no, 15,000-step head |
| k32_r100_09 | 32 | mean | 20260520 | no | 40,000 | 1.4629 | 1.1507 (1.1491 to 1.1507) | +0.3122 | yes |

### The contrastive AUC

| arm | verdict | AUC floor | at step | AUC last | at step |
|---|---|---|---|---|---|
| k32_r100_09 | held | 0.6827 | 20872 | 0.7732 | 40000 |
| k3_r100_09 | held | 0.9931 | 1942 | 0.9988 | 40000 |
| k3_r100_09 | held | 0.9974 | 40001 | 0.9995 | 81800 |
| k3_r100_09b | held | 0.9922 | 1869 | 0.9988 | 32500 |

### The loss by term

| arm | stop | last step | total loss | L_rep | L_align | L_rep weight | EMA momentum | AUC |
|---|---|---|---|---|---|---|---|---|
| k32_r100_09 | 40,000 | 40,000 | 13.4947 | 11.6270 | 1.7583 | 1.00 | 0.9400 | 0.7629 |
| k3_r100_09 | 100,000 | 81,800 | 12.6786 | 11.7612 | 0.2201 | 1.00 | 0.9818 | 0.9998 |
| k3_r100_09 | 40,000 | 40,000 | 12.5323 | 11.7822 | 0.1852 | 1.00 | 0.9400 | 0.9993 |
| k3_r100_09b | 40,000 | 32,400 | 12.7900 | 11.7402 | 0.2282 | 1.00 | 0.9324 | 0.9992 |

### The cost

| run | stage | steps | hours |
|---|---|---|---|
| k3_r100_09 | backbone | 40,000 | 4.6 |
| k3_r100_09_bb40k_h30k_student | head | 30,000 | 1.9 |
| k3_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 2.8 |
| k32_r100_09 | backbone | 40,000 | 9.6 |
| k32_r100_09_bb40k_h30k_student | head | 30,000 | 1.8 |
| k32_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 2.9 |
