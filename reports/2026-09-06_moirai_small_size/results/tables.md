### The scores

| arm | k | reduce | seed | L_rep decay | stop | 11.4M | 1.1M twin | gap | head-matched |
|---|---|---|---|---|---|---|---|---|---|
| k3_r100_09 | 3 | sum | 20260520 | no | 40,000 | 1.3495 | 1.3618 | -0.0123 | no, 15,000-step head |

### The contrastive AUC

| arm | verdict | AUC floor | at step | AUC last | at step |
|---|---|---|---|---|---|
| k32_r100_09 | held | 0.6827 | 20872 | 0.7732 | 40000 |
| k3_r100_09 | held | 0.9931 | 1942 | 0.9988 | 40000 |
| k3_r100_09 | held | 0.9974 | 40001 | 0.9990 | 41300 |
| k3_r100_09b | held | 0.9922 | 1869 | 0.9965 | 3100 |

### The loss by term

| arm | stop | last step | total loss | L_rep | L_align | L_rep weight | EMA momentum | AUC |
|---|---|---|---|---|---|---|---|---|
| k32_r100_09 | 40,000 | 40,000 | 13.4947 | 11.6270 | 1.7583 | 1.00 | 0.9400 | 0.7629 |
| k3_r100_09 | 100,000 | 41,300 | 12.5069 | 11.7613 | 0.1801 | 1.00 | 0.9413 | 0.9984 |
| k3_r100_09 | 40,000 | 40,000 | 12.5323 | 11.7822 | 0.1852 | 1.00 | 0.9400 | 0.9993 |
| k3_r100_09b | 40,000 | 3,100 | 12.7496 | 11.6900 | 0.2152 | 1.00 | 0.9031 | 0.9954 |

### The cost

| run | stage | steps | hours |
|---|---|---|---|
| k3_r100_09 | backbone | 40,000 | 4.6 |
| k3_r100_09_bb40k_h30k_student | head | 30,000 | 1.9 |
| k3_r100_09_bb40k_h30k_student | GIFT-Eval, 97 configs | — | 2.8 |
| k32_r100_09 | backbone | 40,000 | 9.6 |
