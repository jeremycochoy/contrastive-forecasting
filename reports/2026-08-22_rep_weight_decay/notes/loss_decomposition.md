# How to rebuild the loss from the CSV columns

The trainer's total loss closes with three terms:

```
loss = rep_w * l_rep
     + 2 * (cos_err_d0 + cos_err_d1 + cos_err_d2 + cos_err_d3)
     + sigreg_e + sigreg_h
```

Checked on `ctrl_s20` at step 8,500: 11.656 + 3.323 + 0.004 = 14.983, and the
`loss` column gives 14.983.

## The trap

The `l_align` column is the depth-0 copy alone, and it equals `2 * cos_err_d0`.
This card runs k = 3 under the `sum` reduction, so the loss holds four align
copies. A share computed from `l_align` gives 93 percent for `L_rep`. The true
share is 77 percent.

The 93 to 7 split that the issue card states comes from the k = 32 sweep under
the `mean` reduction. It does not hold on this cell. Do not carry that number
into the report.

## What the control arms show to step 8,500

| term | share of the loss | moves after step 500 |
|---|---|---|
| L_rep | 77 percent | no, flat at 11.66 |
| L_align, four copies | 23 percent | yes, 5.2 down to 3.3 |
| SIGReg | 0.03 percent | no, below 0.01 after step 2,000 |

The total loss falls from 17.0 to 15.2. L_align carries all of that fall.
