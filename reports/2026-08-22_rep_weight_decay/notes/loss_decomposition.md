# How to rebuild the loss by term from the CSV columns

This card runs k = 32 under the `mean` reduction, against the EMA **teacher**.
The formula below holds on that cell. It does not hold on a `sum` cell or on a
student-target cell.

## The formula

```
loss = rep_w * l_rep
     + align_w * mean(L_align_d0 .. L_align_d32)
     + sigreg_e + sigreg_h
```

The cell's `align_w` is 1.0, its CPC weight is 0.0, and both SIGReg weights are
1.0. So the align part is a residual of columns the CSV holds:

```
L_align, reduced = loss - rep_w * l_rep - sigreg_e - sigreg_h
```

`scripts/plot_loss_terms.py` draws that residual, and
`tests/test_409_score_pipeline.py::TestTheLossByTermFormula` pins it.

## Two traps

**The `l_align` column is the depth-0 copy alone.** The loss holds the MEAN of
33 copies, one for each rollout depth. A share computed from `l_align` reads
one copy as all 33.

**`l_align` is not `2 * cos_err_d0` here.** That identity holds under
`--align-target student`. This cell aligns on the teacher, so the align term
reads the teacher's next latent and `cos_err_dj` reads the student's. The
`cos_err_d*` columns therefore cannot rebuild the align part. #404's own
`plot_loss_terms.py` used the identity on a teacher run.

`l_rep` goes blank at weight 0.0, where the trainer computes no L_rep. The
residual closes there with `rep_w * l_rep = 0`.

## The share the card states

The issue card states that L_rep holds 93 percent of the total at step 40,000
and that L_align holds the other 7. Both numbers come from this cell, measured
on the sweep's own losses CSVs. Read them again from this card's arms before
you quote them: the sweep measured them at weight 1.0, and every arm here
decays the weight to 0.0.
