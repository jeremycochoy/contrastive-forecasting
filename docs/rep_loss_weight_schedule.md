# `--rep-loss-weight-end` — decay the weight on `L_rep`

Flags: `--rep-loss-weight`, `--rep-loss-weight-end` and
`--rep-loss-weight-ramp-steps` on
`experiments/2026-04-27_freq-embedding/scripts/train.py`.
Config key: `rep_loss_weight` on `spec.train_configuration`, and the
`rep_loss_weight` argument of `contrastive_latent_loss`.
Default: no end value, which holds the weight at `--rep-loss-weight` and is
the historical objective, byte for byte.

Issue: [#409](https://github.com/jeremycochoy/contrastive-forecasting/issues/409).

## What `L_rep` is

`L_rep` is the h-anchored term of the objective: a pooled logsumexp over three
families of negatives, with no positive of its own unless `--moco-rep-keys`
adds one. It pushes the representations apart. Two loss shapes carry it:

| `--loss-shape` | what `L_rep` is |
|---|---|
| `cosine_similarity_batch_split_pred_rep` | one of two terms, beside `L_pred` |
| `cosine_similarity_batch_rep_only` | the whole main loss |

Every other shape ignores the weight, and the training script refuses a
schedule on one of them.

## What the schedule changes

The weight falls linearly from `--rep-loss-weight` at step 0 to
`--rep-loss-weight-end`, then holds:

```
--rep-loss-weight 1.0 --rep-loss-weight-end 0.0 --rep-loss-weight-ramp-steps 10000

  w(0)      = 1.0
  w(5000)   = 0.5
  w(10000)  = 0.0
  w(40000)  = 0.0
```

Give `--rep-loss-weight-ramp-steps` when the run trains in legs. A ladder
resumes each leg with a new `--total-steps`, and without the anchor each leg
ramps over its own budget, so no two stops sit on one curve. This is the
contract of `--ema-tau-ramp-steps`, and both flags call
`src.models.linear_schedule_at_step`.

At weight 0.0 the term is skipped whole: no Gram matmul, no chunked
cross-series accumulator, and no gradient. The run then trains on the add-ons
alone — `L_align`, SIGReg, the CPC auxiliary and `align_moco`.

## Warning: at weight 0.0 nothing pushes the representations apart

`L_rep` carries the negatives of this objective. The training script refuses
`--rep-loss-weight-end 0.0` when the run keeps no other term with a gradient,
because the backward pass would have nothing to differentiate. It cannot
refuse a collapse, so watch these columns of `<run>_losses.csv`:

- `auc`, the contrastive separation of a positive from a negative. A fall
  toward 0.5 is a lost contrastive task.
- `u_batchtime`, the dimension usage of `h_t`.
- `l_rep`, which goes blank at the step the term switches off.

## The per-term columns

`<run>_losses.csv` carries four columns that #409 added:

| column | value |
|---|---|
| `rep_w` | the live weight on `L_rep`. Blank for a shape that reads no rep weight |
| `l_pred` | `L_pred`, unweighted |
| `l_rep` | `L_rep`, unweighted. Blank at weight 0.0, where the term is not computed |
| `l_align` | `L_align`, unweighted, the depth-0 copy |

The three term columns hold the raw term. The weights come from the command
line, which is the convention of the `sigreg_e`, `sigreg_h` and `cpc_aux`
columns. So the total of a `rep_only` run reads:

```
loss = rep_w * l_rep + align_w * l_align + w_e * sigreg_e + w_h * sigreg_h + cpc_w * cpc_aux
```

`l_align` is the depth-0 copy. On a `--train-rollout-depth` run the
`cos_err_d*` columns carry the other depths. Under `--align-target student`
the term reads the same pair as `cos_err_d0`, so `l_align = 2 * cos_err_d0`.
Under `--align-target teacher` the term reads the teacher's next latent and
`cos_err_d0` reads the student's, so the two are different numbers.

The column is blank under `--no-main-contrastive-loss`, where the training
script adds `L_align` outside the main loss.

Before #409 the CSV held the total alone. A report had to read `L_rep` as the
residual of every other term, which put the error of all of them into one
panel. The same script read `L_align` as `2 * mean(cos_err_d0 .. cos_err_dk)`
on a teacher-target run, where that identity does not hold — see
`reports/2026-08-19_ema_momentum_k32/scripts/plot_loss_terms.py`.

## What it does not change

- **A run with no end value.** The weight holds at `--rep-loss-weight` and
  every published number stays a valid baseline.
- **The `loss_tau_ref` diagnostic.** The schedule travels as a function
  argument of `contrastive_latent_loss`, and the diagnostic reads the config
  key, so its reference curve holds the run's fixed base weight.
- **Eval.** The schedule changes the training objective only.

## Resume

The four new columns change the header of `<run>_losses.csv`. The schema guard
refuses to append rows to a CSV that a pre-#409 trainer wrote. To resume such a
run, move the old CSV aside or take a fresh `--run-name`.
