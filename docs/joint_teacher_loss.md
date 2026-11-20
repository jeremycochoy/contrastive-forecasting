# `--loss-shape cosine_similarity_batch_joint_teacher`

Script: `experiments/2026-04-27_freq-embedding/scripts/train.py`.
Function: `joint_teacher_term` in `src/loss.py`.

This shape replaces `L_align` and `L_rep` with one InfoNCE fraction for each
rollout depth. The forecast pair and the representation pair are both in the
numerator and in the denominator. Every h key and every h target comes from
the EMA teacher. One temperature τ divides every pair.

## The fraction

| symbol | tensor |
|---|---|
| `h` | the student encoder latent (`original_latent`) |
| `hT` | the EMA teacher latent (`teacher_original_latent`). It takes no gradient |
| `f^(0)` | the forecaster output (`forecasted_latent`) |
| `f^(j)` | the forecaster applied to its own output j more times (`rollout_latents[j-1]`) |

The loss normalizes each vector to unit length. τ is `--tau`, or the learned
τ of `--learnable-tau`.

At depth j, the anchors are the positions t = 0 .. T-2-j. The anchor
(b, t, c) holds two positives:

```
F_j = cos(f^(j)_{b,t,c}, hT_{b,t+1+j,c}) / τ      the forecast pair
R   = cos(h_{b,t,c},     hT_{b,t,c})     / τ      the representation pair
```

The negatives of the anchor come from five families:

| family | pairs |
|---|---|
| xx | `h_{b,t,c}` with `hT_{b,t,c'}`, c' ≠ c |
| hh_all | `h_{b,t,c}` with `hT_{b,l,c}`, l ≠ t |
| xs_allt | `h_{b,t,c}` with `hT_{b',l,c}`, b' ≠ b, every l |
| cross-batch | `f^(j)_{b,t,c}` with `hT_{b',t+1+j,c}`, b' ≠ b |
| next forecast | `f^(j)_{b,t,c}` with `f^(j)_{b,t+1,c'}`, every c' |

The first three are the families of `L_rep`. The last two are the families
of `L_pred` of the split shape. As in those terms, the denominator pools the
negatives of every real anchor at (t, c) over the rows of the batch:

```
total_j[t,c] = log Σ_b Σ_{negatives n of the anchor (b,t,c)} exp(n)

L_j = mean over the real anchors of
      logsumexp(F_j, R, total_j[t,c]) − logsumexp(F_j, R)
```

So L_j is the mean of −log((e^F + e^R) / (e^F + e^R + Σ e^n)). It is 0 or
more.

The three h families do not change with j. The loss computes them once, for
t = 0 .. T-2, and depth j reads the first T-1-j positions. xs_allt costs the
most, so the loss computes it once for each step.

## The rollout depths

`--train-rollout-depth k` builds the k + 1 copies L_0 .. L_k.
`--train-rollout-reduce` combines them: `sum`, `mean`, `halving`,
`harmonic` or `hill` ([`train_rollout_depth.md`](train_rollout_depth.md)). Each copy
holds both positives, so a reduction weighs whole copies.
`--train-rollout-distances` keeps the copies of the listed distances.

## Padding

The shape takes a row mask. With `--gift-pretrain` (#419) or the contrastive
objective by row (#412), no padded position enters the fraction as an
anchor, a key or a positive.

## Weights and add-ons

The fraction has weight 1. It reads no `--rep-loss-weight` and no
`--pred-loss-weight`. `L_align` is an add-on through `--align-loss-weight`,
as for the other shapes. At 0 the main loss is the fraction alone. SIGReg,
the CPC auxiliary and `align_moco` stay add-ons.

## What the trainer refuses

The trainer refuses these command lines before it builds the model. The
message lists each clash.

| command line | reason |
|---|---|
| no `--ema-embedding` and no `--ema-encoder` | every h key and target comes from the teacher |
| `--multi-patch-sizes` | the fraction runs on one patch size |
| `WORLD_SIZE > 1` | the fraction runs in one process |
| `--subtract-contrastive-floor` | the fraction has no floor |
| `--pos-in-denominator` | the positives are always in the denominator |
| `--moco-negatives`, `--moco-rep-keys` | the keys always come from the teacher |
| `--rep-loss-weight` other than 1, `--rep-loss-weight-end` | the fraction has weight 1 and reads no rep weight |
| `--pred-loss-weight` other than 1 | the fraction has weight 1 |
| `--tau-rep` with a value other than `--tau` | one τ divides every pair |
| `--forecaster-kind` other than `transformer` | the fraction reads one forecast for each position |
| `XSHH_ALLT_FUSED=1` or `XSHH_ALLT_SHARD=1` | these kernels take no padding mask |

## What it logs

| column of `<run>_losses.csv` | value |
|---|---|
| `l_joint` | L_0, the depth-0 copy, unweighted. Only on a run of this shape |
| `l_joint_d0` .. `l_joint_dk` | L_0 .. L_k, unweighted. The last columns, only at `--train-rollout-depth k` above 0 ([`train_rollout_depth.md`](train_rollout_depth.md#the-terms-at-each-depth)) |
| `loss_tau_ref` | L_0 at τ = 0.07, with no gradient |
| `cos_err_d*` | 1 − cos(f^(j)_t, h_{t+1+j}) of each depth. They read the latents, so they also work at `--align-loss-weight 0` |
| `l_pred`, `l_rep`, `rep_w` | blank |
| `l_align` | blank at `--align-loss-weight 0` |

At depth 0 with no add-on, `loss` equals `l_joint`. At depth k with
`--train-rollout-reduce sum` and no add-on, `loss` equals
`l_joint_d0 + ... + l_joint_dk`.

## The xx family

`L_rep` with MoCo keys calls `_xx_lse_masked(student, teacher)`. At channel
c, that call sums over the student channels c' ≠ c against the teacher key
of channel c. The joint fraction calls it with the teacher first. At channel
c it then sums over the teacher keys c' ≠ c of the student anchor of
channel c. This is the xx row of the table above. With one channel the
family is empty, and the two calls give the same loss.
