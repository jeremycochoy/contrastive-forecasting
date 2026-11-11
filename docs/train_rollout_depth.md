# `--train-rollout-depth` — train the composed forecaster

Flag: `--train-rollout-depth K` on
`experiments/2026-04-27_freq-embedding/scripts/train.py`.
Config key: `train_rollout_depth` on `spec.train_configuration`.
Default: `0`, which is the historical objective, byte for byte.

Issue: [#373](https://github.com/jeremycochoy/contrastive-forecasting/issues/373).

## What it changes

At eval the forecaster rolls out on its own output. Training ties one step
only: `f_t ≈ h_{t+1}`.

The flag duplicates every loss term that ties `f` to `h`. Copy `j` ties
`f^(j)_t` to `h_{t+1+j}`, where `f^(j)` is the forecaster applied to its own
output `j` more times:

```
f^(0) = F(h)          f^(0)_t  ties to  h_{t+1}
f^(1) = F(f^(0))      f^(1)_t  ties to  h_{t+2}
...
f^(k) = F(f^(k-1))    f^(k)_t  ties to  h_{t+1+k}
```

`--train-rollout-reduce` says how the `k + 1` copies combine. The default is
the **sum**, which is what #373 ran: a `k = 3` run is the baseline plus three
added terms, not a re-weighted baseline. At depth `j` the anchors run
`t = 0 .. T-2-j`.

`F` is `TransformerBlock.forecaster_forward` — the operator the eval rollout
composes. Gradient flows through the whole chain. No detach between passes.

Two arguments let each caller keep the policy it always ran at:

| argument | training forward and rollout depth | eval `rollout_latent` |
|---|---|---|
| `fp32_tail` | `True` — last layer and output in fp32, what the contrastive loss reads. `BACKBONE_CKPT=1` still checkpoints the non-last layers | `False` — every layer at the ambient precision, no cast, no gradient-checkpointing |
| `cache_mask` | `True` — fixed `T`, so the cache hits every call | `False` — the sequence grows per token, so the cache would never hit and the write would leave module state behind |

## What it does not change

- **Eval.** `rollout_latent` and every eval strategy are untouched. It reads
  the same weights and writes nothing to the backbone.
- **`k = 0` runs.** Published `k = 0` numbers stay a valid baseline.
- **Terms that carry no `f`.** `L_rep`, `L_rep_moco`, `align_moco_loss`,
  SIGReg and the `− mse(h_t, h_{t+1})` half of `mse` enter the total once at
  any `k`, at their configured weight.

Under the sum, `k = 3` therefore gives the f-side four times its baseline
weight against the h-side terms. Watch `u_batchtime` on `h_t` and the loss
curve. `--train-rollout-reduce mean` removes that weight change — see the
next section.

## `--train-rollout-reduce {sum,mean,halving}`

Flag: `--train-rollout-reduce sum|mean|halving`. Config key: `train_rollout_reduce`.
Default: `sum`. Issue:
[#401](https://github.com/jeremycochoy/contrastive-forecasting/issues/401).

| reduction | total | f-side weight at depth `k` |
|---|---|---|
| `sum` (default) | `H + Σ_{j=0..k} F_j` | `k + 1` times its `k = 0` weight |
| `mean` | `H + (Σ_{j=0..k} F_j) / (k + 1)` | its `k = 0` weight, at every `k` |
| `halving` | `H + Σ_{j=0..k} F_j / 2^j` | `2 − 1/2^k` times its `k = 0` weight |

`F_j` is the depth-`j` copy of the terms that tie `f` to `h`. `H` is every
term that carries no `f`. The mean covers the copies only, so `H` enters
once at its configured weight under both reductions. The depth then changes
what the model trains on, and not how much the f-side outweighs the rest.

At `k = 0` there is one copy, so the three reductions return the same
number. Every published `k = 0` run reproduces under each word, and #373's
`k > 0` numbers reproduce under `sum`.

`halving` gives the depths 0 to 3 the weights 1, 1/2, 1/4 and 1/8, so the
nearest depths weigh the most. Depth 0 keeps the weight 1 of both sides, so
the halving needs no f-side pass: the copies of depths 1 to `k` are added to
the depth-0 total at their weights.

The mean costs one more pass over the f-bearing terms at depth 0. The
depth-0 call returns `H + F_0` added together, and the two sides must be
told apart before one of them is divided, so
`contrastive_latent_loss(..., f_terms_only=True)` reads `F_0` on the same
views. On a `cosine_similarity_batch_rep_only` cell that pass is the
`L_align` add-on alone, because the shape body returns zeros for an f-only
call.

Two limits:

- `contrastive_latent_noise` with `mean` raises. The f-side pass would draw
  its own noise, so `loss − F_0` would not be the h-side. No trainer sets
  that key.
- A word that is not `sum`, `mean` or `halving` raises and names itself, at
  the CLI and in `resolve_rollout_reduce`.
- `--value-space-objective` takes `sum` and `mean` only. `halving` raises at
  its first step and names itself.

## `--train-rollout-distances`

Flag: `--train-rollout-distances 1,2,4,8,16,32`, a list of integers with
commas. Config key: `train_rollout_distances` on `spec.train_configuration`.
Default: off. Then every depth `0..k` enters, as above.

The flag keeps the copies at the listed distances only. Distance `d` is the
copy that ties `f^(d-1)_t` to `h_{t+d}`, so `d = j + 1`:

| distance `d` | copy | ties |
|---|---|---|
| 1 | depth 0 | `f^(0)_t` to `h_{t+1}` |
| 2 | depth 1 | `f^(1)_t` to `h_{t+2}` |
| 4 | depth 3 | `f^(3)_t` to `h_{t+4}` |
| 32 | depth 31 | `f^(31)_t` to `h_{t+32}` |

The rollout composes every depth from 1 to `max(d) - 1`, because each depth
is the input of the next one. Gradient flows through the whole chain, as
above. Then the trainer puts `None` in place of each depth with a distance
that is not in the list. One function does this, `forecaster_depths` in
`train.py`, for the whole batch, for each patch-size group and for the raw
depths of the `l2_raw` `L_align`. So every term skips the same depths.

Each distance averages its pair losses over its own `T - d` pairs, the
anchors `t = 0 .. T-1-d`. At `T = 256` patches, distance 32 averages 224
pairs and distance 16 averages 240. For `L_align` the pair loss is
`2 - 2·cos(f^(d-1)_t, h_{t+d})`.

The reduction reads the copies at the listed distances only:

| reduction | f-side total |
|---|---|
| `sum` (default) | the sum of the copies at the listed distances |
| `mean` | that sum divided by the number of distances |

The `L_align` add-on of a `cosine_similarity_batch_rep_only` run with
`--align-loss-weight w` and the sum is `w` times a sum. The sum adds, for
each distance, the mean of `2 - 2·cos` at that distance. `L_rep` enters
once.

The losses CSV gets a `cos_err_d<j>` column for each listed distance only,
with `j = d - 1`. The list `1,2,4,8,16,32` writes `cos_err_d0`, `cos_err_d1`,
`cos_err_d3`, `cos_err_d7`, `cos_err_d15` and `cos_err_d31`.

### Checks

The trainer refuses these command lines before step 1, and the message
names the problem. The parse of the flag refuses a word that is not an
integer, and `resolve_rollout_distances` in `train.py` refuses the rest:

| command line | reason |
|---|---|
| a distance below 1, or a word that is not an integer | a distance is a positive integer |
| a list without 1 | distance 1 is depth 0, which enters every term |
| `--train-rollout-depth` other than 0 and `max(d) - 1` | the two flags give two different depths |
| `--train-rollout-reduce halving` | the halving weighs a copy by its place in the list, and the list leaves depths out |
| `--value-space-objective` | the value objective has no latent term |

`--train-rollout-depth 0`, the default, becomes `max(d) - 1`. Each check of
`--train-rollout-depth` then applies to that depth, `k + 2 ≤ T` included.

### Cost

The forecaster runs `max(d)` passes for each step: the forward pass and the
`max(d) - 1` rollout passes. The list `1,2,4,8,16,32` costs 32 passes, as
`--train-rollout-depth 31` does. The loss terms run once for each listed
distance, 6 times for this list.

## Where it applies

Every shape the `contrastive_latent_loss` dispatch accepts, plus `align_loss`,
`cpc_infonce_aux_loss` and `cpc_infonce_all_loss`.

One exception: `cpc_multistep` and `cpc_multistep_cpcnegs` raise
`NotImplementedError` at any `k > 0`. Their forecaster is `K` parallel linear
heads, `f^(k)_t = W_k h_t`, so there is no single operator to apply to its own
output. `--forecaster-kind cpc` / `linear_cpc` are refused up front for the
same reason. No other shape raises. A shape that carries no `f`, such as
`cosine_similarity_batch_rep_only`, adds exactly zero per depth — the
`L_align` add-on it pairs with still repeats, and a run that pairs it with
nothing is refused (see Guards).

## Guards

The trainer refuses three combinations up front. The last two would
otherwise train at `k = 0` while the CSV still wrote `k + 1` plausible
`cos_err_dj` curves, because the diagnostic reads the depth tensors, not the
loss.

| combination | reason |
|---|---|
| `--train-rollout-depth` below 0 | not a depth |
| `--forecaster-kind cpc` / `linear_cpc` | `K` parallel heads compose no operator to re-apply |
| a `k > 0` run with no term that ties `f` to `h` | the depths enter nothing |

Three terms can take a depth: the main contrastive term, `L_align`
(`--align-loss-weight`) and the CPC auxiliary (`--cpc-infonce-weight`).
SIGReg, `L_rep` and `align_moco_loss` carry no `f`. So the third guard fires
when both weights are 0 **and** the main term takes no depth either, which
happens three ways:

| route | why the main term adds zero per depth |
|---|---|
| `--no-main-contrastive-loss` | the term never runs |
| `--loss-shape cosine_similarity_batch_rep_only` | h-anchored end to end, so the depth copy returns zeros |
| `--loss-shape cosine_similarity_batch_split_pred_rep --pred-loss-weight 0` | `L_pred` is the f-bearing half. At weight 0 every depth copy is zero |

`train.py` states the rule once, in `main_term_depth_gap()` and
`rollout_depth_has_no_consumer()`, and the message names the route it
refused.

The guard refuses no cell of the study. Twelve of the 14 run `rep_only` on
the main arm with `--align-loss-weight` on, so their whole depth contribution
is the `L_align` add-on in the shared tail of `contrastive_latent_loss`. That
is a consumer. The other two run f-bearing shapes.

## What it does with the other flags

`--shard-loss-on-batch` picks the all-gather and nothing else. The depths are
built before that branch and reach the loss on both paths, so a sharded run
trains at the `k` it was given, over its own local shard. Each depth takes
one all-gather on the gathered path.

`--no-main-contrastive-loss` routes the depths through the standalone
`align_loss` and the CPC auxiliary instead of the main term. Those three
functions take the reduction as the `depth_reduce` argument, because they
read no training config. They carry no h-only half, so their mean is the
plain mean of the `k + 1` copies, and their halving adds copy `j` at
`1 / 2^j`. The trainer passes
`--train-rollout-reduce` to each of them.

`--align-loss-weight` on the main arm is a third path, and the one 12 of the
14 cells run: the shape body of a `rep_only` depth copy returns zeros, so the
whole depth contribution is the `L_align` add-on in the shared tail of
`contrastive_latent_loss`. All three paths have a run of the real trainer
behind them.

## What it logs

`k > 0` adds one column per depth to `<run>_losses.csv`:

| column | value |
|---|---|
| `cos_err_d0` | `1 − cos(f^(0)_t, h_{t+1})`, the same quantity as `1 − ff` |
| `cos_err_dj` | `1 − cos(f^(j)_t, h_{t+1+j})` |

A `k = 0` run writes no such column. The extra columns change the CSV
schema, so a `k = 0` run cannot resume into a `k > 0` CSV: the logger refuses
and asks for a fresh run name. Under `--train-rollout-distances` only the
listed distances get a column (see that section).

`loss_tau_ref` stays a depth-0 reference at every `k`, so that curve stays
comparable across runs.

## Cost

Each depth costs one forecaster forward and backward over `[B·C, T, H]`, plus
the negative families the depth rebuilds. `FCST_GRAD_CKPT=1` gradient-
checkpoints each depth's non-last forecaster layers when memory is tight.
