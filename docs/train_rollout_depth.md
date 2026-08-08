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

The depths are **summed** on top of the `k = 0` term. A `k = 3` run is the
baseline plus three added terms, not a re-weighted baseline. At depth `j` the
anchors run `t = 0 .. T-2-j`.

`F` is `TransformerBlock.forecaster_forward` — the operator the eval rollout
composes. Gradient flows through the whole chain; no detach between passes.

Two arguments let each caller keep the policy it always ran at:

| argument | training forward and rollout depth | eval `rollout_latent` |
|---|---|---|
| `fp32_tail` | `True` — last layer and output in fp32, what the contrastive loss reads | `False` — every layer at the ambient precision, no cast |
| `cache_mask` | `True` — fixed `T`, so the cache hits every call | `False` — the sequence grows per token, so the cache would never hit and the write would leave module state behind |

## What it does not change

- **Eval.** `rollout_latent` and every eval strategy are untouched. It reads
  the same weights and writes nothing to the backbone.
- **`k = 0` runs.** Published `k = 0` numbers stay a valid baseline.
- **Terms that carry no `f`.** `L_rep`, `L_rep_moco`, `align_moco_loss`,
  SIGReg and the `− mse(h_t, h_{t+1})` half of `mse` enter the total once at
  any `k`, at their configured weight.

At `k = 3` the f-side therefore carries four times its baseline weight
against the h-side terms. Watch `u_batchtime` on `h_t` and the loss curve.

## Where it applies

Every shape the `contrastive_latent_loss` dispatch accepts, plus `align_loss`,
`cpc_infonce_aux_loss` and `cpc_infonce_all_loss`.

One exception: `cpc_multistep` and `cpc_multistep_cpcnegs` raise
`NotImplementedError` at any `k > 0`. Their forecaster is `K` parallel linear
heads, `f^(k)_t = W_k h_t`, so there is no single operator to apply to its own
output. `--forecaster-kind cpc` / `linear_cpc` are refused up front for the
same reason. No other shape raises. A shape that carries no `f`, such as
`cosine_similarity_batch_rep_only`, adds exactly zero per depth.

## What it logs

`k > 0` adds one column per depth to `<run>_losses.csv`:

| column | value |
|---|---|
| `cos_err_d0` | `1 − cos(f^(0)_t, h_{t+1})`, the same quantity as `1 − ff` |
| `cos_err_dj` | `1 − cos(f^(j)_t, h_{t+1+j})` |

A `k = 0` run writes no such column. The extra columns change the CSV
schema, so a `k = 0` run cannot resume into a `k > 0` CSV: the logger refuses
and asks for a fresh run name.

`loss_tau_ref` stays a depth-0 reference at every `k`, so that curve stays
comparable across runs.

## Cost

Each depth costs one forecaster forward and backward over `[B·C, T, H]`, plus
the negative families the depth rebuilds. `FCST_GRAD_CKPT=1` gradient-
checkpoints each depth's non-last forecaster layers when memory is tight.
