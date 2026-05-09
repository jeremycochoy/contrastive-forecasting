# Loss Extensions — Square Loss Design

## Goal

Compare the production 3-axis contrastive loss (`cosine_similarity_batch`) against a
new variant that adds the one missing edge of the (batch × time) square of prediction
pairs: `f_{b,t} <> f_{b',t}` (cross-batch forecaster vs forecaster at the same time).

---

## Tensor layout

```
B, T, C, H  =  batch, time-steps, channels, hidden-dim

hy_hat_norm  =  fore_norm[:, :-1]   →  f_t      [B, T-1, C, H]
hz_hat_norm  =  fore_norm[:,  1:]   →  f_{t+1}  [B, T-1, C, H]
hx_norm      =  orig_norm[:, :-1]   →  h_t      [B, T-1, C, H]
hy_norm      =  orig_norm[:,  1:]   →  h_{t+1}  [B, T-1, C, H]
```

---

## Terms in `cosine_similarity_batch`

### Positive

| expression | meaning |
|---|---|
| `cos(h_{t+1}^{b,c}, f_t^{b,c})` | `f_t ~ h_{t+1}` — forecaster at t predicts encoder at t+1, same (b, c) |

### Negatives (all shape `[B, T-1, C]` before the cross-batch `.sum(dim=0)`)

| variable | expression | axis | meaning |
|---|---|---|---|
| `neg_xy` | `Σ_{c'} cos(h_t^{b,c'}, h_{t+1}^{b,c})` | channel | encoder now vs encoder next, cross-channel |
| `neg_xx` | `Σ_{c'≠c} cos(h_t^{b,c'}, h_t^{b,c})` | channel | encoder now vs encoder now, cross-channel (off-diagonal) |
| `neg_xy_hat` | `Σ_{c'} cos(h_t^{b,c'}, f_t^{b,c})` | channel | encoder now vs forecaster now, cross-channel |
| `neg_zy` | `Σ_{c'} cos(f_{t+1}^{b,c'}, f_t^{b,c})` | time | forecaster adjacent steps, cross-channel (covers same-channel for C=1) |
| `neg_cross_batch` | `Σ_{b'≠b} cos(h_{t+1}^{b',c}, f_t^{b,c})` | **batch** | h-side cross-batch: other batch's encoder-next vs our forecaster |

The denominator is `negatives.sum(dim=0, keepdim=True)` — all batch elements pool their
negatives before the log, making every (t, c) slice a B-way retrieval task.

---

## The square

Each vertex is a prediction pair `(f_{b,t}, h_{b,t+1})` in (batch, time) space.
Going clockwise from top-left:

```
        t-1                          t
  b   [f_{b,t-1} ~ h_{b,t}]  ←───  [f_{b,t} ~ h_{b,t+1}]
            |                               |
            |  neg_cross_batch (h-side)     |  ← MISSING
            |                               |
  b'  [f_{b',t-1} ~ h_{b',t}] ───→  [f_{b',t} ~ h_{b',t+1}]
```

### Edge coverage

| edge | relationship | covered by |
|---|---|---|
| top / bottom (time axis) | `f_t <> f_{t-1}` same batch | `neg_zy` |
| left (batch axis, time t-1) | `h_{b,t} <> h_{b',t}` | `neg_cross_batch` (h-side) |
| **right (batch axis, time t)** | **`f_{b,t} <> f_{b',t}`** | **MISSING** |

---

## New term: `neg_cross_batch_f`

Symmetric counterpart to `neg_cross_batch` but on the **f-side**: push each forecaster
output away from every other batch element's forecaster at the **same time t**.

```python
# hy_hat_norm: [B, T-1, C, H]
hy_hat_norm_anchor = hy_hat_norm.unsqueeze(0)  # [1, B, T-1, C, H]
hy_hat_norm_other  = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

sims_cross_batch_f = cosine_similarity_from_normalized(
    hy_hat_norm_anchor, hy_hat_norm_other
)  # [B, B, T-1, C]

mask_batch_f = ~torch.eye(B, dtype=torch.bool, device=sims_cross_batch_f.device)
mask_batch_f = mask_batch_f.view(B, B, 1, 1)

neg_cross_batch_f = (
    torch.exp(sims_cross_batch_f / tau)
    .masked_fill(~mask_batch_f, 0)
    .sum(dim=1)
)  # [B, T-1, C]
```

Added to the denominator:
```python
negatives = neg_xy + neg_xx + neg_zy + neg_xy_hat + neg_cross_batch + neg_cross_batch_f
```

Loss shape name: **`cosine_similarity_batch_square`**.

---

## Hypothesis

`neg_cross_batch` already pushes `f_t^b` away from `h_{t+1}^{b'}` (other batch's
encoder); `neg_cross_batch_f` pushes `f_t^b` away from `f_t^{b'}` (other batch's
forecaster at the same time). This closes the right edge of the square, forcing the
forecaster to produce batch-discriminative outputs at each timestep — not just
encoder-discriminative.

Expected direction: tighter batch-axis spreading (lower `U_batch`) without regressing
`U_temporal` or the discriminative metrics (AUC, Top-1, R²).
