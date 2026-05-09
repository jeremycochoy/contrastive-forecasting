# Loss Extensions — Square Loss Design

## Goal

Compare the production 3-axis contrastive loss (`cosine_similarity_batch`) against a
new variant (`cosine_similarity_batch_square`) that adds the two missing **clean edges**
of the (batch × time) square of prediction pairs.

---

## Tensor layout

```
B, T, C, H  =  batch, time-steps, channels, hidden-dim

hy_hat_norm  =  fore_norm[:, :-1]   →  f_t      [B, T-1, C, H]   (forecaster at t)
hz_hat_norm  =  fore_norm[:,  1:]   →  f_{t+1}  [B, T-1, C, H]
hx_norm      =  orig_norm[:, :-1]   →  h_t      [B, T-1, C, H]   (encoder at t)
hy_norm      =  orig_norm[:,  1:]   →  h_{t+1}  [B, T-1, C, H]   (encoder at t+1)
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
| `neg_zy` | `Σ_{c'} cos(f_{t+1}^{b,c'}, f_t^{b,c})` | time | forecaster adjacent steps (covers same-channel for C=1) |
| `neg_cross_batch_forecast_embedding` | `Σ_{b'≠b} cos(h_{t+1}^{b'}, f_t^{b,c})` | **diagonal** | forecaster of b vs encoder-next of b' — see below |

The denominator is `negatives.sum(dim=0, keepdim=True)` — all batch elements pool their
negatives before the log, making every (t, c) slice a B-way retrieval task.

---

## The square

### Geometry

The "square" spans two axes:

- **batch axis** (rows b vs b')
- **time axis** (columns t-1 vs t)

Each vertex of the square is a prediction pair `(f_t, h_{t+1})`. Within each vertex,
`f_t` and `h_{t+1}` are drawn as two close nodes — this is the **thin "forecast vs
ground-truth" dimension**, which we expect to collapse to near-zero at convergence
(`f_t ≈ h_{t+1}` when the forecaster is perfect). The square geometry is what matters;
the per-vertex thickness is a training artefact.

```
        time t-1                          time t
  b   (f_{b,t-1}, h_{b,t})  ←────────  (f_{b,t}, h_{b,t+1})
             │                  ↗ diag      │
  b'  (f_{b',t-1}, h_{b',t}) ────────  (f_{b',t}, h_{b',t+1})
```

See `plots/square_diagram.png` for the full annotated diagram.

### Edge inventory

| edge | nodes | term | status |
|---|---|---|---|
| top / bottom (time axis) | f↔f, h↔h adj. t | `neg_zy` | already there |
| diagonal | `f_{b,t} ↔ h_{b',t+1}` (b≠b') | `neg_cross_batch_forecast_embedding` | already there — kept |
| **left / right (batch axis), f-side** | `f_{b,t} ↔ f_{b',t}` | **`neg_cross_batch_forecast`** | **NEW** |
| **left / right (batch axis), h-side** | `h_{b,t+1} ↔ h_{b',t+1}` | **`neg_cross_batch_embedding`** | **NEW** |

The diagonal `neg_cross_batch_forecast_embedding` = `f_{b,t} ↔ h_{b',t+1}` crosses both
the batch dimension (b→b') and the f/h dimension simultaneously. It is kept unchanged.
The two clean batch-axis edges are what this experiment adds.

---

## New terms: `neg_cross_batch_forecast` and `neg_cross_batch_embedding`

### `neg_cross_batch_forecast` — `f_{b,t} <> f_{b',t}`

```python
f_anchor = hy_hat_norm.unsqueeze(0)  # [1, B, T-1, C, H]
f_other  = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

sims_ff = cosine_similarity_from_normalized(f_anchor, f_other)  # [B, B, T-1, C]

mask_b = ~torch.eye(B, dtype=torch.bool, device=sims_ff.device).view(B, B, 1, 1)
neg_cross_batch_forecast = torch.exp(sims_ff / tau).masked_fill(~mask_b, 0).sum(dim=1)
# [B, T-1, C]
```

### `neg_cross_batch_embedding` — `h_{b,t+1} <> h_{b',t+1}`

```python
h_anchor = hy_norm.unsqueeze(0)  # [1, B, T-1, C, H]
h_other  = hy_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

sims_hh = cosine_similarity_from_normalized(h_anchor, h_other)  # [B, B, T-1, C]

mask_b = ~torch.eye(B, dtype=torch.bool, device=sims_hh.device).view(B, B, 1, 1)
neg_cross_batch_embedding = torch.exp(sims_hh / tau).masked_fill(~mask_b, 0).sum(dim=1)
# [B, T-1, C]
```

### Combined denominator

```python
negatives = (neg_xy + neg_xx + neg_zy + neg_xy_hat
             + neg_cross_batch_forecast_embedding   # existing diagonal, kept
             + neg_cross_batch_forecast             # NEW: f-side batch edge
             + neg_cross_batch_embedding)           # NEW: h-side batch edge
```

Loss shape name: **`cosine_similarity_batch_square`**.

---

## Hypothesis

The existing diagonal (`neg_cross_batch_forecast_embedding`) pushes `f_t^b` away from
`h_{t+1}^{b'}`, but forecasters and encoder states from different batches at the **same
time** are not repelled. Adding both clean batch-axis edges forces batch-discriminative
representations at each timestep on both the f-side and the h-side.

Expected direction: lower `U_batch`, no regression on discriminative metrics
(AUC, Top-1, R²).

---

## Possible follow-ups (not in scope here)

- **Remove the diagonal** (`neg_cross_batch_forecast_embedding`) and confirm it does not
  hurt performance — would tell us whether the diagonal or the clean edges are the
  load-bearing cross-batch signal.
- **Add the reverse diagonal** (`h_{b,t+1} ↔ f_{b',t}`) and more off-axis terms to
  test whether additional cross-type terms compound the gain.
