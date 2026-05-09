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

## The square and its diagonals

Each vertex is a prediction pair `(f_{b,t} ~ h_{b,t+1})` in (batch, time) space.
Going clockwise from top-left:

```
            f_t col                   h_{t+1} col
  b      f_{b,t}   ─── positive ───►  h_{b,t+1}
             │  \                         │
  LEFT EDGE  │   \ neg_cross_batch_       │  RIGHT EDGE
  (MISSING)  │    \  forecast_embedding   │  (MISSING)
             │     \  (diagonal, kept)    │
  b'     f_{b',t}  ─── positive ───►  h_{b',t+1}
```

### Edge / diagonal inventory

| name | expression | type | status |
|---|---|---|---|
| **positive** | `f_{b,t} ~ h_{b,t+1}` | edge (positive) | already there |
| `neg_zy` | `f_{b,t} <> f_{b,t+1}` | time edge | already there |
| `neg_cross_batch_forecast_embedding` | `f_{b,t} <> h_{b',t+1}` (b≠b') | **diagonal** | already there — kept unchanged |
| **`neg_cross_batch_forecast`** | `f_{b,t} <> f_{b',t}` (b≠b') | **left edge** | **NEW** |
| **`neg_cross_batch_embedding`** | `h_{b,t+1} <> h_{b',t+1}` (b≠b') | **right edge** | **NEW** |

### Diagonals deliberately NOT added in this experiment

The square has two diagonals. We keep the existing one
(`neg_cross_batch_forecast_embedding` = `f_{b,t} <> h_{b',t+1}`) unchanged.
We are not adding the reverse or any other cross-time/cross-batch diagonals, e.g.:

- `h_{b,t+1} <> f_{b',t}` — reverse of the existing diagonal
- `f_{b,t} <> h_{b',t}` — cross-batch same-time f vs h (not time-shifted)
- `h_{b,t+1} <> f_{b',t+1}` — cross-batch different-side, shifted the other way

### Why "square" (and future "cube")

The positive pair `f_t^b ~ h_{t+1}^b` contracts each `(f, h)` pair to a single point
in embedding space at convergence, so each vertex of the (batch × time) grid collapses
to a point. This lets us reason about the loss geometry as a 2-D square diagram even
though we live in a high-dimensional embedding space.

The square only spans the (batch, time) axes. Extending to (batch, time, channel) would
give a **cube** — but for our current C=1 setup the channel axis is degenerate, so the
square diagram is sufficient for now. The cube is a later experiment.

---

## New terms: `neg_cross_batch_forecast` and `neg_cross_batch_embedding`

### `neg_cross_batch_forecast` — left edge `f_{b,t} <> f_{b',t}`

Pushes each forecaster output away from every other batch element's forecaster at the
**same time t**. Symmetric to the existing diagonal, but on the f–f side.

```python
f_anchor = hy_hat_norm.unsqueeze(0)  # [1, B, T-1, C, H]
f_other  = hy_hat_norm.unsqueeze(1)  # [B, 1, T-1, C, H]

sims_ff = cosine_similarity_from_normalized(f_anchor, f_other)  # [B, B, T-1, C]

mask_b = ~torch.eye(B, dtype=torch.bool, device=sims_ff.device).view(B, B, 1, 1)
neg_cross_batch_forecast = torch.exp(sims_ff / tau).masked_fill(~mask_b, 0).sum(dim=1)
# [B, T-1, C]
```

### `neg_cross_batch_embedding` — right edge `h_{b,t+1} <> h_{b',t+1}`

Pushes each encoder-next embedding away from every other batch element's encoder-next
at the **same time t+1**. Symmetric to the existing diagonal, but on the h–h side.

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
             + neg_cross_batch_forecast             # NEW: left edge
             + neg_cross_batch_embedding)           # NEW: right edge
```

Loss shape name: **`cosine_similarity_batch_square`**.

---

## Hypothesis

`neg_cross_batch_forecast_embedding` (diagonal) already pushes `f_t^b` away from
`h_{t+1}^{b'}`, but the two clean batch-axis edges are absent: forecasters from
different batches at the same time are not repelled, and neither are encoder states from
different batches at the same time. Adding both clean edges should force batch-discriminative
representations at each timestep on both the f-side and the h-side.

Expected direction: lower `U_batch`, no regression on discriminative metrics
(AUC, Top-1, R²).

---

## Possible follow-ups (not in scope here)

- **Remove the diagonal** (`neg_cross_batch_forecast_embedding`) from the baseline and
  confirm it does not hurt performance — would tell us whether the diagonal or the clean
  edges are load-bearing.
- **Add more diagonals** (e.g. the reverse `h_{t+1}^b <> f_t^{b'}`) to see whether
  off-axis terms compound the gain.
- Both follow-ups make sense only after confirming that the square is better than the
  baseline, so they are deferred.
