# exp_init_u_sweep — init scheme sweep for encoder-latent dimension usage

## Goal

In training the contrastive-forecasting backbone we observe `U_b ≈ 0.003` on
the encoder latent `o_lat` after just 100 steps — i.e. the encoder's outputs
are nearly collinear across the batch axis right out of the gate. Backbone-beta
eventually converges to `U_b = 0.0762` at step 167k, but the journey starts
from a nearly-degenerate point. This experiment asks whether a different
parameter init can lift `U_b(o_lat)` at step 0 (no training), so the backbone
starts from a less-collapsed point.

## Recipe

Build `ConfigurableModel(C=1, H=384, W=16, encoder_type="gru", num_layers=6,
nhead=6, ffn_mult=4.0, activation="gelu", depthwise_conv=3, dropout=0.0,
freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind="ewma",
rev_norm_span=128)`, run a single forward pass on synthetic gaussian input
(`B=32, T_raw=4096, C=1`, scaled by 0.5), and measure
`U_t`, `U_b` on both the encoder-latent `o_lat` (post-patch-head, pre-transformer)
and the forecaster latent `f_lat` (post-transformer). The patch head is a
GRU(2 layers, bidirectional, hidden=128) → Linear(256→384) plus a
Linear(22→384) skip path, with a final LayerNorm.

We compare 8 init schemes (default PyTorch, plus 7 alternatives that touch
patch-head, transformer linears, or both). Each scheme is run 3 times with
different model seeds (42, 43, 44); we report mean ± std on each U metric.
Results land in `results/init_u_sweep.csv` (aggregated) and
`results/init_u_sweep_raw.csv` (per-seed). `results/RESULTS.md` summarises and
picks a winner. Total runtime: ~2-3 min on CPU.
