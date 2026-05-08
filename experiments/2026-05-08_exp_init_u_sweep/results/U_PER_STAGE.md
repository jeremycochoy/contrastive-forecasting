# Per-stage U_b inside GRUEncoder at init

**Setup.** `ConfigurableModel(C=1, H=384, W=16, encoder_type="gru",
num_layers=6, nhead=6, ffn_mult=4.0, depthwise_conv=3, dropout=0.0,
freq_emb_dim=3, seasonality_emb_dim=3, rev_norm_kind="ewma",
rev_norm_span=128)`, CPU, eval mode, B=256, T_raw=4096, n=3 seeds
{42, 43, 44}. Forward path instrumented by replacing
`GRUEncoder.forward` with a line-for-line copy that stashes every
intermediate tensor (input concat, GRU output, proj_out, skip_out,
sum_pre_norm, encoder_out). For each stage we reshape so batch axis is 0
and compute `U_b = dim_usage(z, axis=0)`; `effective_rank = U_b *
feature_dim`. GRU internal gate non-linearities (tanh / sigmoid) are
buried in PyTorch's optimised GRU kernel and not directly accessible
without re-implementing GRU step-by-step — skipped per task spec.
GRUEncoder has no optional non-linearity between proj and add.

## Default init (mean over 3 seeds)

| stage          | feature_dim | U_b    | effective_rank |
|:---------------|------------:|-------:|---------------:|
| input_concat   |          22 | 0.7267 |          15.99 |
| gru_output     |         256 | 0.0042 |           1.06 |
| proj_out       |         384 | 0.0027 |           1.04 |
| skip_out       |         384 | 0.0427 |          16.40 |
| sum_pre_norm   |         384 | 0.0425 |          16.32 |
| encoder_out    |         384 | 0.0425 |          16.33 |

## Orthogonal init on `proj` and `skip` (gain=√2, zero bias)

| stage          | feature_dim | U_b    | effective_rank |
|:---------------|------------:|-------:|---------------:|
| input_concat   |          22 | 0.7267 |          15.99 |
| gru_output     |         256 | 0.0042 |           1.06 |
| proj_out       |         384 | 0.0028 |           1.06 |
| skip_out       |         384 | 0.0416 |          15.99 |
| sum_pre_norm   |         384 | 0.0434 |          16.65 |
| encoder_out    |         384 | 0.0433 |          16.64 |

## Where rank drops (data only)

The drop is at **`gru_output`**: mean effective rank goes from 15.99
at `input_concat` (d=22) to 1.06 at `gru_output` (d=256). `proj_out`
(d=384) sits at eff_rank 1.04 under default init and 1.06 under ortho
— same order as `gru_output`. `skip_out` is at eff_rank 16.40
(default) / 15.99 (ortho); `sum_pre_norm` and `encoder_out` are at
16.32 / 16.33 (default) and 16.65 / 16.64 (ortho). Orthogonal init on
`proj` and `skip` leaves `gru_output` unchanged at 1.06; `proj_out`
moves from 1.04 to 1.06 (+0.03); `skip_out` moves from 16.40 to 15.99
(matching `input_concat` to 4 decimals). `encoder_out` gains a mean
+0.31 in eff_rank (16.33 → 16.64; per-seed Δ ∈ {+0.29, +0.30, +0.35}).

## Where to target an init change

Bounded by these data: the only stage where current init produces a
collapsed rank (eff_rank ≈ 1) is the **GRU itself** (`gru_output`,
which `proj_out` then inherits); changing `proj` / `skip` init alone
moves `proj_out` by only ~0.03 in eff_rank and `encoder_out` by
~+0.31, so a meaningful lift in encoder-output rank requires init
changes that affect `gru_output`.
