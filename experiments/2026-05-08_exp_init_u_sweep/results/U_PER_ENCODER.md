# U_b at init across encoder variants (B=256, default PyTorch init)

Inputs: `torch.randn(256, 4096, 1) * 0.5`, `freq_ids ~ U{0..3}`, `seas_ids ~ U{0..3}`,
RevEWMA span=128 normalisation, freq_emb_dim=3, seasonality_emb_dim=3 → encoder
input width = 22. Encoder output is `[B, T, C, H]` with H=384, T=4096/16=256.
U_b = `dim_usage(o_lat, axis=0)`. Effective rank = U_b · 384. 3 model seeds {42,43,44}.

## Encoder-output U_b (mean ± std over 3 seeds)

| encoder_type   | U_b (mean ± std) | eff_rank (mean ± std) |
|----------------|------------------|------------------------|
| mlp            | 0.0408 ± 0.0013  | 15.65 ± 0.49           |
| mlp_wide       | 0.0430 ± 0.0002  | 16.53 ± 0.06           |
| residual_silu  | 0.0426 ± 0.0001  | 16.36 ± 0.03           |
| gru            | 0.0425 ± 0.0001  | 16.33 ± 0.02           |
| conv           | 0.0423 ± 0.0003  | 16.24 ± 0.11           |

## Per-stage probe — `residual_silu` (3 seeds)

| stage             | feature_dim | U_b (mean ± std) | eff_rank (mean ± std) |
|-------------------|-------------|------------------|------------------------|
| input_concat      | 22          | 0.7267 ± 0.0002  | 15.99 ± 0.00           |
| proj_out          | 384         | 0.0426 ± 0.0001  | 16.36 ± 0.04           |
| mlp_pre_silu      | 384         | 0.0410 ± 0.0002  | 15.73 ± 0.06           |
| mlp_post_silu     | 384         | 0.0431 ± 0.0004  | 16.56 ± 0.16           |
| mlp_out           | 384         | 0.0335 ± 0.0015  | 12.87 ± 0.56           |
| h_after_residual  | 384         | 0.0426 ± 0.0002  | 16.35 ± 0.06           |
| skip_out          | 384         | 0.0429 ± 0.0002  | 16.46 ± 0.07           |
| pre_norm          | 384         | 0.0426 ± 0.0001  | 16.36 ± 0.03           |
| encoder_out       | 384         | 0.0426 ± 0.0001  | 16.36 ± 0.03           |

## Facts

**Highest encoder_out eff_rank.** `mlp_wide` ranks first at 16.53 ± 0.06,
followed by `residual_silu` at 16.36 ± 0.03, then `gru` at 16.33 ± 0.02,
then `conv` at 16.24 ± 0.11, with `mlp` last at 15.65 ± 0.49. The gap
between `residual_silu` and the GRU baseline is +0.03 in eff_rank
(≈8e-5 in U_b) — within seed-to-seed noise. `mlp_wide` exceeds the GRU
by +0.20 in eff_rank. `mlp` is the only encoder noticeably below GRU
(−0.68 eff_rank, also the highest variance across seeds at std 0.49).
At default init, none of the encoders preserve substantially more rank
than the GRU at the encoder output; all five sit in the 15.65–16.53
range and effective rank is dominated by the input concat's eff_rank
of 15.99.

**Per-stage `residual_silu`.** The 22-D `input_concat` has
U_b = 0.7267 (eff_rank 15.99). `proj_out = Linear(W→H)(x)` lifts the
representation into 384-D and lands at eff_rank 16.36 (U_b 0.0426); the
linear projection cannot increase rank above the input's 15.99 except
by the small contribution of the bias and the resulting renormalisation
inside `dim_usage`. `mlp_pre_silu` drops to eff_rank 15.73 (a slight
loss vs `proj_out`'s 16.36). `mlp_post_silu` rises to 16.56 — a measured
+0.82 lift across the SiLU non-linearity (mean of `mlp_post_silu` −
mean of `mlp_pre_silu` = 16.5578 − 15.7331), which does take the rank
above the input's 15.99 by ≈0.57. However `mlp_out` (the second linear
inside `self.mlp`) collapses to 12.87 ± 0.56, so the residual MLP branch
contributes a *lower-rank* update than the input itself. `h_after_residual`
(= `proj_out + mlp_out`) is 16.35, dominated by `proj_out`. `skip_out`
sits at 16.46 (a parallel `Linear(22→384)` of the same input).
`pre_norm` (= `h_after_residual + skip_out`) is 16.36, and `encoder_out`
(after LayerNorm) is unchanged at 16.36 — LayerNorm is a per-token
affine and does not move U_b on the batch axis. Net: the SiLU
non-linearity is the only stage that lifts rank above the input's
15.99, but the gain is washed out by the lower-rank second linear
(`mlp_out` ≈ 12.87) and the dominant skip/proj branches.
