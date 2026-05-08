# U_b under orthogonal-subspace inits for ResidualSiLUEncoder

Setup: `ConfigurableModel(encoder_type="residual_silu", W=16, H=384, freq=3,
seas=3, dropout=0, rev_norm=ewma span=128)`, B=256, T_raw=4096, CPU, eval mode,
3 seeds (42, 43, 44). Synthetic input identical to `u_per_stage.py` (input
seed = model_seed + 10_000, INPUT_SCALE=0.5). For each scheme, the orthogonal
basis Q is sampled per-seed via `torch.linalg.qr(torch.randn(384, k))` with
generator seed = `model_seed + 99_000`, and proj/skip column slices are scaled
by `sqrt(1/22)`.

CSV: `init_u_orthogonal_subspaces.csv` (108 rows = 4 schemes x 3 seeds x 9
stages). All numbers below are means over the 3 seeds.

### default
| stage | feature_dim | U_b | eff_rank |
|---|---|---|---|
| input_concat | 22 | 0.7267 | 15.99 |
| proj_out | 384 | 0.0426 | 16.36 |
| mlp_pre_silu | 384 | 0.0410 | 15.73 |
| mlp_post_silu | 384 | 0.0431 | 16.56 |
| mlp_out | 384 | 0.0335 | 12.87 |
| h_after_residual | 384 | 0.0426 | 16.35 |
| skip_out | 384 | 0.0429 | 16.46 |
| pre_norm | 384 | 0.0426 | 16.36 |
| encoder_out | 384 | 0.0426 | 16.36 |

### ortho_skip_proj_orthogonal_images
| stage | feature_dim | U_b | eff_rank |
|---|---|---|---|
| input_concat | 22 | 0.7267 | 15.99 |
| proj_out | 384 | 0.0416 | 15.99 |
| mlp_pre_silu | 384 | 0.0077 | 2.95 |
| mlp_post_silu | 384 | 0.0077 | 2.95 |
| mlp_out | 384 | 0.0029 | 1.12 |
| h_after_residual | 384 | 0.0183 | 7.01 |
| skip_out | 384 | 0.0416 | 15.99 |
| pre_norm | 384 | 0.0312 | 11.97 |
| encoder_out | 384 | 0.0311 | 11.95 |

### ortho_skip_proj_only_random_mlp
| stage | feature_dim | U_b | eff_rank |
|---|---|---|---|
| input_concat | 22 | 0.7267 | 15.99 |
| proj_out | 384 | 0.0416 | 15.99 |
| mlp_pre_silu | 384 | 0.0077 | 2.95 |
| mlp_post_silu | 384 | 0.0077 | 2.95 |
| mlp_out | 384 | 0.0029 | 1.12 |
| h_after_residual | 384 | 0.0183 | 7.01 |
| skip_out | 384 | 0.0416 | 15.99 |
| pre_norm | 384 | 0.0312 | 11.97 |
| encoder_out | 384 | 0.0311 | 11.95 |

(Identical numbers to `ortho_skip_proj_orthogonal_images` because the two
schemes apply the same weight surgery — the mlp is left at default in both.)

### ortho_all_three_subspaces
| stage | feature_dim | U_b | eff_rank |
|---|---|---|---|
| input_concat | 22 | 0.7267 | 15.99 |
| proj_out | 384 | 0.0416 | 15.99 |
| mlp_pre_silu | 384 | 0.0075 | 2.87 |
| mlp_post_silu | 384 | 0.0075 | 2.87 |
| mlp_out | 384 | 0.0092 | 3.54 |
| h_after_residual | 384 | 0.0418 | 16.05 |
| skip_out | 384 | 0.0416 | 15.99 |
| pre_norm | 384 | 0.0417 | 16.02 |
| encoder_out | 384 | 0.0417 | 16.02 |

## Findings (facts only)

**1) Did `ortho_skip_proj_orthogonal_images` lift `encoder_out` eff_rank above
default?** No. Default `encoder_out` eff_rank = 16.36; under
`ortho_skip_proj_orthogonal_images` it is 11.95, a decrease of 4.41
(11.95 - 16.36 = -4.41; 11.95 / 16.36 = 0.730). The hypothesised lift to ~32
is not observed. The downstream stages show that under this scheme `mlp_out`
collapses to eff_rank 1.12 (from 12.87 default) and `h_after_residual` to
7.01 (from 16.35 default), so the proj+mlp branch contributes a near-rank-1
signal, while `skip_out` stays at 15.99.

**2) For `ortho_all_three_subspaces`: did adding orthogonality on the mlp
output further lift eff_rank?** Relative to
`ortho_skip_proj_orthogonal_images`, yes: `encoder_out` rose from 11.95 to
16.02 (+4.07), `h_after_residual` from 7.01 to 16.05 (+9.03), and `mlp_out`
from 1.12 to 3.54 (+2.43). Relative to the default 16.36, however,
`encoder_out` is still 0.34 lower (16.02 vs 16.36). The CSV shows no
orthogonal-subspace scheme reaching the default `encoder_out` eff_rank, and
none approached the hypothesised ~32.

## Best scheme

The scheme with the highest `encoder_out` eff_rank at init was **`default`**
at 16.36. Among the orthogonal-subspace schemes, the best was
`ortho_all_three_subspaces` at 16.02, which is 0.34 below `default`. Across
all four schemes, the spread of `encoder_out` eff_rank is 11.95 to 16.36
(width 4.41). The hypothesis that orthogonal proj/skip image subspaces lift
`encoder_out` eff_rank to ~32 is not supported by these measurements.
